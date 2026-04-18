# rl_server.py
# pip install flask torch numpy pyyaml
from flask import Flask, request, jsonify
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, math
import time, csv, logging
import os, platform, subprocess, shutil
import warnings
from collections import deque, defaultdict
from config_manager import config

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if platform.system() != "Linux":
    warnings.filterwarnings(
        "ignore",
        message=r".*record_context_cpp is not support.*",
        category=UserWarning,
    )

class PerformanceTracker:
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.action_counts = defaultdict(int)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_length = 0
        self.current_episode_reward = 0.0
        self.step_start_time = None
        
    def start_request(self):
        self.step_start_time = time.time()
        
    def end_request(self):
        if self.step_start_time:
            duration = time.time() - self.step_start_time
            self.request_times.append(duration)
            self.step_start_time = None
            return duration
        return 0
        
    def record_action(self, action):
        self.action_counts[action] += 1
        
    def record_step(self, reward, done):
        self.current_episode_length += 1
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
            
    def get_stats(self):
        stats = {
            'avg_request_time_ms': np.mean(self.request_times) * 1000 if self.request_times else 0,
            'request_count': len(self.request_times),
            'action_distribution': dict(self.action_counts),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'recent_episodes': len(self.episode_rewards)
        }
        return stats

perf_tracker = PerformanceTracker()

CSV_LOG_FILE = config.get('performance', 'csv_filename', 'training_metrics.csv')
csv_fieldnames = ['timestamp', 'step_count', 'episode', 'action', 'reward', 'episode_reward', 
                  'request_time_ms', 'eps', 'done', 'q_loss']
csv_file = None
csv_writer = None

def init_csv_logging():
    global csv_file, csv_writer
    if not config.get('performance', 'csv_enabled', True):
        logger.info("CSV logging disabled in config")
        return
    try:
        csv_file = open(CSV_LOG_FILE, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        csv_file.flush()
        logger.info(f"CSV logging initialized: {CSV_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize CSV logging: {e}")

model_config = config.get_model_config()
training_config = config.get_training_config()

N_OBS = model_config.get('n_observations', 25)
N_ACT = model_config.get('n_actions', 7)

OBS_KEYS = [
    "dx","dy","dz","vx","vy","vz","down","forward","angle","grounded","speed","tJump",
    "r0","r1","r2","r3","r4","r5","r6","r7",
    "dropF","dropR","dropL"
    ,"hazardDist","lastDeathType"
]
N_OBS, N_ACT = len(OBS_KEYS), N_ACT

device = torch.device("cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = model_config.get('hidden_size', 512)
        val_hidden = model_config.get('value_hidden', 256)
        adv_hidden = model_config.get('advantage_hidden', 256)
        use_noisy = model_config.get('use_noisy_nets', True)

        self.feature = nn.Sequential(
            nn.Linear(N_OBS, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        if use_noisy:
            self.val = nn.Sequential(
                NoisyLinear(hidden, val_hidden),
                nn.ReLU(),
                NoisyLinear(val_hidden, 1)
            )
            self.adv = nn.Sequential(
                NoisyLinear(hidden, adv_hidden),
                nn.ReLU(),
                NoisyLinear(adv_hidden, N_ACT)
            )
        else:
            self.val = nn.Sequential(
                nn.Linear(hidden, val_hidden), nn.ReLU(),
                nn.Linear(val_hidden, 1)
            )
            self.adv = nn.Sequential(
                nn.Linear(hidden, adv_hidden), nn.ReLU(),
                nn.Linear(adv_hidden, N_ACT)
            )

        self.use_noisy = use_noisy

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.val(f)                # (B,1)
        a = self.adv(f)                # (B,A)
        q = v + a - a.mean(1, keepdim=True)
        return q.squeeze(0) if q.size(0) == 1 else q

    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

q = QNet().to(device)
q_tgt = QNet().to(device)
q_tgt.load_state_dict(q.state_dict())

def _windows_compiler_available() -> bool:
    if platform.system() != 'Windows':
        return True
    cxx = os.environ.get('CXX', 'cl')
    # If env points to a path, ensure it exists
    exe = shutil.which(cxx) or cxx
    try:
        # Using '/help' mirrors PyTorch's own check
        subprocess.check_output([exe, '/help'], stderr=subprocess.STDOUT)
        return True
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        # Compiler exists but may not support '/help' (acceptable)
        return True

compile_enabled = bool(config.get('optimization', 'compile_model', False))
if compile_enabled and not _windows_compiler_available():
    logger.warning("torch.compile requested but no C++ compiler found on Windows; running without compilation.")
    compile_enabled = False

if compile_enabled:
    try:
        q = torch.compile(q)
        q_tgt = torch.compile(q_tgt)
        logger.info("torch.compile enabled")
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")

opt = optim.AdamW(q.parameters(), 
                  lr=training_config.get('learning_rate', 1e-3), 
                  weight_decay=training_config.get('weight_decay', 1e-4))

gamma = training_config.get('gamma', 0.99)
eps = training_config.get('eps_start', 1.0)
eps_min = training_config.get('eps_min', 0.05)
eps_decay = training_config.get('eps_decay', 0.999)

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        self.beta = self.beta_by_frame(self.frame)
        self.frame += 1

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

buf = PrioritizedReplayBuffer(
    capacity=training_config.get('replay_buffer_size', 100000),
    alpha=training_config.get('per_alpha', 0.6),
    beta_start=training_config.get('per_beta_start', 0.4)
)
bsz = training_config.get('batch_size', 256)
update_every = training_config.get('update_every', 2)
step_count = 0
current_ep_return = 0.0
best_return = -1e9
current_episode_transitions = []  # holds transitions (s,a,r,sp,d)
current_episode_number = 1
last_loss = 0.0

n_step = training_config.get('n_step', 3)
n_step_buffer = deque(maxlen=n_step)

tau = training_config.get('target_update_tau', 0.005)

class RunningNorm:
    def __init__(self, size: int, eps: float = 1e-5, warmup: int = 100):
        self.size = size
        self.eps = eps
        self.warmup = warmup
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.M2 = np.zeros(size, dtype=np.float64)
    def update(self, x: np.ndarray):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def variance(self):
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return self.M2 / (self.count - 1)
    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < self.warmup:
            return x  # skip normalization during warmup
        var = self.variance()
        return (x - self.mean) / np.sqrt(var + self.eps)
    def state_dict(self):
        return { 'count': self.count, 'mean': self.mean, 'M2': self.M2 }
    def load_state_dict(self, state):
        self.count = state.get('count', self.count)
        self.mean = state.get('mean', self.mean)
        self.M2 = state.get('M2', self.M2)

obs_norm = RunningNorm(N_OBS)

criterion = nn.SmoothL1Loss()
max_grad_norm = training_config.get('max_grad_norm', 10.0)

import os, time
from threading import Lock

train_lock = Lock()

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load best model if exists
best_path = os.path.join(SAVE_DIR, "best.pt")
if os.path.exists(best_path):
    # PyTorch >=2.6 defaults weights_only=True which can fail for older pickled checkpoints.
    try:
        ckpt = torch.load(best_path, map_location=device)
    except Exception as e_safe:
        print(f"Safe load failed ({e_safe}); retrying with weights_only=False (local file assumed trusted).")
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except Exception as e_full:
            print(f"Fallback load also failed: {e_full}. Starting fresh (delete {best_path} if corrupted).")
            ckpt = None
    if ckpt is None:
        print("Proceeding without loading checkpoint.")
    else:
    # Backward compatibility: old checkpoints had a flat 'net.*' key layout.
        try:
            q.load_state_dict(ckpt['q'], strict=False)
            q_tgt.load_state_dict(ckpt['q_tgt'], strict=False)
        except Exception as e:
            print(f"Model state load mismatch (expected new dueling architecture). Continuing with fresh weights. Details: {e}")
        if 'opt' in ckpt:
            try:
                opt.load_state_dict(ckpt['opt'])
            except Exception as e:
                print(f"Opt state load failed: {e}")
        eps = ckpt.get('eps', eps)
        step_count = ckpt.get('step_count', 0)
        best_return = ckpt.get('best_return', best_return)
        if 'obs_norm' in ckpt:
            try:
                obs_norm.load_state_dict(ckpt['obs_norm'])
            except Exception as e:
                print(f"Obs norm load failed: {e}")
        print(f"Loaded best (compat) from {best_path} (eps={eps:.3f}, steps={step_count}, best_return={best_return:.1f}, norm_count={obs_norm.count})")
else:
    print("No saved model found, starting fresh")

def save_checkpoint(name: str, is_best=False):
    path = os.path.join(SAVE_DIR, name)
    torch.save({
        'q': q.state_dict(),
        'q_tgt': q_tgt.state_dict(),
        'opt': opt.state_dict(),
        'eps': eps,
        'step_count': step_count,
        'best_return': best_return,
        'obs_keys': OBS_KEYS,
        'timestamp': time.time(),
        'is_best': is_best,
        'obs_norm': obs_norm.state_dict(),
        'version': 'phase1'
    }, path)


last_obs = None
last_action = None

class ActionCache:
    def __init__(self, enabled=True, max_size=100):
        self.enabled = enabled
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _state_key(self, state):
        if not self.enabled:
            return None
        rounded = tuple(round(float(x), 2) for x in state)
        return rounded
    
    def get(self, state):
        key = self._state_key(state)
        if key and key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, state, action):
        key = self._state_key(state)
        if not key:
            return
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = action
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self):
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'enabled': self.enabled
        }

action_cache = ActionCache(enabled=False)

def to_vec(obs):
    return np.array([float(obs[k]) for k in OBS_KEYS], dtype=np.float32)

def preprocess_state(raw: np.ndarray) -> np.ndarray:
    obs_norm.update(raw)
    return obs_norm.normalize(raw).astype(np.float32)

def select_action(s):
    global eps

    cached_action = action_cache.get(s)
    if cached_action is not None and random.random() >= eps:
        return cached_action

    use_noisy = model_config.get('use_noisy_nets', True)
    effective_eps = eps if not use_noisy else eps * 0.1

    if random.random() < effective_eps:
        action = random.randrange(N_ACT)
    else:
        with torch.no_grad():
            s_tensor = torch.from_numpy(s).to(device)
            q.reset_noise()
            qv = q(s_tensor)
            if qv.dim() > 1:
                qv = qv[0]
            action = int(torch.argmax(qv).item())

    # Cache the action for future use
    action_cache.put(s, action)
    return action

def train_step():
    global last_loss
    if not train_lock.acquire(blocking=False):
        return
    try:
        if len(buf) < bsz:
            return

        batch, indices, weights = buf.sample(bsz)

        s = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        sp = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

        q_s = q(s)
        q_sa = q_s.gather(1, a)
        with torch.no_grad():
            next_online = q(sp)
            next_actions = next_online.argmax(1, keepdim=True)
            next_target = q_tgt(sp).gather(1, next_actions)
            target = r + gamma * (1 - d) * next_target

        td_errors = torch.abs(q_sa - target).detach().cpu().numpy()

        element_wise_loss = nn.functional.smooth_l1_loss(q_sa, target, reduction='none')
        loss = (element_wise_loss * weights).mean()

        last_loss = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
        opt.step()

        q.reset_noise()
        q_tgt.reset_noise()

        new_priorities = td_errors.flatten() + 1e-6
        buf.update_priorities(indices, new_priorities)

        with torch.no_grad():
            for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                p_tgt.mul_(1 - tau).add_(p, alpha=tau)
    finally:
        train_lock.release()

@app.route("/step", methods=["POST"])
def step():
    global last_obs, last_action, eps, step_count, current_ep_return, best_return, current_episode_number
    
    perf_tracker.start_request()
    
    data = request.get_json(force=True)
    obs_raw = to_vec(data["obs"])
    obs = preprocess_state(obs_raw)
    reward = float(data.get("reward", 0.0))
    reward = float(np.clip(reward, -5.0, 5.0))
    done = bool(data.get("done", False))
    current_ep_return += reward

    if last_obs is not None and last_action is not None:
        n_step_buffer.append((last_obs, last_action, reward, obs, done))

        if len(n_step_buffer) == n_step or done:
            n_step_reward = 0.0
            for i, (_, _, r, _, _) in enumerate(n_step_buffer):
                n_step_reward += (gamma ** i) * r

            s0, a0, _, _, _ = n_step_buffer[0]
            sn = obs
            dn = 1.0 if done else 0.0

            transition = (s0, a0, n_step_reward, sn, dn)
            buf.push(transition)
            current_episode_transitions.append(transition)

        if step_count % update_every == 0:
            train_step()
        eps = max(eps_min, eps * eps_decay)

    action = select_action(obs)
    
    perf_tracker.record_action(action)
    perf_tracker.record_step(reward, done)
    request_time = perf_tracker.end_request()

    log_frequency = config.get('performance', 'log_every_n_steps', 50)
    if step_count % log_frequency == 0 or done:
        stats = perf_tracker.get_stats()
        logger.info(f"Step {step_count}: action={action}, reward={reward:.3f}, "
                   f"eps={eps:.3f}, avg_request_time={stats['avg_request_time_ms']:.1f}ms, "
                   f"episode_reward={current_ep_return:.2f}")

    if csv_writer:
        try:
            csv_writer.writerow({
                'timestamp': time.time(),
                'step_count': step_count,
                'episode': current_episode_number,
                'action': action,
                'reward': reward,
                'episode_reward': current_ep_return,
                'request_time_ms': request_time * 1000,
                'eps': eps,
                'done': done,
                'q_loss': last_loss
            })
            csv_file.flush()
        except Exception as e:
            logger.error(f"CSV logging error: {e}")

    if done:
        save_checkpoint("last.pt", is_best=False)
        if current_ep_return > best_return:
            best_return = current_ep_return
            save_checkpoint("best.pt", is_best=True)
            eps = max(eps_min, eps * 0.5)
            logger.info(f"new best: {best_return:.2f}")
        
        stats = perf_tracker.get_stats()
        logger.info(f"ep {current_episode_number} done: reward={current_ep_return:.2f}, "
                   f"len={len(current_episode_transitions)}, "
                   f"avg_req={stats['avg_request_time_ms']:.1f}ms")
        
        current_ep_return = 0.0
        current_episode_transitions.clear()
        current_episode_number += 1
        n_step_buffer.clear()
        last_obs = None
        last_action = None
    else:
        last_obs = obs
        last_action = action
    step_count += 1

    return jsonify({"action": action})

@app.route("/stats", methods=["GET"])
def get_stats():
    stats = perf_tracker.get_stats()
    stats.update({
        'step_count': step_count,
        'current_episode': current_episode_number,
        'eps': eps,
        'best_return': best_return,
        'buffer_size': len(buf),
        'last_loss': last_loss,
        'action_cache': action_cache.stats(),
    })
    return jsonify(stats)

@app.route("/config/reward", methods=["GET"])
def get_reward_config():
    return jsonify(config.get_reward_config())

@app.route("/config/model", methods=["GET"])  
def get_model_config():
    return jsonify({
        'training': config.get_training_config(),
        'model': config.get_model_config(),
        'performance': config.model_config.get('performance', {}),
        'optimization': config.model_config.get('optimization', {})
    })

@app.route("/config/reload", methods=["POST"])
def reload_config():
    try:
        config.reload()
        return jsonify({"status": "success", "message": "config reloaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    init_csv_logging()
    logger.info(f"starting, csv={CSV_LOG_FILE}")
    app.run(host="127.0.0.1", port=5000, debug=False)

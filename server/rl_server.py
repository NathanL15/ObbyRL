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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy PyTorch warning on Windows about record_context_cpp (harmless)
if platform.system() != "Linux":
    warnings.filterwarnings(
        "ignore",
        message=r".*record_context_cpp is not support.*",
        category=UserWarning,
    )

# Performance tracking
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

# CSV logging setup
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

# Load configuration values
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

class QNet(nn.Module):
    """Dueling network architecture for Double DQN."""
    def __init__(self):
        super().__init__()
        hidden = model_config.get('hidden_size', 256)
        val_hidden = model_config.get('value_hidden', 128) 
        adv_hidden = model_config.get('advantage_hidden', 128)
        
        self.feature = nn.Sequential(
            nn.Linear(N_OBS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, val_hidden), nn.ReLU(),
            nn.Linear(val_hidden, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, adv_hidden), nn.ReLU(),
            nn.Linear(adv_hidden, N_ACT)
        )
    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.val(f)                # (B,1)
        a = self.adv(f)                # (B,A)
        q = v + a - a.mean(1, keepdim=True)
        return q.squeeze(0) if q.size(0) == 1 else q

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

# Apply model compilation if enabled and compiler is available
compile_enabled = bool(config.get('optimization', 'compile_model', False))
if compile_enabled and not _windows_compiler_available():
    logger.warning("torch.compile requested but no C++ compiler found on Windows; running without compilation.")
    compile_enabled = False

if compile_enabled:
    try:
        q = torch.compile(q)
        q_tgt = torch.compile(q_tgt)
        logger.info("Model compilation enabled for both Q-networks")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")

opt = optim.AdamW(q.parameters(), 
                  lr=training_config.get('learning_rate', 1e-3), 
                  weight_decay=training_config.get('weight_decay', 1e-4))

gamma = training_config.get('gamma', 0.99)
eps = training_config.get('eps_start', 1.0)              # start high exploration (will decay)
eps_min = training_config.get('eps_min', 0.05)
eps_decay = training_config.get('eps_decay', 0.999)      # slightly faster decay; will apply adaptive bumps on improvements
buf = deque(maxlen=training_config.get('replay_buffer_size', 100000))
elite_buf = deque(maxlen=training_config.get('elite_buffer_size', 5000))  # preserved for now (will remove/replace with PER in later phase)
bsz = training_config.get('batch_size', 128)
update_every = training_config.get('update_every', 2)
step_count = 0
current_ep_return = 0.0
best_return = -1e9
current_episode_transitions = []  # holds transitions (s,a,r,sp,d)
current_episode_number = 1
last_loss = 0.0  # Track training loss

# Soft target update factor
tau = training_config.get('target_update_tau', 0.005)

# Running observation normalization -------------------------------------------------
class RunningNorm:
    def __init__(self, size: int, eps: float = 1e-5, warmup: int = 100):
        self.size = size
        self.eps = eps
        self.warmup = warmup
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.M2 = np.zeros(size, dtype=np.float64)
    def update(self, x: np.ndarray):
        # x shape (size,)
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
            return x  # do not distort early exploration
        var = self.variance()
        return (x - self.mean) / np.sqrt(var + self.eps)
    def state_dict(self):
        return { 'count': self.count, 'mean': self.mean, 'M2': self.M2 }
    def load_state_dict(self, state):
        self.count = state.get('count', self.count)
        self.mean = state.get('mean', self.mean)
        self.M2 = state.get('M2', self.M2)

obs_norm = RunningNorm(N_OBS)

# Loss & utility
criterion = nn.SmoothL1Loss()
max_grad_norm = training_config.get('max_grad_norm', 10.0)

import os, time
from threading import Lock

# Optional: enable for debugging gradient issues (set to True if still errors)
ENABLE_ANOMALY_DETECT = False
if ENABLE_ANOMALY_DETECT:
    torch.autograd.set_detect_anomaly(True)

train_lock = Lock()

# Request batching for performance optimization
class RequestBatcher:
    def __init__(self, enabled=False, timeout_ms=10):
        self.enabled = enabled
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.lock = Lock()
        
    def add_request(self, request_data):
        if not self.enabled:
            return None
            
        with self.lock:
            self.pending_requests.append(request_data)
            
        # For now, just return None (single request processing)
        # Future: implement actual batching with threading
        return None
    
    def process_batch(self):
        # Future implementation for batch processing
        pass

request_batcher = RequestBatcher(
    enabled=config.get('optimization', 'enable_request_batching', False),
    timeout_ms=config.get('optimization', 'batch_timeout_ms', 10)
)

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

# Simple action caching for performance
class ActionCache:
    def __init__(self, enabled=True, max_size=100):
        self.enabled = enabled
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _state_key(self, state):
        """Create a cache key from state (rounded for fuzzy matching)."""
        if not self.enabled:
            return None
        # Round to 2 decimal places for fuzzy matching of similar states
        rounded = tuple(round(float(x), 2) for x in state)
        return rounded
    
    def get(self, state):
        """Get cached action for similar state."""
        key = self._state_key(state)
        if key and key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, state, action):
        """Cache action for state."""
        key = self._state_key(state)
        if not key:
            return
            
        # Remove oldest if at capacity
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

action_cache = ActionCache(enabled=False)  # Disabled by default for safety

def to_vec(obs):
    return np.array([float(obs[k]) for k in OBS_KEYS], dtype=np.float32)

def preprocess_state(raw: np.ndarray) -> np.ndarray:
    obs_norm.update(raw)
    return obs_norm.normalize(raw).astype(np.float32)

def select_action(s):
    global eps
    
    # Try cache first (if enabled)
    cached_action = action_cache.get(s)
    if cached_action is not None and random.random() >= eps:
        return cached_action
    
    if random.random() < eps:
        action = random.randrange(N_ACT)
    else:
        with torch.no_grad():
            s_tensor = torch.from_numpy(s).to(device)
            qv = q(s_tensor)
            if qv.dim() > 1:
                qv = qv[0]
            action = int(torch.argmax(qv).item())
    
    # Cache the action for future use
    action_cache.put(s, action)
    return action

def train_step():
    global last_loss
    # Ensure only one backward/optimizer step at a time (Flask may be threaded)
    if not train_lock.acquire(blocking=False):
        return  # skip if another thread is training; reduces race risk
    try:
        # Need enough base samples
        if len(buf) < bsz:
            return
        elite_take = 0
        if len(elite_buf) > 0:
            elite_take = min(len(elite_buf), bsz // 4)  # reduce elite influence
        base_take = bsz - elite_take
        batch = []
        if elite_take > 0:
            batch.extend(random.sample(elite_buf, elite_take))
        batch.extend(random.sample(buf, base_take))
        random.shuffle(batch)

        s = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        sp = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

        # Double DQN target
        q_s = q(s)
        q_sa = q_s.gather(1, a)
        with torch.no_grad():
            next_online = q(sp)
            next_actions = next_online.argmax(1, keepdim=True)
            next_target = q_tgt(sp).gather(1, next_actions)
            target = r + gamma * (1 - d) * next_target

        loss = criterion(q_sa, target)
        last_loss = loss.item()  # Track loss for logging
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
        opt.step()

        # Soft target update (no gradients tracked)
        with torch.no_grad():
            for p_tgt, p in zip(q_tgt.parameters(), q.parameters()):
                p_tgt.mul_(1 - tau).add_(p, alpha=tau)
    finally:
        train_lock.release()

@app.route("/step", methods=["POST"])
def step():
    global last_obs, last_action, eps, step_count, current_ep_return, best_return, current_episode_number
    
    # Start performance tracking
    perf_tracker.start_request()
    
    data = request.get_json(force=True)
    obs_raw = to_vec(data["obs"])
    obs = preprocess_state(obs_raw)
    reward = float(data.get("reward", 0.0))
    # Optional reward clipping (robustness)
    reward = float(np.clip(reward, -5.0, 5.0))
    done = bool(data.get("done", False))
    current_ep_return += reward

    if last_obs is not None and last_action is not None:
        transition = (last_obs, last_action, reward, obs, 1.0 if done else 0.0)
        buf.append(transition)
        current_episode_transitions.append(transition)
        if step_count % update_every == 0:
            train_step()
        # epsilon decay
        eps = max(eps_min, eps * eps_decay)

    action = select_action(obs)
    
    # Record performance metrics
    perf_tracker.record_action(action)
    perf_tracker.record_step(reward, done)
    request_time = perf_tracker.end_request()

    # Structured logging every N steps or on episode end
    log_frequency = config.get('performance', 'log_every_n_steps', 50)
    if step_count % log_frequency == 0 or done:
        stats = perf_tracker.get_stats()
        logger.info(f"Step {step_count}: action={action}, reward={reward:.3f}, "
                   f"eps={eps:.3f}, avg_request_time={stats['avg_request_time_ms']:.1f}ms, "
                   f"episode_reward={current_ep_return:.2f}")

    # CSV logging
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
        # episode finished: save latest checkpoint
        save_checkpoint("last.pt", is_best=False)
        # Elite criteria: improve best return or reach at least 95% of best (if best established)
        is_new_best = current_ep_return > best_return
        meets_threshold = (best_return > -1e8) and (current_ep_return >= 0.95 * best_return)
        if is_new_best or meets_threshold:
            # Copy transitions to elite buffer
            for tr in current_episode_transitions:
                elite_buf.append(tr)
        if is_new_best:
            best_return = current_ep_return
            save_checkpoint("best.pt", is_best=True)
            # Exploration bump-down on genuine improvement
            eps = max(eps_min, eps * 0.5)
            logger.info(f"New best episode reward: {best_return:.2f}")
        elif meets_threshold:
            eps = max(eps_min, eps * 0.9)
        
        # Log episode completion
        stats = perf_tracker.get_stats()
        logger.info(f"Episode {current_episode_number} completed: reward={current_ep_return:.2f}, "
                   f"length={len(current_episode_transitions)}, "
                   f"avg_request_time={stats['avg_request_time_ms']:.1f}ms")
        
        # reset episode accumulator
        current_ep_return = 0.0
        current_episode_transitions.clear()
        current_episode_number += 1
        last_obs = None
        last_action = None
    else:
        last_obs = obs
        last_action = action
    step_count += 1

    return jsonify({"action": action})

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get current performance statistics"""
    stats = perf_tracker.get_stats()
    stats.update({
        'step_count': step_count,
        'current_episode': current_episode_number,
        'eps': eps,
        'best_return': best_return,
        'buffer_size': len(buf),
        'elite_buffer_size': len(elite_buf),
        'last_loss': last_loss,
        'action_cache': action_cache.stats(),
        'request_batcher': {
            'enabled': request_batcher.enabled,
            'pending': len(request_batcher.pending_requests) if request_batcher.enabled else 0
        }
    })
    return jsonify(stats)

@app.route("/config/reward", methods=["GET"])
def get_reward_config():
    """Get current reward shaping configuration"""
    return jsonify(config.get_reward_config())

@app.route("/config/model", methods=["GET"])  
def get_model_config():
    """Get current model and training configuration"""
    return jsonify({
        'training': config.get_training_config(),
        'model': config.get_model_config(),
        'performance': config.model_config.get('performance', {}),
        'optimization': config.model_config.get('optimization', {})
    })

@app.route("/config/reload", methods=["POST"])
def reload_config():
    """Reload configuration from files"""
    try:
        config.reload()
        return jsonify({"status": "success", "message": "Configuration reloaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    init_csv_logging()
    logger.info("RL Server starting with performance tracking enabled")
    logger.info(f"CSV logging to: {CSV_LOG_FILE}")
    app.run(host="127.0.0.1", port=5000, debug=False)

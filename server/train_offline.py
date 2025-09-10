#!/usr/bin/env python3
"""
Offline training CLI for the RL system.
Allows training without running the Roblox client for rapid experimentation.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import platform, shutil, subprocess
import warnings
from collections import deque
import json
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # ensure symbol exists for type checkers
    HAS_MATPLOTLIB = False

# Import the RL components
import sys
sys.path.append('.')
from rl_server import QNet, obs_norm, N_OBS, N_ACT, device

# Suppress noisy PyTorch warning on Windows about record_context_cpp (harmless)
if platform.system() != "Linux":
    warnings.filterwarnings(
        "ignore",
        message=r".*record_context_cpp is not support.*",
        category=UserWarning,
    )
from config_manager import config

class SyntheticEnv:
    """Simplified synthetic environment for offline training."""
    
    def __init__(self):
        self.reset()
        self.action_space = N_ACT
        self.observation_space = N_OBS
        
    def reset(self):
        """Reset environment to initial state."""
        self.position = np.array([0.0, 5.0, 0.0])  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target = np.array([50.0, 5.0, 0.0])   # Simple target
        self.steps = 0
        self.max_steps = 1000
        return self._get_obs()
    
    def _get_obs(self):
        """Generate observation similar to Roblox client."""
        # Distance to target
        delta = self.target - self.position
        dx, dy, dz = delta
        dist = np.linalg.norm(delta)
        
        # Velocity
        vx, vy, vz = self.velocity
        
        # Simple environment sensors
        down = 5.0 - self.position[1]  # distance to ground
        forward = max(0, 30 - abs(dx))  # forward ray
        
        # Angle to target
        if dist > 0:
            angle = np.dot([1, 0, 0], delta / dist)  # cosine similarity
        else:
            angle = 1.0
            
        grounded = 1.0 if self.position[1] <= 5.5 else 0.0
        speed = np.linalg.norm(self.velocity[:2])  # horizontal speed
        
        # Radial sensors (simplified)
        radials = [max(0, 25 - np.random.normal(5, 2)) for _ in range(8)]
        
        # Edge probes
        drops = [down + np.random.normal(0, 1) for _ in range(3)]
        
        # Hazard distance (simplified)
        hazard_dist = min(1.0, float(dist) / 50.0)
            
        obs = {
            'dx': dx, 'dy': dy, 'dz': dz,
            'vx': vx, 'vy': vy, 'vz': vz,
            'down': down, 'forward': forward,
            'angle': angle, 'grounded': grounded,
            'speed': speed, 'tJump': 0.0,
            'r0': radials[0], 'r1': radials[1], 'r2': radials[2], 'r3': radials[3],
            'r4': radials[4], 'r5': radials[5], 'r6': radials[6], 'r7': radials[7],
            'dropF': drops[0], 'dropR': drops[1], 'dropL': drops[2],
            'hazardDist': hazard_dist, 'lastDeathType': 0
        }
        
        return np.array([obs[k] for k in ['dx','dy','dz','vx','vy','vz','down','forward',
                                            'angle','grounded','speed','tJump','r0','r1','r2','r3',
                                            'r4','r5','r6','r7','dropF','dropR','dropL',
                                            'hazardDist','lastDeathType']], dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done."""
        self.steps += 1
        
        # Simple physics
        dt = 0.1
        
        # Action effects (simplified)
        if action == 1:  # Forward
            self.velocity[0] += 2.0
        elif action == 2:  # Left  
            self.velocity[2] -= 1.0
        elif action == 3:  # Right
            self.velocity[2] += 1.0
        elif action == 4:  # Jump
            if self.position[1] <= 5.5:  # grounded
                self.velocity[1] = 8.0
        elif action == 5:  # Forward + Jump
            if self.position[1] <= 5.5:
                self.velocity[0] += 3.0
                self.velocity[1] = 8.0
        elif action == 6:  # Backward
            self.velocity[0] -= 1.0
        
        # Physics update
        self.velocity[1] -= 9.8 * dt  # gravity
        self.velocity *= 0.9  # friction
        self.position += self.velocity * dt
        
        # Ground collision
        if self.position[1] < 5.0:
            self.position[1] = 5.0
            self.velocity[1] = 0.0
        
        # Calculate reward
        old_dist = np.linalg.norm(self.target - (self.position - self.velocity * dt))
        new_dist = np.linalg.norm(self.target - self.position)
        progress = old_dist - new_dist
        
        reward = progress * 2.0 - 0.01  # progress reward + small step penalty
        
        # Check if done
        done = False
        if new_dist < 2.0:  # reached target
            reward += 50.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
        elif self.position[1] < 0:  # fell off
            reward -= 10.0
            done = True
            
        return self._get_obs(), reward, done, {}

def train_offline(episodes=100, render=False):
    """Train the model offline using synthetic environment."""
    
    # Initialize environment and model
    env = SyntheticEnv()
    
    # Load or create model
    base_q = QNet().to(device)
    base_tgt = QNet().to(device)
    base_tgt.load_state_dict(base_q.state_dict())

    # Optional model compile for speed (forward wrappers only)
    q_fwd, q_tgt_fwd = base_q, base_tgt
    def _windows_compiler_available() -> bool:
        if platform.system() != 'Windows':
            return True
        cxx = os.environ.get('CXX', 'cl')
        exe = shutil.which(cxx) or cxx
        try:
            subprocess.check_output([exe, '/help'], stderr=subprocess.STDOUT)
            return True
        except FileNotFoundError:
            return False
        except subprocess.SubprocessError:
            return True
    try:
        compile_enabled = bool(config.get('optimization', 'compile_model', False))
        if compile_enabled and not _windows_compiler_available():
            print("[offline] torch.compile requested but no C++ compiler found on Windows; running without compilation.")
            compile_enabled = False
        if compile_enabled:
            q_fwd = torch.compile(base_q)
            q_tgt_fwd = torch.compile(base_tgt)
    except Exception as e:
        print(f"Model compilation disabled for offline training: {e}")

    # Hyperparameters (align with server defaults/config)
    training_cfg = config.get_training_config() or {}
    lr = float(training_cfg.get('learning_rate', 1e-3))
    gamma = float(training_cfg.get('gamma', 0.99))
    batch_size = int(training_cfg.get('batch_size', 256))
    max_grad_norm = float(training_cfg.get('max_grad_norm', 10.0))
    tau = float(training_cfg.get('target_update_tau', 0.005))
    buffer_size = int(training_cfg.get('replay_buffer_size', 10000))
    # Epsilon schedule
    eps = float(training_cfg.get('eps_start', 1.0))
    eps_min = float(training_cfg.get('eps_min', 0.05))
    eps_decay = float(training_cfg.get('eps_decay', 0.995))

    optimizer = torch.optim.AdamW(base_q.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber loss like server

    # Replay buffer
    buffer = deque(maxlen=buffer_size)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print(f"Starting offline training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        # Update running norm then normalize (match server preprocess)
        obs_norm.update(state)
        state = obs_norm.normalize(state)
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            if np.random.random() < eps:
                action = np.random.randint(N_ACT)
            else:
                with torch.no_grad():
                    q_values = q_fwd(torch.FloatTensor(state).to(device))
                    action = q_values.argmax().item()
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            # Reward clipping like server for robustness
            reward = float(np.clip(reward, -5.0, 5.0))
            obs_norm.update(next_state)
            next_state = obs_norm.normalize(next_state)
            
            # Store transition
            buffer.append((state, action, reward, next_state, 1.0 if done else 0.0))
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Train if enough samples
            if len(buffer) > batch_size:
                batch = np.random.choice(len(buffer), batch_size, replace=False)
                batch_transitions = [buffer[i] for i in batch]

                # Fast tensorization via numpy stack/array
                states_np = np.stack([t[0] for t in batch_transitions]).astype(np.float32)
                actions_np = np.array([t[1] for t in batch_transitions], dtype=np.int64)
                rewards_np = np.array([t[2] for t in batch_transitions], dtype=np.float32)
                next_states_np = np.stack([t[3] for t in batch_transitions]).astype(np.float32)
                dones_np = np.array([t[4] for t in batch_transitions], dtype=np.float32)

                states = torch.from_numpy(states_np).to(device)
                actions = torch.from_numpy(actions_np).unsqueeze(1).to(device)
                rewards_t = torch.from_numpy(rewards_np).unsqueeze(1).to(device)
                next_states = torch.from_numpy(next_states_np).to(device)
                dones_t = torch.from_numpy(dones_np).unsqueeze(1).to(device)

                # Double DQN target (align with server)
                with torch.no_grad():
                    next_online = q_fwd(next_states)
                    next_actions = next_online.argmax(1, keepdim=True)
                    next_target = q_tgt_fwd(next_states).gather(1, next_actions)
                    targets = rewards_t + gamma * (1.0 - dones_t) * next_target

                # Current Q(s,a)
                q_s = q_fwd(states)
                q_sa = q_s.gather(1, actions)

                # Loss + step with grad clip
                loss = criterion(q_sa, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(base_q.parameters(), max_grad_norm)
                optimizer.step()

                # Soft target update
                with torch.no_grad():
                    for p_tgt, p in zip(base_tgt.parameters(), base_q.parameters()):
                        p_tgt.mul_(1 - tau).add_(p, alpha=tau)

                losses.append(loss.item())
            
            if done:
                break
        
        # Decay epsilon
        eps = max(eps_min, eps * eps_decay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode}: avg_reward={avg_reward:.2f}, "
                  f"avg_length={avg_length:.1f}, eps={eps:.3f}, loss={avg_loss:.4f}")
    
    # Save trained model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'q': base_q.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'config': config.get_training_config()
    }, 'checkpoints/offline_trained.pt')
    
    print(f"Training completed. Model saved to checkpoints/offline_trained.pt")
    
    # Plot results if requested
    if render and HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.subplot(1, 3, 3)
        if losses:
            plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('offline_training_results.png')
        print("Results plotted to offline_training_results.png")
    elif render and not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot generation")
    
    return {
        'final_avg_reward': np.mean(episode_rewards[-10:]),
        'total_episodes': episodes,
        'final_epsilon': eps
    }

def main():
    parser = argparse.ArgumentParser(description="Offline RL training")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Generate plots")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing synthetic environment...")
        env = SyntheticEnv()
        
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}:")
            while True:
                action = np.random.randint(N_ACT)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if steps % 50 == 0:
                    pos = env.position
                    target = env.target
                    dist = np.linalg.norm(target - pos)
                    print(f"  Step {steps}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
                          f"dist={dist:.1f}, reward={total_reward:.2f}")
                
                if done:
                    break
            
            print(f"  Final: steps={steps}, reward={total_reward:.2f}")
        
    else:
        results = train_offline(args.episodes, args.render)
        print(f"\nTraining Summary:")
        print(f"Final average reward: {results['final_avg_reward']:.2f}")
        print(f"Total episodes: {results['total_episodes']}")
        print(f"Final epsilon: {results['final_epsilon']:.3f}")

if __name__ == "__main__":
    main()
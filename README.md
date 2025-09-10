# 🧠 RL on Roblox: Classic Obby

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Roblox](https://img.shields.io/badge/roblox-studio-red.svg)](https://create.roblox.com/)

Train a **Deep Q-Network (DQN) reinforcement learning agent** to navigate a **Roblox Classic Obby** environment. The agent learns to jump, move, and avoid hazards through trial and error, communicating in real-time between Roblox Studio and a Python server.

## ✨ Features

- **Real-time RL Training**: Agent interacts with Roblox game world via HTTP API
- **Advanced Observations**: Includes kinematics, radial rays, edge probes, hazard detection
- **Smart Reward Shaping**: Distance-based progress, leap bonuses, milestone rewards, hazard avoidance
- **Automatic Checkpointing**: Saves best and latest models during training
- **Elite Replay Buffer**: Retains high-performing episode trajectories to prevent forgetting good strategies
- **Adaptive Exploration**: Epsilon decay adjusts based on performance
- **Hazard Awareness**: Detects and avoids lethal blocks with safe respawn logic
- **🚀 Performance Optimizations**: Torch compilation, action caching, request timing
- **📊 Comprehensive Monitoring**: Real-time dashboard, CSV logging, performance metrics
- **⚙️ Configuration Management**: Runtime tunable parameters without code changes
- **🔬 Offline Training**: Synthetic environment for rapid experimentation

## 🏗️ Architecture

```
Roblox Client (AgentClient.client.lua)
    ↓ RemoteFunction RLStep [MONITORED]
Roblox Server (RLServer.lua)
    ↓ HTTP POST [TIMED]
Python Flask Server (rl_server.py)
    ↓ DQN Training Loop [OPTIMIZED]
PyTorch Q-Network [COMPILED]
    ↓ Metrics & Logs
Dashboard & CSV Export
```

## 📋 Prerequisites

- **Roblox Studio**: For running the obby environment
- **Python 3.8+**: For the RL server
- **PyYAML**: For configuration management (`pip install pyyaml`)
- **Roblox Game Setup**: Classic Obby with checkpoints (CP_0, CP_1, CP_2, ...) and hazard-tagged parts

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/NathanL15/ClassicObby-RL.git
cd ClassicObby-RL
```

### 2. Set up the Python RL Server

```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the DQN server
python rl_server.py
```

### 3. Set up Roblox Environment

1. Open `place/ClassicObby.rbxl` in Roblox Studio
2. Ensure checkpoints are named `CP_0`, `CP_1`, `CP_2`, etc.
3. Tag hazard parts with `CollectionService` tag "Hazard"
4. Add `RemoteFunction` named "RLStep" in `ReplicatedStorage`
5. Insert `roblox/RLServer.lua` as a ServerScript in `ServerScriptService`
6. Insert `roblox/AgentClient.client.lua` as a LocalScript in `StarterPlayerScripts`

### 4. Start Training & Monitoring

1. **Run the Python server** (from step 2)
2. **Start the dashboard** (optional):
   ```bash
   python dashboard.py
   ```
   Open http://127.0.0.1:5001 in your browser
3. **Play the Roblox game** in Studio
4. **Watch the agent learn** to navigate the obby!

## 📊 New Monitoring & Performance Features

### Real-time Dashboard
```bash
cd server
python dashboard.py
# Open http://127.0.0.1:5001
```

- **Performance metrics**: Request timing, throughput analysis
- **Training progress**: Steps, episodes, rewards, exploration rate
- **Action distribution**: Analyze policy behavior
- **System status**: Buffer sizes, optimizations, configuration

### Performance Statistics
```bash
# Get current stats via API
curl http://127.0.0.1:5000/stats

# View configuration
python config_tool.py --action view --config-type reward
python config_tool.py --action view --config-type model
```

### CSV Data Export
Training metrics are automatically exported to `training_metrics.csv`:
- Timestamp, step count, episode number
- Action taken, reward received
- Request timing, exploration rate
- Training loss values

## ⚙️ Configuration Management

### Reward Shaping Configuration
Edit `config/reward_shaping.conf` to tune reward parameters:

```ini
[progress_rewards]
base_reward_per_step = -0.005
progress_reward_scale = 3.0
leap_bonus = 1.0
milestone_bonus = 2.0

[checkpoints]
checkpoint_bonus = 20.0
completion_base_bonus = 50.0

[penalties]
death_penalty_hazard = 15.0
stuck_penalty = 8.0
```

### Model & Training Configuration
Edit `config/model_config.yaml`:

```yaml
training:
  learning_rate: 0.001
  batch_size: 128
  eps_start: 1.0
  eps_min: 0.05

optimization:
  compile_model: true  # Enable torch.compile()
  enable_action_cache: false
```

### Runtime Configuration Updates
```bash
# Update reward parameters
python config_tool.py --action update-rewards

# Reload configuration on server
python config_tool.py --action reload
```

## 🔬 Offline Training & Experimentation

Train without Roblox for rapid iteration:

```bash
cd server

# Quick test of synthetic environment
python train_offline.py --test

# Train for 100 episodes offline
python train_offline.py --episodes 100

# Train with visualization (requires matplotlib)
python train_offline.py --episodes 50 --render
```

Benefits:
- **Fast iteration**: No Roblox startup time
- **Controlled experiments**: Reproducible synthetic environment
- **Hyperparameter tuning**: Quick testing of different configurations
- **Algorithm development**: Test new RL approaches safely

## 🎯 Performance Targets & Results

| Metric | Previous | Current | Target |
|--------|----------|---------|--------|
| Action step latency | ~250ms | **~180ms** | <120ms |
| Model compilation | ❌ | ✅ **torch.compile()** | ✅ |
| Request timing | ❌ | ✅ **Full monitoring** | ✅ |
| Reward observability | Basic logs | ✅ **Component breakdown** | ✅ |
| Configuration management | Hardcoded | ✅ **Runtime tunable** | ✅ |
| Offline training | ❌ | ✅ **Synthetic environment** | ✅ |

### Performance Optimizations Applied
- **Torch Compilation**: ~15-20% inference speedup
- **Action Caching**: Reduces redundant computation for similar states
- **Request Monitoring**: Full timing instrumentation for bottleneck identification
- **Configuration System**: Easy parameter tuning without code changes

## ⚙️ Legacy Configuration (Agent Parameters)

### AgentClient.client.lua Parameters

- `STEP_DT`: Internal update interval (0.08s)
- `ACTION_DECISION_DT`: Decision frequency (0.25s)
- `HAZARD_NEAR_RADIUS`: Distance to trigger hazard avoidance (15 studs)
- `CHECK_RADIUS`: Checkpoint reach distance (6 studs)
- `TIMING_LOG_EVERY`: Log timing stats every N decisions (10)

### Server Parameters (rl_server.py)

- `N_ACT`: Number of actions (7: idle, forward, left, right, jump, forward+jump, backward)
- `gamma`: Discount factor (0.99)
- `eps_min`: Minimum exploration rate (0.05)
- `buf`: Replay buffer size (100,000)
- `elite_buf`: Elite buffer size (5,000)

## 🎮 Actions

| ID | Action | Description |
|----|--------|-------------|
| 0  | Idle   | No movement |
| 1  | Forward| Move forward |
| 2  | Left   | Strafe left |
| 3  | Right  | Strafe right |
| 4  | Jump   | Jump in place |
| 5  | Forward + Jump | Jump while moving forward |
| 6  | Backward | Move backward |

## 📊 Observations

The agent receives 25-dimensional observations:

- **Kinematics**: dx, dy, dz, vx, vy, vz (position/velocity to target)
- **Environment**: down, forward (ray distances)
- **Orientation**: angle (cosine to target), grounded, speed, tJump
- **Radial Rays**: r0-r7 (8 directions for obstacle sensing)
- **Edge Probes**: dropF, dropR, dropL (gap detection ahead/sides)
- **Hazards**: hazardDist (normalized distance to nearest hazard), lastDeathType

## 🏆 Reward Structure (Configurable)

- **Progress**: +3.0 * distance improvement (capped at -2 regress)
- **Leap Bonus**: +1 for jumps >2 studs closer
- **Milestone**: +2 every 1-stud improvement on best distance
- **Checkpoint**: +20 for reaching next CP
- **Completion**: +50 + 10 * num_checkpoints for finishing
- **Penalties**: -15 hazard death, -8 fall death, -8 stuck, -0.02 hazard approach
- **Base**: -0.005 per step

All reward parameters are configurable via `config/reward_shaping.conf`.

## 🛠️ Development Tools

### Configuration Management
```bash
# View current configuration
python config_tool.py --action view

# Update reward parameters with optimized values
python config_tool.py --action update-rewards

# Reload server configuration
python config_tool.py --action reload
```

### Performance Testing
```bash
# Test server performance
python test_performance.py

# Monitor via dashboard
python dashboard.py
# Open http://127.0.0.1:5001
```

### Offline Development
```bash
# Test synthetic environment
python train_offline.py --test

# Rapid training iteration
python train_offline.py --episodes 20
```

## 📁 Project Structure

```
ClassicObby-RL/
├── place/
│   └── ClassicObby.rbxl          # Roblox place file
├── roblox/
│   ├── AgentClient.client.lua    # Client-side agent logic [ENHANCED]
│   ├── RLServer.lua              # Server-side HTTP bridge
│   └── RewardConfig.lua          # Configuration loading utility
├── server/
│   ├── rl_server.py             # DQN training server [OPTIMIZED]
│   ├── config_manager.py        # Configuration management
│   ├── dashboard.py             # Real-time monitoring dashboard
│   ├── train_offline.py         # Offline training CLI
│   ├── test_performance.py      # Performance testing
│   ├── config_tool.py           # Configuration utility
│   └── requirements.txt         # Python dependencies [UPDATED]
├── config/
│   ├── model_config.yaml        # Model & training parameters
│   └── reward_shaping.conf      # Reward function configuration
├── checkpoints/                  # Auto-saved models (created on run)
│   ├── best.pt                   # Best performing model
│   ├── last.pt                   # Most recent model
│   └── offline_trained.pt       # Offline training results
└── README.md                     # This file [ENHANCED]
```

## 🧪 Performance Monitoring

The system now provides comprehensive performance monitoring:

1. **HTTP Request Timing**: Both client and server side measurement
2. **Reward Component Breakdown**: Detailed logging of reward calculations
3. **Action Distribution Analysis**: Track policy behavior over time
4. **Training Progress Metrics**: Episode rewards, lengths, exploration rate
5. **System Resource Usage**: Buffer sizes, cache utilization
6. **CSV Data Export**: Complete training log for external analysis

## 🎯 Next Steps & Future Enhancements

- **Request Batching**: Implement full batching with threading for higher throughput
- **TensorBoard Integration**: Add TensorBoard logging for advanced visualization
- **Multi-agent Training**: Support multiple agents training simultaneously
- **Advanced RL Algorithms**: PPO, SAC for continuous control and better sample efficiency
- **Curriculum Learning**: Automatic difficulty progression based on performance
- **Model Compression**: ONNX export for even faster inference

## 🤝 Contributing

Contributions are welcome! The new configuration and monitoring systems make it easy to experiment with:

- New reward shaping strategies
- Alternative RL algorithms
- Performance optimizations
- Additional synthetic environments

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


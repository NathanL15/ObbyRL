# Offline RL Autotuning System

## Overview

The autotuning system automatically optimizes hyperparameters for the offline RL agent by running iterative training loops, evaluating performance, and making safe parameter adjustments.

## Quick Start

### Basic Usage (Recommended)
```bash
# Run autotuning with default settings (250 episodes per run, 60min timeout)
python autotune.py

# Custom episodes per run
python autotune.py --episodes 100

# Custom timeout
python autotune.py --timeout 30

# Custom output directory
python autotune.py --autotune-dir my_autotune_results
```

### Manual Control
```python
from autotune_offline import AutotuneManager

manager = AutotuneManager()
manager.episodes_per_run = 250
manager.timeout_minutes = 60
manager.run_autotune_loop()
```

## Directory Structure

The system creates an `autotune/` directory with:

```
autotune/
├── run_001.log              # Training logs for each run
├── run_002.log
├── config_run_001.json      # Configuration used for each run
├── config_run_002.json
├── metrics.csv              # Performance metrics across all runs
├── offline_trained_run_001.pt  # Best models saved
├── offline_trained_run_002.pt
└── SUMMARY.md               # Final summary report
```

## Git Integration

- Creates and switches to branch: `autotune/YYYY-MM-DD`
- Commits each configuration change with descriptive messages
- Example: `"autotune: change learning_rate from 0.001 to 0.0015 (run 003)"`

## System Behavior

### Runtime & Timeout Policy
- **Per-run timeout**: 60 minutes (configurable)
- **Timeout handling**: Automatically applies performance optimizations:
  1. Disable model compilation if enabled
  2. Reduce batch size (256 → 128)
  3. Increase learning rate if < 1e-3
- **Timeout recovery**: After 3 consecutive timeouts, applies alternative hyperparameter change
- **No episode reduction**: Maintains 250 episodes per run regardless of timeouts

### Improvement Criteria
A run is considered "improving" if ANY of:
- Current reward ≥ previous × 1.10
- Current reward - previous ≥ 75  
- Last 3 reward slopes are positive

### Hyperparameter Changes (applied one at a time)
1. **Epsilon decay**: Adjust exploration vs exploitation
2. **Learning rate**: Increase if stagnating, decrease if unstable
3. **Batch size**: Try 128, 256, or 512
4. **Target update tau**: Try 0.003, 0.01 (default 0.005)
5. **Gamma (discount factor)**: Try 0.995, 0.97 (default 0.99)
6. **Reward shaping**: Modify step penalties and progress multipliers

### Stopping Conditions
Autotuning stops when ANY condition is met:
- Final reward ≥ 500 (target reached)
- Two consecutive improving runs
- 10 total runs with no improvement
- 3 consecutive crashes (timeouts don't count as crashes)

## Output Formats

### Console Output
```
=== RUN 1 ===
Command: /usr/bin/python train_offline.py --episodes 250
Timeout: 60 minutes
Log file: autotune/run_001.log

Run 1 complete:
  Status: ok
  Reward: 125.45
  Best: 125.45
  Note: new_best
```

### Final Summary
```
BEST_REWARD=312.4
BEST_CONFIG={"learning_rate": 0.0015, "eps_decay": 0.997, "batch_size": 512}
RUNS=6
RECOMMENDATION=Test this config in real-time or expand reward target.
```

### Metrics CSV
```csv
run_id,timestamp,status,avg_reward,eps_decay,lr,batch_size,gamma,tau,note
1,2025-09-10T10:00:00,ok,125.45,0.999,0.001,128,0.99,0.005,new_best
2,2025-09-10T10:15:00,ok,156.78,0.997,0.001,128,0.99,0.005,improved
```

## Configuration Integration

The system automatically:
- Reads current parameters from `config/model_config.yaml`
- Updates configuration files when making changes
- Reloads the config manager to apply changes
- Saves configuration snapshots for each run

## Error Handling

- **Crashes**: Logged as "crashed" status, system continues
- **Timeouts**: Handled with performance optimizations
- **Invalid rewards**: Marked as status="invalid", system retries once
- **Config errors**: Warnings logged, system uses defaults where possible

## Safety Features

- **Single change per run**: Only one parameter modified at a time
- **Bounded changes**: Learning rate ≤ 0.01, batch size ∈ {128, 256, 512}
- **No architecture changes**: QNet structure remains untouched
- **Rollback support**: Git commits allow easy reversion
- **Graceful timeouts**: Proper process termination and cleanup

## Troubleshooting

### Common Issues

1. **Config file not found**
   - Ensure you're running from the `server/` directory
   - Check that `config/model_config.yaml` exists

2. **Git branch creation fails**
   - Ensure git is properly configured
   - Check repository status and permissions

3. **Training consistently times out**
   - System will automatically apply performance optimizations
   - Consider increasing timeout with `--timeout` parameter

4. **No improvement after many runs**
   - Check reward function configuration
   - Verify synthetic environment is providing reasonable rewards
   - Consider manual reward function tuning

### Debug Mode
For detailed logging, monitor the individual run logs:
```bash
tail -f autotune/run_001.log
```

## Performance Notes

- Model compilation is automatically disabled on timeout to speed up training
- Batch size reduction helps with memory and speed
- Learning rate adjustments can significantly impact convergence speed
- The system prioritizes forward progress over perfect optimization
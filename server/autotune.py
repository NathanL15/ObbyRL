#!/usr/bin/env python3
"""
Main autotune entry point - matches the interface specified in requirements.
This script can be invoked directly or import the AutotuneManager for custom use.
"""

import sys
import argparse

# Add current directory to path for imports
sys.path.append('.')
from autotune_offline import AutotuneManager


def main():
    """Main entry point for autotuning."""
    parser = argparse.ArgumentParser(description="Offline RL Autotuning System")
    parser.add_argument("--episodes", type=int, default=250, 
                        help="Episodes per training run (default: 250)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout per run in minutes (default: 60)")
    parser.add_argument("--autotune-dir", type=str, default="autotune",
                        help="Directory for autotune outputs (default: autotune)")
    
    args = parser.parse_args()
    
    # Create and configure autotuning manager
    manager = AutotuneManager(autotune_dir=args.autotune_dir)
    manager.episodes_per_run = args.episodes
    manager.timeout_minutes = args.timeout
    
    # Run the autotuning loop
    manager.run_autotune_loop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Configuration management utility for the RL system.
Allows viewing and updating configuration without restarting the server.
"""

import argparse
import json
import requests
import yaml
import configparser

def get_config(base_url, config_type):
    """Get current configuration from server."""
    url = f"{base_url}/config/{config_type}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.RequestException as e:
        print(f"Failed to connect to server: {e}")
        return None

def reload_config(base_url):
    """Reload configuration on server."""
    url = f"{base_url}/config/reload"
    try:
        response = requests.post(url, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"Config reload: {result['status']} - {result['message']}")
            return True
        else:
            print(f"Error {response.status_code}: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Failed to connect to server: {e}")
        return False

def update_reward_config(config_file):
    """Update reward configuration file with new values."""
    config = configparser.ConfigParser()
    
    # Example: increase checkpoint bonus
    config['checkpoints'] = {
        'checkpoint_bonus': '25.0',  # increased from 20.0
        'completion_base_bonus': '60.0',  # increased from 50.0
        'completion_cp_bonus': '12.0'
    }
    
    # Example: adjust progress rewards for faster learning
    config['progress_rewards'] = {
        'base_reward_per_step': '-0.003',  # less negative
        'progress_reward_scale': '4.0',    # higher scale
        'progress_reward_cap': '-1.5',
        'leap_threshold': '1.5',           # easier to trigger
        'leap_bonus': '1.5',               # bigger bonus
        'milestone_threshold': '0.8',      # more frequent milestones
        'milestone_bonus': '2.5',
        'sustained_threshold': '0.3',     # easier sustained progress
        'sustained_bonus': '0.8'
    }
    
    # Keep other sections unchanged from original
    config['penalties'] = {
        'death_penalty_hazard': '15.0',
        'death_penalty_fall': '8.0',
        'death_penalty_other': '10.0',
        'stuck_penalty': '8.0'
    }
    
    config['movement_rewards'] = {
        'heading_shape_scale': '0.002',
        'horizontal_progress_scale': '0.15',
        'jump_up_bonus': '1.0',
        'idle_speed_threshold': '1.0',
        'idle_penalty_per_step': '0.02'
    }
    
    config['hazards'] = {
        'hazard_near_radius': '15.0',
        'hazard_avoid_penalty': '0.02',
        'safe_progress_threshold': '0.25'
    }
    
    config['thresholds'] = {
        'backtrack_threshold': '1.0',
        'backtrack_penalty_scale': '0.05',
        'min_progress_eps': '0.5',
        'stuck_steps': '200'
    }
    
    with open(config_file, 'w') as f:
        config.write(f)
    
    print(f"Updated reward configuration saved to {config_file}")
    print("Key changes:")
    print("- Increased checkpoint bonuses for stronger goal-seeking")
    print("- Reduced base penalty and increased progress scale")
    print("- Lowered thresholds for bonuses to trigger more frequently")

def main():
    parser = argparse.ArgumentParser(description="Manage RL system configuration")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="Server base URL")
    parser.add_argument("--action", choices=["view", "reload", "update-rewards"], 
                       default="view", help="Action to perform")
    parser.add_argument("--config-type", choices=["reward", "model"], 
                       default="reward", help="Configuration type to view")
    parser.add_argument("--config-file", default="../config/reward_shaping.conf",
                       help="Config file path for updates")
    
    args = parser.parse_args()
    
    if args.action == "view":
        print(f"=== Current {args.config_type.title()} Configuration ===")
        config = get_config(args.url, args.config_type)
        if config:
            print(json.dumps(config, indent=2))
    
    elif args.action == "reload":
        print("=== Reloading Server Configuration ===")
        reload_config(args.url)
    
    elif args.action == "update-rewards":
        print("=== Updating Reward Configuration ===")
        update_reward_config(args.config_file)
        print("\nTo apply changes, run:")
        print(f"python config_tool.py --action reload --url {args.url}")

if __name__ == "__main__":
    main()
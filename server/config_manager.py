"""
Configuration management for the RL server.
Loads settings from config files and provides easy access.
"""
import os
import yaml
import configparser
from typing import Dict, Any

class Config:
    """Unified configuration manager for the RL system."""
    
    def __init__(self, config_dir: str = None):
        # Auto-detect config directory
        if config_dir is None:
            # Try relative to current working directory first
            if os.path.exists("config"):
                config_dir = "config"
            # Try relative to this file's directory
            elif os.path.exists(os.path.join(os.path.dirname(__file__), "..", "config")):
                config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
            else:
                config_dir = "config"  # fallback
        
        self.config_dir = config_dir
        self.model_config = {}
        self.reward_config = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files."""
        # Load model config (YAML)
        model_path = os.path.join(self.config_dir, "model_config.yaml")
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                self.model_config = yaml.safe_load(f)
        else:
            print(f"Warning: Model config not found at {model_path}, using defaults")
            self.model_config = self._get_default_model_config()
        
        # Load reward shaping config (INI format)
        reward_path = os.path.join(self.config_dir, "reward_shaping.conf")
        if os.path.exists(reward_path):
            parser = configparser.ConfigParser()
            parser.read(reward_path)
            # Convert to nested dict
            self.reward_config = {section: dict(parser.items(section)) 
                                for section in parser.sections()}
            # Convert numeric values
            self._convert_reward_config_types()
        else:
            print(f"Warning: Reward config not found at {reward_path}, using defaults")
            self.reward_config = self._get_default_reward_config()
    
    def _convert_reward_config_types(self):
        """Convert string values to appropriate numeric types."""
        for section_name, section in self.reward_config.items():
            for key, value in section.items():
                try:
                    # Try int first, then float
                    if '.' in value:
                        section[key] = float(value)
                    else:
                        section[key] = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Default model configuration if file not found."""
        return {
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'gamma': 0.99,
                'eps_start': 1.0,
                'eps_min': 0.05,
                'eps_decay': 0.999,
                'batch_size': 128,
                'update_every': 2,
                'target_update_tau': 0.005,
                'replay_buffer_size': 100000,
                'elite_buffer_size': 5000,
                'max_grad_norm': 10.0
            },
            'model': {
                'hidden_size': 256,
                'value_hidden': 128,
                'advantage_hidden': 128,
                'n_observations': 25,
                'n_actions': 7
            },
            'performance': {
                'log_every_n_steps': 50,
                'timing_log_every': 10,
                'max_stored_times': 100,
                'csv_enabled': True,
                'csv_filename': 'training_metrics.csv'
            },
            'optimization': {
                'compile_model': False,
                'use_mixed_precision': False,
                'enable_request_batching': False,
                'batch_timeout_ms': 10
            }
        }
    
    def _get_default_reward_config(self) -> Dict[str, Any]:
        """Default reward configuration if file not found."""
        return {
            'progress_rewards': {
                'base_reward_per_step': -0.005,
                'progress_reward_scale': 3.0,
                'progress_reward_cap': -2.0,
                'leap_threshold': 2.0,
                'leap_bonus': 1.0,
                'milestone_threshold': 1.0,
                'milestone_bonus': 2.0,
                'sustained_threshold': 0.5,
                'sustained_bonus': 0.5
            },
            'checkpoints': {
                'checkpoint_bonus': 20.0,
                'completion_base_bonus': 50.0,
                'completion_cp_bonus': 10.0
            },
            'penalties': {
                'death_penalty_hazard': 15.0,
                'death_penalty_fall': 8.0,
                'death_penalty_other': 10.0,
                'stuck_penalty': 8.0
            }
        }
    
    def get(self, section: str, key: str, default=None):
        """Get a configuration value with fallback to default."""
        if section in self.model_config:
            return self.model_config[section].get(key, default)
        elif section in self.reward_config:
            return self.reward_config[section].get(key, default)
        return default
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get all training-related configuration."""
        return self.model_config.get('training', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return self.model_config.get('model', {})
    
    def get_reward_config(self) -> Dict[str, Any]:
        """Get reward shaping configuration."""
        return self.reward_config
    
    def reload(self):
        """Reload configuration from files."""
        self._load_configs()
        
    def __repr__(self):
        return f"Config(model_sections={list(self.model_config.keys())}, reward_sections={list(self.reward_config.keys())})"

# Global config instance
config = Config()
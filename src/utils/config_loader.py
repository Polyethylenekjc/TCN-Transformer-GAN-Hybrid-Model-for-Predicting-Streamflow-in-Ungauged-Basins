"""Configuration loader from YAML files."""

import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage YAML configuration."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Dictionary containing all configuration parameters
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save config.yaml
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def get_nested(config: Dict[str, Any], keys: str, default=None):
        """
        Get nested value from config using dot notation.
        
        Args:
            config: Configuration dictionary
            keys: Dot-separated keys (e.g., 'data.image_dir')
            default: Default value if key not found
            
        Returns:
            Value from config or default
        """
        parts = keys.split('.')
        value = config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
        return value if value is not None else default

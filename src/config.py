"""
Configuration management for DermaLens
"""
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_path: str = "./configs/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access"""
        return self.get(key)


def get_config() -> Config:
    """Get global config instance"""
    return Config()

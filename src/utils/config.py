import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimpleConfig:
    def __init__(self):
        # Direct path to configs from project root
        self.config_dir = Path("configs")
        self.configs = {}
        self._load_configs()
    
    def _load_configs(self):
        config_files = {
            'system': 'system_config.yaml',
            'asr': 'asr_config.yaml', 
            'diarization': 'diarization_config.yaml',
            'audio': 'audio_preprocess_config.yaml',
            'streaming': 'streaming_config.yaml',
            'punctuation': 'punctuation_config.yaml',
            'output': 'output_config.yaml'
        }
        
        for config_name, filename in config_files.items():
            try:
                config_path = self.config_dir / filename
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.configs[config_name] = yaml.safe_load(f)
                    logger.info(f'Loaded config: {filename}')
                else:
                    logger.warning(f'Config file not found: {config_path}')
                    self.configs[config_name] = {}
            except Exception as e:
                logger.error(f'Error loading config {filename}: {e}')
                self.configs[config_name] = {}
    
    def get(self, section, key=None, default=None):
        config = self.configs.get(section, {})
        return config.get(key, default) if key else config

# Global config instance
config = SimpleConfig()

# Create config_manager alias for backward compatibility
config_manager = config

# Convenience functions
def get_system_config(): return config.get('system')
def get_asr_config(): return config.get('asr')
def get_diarization_config(): return config.get('diarization')
def get_audio_config(): return config.get('audio')
def get_streaming_config(): return config.get('streaming')
def get_punctuation_config(): return config.get('punctuation')
def get_output_config(): return config.get('output')
def get_audio_preprocess_config(): return config.get('audio')

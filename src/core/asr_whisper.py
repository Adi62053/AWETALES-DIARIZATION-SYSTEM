import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class OfflineASR:
    """Offline ASR implementation using Whisper"""
    
    def __init__(self, model_size="base"):
        logger.info(f"Offline ASR initialized with model: {model_size}")
        self.model_size = model_size
        self.initialized = True
    
    def transcribe(self, audio_path: str, language: str = "zh") -> List[str]:
        """Transcribe audio file to text"""
        logger.info(f"Transcribing: {audio_path}")
        return [f"Offline ASR transcription for {audio_path} (language: {language})"]
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[str]:
        """Transcribe audio data to text"""
        duration = len(audio_data) / sample_rate
        return [f"Offline ASR: {duration:.1f}s audio processed"]

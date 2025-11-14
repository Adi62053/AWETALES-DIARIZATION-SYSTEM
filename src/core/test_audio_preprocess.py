"""
Tests for audio preprocessing components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.audio_preprocess import AudioPreprocessor


class TestAudioPreprocessor:
    """Test audio preprocessing functionality"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create audio preprocessor instance"""
        return AudioPreprocessor()
    
    @pytest.mark.asyncio
    async def test_denoise_audio(self, preprocessor, temp_audio_file):
        """Test audio denoising functionality"""
        with patch('src.core.audio_preprocess.uvr_denoise') as mock_denoise:
            mock_denoise.return_value = temp_audio_file
            
            result = await preprocessor.denoise_audio(temp_audio_file)
            
            assert result == temp_audio_file
            mock_denoise.assert_called_once_with(temp_audio_file)
    
    @pytest.mark.asyncio
    async def test_vad_detection(self, preprocessor, temp_audio_file):
        """Test voice activity detection"""
        with patch('src.core.audio_preprocess.fsmn_vad') as mock_vad:
            mock_vad.return_value = [
                {"start": 0.0, "end": 1.5, "confidence": 0.95},
                {"start": 3.0, "end": 4.5, "confidence": 0.92}
            ]
            
            segments = await preprocessor.detect_voice_activity(temp_audio_file)
            
            assert len(segments) == 2
            assert segments[0]["start"] == 0.0
            assert segments[0]["end"] == 1.5
            mock_vad.assert_called_once_with(temp_audio_file)
    
    @pytest.mark.asyncio
    async def test_endpoint_detection(self, preprocessor, temp_audio_file):
        """Test endpoint detection"""
        with patch('src.core.audio_preprocess.campp_endpoint') as mock_endpoint:
            mock_endpoint.return_value = {
                "speech_starts": [0.0, 3.0],
                "speech_ends": [1.5, 4.5],
                "confidence": 0.94
            }
            
            endpoints = await preprocessor.detect_endpoints(temp_audio_file)
            
            assert "speech_starts" in endpoints
            assert len(endpoints["speech_starts"]) == 2
            mock_endpoint.assert_called_once_with(temp_audio_file)
    
    def test_audio_validation_valid_file(self, preprocessor, temp_audio_file):
        """Test valid audio file validation"""
        is_valid, message = preprocessor.validate_audio_file(temp_audio_file)
        assert is_valid is True
        assert "valid" in message.lower()
    
    def test_audio_validation_invalid_file(self, preprocessor):
        """Test invalid audio file validation"""
        is_valid, message = preprocessor.validate_audio_file("nonexistent.wav")
        assert is_valid is False
        assert "not exist" in message.lower() or "invalid" in message.lower()
"""
Tests for diarization components
"""

import pytest
from unittest.mock import Mock, patch

from src.core.diarization import DiarizationEngine


class TestDiarizationEngine:
    """Test diarization functionality"""
    
    @pytest.fixture
    def diarization_engine(self):
        """Create diarization engine instance"""
        return DiarizationEngine()
    
    @pytest.mark.asyncio
    async def test_diarize_audio(self, diarization_engine, temp_audio_file):
        """Test audio diarization"""
        with patch('src.core.diarization.pyannote_diarize') as mock_diarize:
            mock_diarize.return_value = [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.5,
                    "confidence": 0.92
                },
                {
                    "speaker": "SPEAKER_01", 
                    "start": 2.5,
                    "end": 5.0,
                    "confidence": 0.88
                }
            ]
            
            segments = await diarization_engine.diarize(temp_audio_file)
            
            assert len(segments) == 2
            assert segments[0]["speaker"] == "SPEAKER_00"
            assert segments[1]["speaker"] == "SPEAKER_01"
            mock_diarize.assert_called_once_with(temp_audio_file)
    
    @pytest.mark.asyncio
    async def test_speaker_count_estimation(self, diarization_engine, temp_audio_file):
        """Test speaker count estimation"""
        with patch('src.core.diarization.estimate_speaker_count') as mock_estimate:
            mock_estimate.return_value = 3
            
            count = await diarization_engine.estimate_speaker_count(temp_audio_file)
            
            assert count == 3
            mock_estimate.assert_called_once_with(temp_audio_file)
    
    @pytest.mark.asyncio 
    async def test_overlap_detection(self, diarization_engine, temp_audio_file):
        """Test speaker overlap detection"""
        with patch('src.core.diarization.detect_overlap') as mock_overlap:
            mock_overlap.return_value = [
                {
                    "start": 1.0,
                    "end": 1.5,
                    "speakers": ["SPEAKER_00", "SPEAKER_01"],
                    "confidence": 0.85
                }
            ]
            
            overlaps = await diarization_engine.detect_overlaps(temp_audio_file)
            
            assert len(overlaps) == 1
            assert len(overlaps[0]["speakers"]) == 2
            mock_overlap.assert_called_once_with(temp_audio_file)
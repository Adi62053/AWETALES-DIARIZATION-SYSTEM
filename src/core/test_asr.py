"""
Tests for ASR components
"""

import pytest
from unittest.mock import Mock, patch

from src.core.asr_paraformer import ParaformerASR
from src.core.asr_whisper import WhisperASR


class TestParaformerASR:
    """Test Paraformer streaming ASR"""
    
    @pytest.fixture
    def paraformer_asr(self):
        """Create Paraformer ASR instance"""
        return ParaformerASR()
    
    @pytest.mark.asyncio
    async def test_streaming_transcribe(self, paraformer_asr, sample_audio_chunk):
        """Test streaming transcription"""
        with patch('src.core.asr_paraformer.paraformer_transcribe') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "test transcription",
                "confidence": 0.95,
                "words": [
                    {"word": "test", "start": 0.0, "end": 0.5, "confidence": 0.96},
                    {"word": "transcription", "start": 0.5, "end": 1.2, "confidence": 0.94}
                ]
            }
            
            result = await paraformer_asr.transcribe_stream(sample_audio_chunk)
            
            assert result["text"] == "test transcription"
            assert result["confidence"] >= 0.9
            mock_transcribe.assert_called_once_with(sample_audio_chunk)
    
    @pytest.mark.asyncio
    async def test_language_detection(self, paraformer_asr, sample_audio_chunk):
        """Test language detection"""
        with patch('src.core.asr_paraformer.detect_language') as mock_detect:
            mock_detect.return_value = {"language": "en", "confidence": 0.98}
            
            lang_info = await paraformer_asr.detect_language(sample_audio_chunk)
            
            assert lang_info["language"] == "en"
            assert lang_info["confidence"] >= 0.9
            mock_detect.assert_called_once_with(sample_audio_chunk)


class TestWhisperASR:
    """Test Whisper offline ASR"""
    
    @pytest.fixture
    def whisper_asr(self):
        """Create Whisper ASR instance"""
        return WhisperASR()
    
    @pytest.mark.asyncio
    async def test_offline_transcribe(self, whisper_asr, temp_audio_file):
        """Test offline transcription"""
        with patch('src.core.asr_whisper.whisper_transcribe') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "complete offline transcription",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "complete offline",
                        "confidence": 0.96
                    },
                    {
                        "start": 2.0,
                        "end": 4.0, 
                        "text": "transcription",
                        "confidence": 0.94
                    }
                ],
                "language": "en"
            }
            
            result = await whisper_asr.transcribe_offline(temp_audio_file)
            
            assert "complete offline transcription" in result["text"]
            assert len(result["segments"]) == 2
            assert result["language"] == "en"
            mock_transcribe.assert_called_once_with(temp_audio_file)
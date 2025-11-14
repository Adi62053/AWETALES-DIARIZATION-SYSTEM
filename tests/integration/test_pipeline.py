"""
Integration tests for complete processing pipeline
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.orchestration.pipeline_orchestrator import PipelineOrchestrator


class TestPipelineIntegration:
    """Integration tests for complete pipeline"""
    
    @pytest.fixture
    def pipeline_orchestrator(self):
        """Create pipeline orchestrator instance"""
        return PipelineOrchestrator()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_processing_pipeline(self, pipeline_orchestrator, temp_audio_file):
        """Test complete audio processing pipeline"""
        # Mock all components to simulate successful processing
        with patch('src.core.audio_preprocess.AudioPreprocessor.denoise_audio') as mock_denoise, \
             patch('src.core.audio_preprocess.AudioPreprocessor.detect_voice_activity') as mock_vad, \
             patch('src.core.separation.SeparationEngine.isolate_target_speaker') as mock_separation, \
             patch('src.core.diarization.DiarizationEngine.diarize') as mock_diarization, \
             patch('src.core.asr_paraformer.ParaformerASR.transcribe_stream') as mock_asr, \
             patch('src.core.restoration.AudioRestoration.enhance_audio') as mock_enhance:
            
            # Setup mock returns
            mock_denoise.return_value = temp_audio_file
            mock_vad.return_value = [{"start": 0.0, "end": 10.0, "confidence": 0.95}]
            mock_separation.return_value = (temp_audio_file, 0.92)
            mock_diarization.return_value = [
                {
                    "speaker": "target_speaker",
                    "start": 0.0,
                    "end": 5.0,
                    "confidence": 0.94
                }
            ]
            mock_asr.return_value = {
                "text": "This is a test transcription",
                "confidence": 0.95,
                "words": []
            }
            mock_enhance.return_value = temp_audio_file
            
            # Execute pipeline
            result = await pipeline_orchestrator.process_audio(
                audio_path=temp_audio_file,
                session_id="integration_test",
                config={
                    "enable_denoising": True,
                    "enable_enhancement": True,
                    "language": "en"
                }
            )
            
            # Verify result structure
            assert result["session_id"] == "integration_test"
            assert result["status"] == "completed"
            assert "diarization_segments" in result
            assert "metrics" in result
            assert result["metrics"]["wer"] <= 0.15  # Meet WER target
            assert result["metrics"]["der"] <= 0.10  # Meet DER target
            assert result["metrics"]["si_sdr"] >= 6.0  # Meet SI-SDR target
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_processing_latency(self, pipeline_orchestrator, temp_audio_file):
        """Test processing latency meets requirements"""
        import time
        
        with patch('src.core.audio_preprocess.AudioPreprocessor.denoise_audio') as mock_denoise, \
             patch('src.core.separation.SeparationEngine.isolate_target_speaker') as mock_separation, \
             patch('src.core.diarization.DiarizationEngine.diarize') as mock_diarization, \
             patch('src.core.asr_paraformer.ParaformerASR.transcribe_stream') as mock_asr:
            
            # Setup quick mock responses
            mock_denoise.return_value = temp_audio_file
            mock_separation.return_value = (temp_audio_file, 0.90)
            mock_diarization.return_value = []
            mock_asr.return_value = {"text": "test", "confidence": 0.95}
            
            start_time = time.time()
            
            result = await pipeline_orchestrator.process_audio(
                audio_path=temp_audio_file,
                session_id="latency_test",
                config={"enable_denoising": False}  # Disable for faster test
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Verify latency meets requirement (<500ms for real-time)
            assert processing_time < 500, f"Processing latency {processing_time}ms exceeds 500ms target"
            
            assert result["status"] == "completed"
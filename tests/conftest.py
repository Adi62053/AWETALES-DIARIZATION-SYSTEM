"""
Pytest configuration and fixtures for Awetales Diarization System tests
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import numpy as np

# Configure test environment
os.environ["TESTING"] = "true"
os.environ["MODEL_CACHE_DIR"] = "./test_model_cache"


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Create a minimal WAV file header (44 bytes + some silent audio)
        sample_rate = 16000
        duration = 2.0  # seconds
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
        
        # Simple WAV header (simplified)
        f.write(b"RIFF")
        f.write((36 + len(samples) * 2).to_bytes(4, 'little'))  # File size
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write((16).to_bytes(4, 'little'))  # fmt chunk size
        f.write((1).to_bytes(2, 'little'))   # PCM format
        f.write((1).to_bytes(2, 'little'))   # Mono
        f.write(sample_rate.to_bytes(4, 'little'))  # Sample rate
        f.write((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        f.write(b"data")
        f.write((len(samples) * 2).to_bytes(4, 'little'))  # Data chunk size
        f.write(samples.tobytes())
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_audio_chunk():
    """Generate sample audio chunk data"""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = np.random.randint(-32768, 32767, int(sample_rate * duration), dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def mock_session_config():
    """Mock session configuration"""
    return {
        "session_id": "test_session_123",
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration": 2.0,
        "language": "en",
        "enable_denoising": True,
        "enable_enhancement": True
    }


@pytest.fixture
def mock_diarization_result():
    """Mock diarization result"""
    return {
        "speaker": "speaker_1",
        "start_time": 0.0,
        "end_time": 1.5,
        "text": "This is a test transcription.",
        "confidence": 0.95
    }


@pytest.fixture
def mock_asr_result():
    """Mock ASR result"""
    return {
        "text": "This is a test transcription.",
        "start_time": 0.0,
        "end_time": 1.5,
        "confidence": 0.92,
        "words": [
            {"word": "This", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "is", "start": 0.3, "end": 0.5, "confidence": 0.93},
            {"word": "a", "start": 0.5, "end": 0.6, "confidence": 0.90},
            {"word": "test", "start": 0.6, "end": 1.0, "confidence": 0.94},
            {"word": "transcription", "start": 1.0, "end": 1.5, "confidence": 0.91}
        ]
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_pipeline_orchestrator():
    """Mock pipeline orchestrator"""
    orchestrator = Mock()
    orchestrator.process_audio = AsyncMock(return_value={
        "session_id": "test_session",
        "status": "completed",
        "processing_time": 2.5,
        "audio_duration": 10.0,
        "diarization_segments": [
            {
                "speaker": "speaker_1",
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "First segment text",
                "confidence": 0.95
            }
        ],
        "target_speaker_audio_path": "/tmp/target_speaker.wav",
        "metrics": {
            "wer": 0.12,
            "der": 0.08,
            "si_sdr": 8.5,
            "latency": 450.0
        }
    })
    return orchestrator


@pytest.fixture
async def mock_resource_manager():
    """Mock resource manager"""
    manager = Mock()
    manager.estimate_processing_time = AsyncMock(return_value=3.0)
    manager.get_system_status = AsyncMock(return_value={
        "overall_status": "healthy",
        "components": {
            "audio_preprocess": "healthy",
            "separation": "healthy",
            "diarization": "healthy",
            "asr": "healthy"
        },
        "queue_size": 0,
        "gpu_available": True
    })
    return manager
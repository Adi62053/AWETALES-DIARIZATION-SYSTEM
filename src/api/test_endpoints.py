"""
Tests for API endpoints
"""

import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.endpoints import initialize_components


class TestAPIEndpoints:
    """Test API endpoints functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # Initialize components with mocks
        initialize_components()
        
        with TestClient(app) as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "components" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Awetales Diarization System API" in data["message"]
    
    @patch('src.api.endpoints.pipeline_orchestrator')
    def test_process_audio_endpoint(self, mock_orchestrator, client, temp_audio_file):
        """Test audio processing endpoint"""
        mock_orchestrator.process_audio = AsyncMock(return_value={
            "session_id": "test_123",
            "status": "completed",
            "processing_time": 2.5,
            "diarization_segments": [],
            "metrics": {}
        })
        
        with open(temp_audio_file, 'rb') as audio_file:
            response = client.post(
                "/api/v1/process",
                files={"audio_file": ("test.wav", audio_file, "audio/wav")},
                data={
                    "language": "en",
                    "enable_denoising": "true",
                    "priority": "5"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "session_id" in data
    
    def test_process_audio_invalid_file(self, client):
        """Test audio processing with invalid file"""
        response = client.post(
            "/api/v1/process",
            files={"audio_file": ("test.txt", b"not audio data", "text/plain")}
        )
        
        assert response.status_code == 400
    
    @patch('src.api.endpoints.output_manager')
    def test_get_processing_result(self, mock_output, client):
        """Test getting processing results"""
        mock_output.get_results = AsyncMock(return_value={
            "session_id": "test_123",
            "status": "completed",
            "processing_time": 2.5,
            "audio_duration": 10.0,
            "diarization_segments": [
                {
                    "speaker": "speaker_1",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": "Test transcription",
                    "confidence": 0.95
                }
            ],
            "metrics": {
                "wer": 0.12,
                "der": 0.08,
                "si_sdr": 8.5
            }
        })
        
        response = client.get("/api/v1/process/test_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["results"]) == 1
        assert data["results"][0]["speaker"] == "speaker_1"
    
    def test_get_processing_result_not_found(self, client):
        """Test getting non-existent processing results"""
        response = client.get("/api/v1/process/nonexistent")
        
        assert response.status_code == 404
    
    @patch('src.api.endpoints.resource_manager')
    def test_system_status(self, mock_resource, client):
        """Test system status endpoint"""
        mock_resource.get_system_status = AsyncMock(return_value={
            "overall_status": "healthy",
            "components": {
                "audio_preprocess": "healthy",
                "separation": "healthy",
                "diarization": "healthy"
            },
            "queue_size": 0,
            "gpu_available": True
        })
        
        response = client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "healthy"
        assert "components" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/api/v1/metrics/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "performance_metrics" in data
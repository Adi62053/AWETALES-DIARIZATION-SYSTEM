"""
Tests for WebSocket functionality
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from src.api.websocket_handler import ConnectionManager, websocket_endpoint


class TestWebSocketHandler:
    """Test WebSocket handler functionality"""
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager instance"""
        return ConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.receive = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    async def test_connection_management(self, connection_manager, mock_websocket):
        """Test WebSocket connection management"""
        session_id = "test_session_123"
        
        await connection_manager.connect(mock_websocket, session_id)
        
        assert session_id in connection_manager.active_connections
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_initialization(self, connection_manager, mock_websocket):
        """Test session initialization"""
        session_id = "test_session_123"
        
        config = {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_duration": 2.0,
            "language": "en"
        }
        
        with patch.object(connection_manager, 'initialize_session') as mock_init:
            mock_init.return_value = None
            
            await connection_manager.handle_control_message(session_id, {
                "type": "initialize",
                "config": config
            })
            
            mock_init.assert_called_once_with(session_id, config)
    
    @pytest.mark.asyncio
    async def test_audio_chunk_processing(self, connection_manager, mock_websocket, sample_audio_chunk):
        """Test audio chunk processing"""
        session_id = "test_session_123"
        
        # First initialize session
        connection_manager.buffer_managers[session_id] = Mock()
        connection_manager.buffer_managers[session_id].add_audio_chunk = AsyncMock()
        
        connection_manager.session_managers[session_id] = Mock()
        connection_manager.session_managers[session_id].update_activity = Mock()
        
        await connection_manager.process_audio_chunk(session_id, sample_audio_chunk)
        
        connection_manager.buffer_managers[session_id].add_audio_chunk.assert_called_once_with(sample_audio_chunk)
        connection_manager.session_managers[session_id].update_activity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, connection_manager, mock_websocket):
        """Test error handling"""
        session_id = "test_session_123"
        
        await connection_manager.send_error(session_id, "Test error message")
        
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "error"
        assert call_args["error"] == "Test error message"
    
    @pytest.mark.asyncio
    async def test_session_finalization(self, connection_manager, mock_websocket):
        """Test session finalization"""
        session_id = "test_session_123"
        
        connection_manager.active_connections[session_id] = mock_websocket
        connection_manager.stream_processors[session_id] = Mock()
        connection_manager.stream_processors[session_id].stop_processing = AsyncMock(return_value={})
        
        await connection_manager.finalize_session(session_id)
        
        mock_websocket.send_json.assert_called()
        # Check that final_results message was sent
        call_args = [call[0][0] for call in mock_websocket.send_json.call_args_list]
        message_types = [msg["type"] for msg in call_args]
        assert "final_results" in message_types
        assert "session_ended" in message_types
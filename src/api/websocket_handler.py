"""
WebSocket Handler for Real-time Audio Streaming
for Awetales Diarization System

This module implements the WebSocket endpoint for real-time
audio streaming, processing, and incremental result delivery
for the Target Speaker Diarization + ASR System.
"""

import asyncio
import json
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import struct

from fastapi import WebSocket, WebSocketDisconnect, WebSocketException, status
from fastapi.routing import APIRouter

from src.streaming.buffer_manager import BufferManager
from src.streaming.session_manager import SessionManager
from src.streaming.stream_processor import StreamProcessor
from src.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.health_check import HealthChecker

# Configure logging
logger = logging.getLogger("awetales_websocket")


@dataclass
class ConnectionConfig:
    """Configuration for WebSocket connection"""
    session_id: str
    target_speaker_embedding: Optional[str] = None
    sample_rate: int = 16000
    sample_width: int = 2  # 16-bit PCM
    channels: int = 1
    chunk_duration: float = 2.0  # seconds
    enable_denoising: bool = True
    enable_enhancement: bool = True
    language: str = "en"
    max_silence_duration: float = 10.0  # seconds


class ConnectionManager:
    """Manages WebSocket connections and session state"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_configs: Dict[str, ConnectionConfig] = {}
        self.buffer_managers: Dict[str, BufferManager] = {}
        self.session_managers: Dict[str, SessionManager] = {}
        self.stream_processors: Dict[str, StreamProcessor] = {}
        
        # Initialize core components
        self.pipeline_orchestrator = PipelineOrchestrator()
        self.performance_monitor = PerformanceMonitor()
        self.health_checker = HealthChecker()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and initialize session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        logger.info(f"WebSocket connected: {session_id}")
        
        # Send connection acknowledgement
        await self.send_message(session_id, {
            "type": "connection_ack",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "connected"
        })
    
    def disconnect(self, session_id: str):
        """Clean up connection resources"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        if session_id in self.connection_configs:
            del self.connection_configs[session_id]
        
        # Clean up processing components
        if session_id in self.buffer_managers:
            self.buffer_managers[session_id].cleanup()
            del self.buffer_managers[session_id]
        
        if session_id in self.session_managers:
            self.session_managers[session_id].cleanup()
            del self.session_managers[session_id]
        
        if session_id in self.stream_processors:
            del self.stream_processors[session_id]
        
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send JSON message to WebSocket client"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {str(e)}")
                self.disconnect(session_id)
    
    async def send_audio_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """Send audio processing metrics to client"""
        await self.send_message(session_id, {
            "type": "audio_metrics",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
    
    async def send_diarization_result(self, session_id: str, result: Dict[str, Any]):
        """Send incremental diarization result to client"""
        await self.send_message(session_id, {
            "type": "diarization_result",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    
    async def send_asr_result(self, session_id: str, result: Dict[str, Any]):
        """Send incremental ASR result to client"""
        await self.send_message(session_id, {
            "type": "asr_result",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
    
    async def send_speaker_change(self, session_id: str, speaker_info: Dict[str, Any]):
        """Send speaker change notification to client"""
        await self.send_message(session_id, {
            "type": "speaker_change",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker_info
        })
    
    async def send_error(self, session_id: str, error: str, fatal: bool = False):
        """Send error message to client"""
        await self.send_message(session_id, {
            "type": "error",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "fatal": fatal
        })
        
        if fatal:
            logger.error(f"Fatal error for {session_id}: {error}")
            self.disconnect(session_id)
    
    async def initialize_session(self, session_id: str, config: Dict[str, Any]):
        """Initialize processing components for a session"""
        try:
            # Create connection configuration
            connection_config = ConnectionConfig(
                session_id=session_id,
                target_speaker_embedding=config.get("target_speaker_embedding"),
                sample_rate=config.get("sample_rate", 16000),
                sample_width=config.get("sample_width", 2),
                channels=config.get("channels", 1),
                chunk_duration=config.get("chunk_duration", 2.0),
                enable_denoising=config.get("enable_denoising", True),
                enable_enhancement=config.get("enable_enhancement", True),
                language=config.get("language", "en"),
                max_silence_duration=config.get("max_silence_duration", 10.0)
            )
            
            self.connection_configs[session_id] = connection_config
            
            # Initialize buffer manager
            buffer_manager = BufferManager(
                session_id=session_id,
                sample_rate=connection_config.sample_rate,
                chunk_duration=connection_config.chunk_duration,
                channels=connection_config.channels
            )
            self.buffer_managers[session_id] = buffer_manager
            
            # Initialize session manager
            session_manager = SessionManager(
                session_id=session_id,
                buffer_manager=buffer_manager,
                max_silence_duration=connection_config.max_silence_duration
            )
            self.session_managers[session_id] = session_manager
            
            # Initialize stream processor
            stream_processor = StreamProcessor(
                session_id=session_id,
                buffer_manager=buffer_manager,
                session_manager=session_manager,
                pipeline_orchestrator=self.pipeline_orchestrator,
                performance_monitor=self.performance_monitor
            )
            self.stream_processors[session_id] = stream_processor
            
            # Set up result callbacks
            stream_processor.set_diarization_callback(
                lambda result: self.send_diarization_result(session_id, result)
            )
            stream_processor.set_asr_callback(
                lambda result: self.send_asr_result(session_id, result)
            )
            stream_processor.set_speaker_change_callback(
                lambda speaker: self.send_speaker_change(session_id, speaker)
            )
            stream_processor.set_metrics_callback(
                lambda metrics: self.send_audio_metrics(session_id, metrics)
            )
            stream_processor.set_error_callback(
                lambda error: self.send_error(session_id, error)
            )
            
            # Start stream processing
            await stream_processor.start_processing(connection_config.__dict__)
            
            logger.info(f"Session initialized: {session_id}")
            
            await self.send_message(session_id, {
                "type": "session_initialized",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "config": config
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {str(e)}")
            await self.send_error(session_id, f"Session initialization failed: {str(e)}", fatal=True)
            raise
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes):
        """Process incoming audio chunk"""
        if session_id not in self.buffer_managers:
            await self.send_error(session_id, "Session not initialized", fatal=True)
            return
        
        try:
            buffer_manager = self.buffer_managers[session_id]
            
            # Add audio chunk to buffer
            await buffer_manager.add_audio_chunk(audio_data)
            
            # Update session activity
            session_manager = self.session_managers[session_id]
            session_manager.update_activity()
            
            # Send audio receipt acknowledgement
            await self.send_message(session_id, {
                "type": "audio_ack",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "chunk_size": len(audio_data)
            })
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for {session_id}: {str(e)}")
            await self.send_error(session_id, f"Audio processing error: {str(e)}")
    
    async def handle_control_message(self, session_id: str, message: Dict[str, Any]):
        """Handle control messages from client"""
        message_type = message.get("type")
        
        try:
            if message_type == "initialize":
                await self.initialize_session(session_id, message.get("config", {}))
            
            elif message_type == "pause":
                if session_id in self.stream_processors:
                    await self.stream_processors[session_id].pause_processing()
                    await self.send_message(session_id, {
                        "type": "processing_paused",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "resume":
                if session_id in self.stream_processors:
                    await self.stream_processors[session_id].resume_processing()
                    await self.send_message(session_id, {
                        "type": "processing_resumed",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "update_config":
                if session_id in self.connection_configs:
                    # Update configuration
                    config = self.connection_configs[session_id]
                    new_config = message.get("config", {})
                    
                    if "target_speaker_embedding" in new_config:
                        config.target_speaker_embedding = new_config["target_speaker_embedding"]
                    
                    if "language" in new_config:
                        config.language = new_config["language"]
                    
                    await self.send_message(session_id, {
                        "type": "config_updated",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "config": new_config
                    })
            
            elif message_type == "get_status":
                status_info = await self.get_session_status(session_id)
                await self.send_message(session_id, {
                    "type": "status_report",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": status_info
                })
            
            elif message_type == "end_stream":
                await self.finalize_session(session_id)
            
            else:
                await self.send_error(session_id, f"Unknown control message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling control message for {session_id}: {str(e)}")
            await self.send_error(session_id, f"Control message error: {str(e)}")
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status"""
        if session_id not in self.session_managers:
            return {"status": "not_initialized"}
        
        try:
            session_manager = self.session_managers[session_id]
            buffer_manager = self.buffer_managers[session_id]
            
            return {
                "status": "active",
                "audio_buffer_duration": buffer_manager.get_buffer_duration(),
                "processed_chunks": session_manager.get_processed_chunk_count(),
                "active_speakers": session_manager.get_active_speakers(),
                "last_activity": session_manager.get_last_activity().isoformat(),
                "performance_metrics": self.performance_monitor.get_session_metrics(session_id)
            }
        except Exception as e:
            logger.error(f"Error getting status for {session_id}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def finalize_session(self, session_id: str):
        """Finalize session and send final results"""
        try:
            if session_id in self.stream_processors:
                # Get final results from stream processor
                final_results = await self.stream_processors[session_id].stop_processing()
                
                # Send final results
                await self.send_message(session_id, {
                    "type": "final_results",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "results": final_results
                })
                
                logger.info(f"Session finalized: {session_id}")
            
            # Send session end message
            await self.send_message(session_id, {
                "type": "session_ended",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error finalizing session {session_id}: {str(e)}")
            await self.send_error(session_id, f"Session finalization error: {str(e)}")
        
        finally:
            # Clean up connection
            self.disconnect(session_id)


# Global connection manager instance
connection_manager = ConnectionManager()

# WebSocket router
websocket_router = APIRouter()


@websocket_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming and processing
    
    Protocol:
    - Client connects and sends initialization message
    - Audio chunks are sent as binary messages
    - Control messages are sent as JSON messages
    - Results are streamed back as JSON messages
    
    Message Types:
    - Client to Server:
        * initialize: Initialize session with configuration
        * audio_chunk: Binary audio data
        * pause/resume: Control processing
        * update_config: Update processing parameters
        * get_status: Request current status
        * end_stream: End session gracefully
    
    - Server to Client:
        * connection_ack: Connection established
        * session_initialized: Session ready
        * audio_ack: Audio chunk received
        * diarization_result: Incremental diarization
        * asr_result: Incremental transcription
        * speaker_change: Speaker activity change
        * audio_metrics: Processing metrics
        * status_report: Current session status
        * final_results: Complete results
        * error: Error notification
    """
    
    session_id = str(uuid.uuid4())
    
    try:
        # Accept connection
        await connection_manager.connect(websocket, session_id)
        
        # Main message handling loop
        while True:
            # Wait for message with timeout
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=300.0  # 5-minute timeout
                )
            except asyncio.TimeoutError:
                await connection_manager.send_error(
                    session_id, 
                    "Connection timeout - no activity", 
                    fatal=True
                )
                break
            
            # Handle different message types
            if message["type"] == "websocket.receive":
                if "text" in message:
                    # JSON control message
                    try:
                        data = json.loads(message["text"])
                        await connection_manager.handle_control_message(session_id, data)
                    
                    except json.JSONDecodeError as e:
                        await connection_manager.send_error(
                            session_id, 
                            f"Invalid JSON message: {str(e)}"
                        )
                
                elif "bytes" in message:
                    # Binary audio data
                    audio_data = message["bytes"]
                    await connection_manager.process_audio_chunk(session_id, audio_data)
                
                else:
                    await connection_manager.send_error(
                        session_id, 
                        "Invalid message format"
                    )
            
            elif message["type"] == "websocket.disconnect":
                logger.info(f"Client disconnected: {session_id}")
                break
            
            else:
                await connection_manager.send_error(
                    session_id, 
                    f"Unexpected message type: {message['type']}"
                )
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {session_id}")
    
    except WebSocketException as e:
        logger.error(f"WebSocket error for {session_id}: {str(e)}")
        await connection_manager.send_error(session_id, f"WebSocket error: {str(e)}", fatal=True)
    
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler for {session_id}: {str(e)}")
        await connection_manager.send_error(session_id, f"Unexpected error: {str(e)}", fatal=True)
    
    finally:
        # Clean up connection
        connection_manager.disconnect(session_id)


@websocket_router.get("/stream/info")
async def get_websocket_info():
    """Get WebSocket endpoint information and requirements"""
    return {
        "endpoint": "/api/v1/stream",
        "protocol": "WebSocket (RFC 6455)",
        "audio_format": "16-bit PCM, 16kHz, mono",
        "chunk_size": "Recommended 1-2 seconds",
        "message_types": {
            "client_to_server": [
                "initialize", "audio_chunk", "pause", "resume", 
                "update_config", "get_status", "end_stream"
            ],
            "server_to_client": [
                "connection_ack", "session_initialized", "audio_ack",
                "diarization_result", "asr_result", "speaker_change",
                "audio_metrics", "status_report", "final_results", "error"
            ]
        },
        "requirements": {
            "sample_rate": 16000,
            "sample_width": 2,
            "channels": 1,
            "max_silence": 30.0
        },
        "performance_targets": {
            "latency": "<500ms",
            "throughput": "Real-time",
            "concurrent_sessions": "Configurable"
        }
    }


# Health check for WebSocket service
@websocket_router.get("/stream/health")
async def websocket_health_check():
    """Health check for WebSocket streaming service"""
    try:
        # Check if core components are available
        health_status = {
            "websocket_service": "healthy",
            "active_connections": len(connection_manager.active_connections),
            "buffer_managers": len(connection_manager.buffer_managers),
            "session_managers": len(connection_manager.session_managers),
            "stream_processors": len(connection_manager.stream_processors),
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
    
    except Exception as e:
        logger.error(f"WebSocket health check failed: {str(e)}")
        return {
            "websocket_service": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Utility function for testing WebSocket connections
@websocket_router.post("/stream/test")
async def test_websocket_configuration(config: Dict[str, Any]):
    """Test WebSocket configuration without establishing connection"""
    try:
        # Validate configuration
        required_params = ["sample_rate", "channels", "chunk_duration"]
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            return {
                "valid": False,
                "missing_parameters": missing_params,
                "message": "Required parameters missing"
            }
        
        # Check parameter ranges
        sample_rate = config["sample_rate"]
        if sample_rate not in [8000, 16000, 22050, 44100]:
            return {
                "valid": False,
                "message": "Sample rate must be 8000, 16000, 22050, or 44100"
            }
        
        channels = config["channels"]
        if channels not in [1, 2]:
            return {
                "valid": False,
                "message": "Channels must be 1 (mono) or 2 (stereo)"
            }
        
        chunk_duration = config["chunk_duration"]
        if not (0.5 <= chunk_duration <= 5.0):
            return {
                "valid": False,
                "message": "Chunk duration must be between 0.5 and 5.0 seconds"
            }
        
        return {
            "valid": True,
            "message": "Configuration is valid",
            "recommended_settings": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_duration": 2.0,
                "buffer_size": sample_rate * chunk_duration * channels * 2  # 16-bit
            }
        }
    
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "Configuration validation failed"
        }
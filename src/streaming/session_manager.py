"""
Real-time Session Manager for Awetales Diarization System

Manages audio session lifecycle, WebSocket integration, and state tracking
for real-time streaming diarization and ASR.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import aiohttp
from aiohttp import web
import redis.asyncio as redis
from datetime import datetime, timedelta
import weakref

# Import from buffer_manager instead of trying to use undefined function
from src.streaming.buffer_manager import BufferManager, AudioBufferManager, BufferState, create_buffer_manager

logger = logging.getLogger(__name__)

class SessionState(Enum):
    """Session state enumeration"""
    CREATED = "created"
    CONNECTING = "connecting"
    ACTIVE = "active"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"

class WebSocketState(Enum):
    """WebSocket connection state"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"

@dataclass
class SessionConfig:
    """Session configuration"""
    session_id: str
    user_id: Optional[str] = None
    audio_config: Dict[str, Any] = field(default_factory=lambda: {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_size": 2.0,
        "overlap_size": 0.5,
        "codec": "pcm"
    })
    processing_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_diarization": True,
        "enable_asr": True,
        "enable_speaker_recognition": True,
        "language": "en",
        "max_speakers": 4
    })
    web_socket_config: Dict[str, Any] = field(default_factory=lambda: {
        "ping_interval": 30,
        "ping_timeout": 10,
        "max_message_size": 1024 * 1024,  # 1MB
        "reconnect_attempts": 3
    })
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class WebSocketConnection:
    """WebSocket connection details"""
    session_id: str
    websocket: Optional[web.WebSocketResponse] = None
    state: WebSocketState = WebSocketState.DISCONNECTED
    connected_at: Optional[float] = None
    last_message_time: Optional[float] = None
    message_count: int = 0
    error_count: int = 0
    client_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionMetrics:
    """Session performance metrics"""
    audio_chunks_received: int = 0
    audio_chunks_processed: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    total_audio_duration: float = 0.0
    average_latency: float = 0.0
    max_latency: float = 0.0
    last_activity: float = field(default_factory=time.time)
    error_count: int = 0

class SessionManager:
    """
    Manages real-time audio sessions with WebSocket integration.
    
    Features:
    - Session lifecycle management
    - WebSocket connection handling
    - State tracking and persistence
    - Concurrent session management
    - Integration with buffer manager
    """
    
    def __init__(self, buffer_manager: AudioBufferManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SessionManager.
        
        Args:
            buffer_manager: AudioBufferManager instance
            config: Configuration dictionary
        """
        self.buffer_manager = buffer_manager
        self.config = self._load_config(config)
        
        # Session storage
        self.sessions: Dict[str, SessionConfig] = {}
        self.session_states: Dict[str, SessionState] = {}
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        
        # Callback handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.state_change_handlers: Dict[str, Callable] = {}
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Redis for persistence (optional)
        self.redis_client: Optional[redis.Redis] = None
        self._setup_redis()
        
        logger.info("SessionManager initialized")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            "max_sessions": 50,
            "session_timeout": 1800,  # 30 minutes
            "cleanup_interval": 60,   # 1 minute
            "health_check_interval": 30,
            "enable_redis": False,
            "redis_url": "redis://localhost:6379",
            "max_reconnect_attempts": 3,
            "enable_metrics": True,
            "log_level": "INFO"
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _setup_redis(self):
        """Setup Redis connection for session persistence"""
        if self.config["enable_redis"]:
            try:
                self.redis_client = redis.from_url(
                    self.config["redis_url"],
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Redis client initialized for session persistence")
            except Exception as e:
                logger.warning("Failed to initialize Redis: %s. Using in-memory storage only.", e)
                self.redis_client = None
    
    async def start(self):
        """Start session manager and background tasks"""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        self.health_check_task = asyncio.create_task(self._health_check_worker())
        logger.info("SessionManager started")
    
    async def stop(self):
        """Stop session manager and cleanup"""
        self.is_running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Close all WebSocket connections
        for session_id in list(self.websocket_connections.keys()):
            await self._close_websocket(session_id)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("SessionManager stopped")
    
    async def create_session(self, session_id: Optional[str] = None, 
                           user_id: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new audio session.
        
        Args:
            session_id: Optional custom session ID
            user_id: Optional user identifier
            config: Session configuration overrides
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            logger.warning("Session %s already exists", session_id)
            return session_id
        
        if len(self.sessions) >= self.config["max_sessions"]:
            raise RuntimeError(f"Maximum sessions ({self.config['max_sessions']}) reached")
        
        # Create session configuration
        session_config = SessionConfig(
            session_id=session_id,
            user_id=user_id
        )
        
        # Apply configuration overrides
        if config:
            if "audio_config" in config:
                session_config.audio_config.update(config["audio_config"])
            if "processing_config" in config:
                session_config.processing_config.update(config["processing_config"])
            if "web_socket_config" in config:
                session_config.web_socket_config.update(config["web_socket_config"])
        
        # Create buffer session
        buffer_config = {
            "chunk_size": session_config.audio_config["chunk_size"],
            "overlap_size": session_config.audio_config["overlap_size"],
            "sample_rate": session_config.audio_config["sample_rate"]
        }
        
        self.buffer_manager.buffer_manager.create_session(
            session_id, 
            buffer_config
        )
        
        # Store session data
        self.sessions[session_id] = session_config
        self.session_states[session_id] = SessionState.CREATED
        self.session_metrics[session_id] = SessionMetrics()
        self.websocket_connections[session_id] = WebSocketConnection(session_id=session_id)
        
        # Persist to Redis if enabled
        if self.redis_client:
            await self._persist_session(session_id)
        
        logger.info("Created session %s for user %s", session_id, user_id)
        await self._notify_state_change(session_id, SessionState.CREATED)
        
        return session_id
    
    async def register_websocket(self, session_id: str, websocket: web.WebSocketResponse,
                               client_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register WebSocket connection for a session.
        
        Args:
            session_id: Session identifier
            websocket: WebSocket response object
            client_info: Optional client information
            
        Returns:
            bool: True if registration successful
        """
        if session_id not in self.sessions:
            logger.error("Session %s not found for WebSocket registration", session_id)
            return False
        
        ws_conn = self.websocket_connections[session_id]
        ws_conn.websocket = websocket
        ws_conn.state = WebSocketState.CONNECTED
        ws_conn.connected_at = time.time()
        ws_conn.last_message_time = time.time()
        ws_conn.client_info = client_info or {}
        
        # Update session state
        await self._update_session_state(session_id, SessionState.ACTIVE)
        
        logger.info("WebSocket registered for session %s", session_id)
        return True
    
    async def handle_audio_data(self, session_id: str, audio_data: bytes, 
                              timestamp: Optional[float] = None) -> bool:
        """
        Handle incoming audio data for a session.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio data
            timestamp: Optional audio timestamp
            
        Returns:
            bool: True if processing successful
        """
        if session_id not in self.sessions:
            logger.error("Session %s not found", session_id)
            return False
        
        try:
            # Convert audio data to numpy array
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Update metrics
            metrics = self.session_metrics[session_id]
            metrics.audio_chunks_received += 1
            metrics.last_activity = time.time()
            metrics.total_audio_duration += len(audio_array) / self.sessions[session_id].audio_config["sample_rate"]
            
            # Send to buffer manager
            success = await self.buffer_manager.buffer_manager.add_audio_chunk(
                session_id, audio_array, timestamp
            )
            
            if success:
                metrics.audio_chunks_processed += 1
                await self._update_session_state(session_id, SessionState.PROCESSING)
            
            return success
            
        except Exception as e:
            logger.error("Error handling audio data for session %s: %s", session_id, str(e))
            await self._handle_session_error(session_id, str(e))
            return False
    
    async def send_message(self, session_id: str, message: Dict[str, Any], 
                         message_type: str = "processing_update") -> bool:
        """
        Send message to WebSocket client.
        
        Args:
            session_id: Session identifier
            message: Message data
            message_type: Message type identifier
            
        Returns:
            bool: True if message sent successfully
        """
        if session_id not in self.websocket_connections:
            logger.error("No WebSocket connection for session %s", session_id)
            return False
        
        ws_conn = self.websocket_connections[session_id]
        
        if ws_conn.state != WebSocketState.CONNECTED or not ws_conn.websocket:
            logger.warning("WebSocket not connected for session %s", session_id)
            return False
        
        try:
            # Prepare message
            full_message = {
                "type": message_type,
                "session_id": session_id,
                "timestamp": time.time(),
                "data": message
            }
            
            message_json = json.dumps(full_message)
            
            # Check message size
            if len(message_json) > self.sessions[session_id].web_socket_config["max_message_size"]:
                logger.warning("Message too large for session %s: %d bytes", 
                             session_id, len(message_json))
                return False
            
            # Send message
            await ws_conn.websocket.send_str(message_json)
            
            # Update metrics
            ws_conn.message_count += 1
            ws_conn.last_message_time = time.time()
            self.session_metrics[session_id].messages_sent += 1
            
            logger.debug("Sent %s message to session %s", message_type, session_id)
            return True
            
        except Exception as e:
            logger.error("Error sending message to session %s: %s", session_id, str(e))
            ws_conn.error_count += 1
            await self._handle_websocket_error(session_id)
            return False
    
    async def handle_client_message(self, session_id: str, message: str) -> bool:
        """
        Handle incoming message from WebSocket client.
        
        Args:
            session_id: Session identifier
            message: Raw message string
            
        Returns:
            bool: True if message handled successfully
        """
        if session_id not in self.websocket_connections:
            return False
        
        try:
            message_data = json.loads(message)
            message_type = message_data.get("type", "unknown")
            
            # Update metrics
            ws_conn = self.websocket_connections[session_id]
            ws_conn.last_message_time = time.time()
            ws_conn.message_count += 1
            self.session_metrics[session_id].messages_received += 1
            
            # Call message handler if registered
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](session_id, message_data)
            else:
                await self._handle_default_message(session_id, message_data)
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON message from session %s: %s", session_id, str(e))
            return False
        except Exception as e:
            logger.error("Error handling client message for session %s: %s", session_id, str(e))
            return False
    
    async def _handle_default_message(self, session_id: str, message_data: Dict[str, Any]):
        """Handle default message types"""
        message_type = message_data.get("type")
        
        if message_type == "ping":
            await self.send_message(session_id, {"status": "pong"}, "pong")
        
        elif message_type == "pause":
            await self._update_session_state(session_id, SessionState.PAUSED)
            await self.send_message(session_id, {"status": "paused"}, "session_update")
        
        elif message_type == "resume":
            await self._update_session_state(session_id, SessionState.ACTIVE)
            await self.send_message(session_id, {"status": "resumed"}, "session_update")
        
        elif message_type == "terminate":
            await self.terminate_session(session_id)
    
    async def terminate_session(self, session_id: str, reason: str = "client_request") -> bool:
        """
        Terminate a session.
        
        Args:
            session_id: Session identifier
            reason: Termination reason
            
        Returns:
            bool: True if termination successful
        """
        if session_id not in self.sessions:
            return False
        
        try:
            # Close WebSocket connection
            await self._close_websocket(session_id)
            
            # Cleanup buffer session
            self.buffer_manager.buffer_manager.delete_session(session_id)
            
            # Send final message if possible
            await self.send_message(session_id, {
                "status": "terminated",
                "reason": reason,
                "metrics": self._get_session_metrics_summary(session_id)
            }, "session_terminated")
            
            # Remove session data
            del self.sessions[session_id]
            del self.session_states[session_id]
            del self.session_metrics[session_id]
            del self.websocket_connections[session_id]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
            
            logger.info("Terminated session %s: %s", session_id, reason)
            return True
            
        except Exception as e:
            logger.error("Error terminating session %s: %s", session_id, str(e))
            return False
    
    async def _update_session_state(self, session_id: str, new_state: SessionState):
        """Update session state and notify handlers"""
        old_state = self.session_states.get(session_id)
        
        if old_state != new_state:
            self.session_states[session_id] = new_state
            self.sessions[session_id].updated_at = time.time()
            
            # Persist state change
            if self.redis_client:
                await self._persist_session(session_id)
            
            # Notify state change handlers
            await self._notify_state_change(session_id, new_state, old_state)
            
            logger.debug("Session %s state changed: %s -> %s", 
                        session_id, old_state, new_state)
    
    async def _notify_state_change(self, session_id: str, new_state: SessionState, 
                                 old_state: Optional[SessionState] = None):
        """Notify state change handlers"""
        for handler_name, handler in self.state_change_handlers.items():
            try:
                await handler(session_id, new_state, old_state)
            except Exception as e:
                logger.error("Error in state change handler %s: %s", handler_name, str(e))
    
    async def _handle_session_error(self, session_id: str, error_message: str):
        """Handle session error"""
        logger.error("Session %s error: %s", session_id, error_message)
        
        self.session_metrics[session_id].error_count += 1
        await self._update_session_state(session_id, SessionState.ERROR)
        
        # Send error message to client
        await self.send_message(session_id, {
            "error": error_message,
            "session_state": "error"
        }, "error")
    
    async def _handle_websocket_error(self, session_id: str):
        """Handle WebSocket error"""
        ws_conn = self.websocket_connections[session_id]
        ws_conn.error_count += 1
        
        if ws_conn.error_count >= self.config["max_reconnect_attempts"]:
            logger.warning("Max WebSocket errors reached for session %s", session_id)
            await self.terminate_session(session_id, "websocket_error")
        else:
            ws_conn.state = WebSocketState.RECONNECTING
            logger.warning("WebSocket error for session %s, attempt %d", 
                         session_id, ws_conn.error_count)
    
    async def _close_websocket(self, session_id: str):
        """Close WebSocket connection"""
        if session_id in self.websocket_connections:
            ws_conn = self.websocket_connections[session_id]
            if ws_conn.websocket and not ws_conn.websocket.closed:
                await ws_conn.websocket.close()
            ws_conn.state = WebSocketState.CLOSED
            ws_conn.websocket = None
    
    async def _persist_session(self, session_id: str):
        """Persist session data to Redis"""
        if not self.redis_client:
            return
        
        try:
            session_data = {
                "config": self.sessions[session_id].__dict__,
                "state": self.session_states[session_id].value,
                "metrics": self.session_metrics[session_id].__dict__,
                "websocket": {
                    "state": self.websocket_connections[session_id].state.value,
                    "connected_at": self.websocket_connections[session_id].connected_at,
                    "message_count": self.websocket_connections[session_id].message_count
                }
            }
            
            await self.redis_client.setex(
                f"session:{session_id}",
                timedelta(seconds=self.config["session_timeout"] * 2),
                json.dumps(session_data)
            )
            
        except Exception as e:
            logger.error("Error persisting session %s: %s", session_id, str(e))
    
    async def _cleanup_worker(self):
        """Background worker for session cleanup"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup worker: %s", str(e))
    
    async def _health_check_worker(self):
        """Background worker for health checks"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["health_check_interval"])
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check worker: %s", str(e))
    
    async def _cleanup_inactive_sessions(self):
        """Cleanup inactive sessions"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, metrics in self.session_metrics.items():
            inactivity_time = current_time - metrics.last_activity
            
            if (inactivity_time > self.config["session_timeout"] and 
                self.session_states[session_id] not in [SessionState.ACTIVE, SessionState.PROCESSING]):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self.terminate_session(session_id, "inactivity_timeout")
        
        if sessions_to_remove:
            logger.info("Cleaned up %d inactive sessions", len(sessions_to_remove))
    
    async def _perform_health_checks(self):
        """Perform health checks on active sessions"""
        for session_id, ws_conn in self.websocket_connections.items():
            if (ws_conn.state == WebSocketState.CONNECTED and 
                ws_conn.websocket and not ws_conn.websocket.closed):
                
                # Send ping to check connection
                try:
                    await self.send_message(session_id, {"ping": True}, "ping")
                except Exception as e:
                    logger.warning("Health check failed for session %s: %s", session_id, str(e))
                    await self._handle_websocket_error(session_id)
    
    def _get_session_metrics_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session metrics summary"""
        if session_id not in self.session_metrics:
            return {}
        
        metrics = self.session_metrics[session_id]
        return {
            "audio_chunks_received": metrics.audio_chunks_received,
            "audio_chunks_processed": metrics.audio_chunks_processed,
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "total_audio_duration": metrics.total_audio_duration,
            "average_latency": metrics.average_latency,
            "error_count": metrics.error_count
        }
    
    # Public API methods
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details"""
        if session_id not in self.sessions:
            return None
        
        session_config = self.sessions[session_id]
        session_state = self.session_states[session_id]
        session_metrics = self.session_metrics[session_id]
        ws_conn = self.websocket_connections[session_id]
        
        return {
            "session_id": session_id,
            "user_id": session_config.user_id,
            "state": session_state.value,
            "created_at": session_config.created_at,
            "updated_at": session_config.updated_at,
            "websocket_state": ws_conn.state.value,
            "metrics": self._get_session_metrics_summary(session_id),
            "config": {
                "audio": session_config.audio_config,
                "processing": session_config.processing_config
            }
        }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        return [self.get_session(session_id) for session_id in self.sessions.keys()]
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register custom message handler"""
        self.message_handlers[message_type] = handler
    
    def register_state_change_handler(self, handler_name: str, handler: Callable):
        """Register state change handler"""
        self.state_change_handlers[handler_name] = handler
    
    def get_session_count(self) -> int:
        """Get total session count"""
        return len(self.sessions)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        return len([s for s in self.session_states.values() 
                   if s in [SessionState.ACTIVE, SessionState.PROCESSING]])

# Factory function for easy creation
async def create_session_manager(buffer_manager: AudioBufferManager, 
                              config: Optional[Dict[str, Any]] = None) -> SessionManager:
    """
    Create and start a SessionManager instance.
    
    Args:
        buffer_manager: AudioBufferManager instance
        config: Configuration dictionary
        
    Returns:
        SessionManager: Started session manager instance
    """
    manager = SessionManager(buffer_manager, config)
    await manager.start()
    return manager

# Example usage and testing
async def example_usage():
    """Example demonstrating session manager usage"""
    
    # Create buffer manager first using the imported function
    buffer_manager = await create_buffer_manager()
    
    # Create session manager
    session_manager = await create_session_manager(buffer_manager)
    
    try:
        # Create a session
        session_id = await session_manager.create_session(
            user_id="test_user_123",
            config={
                "audio_config": {
                    "chunk_size": 2.0,
                    "overlap_size": 0.5
                },
                "processing_config": {
                    "enable_diarization": True,
                    "language": "en"
                }
            }
        )
        
        print(f"Created session: {session_id}")
        
        # Simulate some operations
        import numpy as np
        
        # Simulate audio data
        sample_rate = 16000
        audio_data = np.random.randn(sample_rate).astype(np.float32)
        raw_audio = (audio_data * 32768).astype(np.int16).tobytes()
        
        # Handle audio data
        await session_manager.handle_audio_data(session_id, raw_audio)
        
        # Get session info
        session_info = session_manager.get_session(session_id)
        print(f"Session info: {json.dumps(session_info, indent=2, default=str)}")
        
        # Get all sessions
        all_sessions = session_manager.get_all_sessions()
        print(f"Total sessions: {len(all_sessions)}")
        
    finally:
        await session_manager.stop()
        await buffer_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
"""
Real-time Audio Buffer Manager for Awetales Diarization System

Handles audio chunk buffering, overlap-add processing, and session management
for real-time streaming diarization and ASR.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque, defaultdict
import threading
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class BufferState(Enum):
    """Buffer state enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    PROCESSING = "processing"
    ERROR = "error"

@dataclass
class AudioChunk:
    """Audio chunk data structure"""
    session_id: str
    chunk_id: int
    audio_data: np.ndarray
    timestamp: float
    sample_rate: int = 16000
    duration: float = 0.0
    
    def __post_init__(self):
        if self.duration == 0.0 and len(self.audio_data) > 0:
            self.duration = len(self.audio_data) / self.sample_rate

@dataclass
class SessionConfig:
    """Session configuration"""
    chunk_size: float = 2.0  # seconds
    overlap_size: float = 0.5  # seconds
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    max_buffer_size: int = 10  # maximum chunks in buffer
    preprocess_audio: bool = True

@dataclass
class SessionMetrics:
    """Session performance metrics"""
    total_chunks_processed: int = 0
    average_latency: float = 0.0
    max_latency: float = 0.0
    buffer_utilization: float = 0.0
    last_processed_time: float = 0.0

class BufferManager:
    """
    Manages real-time audio buffers with overlap-add processing for multiple sessions.
    
    Features:
    - Session-based audio buffering
    - Overlap-add for seamless processing
    - Chunk queuing and prioritization
    - Memory management and garbage collection
    - Latency optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BufferManager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._load_config(config)
        
        # Session management
        self.sessions: Dict[str, SessionConfig] = {}
        self.session_buffers: Dict[str, Deque[AudioChunk]] = {}
        self.session_metrics: Dict[str, SessionMetrics] = {}
        self.session_states: Dict[str, BufferState] = {}
        
        # Overlap buffers for each session
        self.overlap_buffers: Dict[str, np.ndarray] = {}
        
        # Processing queues
        self.ready_queues: Dict[str, Deque[AudioChunk]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._processing_lock = threading.RLock()
        
        # Performance monitoring
        self.start_time = time.time()
        self.total_chunks_processed = 0
        
        logger.info("BufferManager initialized with config: %s", self.config)
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            "default_chunk_size": 2.0,
            "default_overlap": 0.5,
            "sample_rate": 16000,
            "max_sessions": 50,
            "max_chunk_queue": 100,
            "cleanup_interval": 300,  # 5 minutes
            "max_session_age": 3600,  # 1 hour
            "enable_metrics": True,
            "log_level": "INFO"
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def create_session(self, session_id: str, config: Optional[SessionConfig] = None) -> bool:
        """
        Create a new audio session.
        
        Args:
            session_id: Unique session identifier
            config: Session-specific configuration
            
        Returns:
            bool: True if session created successfully
        """
        with self._lock:
            if session_id in self.sessions:
                logger.warning("Session %s already exists", session_id)
                return False
            
            if len(self.sessions) >= self.config["max_sessions"]:
                logger.error("Maximum sessions reached. Cannot create session %s", session_id)
                return False
            
            # Use provided config or create default
            session_config = config or SessionConfig(
                chunk_size=self.config["default_chunk_size"],
                overlap_size=self.config["default_overlap"],
                sample_rate=self.config["sample_rate"]
            )
            
            self.sessions[session_id] = session_config
            self.session_buffers[session_id] = deque(maxlen=session_config.max_buffer_size)
            self.ready_queues[session_id] = deque(maxlen=self.config["max_chunk_queue"])
            self.overlap_buffers[session_id] = np.array([], dtype=np.float32)
            self.session_metrics[session_id] = SessionMetrics()
            self.session_states[session_id] = BufferState.IDLE
            
            logger.info("Created session %s with config: %s", session_id, session_config)
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and cleanup resources.
        
        Args:
            session_id: Session identifier to delete
            
        Returns:
            bool: True if session deleted successfully
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.warning("Session %s not found for deletion", session_id)
                return False
            
            # Cleanup resources
            del self.sessions[session_id]
            del self.session_buffers[session_id]
            del self.ready_queues[session_id]
            del self.overlap_buffers[session_id]
            del self.session_metrics[session_id]
            del self.session_states[session_id]
            
            logger.info("Deleted session %s and cleaned up resources", session_id)
            return True
    
    async def add_audio_chunk(self, session_id: str, audio_data: np.ndarray, 
                            timestamp: Optional[float] = None) -> bool:
        """
        Add audio chunk to session buffer with overlap-add processing.
        
        Args:
            session_id: Session identifier
            audio_data: Audio data as numpy array
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            bool: True if chunk processed successfully
        """
        if session_id not in self.sessions:
            logger.error("Session %s not found", session_id)
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            with self._processing_lock:
                session_config = self.sessions[session_id]
                buffer = self.session_buffers[session_id]
                overlap_buffer = self.overlap_buffers[session_id]
                ready_queue = self.ready_queues[session_id]
                metrics = self.session_metrics[session_id]
                
                # Convert audio data to float32 for processing
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                
                # Add to overlap buffer
                if len(overlap_buffer) > 0:
                    audio_data = np.concatenate([overlap_buffer, audio_data])
                
                # Calculate chunk size in samples
                chunk_samples = int(session_config.chunk_size * session_config.sample_rate)
                overlap_samples = int(session_config.overlap_size * session_config.sample_rate)
                
                # Process complete chunks
                chunk_start = 0
                chunk_id = metrics.total_chunks_processed
                
                while chunk_start + chunk_samples <= len(audio_data):
                    chunk_end = chunk_start + chunk_samples
                    chunk_data = audio_data[chunk_start:chunk_end]
                    
                    # Create audio chunk
                    audio_chunk = AudioChunk(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        audio_data=chunk_data.copy(),
                        timestamp=timestamp + (chunk_start / session_config.sample_rate),
                        sample_rate=session_config.sample_rate
                    )
                    
                    # Add to buffer and ready queue
                    buffer.append(audio_chunk)
                    ready_queue.append(audio_chunk)
                    
                    chunk_id += 1
                    chunk_start = chunk_end - overlap_samples
                
                # Update overlap buffer with remaining audio
                if chunk_start < len(audio_data):
                    self.overlap_buffers[session_id] = audio_data[chunk_start:].copy()
                else:
                    self.overlap_buffers[session_id] = np.array([], dtype=np.float32)
                
                # Update metrics
                metrics.total_chunks_processed = chunk_id
                metrics.last_processed_time = time.time()
                
                current_latency = time.time() - timestamp
                metrics.average_latency = (
                    (metrics.average_latency * (chunk_id - 1) + current_latency) / chunk_id
                    if chunk_id > 0 else current_latency
                )
                metrics.max_latency = max(metrics.max_latency, current_latency)
                metrics.buffer_utilization = len(buffer) / session_config.max_buffer_size
                
                self.total_chunks_processed += (chunk_id - metrics.total_chunks_processed)
                
                logger.debug("Processed %d chunks for session %s, latency: %.3fs", 
                           chunk_id - metrics.total_chunks_processed, session_id, current_latency)
                
                return True
                
        except Exception as e:
            logger.error("Error processing audio chunk for session %s: %s", session_id, str(e))
            self.session_states[session_id] = BufferState.ERROR
            return False
    
    async def get_ready_chunk(self, session_id: str, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Get next ready chunk for processing.
        
        Args:
            session_id: Session identifier
            timeout: Optional timeout in seconds
            
        Returns:
            Optional[AudioChunk]: Next audio chunk or None if timeout/no data
        """
        if session_id not in self.ready_queues:
            logger.error("Session %s not found", session_id)
            return None
        
        start_time = time.time()
        ready_queue = self.ready_queues[session_id]
        
        while True:
            with self._lock:
                if ready_queue:
                    chunk = ready_queue.popleft()
                    logger.debug("Retrieved chunk %d for session %s", chunk.chunk_id, session_id)
                    return chunk
            
            if timeout and (time.time() - start_time) > timeout:
                logger.debug("Timeout waiting for chunk in session %s", session_id)
                return None
            
            # Wait before checking again
            await asyncio.sleep(0.01)
    
    def get_session_buffer(self, session_id: str) -> List[AudioChunk]:
        """
        Get current buffer state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List[AudioChunk]: List of audio chunks in buffer
        """
        with self._lock:
            if session_id not in self.session_buffers:
                logger.error("Session %s not found", session_id)
                return []
            
            return list(self.session_buffers[session_id])
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """
        Get performance metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[SessionMetrics]: Session metrics or None if not found
        """
        with self._lock:
            return self.session_metrics.get(session_id)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get overall system metrics.
        
        Returns:
            Dict[str, Any]: System-wide metrics
        """
        with self._lock:
            uptime = time.time() - self.start_time
            active_sessions = len([s for s in self.session_states.values() 
                                 if s == BufferState.ACTIVE])
            
            return {
                "uptime": uptime,
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "total_chunks_processed": self.total_chunks_processed,
                "chunks_per_second": self.total_chunks_processed / uptime if uptime > 0 else 0,
                "session_metrics": {
                    sid: {
                        "total_chunks": metrics.total_chunks_processed,
                        "avg_latency": metrics.average_latency,
                        "max_latency": metrics.max_latency,
                        "buffer_utilization": metrics.buffer_utilization
                    }
                    for sid, metrics in self.session_metrics.items()
                }
            }
    
    def flush_session(self, session_id: str) -> bool:
        """
        Flush session buffers and reset state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if flush successful
        """
        with self._lock:
            if session_id not in self.sessions:
                logger.error("Session %s not found", session_id)
                return False
            
            self.session_buffers[session_id].clear()
            self.ready_queues[session_id].clear()
            self.overlap_buffers[session_id] = np.array([], dtype=np.float32)
            self.session_states[session_id] = BufferState.IDLE
            
            logger.info("Flushed buffers for session %s", session_id)
            return True
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        with self._lock:
            return session_id in self.sessions
    
    def get_session_state(self, session_id: str) -> Optional[BufferState]:
        """Get session buffer state"""
        with self._lock:
            return self.session_states.get(session_id)
    
    def set_session_state(self, session_id: str, state: BufferState) -> bool:
        """Set session buffer state"""
        with self._lock:
            if session_id not in self.sessions:
                return False
            self.session_states[session_id] = state
            return True
    
    async def cleanup_inactive_sessions(self, max_age: Optional[float] = None) -> int:
        """
        Cleanup inactive sessions older than max_age.
        
        Args:
            max_age: Maximum session age in seconds
            
        Returns:
            int: Number of sessions cleaned up
        """
        if max_age is None:
            max_age = self.config["max_session_age"]
        
        current_time = time.time()
        sessions_to_remove = []
        
        with self._lock:
            for session_id, metrics in self.session_metrics.items():
                session_age = current_time - metrics.last_processed_time
                if session_age > max_age:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self.delete_session(session_id)
        
        logger.info("Cleaned up %d inactive sessions", len(sessions_to_remove))
        return len(sessions_to_remove)

class AudioBufferManager:
    """
    High-level audio buffer manager with async support and advanced features.
    
    This class provides a more convenient interface for the BufferManager
    with additional features like automatic session management and
    integration with other system components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.buffer_manager = BufferManager(config)
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self):
        """Start the buffer manager and background tasks"""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        logger.info("AudioBufferManager started")
    
    async def stop(self):
        """Stop the buffer manager and cleanup"""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("AudioBufferManager stopped")
    
    async def _cleanup_worker(self):
        """Background worker for session cleanup"""
        while self.is_running:
            try:
                await asyncio.sleep(self.buffer_manager.config["cleanup_interval"])
                await self.buffer_manager.cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup worker: %s", str(e))
    
    async def process_audio_stream(self, session_id: str, audio_stream: Any, 
                                 chunk_callback: Optional[callable] = None) -> bool:
        """
        Process audio stream and manage buffering.
        
        Args:
            session_id: Session identifier
            audio_stream: Async audio stream
            chunk_callback: Optional callback for processed chunks
            
        Returns:
            bool: True if processing successful
        """
        if not self.buffer_manager.session_exists(session_id):
            self.buffer_manager.create_session(session_id)
        
        self.buffer_manager.set_session_state(session_id, BufferState.ACTIVE)
        
        try:
            async for audio_chunk in audio_stream:
                success = await self.buffer_manager.add_audio_chunk(
                    session_id, audio_chunk
                )
                
                if success and chunk_callback:
                    ready_chunk = await self.buffer_manager.get_ready_chunk(session_id)
                    if ready_chunk:
                        await chunk_callback(ready_chunk)
            
            return True
            
        except Exception as e:
            logger.error("Error processing audio stream for session %s: %s", session_id, str(e))
            self.buffer_manager.set_session_state(session_id, BufferState.ERROR)
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return self.buffer_manager.get_all_metrics()

# Factory function for easy creation
async def create_buffer_manager(config: Optional[Dict[str, Any]] = None) -> AudioBufferManager:
    """
    Create and start an AudioBufferManager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AudioBufferManager: Started buffer manager instance
    """
    manager = AudioBufferManager(config)
    await manager.start()
    return manager

# Example usage and testing
async def example_usage():
    """Example demonstrating buffer manager usage"""
    
    # Create buffer manager
    config = {
        "default_chunk_size": 2.0,
        "default_overlap": 0.5,
        "max_sessions": 10
    }
    
    buffer_manager = await create_buffer_manager(config)
    
    try:
        # Create a session
        session_id = "test_session_123"
        buffer_manager.buffer_manager.create_session(session_id)
        
        # Simulate audio data (3 seconds of audio)
        sample_rate = 16000
        audio_data = np.random.randn(3 * sample_rate).astype(np.float32)
        
        # Process audio in chunks
        chunk_size = int(0.5 * sample_rate)  # 500ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await buffer_manager.buffer_manager.add_audio_chunk(session_id, chunk)
            
            # Get ready chunks for processing
            ready_chunk = await buffer_manager.buffer_manager.get_ready_chunk(session_id, timeout=0.1)
            if ready_chunk:
                print(f"Processing chunk {ready_chunk.chunk_id}, duration: {ready_chunk.duration:.2f}s")
        
        # Get metrics
        metrics = buffer_manager.get_metrics()
        print(f"System metrics: {json.dumps(metrics, indent=2, default=str)}")
        
    finally:
        await buffer_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
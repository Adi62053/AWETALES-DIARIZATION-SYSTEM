"""
Pipeline Orchestrator for Awetales Diarization System

Coordinates the complete audio processing workflow from input to output,
integrating all core, streaming, and orchestration modules.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, PriorityQueue, Empty  # Fixed: Added Empty import

# Import all core processing modules
from src.core.audio_preprocess import AudioPreprocessor
from src.core.separation import VoiceSeparator
from src.core.restoration import AudioRestorer
from src.core.diarization import DiarizationEngine
from src.core.speaker_recog import SpeakerRecognizer
from src.core.asr_paraformer import StreamingASR
from src.core.asr_whisper import OfflineASR
from src.core.punct_restore import PunctuationRestorer
from src.core.output_manager import OutputManager

# Import streaming modules
from src.streaming.buffer_manager import AudioBufferManager, create_buffer_manager
from src.streaming.session_manager import SessionManager, create_session_manager
from src.streaming.stream_processor import StreamProcessor, create_stream_processor

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing mode enumeration"""
    REALTIME_STREAMING = "realtime_streaming"
    BATCH_PROCESSING = "batch_processing"
    HYBRID = "hybrid"

class PipelineState(Enum):
    """Pipeline state enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    pipeline_id: str
    mode: ProcessingMode = ProcessingMode.REALTIME_STREAMING
    max_concurrent_sessions: int = 20
    enable_gpu: bool = True
    gpu_memory_limit: Optional[int] = None
    max_workers: int = 8
    processing_timeout: float = 30.0
    enable_fallback: bool = True
    quality_profile: str = "balanced"  # balanced, high_quality, fast

@dataclass
class ProcessingRequest:
    """Processing request data"""
    request_id: str
    session_id: str
    audio_data: Optional[np.ndarray] = None
    audio_file_path: Optional[str] = None
    sample_rate: int = 16000
    config: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class ProcessingResult:
    """Processing result"""
    request_id: str
    session_id: str
    success: bool
    results: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None
    completed_at: float = field(default_factory=time.time)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_requests_processed: int = 0
    active_sessions: int = 0
    average_processing_time: float = 0.0
    error_count: int = 0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0

class PipelineOrchestrator:
    """
    Main orchestrator for the complete audio processing workflow.
    
    Features:
    - Unified interface for real-time and batch processing
    - Session lifecycle management
    - Resource allocation and load balancing
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PipelineOrchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = self._load_config(config)
        
        # Pipeline state
        self.pipeline_state: PipelineState = PipelineState.INITIALIZING
        self.pipeline_metrics: PipelineMetrics = PipelineMetrics()
        
        # Core processing modules
        self.audio_preprocessor: Optional[AudioPreprocessor] = None
        self.voice_separator: Optional[VoiceSeparator] = None
        self.audio_restorer: Optional[AudioRestorer] = None
        self.diarization_engine: Optional[DiarizationEngine] = None
        self.speaker_recognizer: Optional[SpeakerRecognizer] = None
        self.streaming_asr: Optional[StreamingASR] = None
        self.offline_asr: Optional[OfflineASR] = None
        self.punctuation_restorer: Optional[PunctuationRestorer] = None
        self.output_manager: Optional[OutputManager] = None
        
        # Streaming modules
        self.buffer_manager: Optional[AudioBufferManager] = None
        self.session_manager: Optional[SessionManager] = None
        self.stream_processor: Optional[StreamProcessor] = None
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_configs: Dict[str, Dict[str, Any]] = {}
        
        # Processing queues
        self.request_queue: PriorityQueue = PriorityQueue()
        self.result_queues: Dict[str, Queue] = {}
        
        # Threading and parallelism
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.processing_lock = threading.RLock()
        
        # Background tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.queue_processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Callback handlers
        self.result_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        logger.info("PipelineOrchestrator initialized with config: %s", self.config)
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> PipelineConfig:
        """Load and validate configuration"""
        default_config = {
            "pipeline_id": f"pipeline_{uuid.uuid4().hex[:8]}",
            "mode": "realtime_streaming",
            "max_concurrent_sessions": 20,
            "enable_gpu": True,
            "gpu_memory_limit": None,
            "max_workers": 8,
            "processing_timeout": 30.0,
            "enable_fallback": True,
            "quality_profile": "balanced"
        }
        
        if config:
            default_config.update(config)
        
        return PipelineConfig(**default_config)
    
    async def initialize(self):
        """Initialize the complete pipeline"""
        try:
            self.pipeline_state = PipelineState.INITIALIZING
            logger.info("Initializing pipeline orchestrator...")
            
            # Initialize core processing modules
            await self._initialize_core_modules()
            
            # Initialize streaming modules
            await self._initialize_streaming_modules()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.pipeline_state = PipelineState.READY
            logger.info("Pipeline orchestrator initialized successfully")
            
        except Exception as e:
            self.pipeline_state = PipelineState.ERROR
            logger.error("Failed to initialize pipeline orchestrator: %s", str(e))
            raise
    
    async def _initialize_core_modules(self):
        """Initialize all core processing modules"""
        gpu_config = {
            "use_gpu": self.config.enable_gpu,
            "gpu_memory_limit": self.config.gpu_memory_limit
        }
        
        try:
            self.audio_preprocessor = AudioPreprocessor(config=gpu_config)
            self.voice_separator = VoiceSeparator(config=gpu_config)
            self.audio_restorer = AudioRestorer(config=gpu_config)
            self.diarization_engine = DiarizationEngine(config=gpu_config)
            self.speaker_recognizer = SpeakerRecognizer(config=gpu_config)
            self.streaming_asr = StreamingASR(config=gpu_config)
            self.offline_asr = OfflineASR(config=gpu_config)
            self.punctuation_restorer = PunctuationRestorer(config=gpu_config)
            self.output_manager = OutputManager()
            
            logger.info("All core modules initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize core modules: %s", str(e))
            raise
    
    async def _initialize_streaming_modules(self):
        """Initialize streaming modules"""
        try:
            # Initialize buffer manager
            buffer_config = {
                "max_sessions": self.config.max_concurrent_sessions,
                "default_chunk_size": 2.0,
                "default_overlap": 0.5
            }
            self.buffer_manager = await create_buffer_manager(buffer_config)
            
            # Initialize session manager
            session_config = {
                "max_sessions": self.config.max_concurrent_sessions,
                "session_timeout": 1800
            }
            self.session_manager = await create_session_manager(
                self.buffer_manager, session_config
            )
            
            # Initialize stream processor
            processor_config = {
                "max_concurrent_pipelines": self.config.max_concurrent_sessions,
                "max_workers": self.config.max_workers
            }
            self.stream_processor = await create_stream_processor(
                self.buffer_manager, self.session_manager, processor_config
            )
            
            logger.info("All streaming modules initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize streaming modules: %s", str(e))
            raise
    
    async def _start_background_tasks(self):
        """Start background monitoring and processing tasks"""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_worker())
        self.queue_processor_task = asyncio.create_task(self._queue_processor_worker())
        
        logger.info("Background tasks started")
    
    async def shutdown(self):
        """Shutdown the pipeline orchestrator"""
        self.pipeline_state = PipelineState.SHUTTING_DOWN
        self.is_running = False
        
        logger.info("Shutting down pipeline orchestrator...")
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
        
        # Cancel processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Shutdown streaming modules
        if self.stream_processor:
            await self.stream_processor.stop()
        if self.session_manager:
            await self.session_manager.stop()
        if self.buffer_manager:
            await self.buffer_manager.stop()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Pipeline orchestrator shutdown completed")
    
    async def create_session(self, 
                           session_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new processing session.
        
        Args:
            session_config: Session configuration
            
        Returns:
            str: Session ID
        """
        if len(self.active_sessions) >= self.config.max_concurrent_sessions:
            raise RuntimeError("Maximum concurrent sessions reached")
        
        session_id = str(uuid.uuid4())
        
        try:
            # Create session in session manager
            await self.session_manager.create_session(
                session_id, 
                config=session_config
            )
            
            # Create processing pipeline
            await self.stream_processor.create_processing_pipeline(
                session_id,
                config=session_config
            )
            
            # Store session data
            self.active_sessions[session_id] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "request_count": 0,
                "state": "active"
            }
            
            self.session_configs[session_id] = session_config or {}
            
            # Update metrics
            self.pipeline_metrics.active_sessions = len(self.active_sessions)
            
            logger.info("Created session %s", session_id)
            return session_id
            
        except Exception as e:
            logger.error("Failed to create session %s: %s", session_id, str(e))
            raise
    
    async def process_realtime_audio(self,
                                   session_id: str,
                                   audio_data: np.ndarray,
                                   sample_rate: int = 16000,
                                   timestamp: Optional[float] = None) -> bool:
        """
        Process real-time audio data for a session.
        
        Args:
            session_id: Session identifier
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            timestamp: Optional audio timestamp
            
        Returns:
            bool: True if processing started successfully
        """
        if session_id not in self.active_sessions:
            logger.error("Session %s not found", session_id)
            return False
        
        try:
            # Convert to bytes for transmission
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Send to session manager for processing
            success = await self.session_manager.handle_audio_data(
                session_id, audio_bytes, timestamp
            )
            
            if success:
                # Update session activity
                self.active_sessions[session_id]["last_activity"] = time.time()
                self.active_sessions[session_id]["request_count"] += 1
                
                logger.debug("Real-time audio processing started for session %s", session_id)
            
            return success
            
        except Exception as e:
            logger.error("Error processing real-time audio for session %s: %s", session_id, str(e))
            return False
    
    async def process_batch_audio(self,
                                audio_data: Optional[np.ndarray] = None,
                                audio_file_path: Optional[str] = None,
                                sample_rate: int = 16000,
                                config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process batch audio data.
        
        Args:
            audio_data: Audio data as numpy array
            audio_file_path: Path to audio file
            sample_rate: Sample rate of audio data
            config: Processing configuration
            
        Returns:
            ProcessingResult: Processing result
        """
        request_id = str(uuid.uuid4())
        session_id = f"batch_{request_id[:8]}"
        
        try:
            # Create temporary session for batch processing
            await self.create_session(config)
            
            # Create processing request
            request = ProcessingRequest(
                request_id=request_id,
                session_id=session_id,
                audio_data=audio_data,
                audio_file_path=audio_file_path,
                sample_rate=sample_rate,
                config=config or {},
                priority=TaskPriority.NORMAL
            )
            
            # Process the request
            result = await self._process_batch_request(request)
            
            # Cleanup temporary session
            await self._cleanup_session(session_id)
            
            return result
            
        except Exception as e:
            logger.error("Error processing batch audio: %s", str(e))
            
            # Cleanup session on error
            await self._cleanup_session(session_id)
            
            return ProcessingResult(
                request_id=request_id,
                session_id=session_id,
                success=False,
                results={},
                processing_time=0.0,
                error=str(e)
            )
    
    async def _process_batch_request(self, request: ProcessingRequest) -> ProcessingResult:
        """Process a batch request through the complete pipeline"""
        start_time = time.time()
        
        try:
            # Load audio data
            audio_data = await self._load_audio_data(request)
            
            # Process through pipeline stages
            results = {}
            
            # Stage 1: Preprocessing
            if self.audio_preprocessor:
                preprocessed_audio = await self.audio_preprocessor.process(
                    audio_data, request.sample_rate
                )
                results["preprocessing"] = {"audio": preprocessed_audio}
            
            # Stage 2: Voice Separation
            if self.voice_separator:
                input_audio = results.get("preprocessing", {}).get("audio", audio_data)
                separated_speakers = await self.voice_separator.separate_speakers(
                    input_audio, request.sample_rate
                )
                results["separation"] = {"speakers": separated_speakers}
            
            # Stage 3: Speaker Recognition (if target embedding provided)
            target_embedding = request.config.get("target_speaker_embedding")
            if self.speaker_recognizer and target_embedding is not None:
                speakers = results.get("separation", {}).get("speakers", [])
                target_speaker = await self.speaker_recognizer.identify_target_speaker(
                    speakers, target_embedding, 0.7
                )
                results["speaker_recognition"] = {"target_speaker": target_speaker}
            
            # Stage 4: Audio Restoration
            if self.audio_restorer:
                target_audio = results.get("speaker_recognition", {}).get("target_speaker")
                if target_audio is not None:
                    restored_audio = await self.audio_restorer.enhance_audio(
                        target_audio, request.sample_rate
                    )
                    results["restoration"] = {"audio": restored_audio}
            
            # Stage 5: Diarization
            if self.diarization_engine:
                input_audio = results.get("restoration", {}).get("audio", audio_data)
                diarization_result = await self.diarization_engine.process(
                    input_audio, request.sample_rate, 
                    request.config.get("max_speakers", 4)
                )
                results["diarization"] = {"result": diarization_result}
            
            # Stage 6: ASR (use offline ASR for batch processing)
            if self.offline_asr:
                input_audio = results.get("restoration", {}).get("audio", audio_data)
                asr_result = await self.offline_asr.transcribe(
                    input_audio, request.sample_rate,
                    request.config.get("language", "en")
                )
                results["asr"] = {"transcript": asr_result}
            
            # Stage 7: Punctuation Restoration
            if self.punctuation_restorer and results.get("asr"):
                transcript = results["asr"]["transcript"]
                punctuated_text = await self.punctuation_restorer.restore_punctuation(transcript)
                results["punctuation"] = {"text": punctuated_text}
            
            # Stage 8: Output Generation
            if self.output_manager:
                final_output = await self.output_manager.generate_final_output(
                    request.session_id, results, time.time(), len(audio_data) / request.sample_rate
                )
                results["output"] = {"result": final_output}
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_processing_metrics(processing_time, True)
            
            return ProcessingResult(
                request_id=request.request_id,
                session_id=request.session_id,
                success=True,
                results=results,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_processing_metrics(processing_time, False)
            
            logger.error("Error processing batch request %s: %s", request.request_id, str(e))
            
            return ProcessingResult(
                request_id=request.request_id,
                session_id=request.session_id,
                success=False,
                results={},
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _load_audio_data(self, request: ProcessingRequest) -> np.ndarray:
        """Load audio data from various sources"""
        if request.audio_data is not None:
            return request.audio_data
        elif request.audio_file_path:
            # Load from file (implementation depends on your audio loading library)
            import soundfile as sf
            audio_data, sample_rate = sf.read(request.audio_file_path)
            if sample_rate != request.sample_rate:
                # Resample if necessary
                from scipy import signal
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert to mono
                audio_data = signal.resample(audio_data, 
                                           int(len(audio_data) * request.sample_rate / sample_rate))
            return audio_data.astype(np.float32)
        else:
            raise ValueError("Either audio_data or audio_file_path must be provided")
    
    async def _queue_processor_worker(self):
        """Background worker for processing queued requests"""
        while self.is_running:
            try:
                # Get next request from queue
                priority, request = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.request_queue.get(timeout=1.0)
                )
                
                if request is None:
                    continue
                
                # Process the request
                result = await self._process_batch_request(request)
                
                # Notify result handlers
                for handler in self.result_handlers:
                    try:
                        await handler(result)
                    except Exception as e:
                        logger.error("Error in result handler: %s", str(e))
                
                # Update metrics
                self.pipeline_metrics.total_requests_processed += 1
                
            except Empty:  # Fixed: Empty is now properly imported
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in queue processor worker: %s", str(e))
    
    async def _monitoring_worker(self):
        """Background worker for monitoring pipeline health"""
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                # Monitor active sessions
                current_time = time.time()
                sessions_to_cleanup = []
                
                for session_id, session_data in self.active_sessions.items():
                    inactivity_time = current_time - session_data["last_activity"]
                    
                    # Cleanup inactive sessions (30 minutes timeout)
                    if inactivity_time > 1800 and session_id.startswith("batch_"):
                        sessions_to_cleanup.append(session_id)
                
                # Cleanup inactive sessions
                for session_id in sessions_to_cleanup:
                    await self._cleanup_session(session_id)
                
                # Log metrics periodically
                logger.debug(
                    "Pipeline metrics: %d active sessions, %d total requests, avg time: %.3fs",
                    self.pipeline_metrics.active_sessions,
                    self.pipeline_metrics.total_requests_processed,
                    self.pipeline_metrics.average_processing_time
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring worker: %s", str(e))
    
    async def _cleanup_session(self, session_id: str):
        """Cleanup a session and its resources"""
        try:
            if session_id in self.active_sessions:
                # Stop processing pipeline
                if self.stream_processor:
                    await self.stream_processor.stop_processing_pipeline(session_id)
                
                # Delete session from managers
                if self.session_manager:
                    await self.session_manager.delete_session(session_id)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                if session_id in self.session_configs:
                    del self.session_configs[session_id]
                
                # Update metrics
                self.pipeline_metrics.active_sessions = len(self.active_sessions)
                
                logger.info("Cleaned up session %s", session_id)
                
        except Exception as e:
            logger.error("Error cleaning up session %s: %s", session_id, str(e))
    
    def _update_processing_metrics(self, processing_time: float, success: bool):
        """Update processing metrics"""
        with self.processing_lock:
            if success:
                total_requests = self.pipeline_metrics.total_requests_processed
                self.pipeline_metrics.average_processing_time = (
                    (self.pipeline_metrics.average_processing_time * total_requests + processing_time) 
                    / (total_requests + 1)
                )
            else:
                self.pipeline_metrics.error_count += 1
    
    def register_result_handler(self, handler: Callable):
        """Register a result handler callback"""
        self.result_handlers.append(handler)
    
    def register_error_handler(self, handler: Callable):
        """Register an error handler callback"""
        self.error_handlers.append(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        return {
            "pipeline_state": self.pipeline_state.value,
            "active_sessions": self.pipeline_metrics.active_sessions,
            "total_requests_processed": self.pipeline_metrics.total_requests_processed,
            "average_processing_time": self.pipeline_metrics.average_processing_time,
            "error_count": self.pipeline_metrics.error_count,
            "gpu_utilization": self.pipeline_metrics.gpu_utilization,
            "cpu_utilization": self.pipeline_metrics.cpu_utilization,
            "memory_usage": self.pipeline_metrics.memory_usage
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "last_activity": session_data["last_activity"],
            "request_count": session_data["request_count"],
            "state": session_data["state"],
            "config": self.session_configs.get(session_id, {})
        }

# Factory function for easy creation
async def create_pipeline_orchestrator(config: Optional[Dict[str, Any]] = None) -> PipelineOrchestrator:
    """
    Create and initialize a PipelineOrchestrator instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PipelineOrchestrator: Initialized pipeline orchestrator instance
    """
    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator

# Example usage and testing
async def example_usage():
    """Example demonstrating pipeline orchestrator usage"""
    
    # Create pipeline orchestrator
    orchestrator = await create_pipeline_orchestrator({
        "max_concurrent_sessions": 10,
        "quality_profile": "balanced"
    })
    
    try:
        # Create a session for real-time processing
        session_id = await orchestrator.create_session({
            "audio_config": {
                "chunk_size": 2.0,
                "overlap_size": 0.5
            },
            "processing_config": {
                "enable_diarization": True,
                "enable_asr": True,
                "language": "en"
            }
        })
        
        print(f"Created session: {session_id}")
        
        # Simulate real-time audio processing
        import numpy as np
        
        sample_rate = 16000
        audio_data = np.random.randn(sample_rate).astype(np.float32)  # 1 second of audio
        
        success = await orchestrator.process_realtime_audio(
            session_id, audio_data, sample_rate
        )
        
        print(f"Real-time processing started: {success}")
        
        # Process batch audio
        batch_result = await orchestrator.process_batch_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            config={"language": "en"}
        )
        
        print(f"Batch processing completed: {batch_result.success}")
        print(f"Processing time: {batch_result.processing_time:.3f}s")
        
        # Get metrics
        metrics = orchestrator.get_metrics()
        print(f"Pipeline metrics: {metrics}")
        
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
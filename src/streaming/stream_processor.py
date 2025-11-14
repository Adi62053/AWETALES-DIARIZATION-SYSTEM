"""
Real-time Stream Processor for Awetales Diarization System

Orchestrates the complete audio processing pipeline for real-time
target speaker diarization and ASR with incremental results.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty

# Import core processing modules
from src.core.audio_preprocess import AudioPreprocessor
from src.core.separation import VoiceSeparator
from src.core.restoration import AudioRestorer
from src.core.diarization import DiarizationEngine
from src.core.speaker_recog import SpeakerRecognizer
from src.core.asr_paraformer import StreamingASR
from src.core.punct_restore import PunctuationRestorer
from src.core.output_manager import OutputManager

# Import streaming modules
from src.streaming.buffer_manager import AudioBufferManager, AudioChunk
from src.streaming.session_manager import SessionManager

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    """Processing stage enumeration"""
    PREPROCESSING = "preprocessing"
    SEPARATION = "separation"
    RESTORATION = "restoration"
    DIARIZATION = "diarization"
    SPEAKER_RECOGNITION = "speaker_recognition"
    ASR = "asr"
    PUNCTUATION = "punctuation"
    OUTPUT = "output"

class ProcessingState(Enum):
    """Processing state enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ProcessingConfig:
    """Processing pipeline configuration"""
    session_id: str
    enable_preprocessing: bool = True
    enable_separation: bool = True
    enable_restoration: bool = True
    enable_diarization: bool = True
    enable_speaker_recognition: bool = True
    enable_asr: bool = True
    enable_punctuation: bool = True
    
    # Performance settings
    chunk_size: float = 2.0
    overlap_size: float = 0.5
    max_workers: int = 4
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None
    
    # Model specific settings
    target_speaker_embedding: Optional[np.ndarray] = None
    language: str = "en"
    max_speakers: int = 4
    
    # Quality settings
    separation_threshold: float = 0.7
    diarization_threshold: float = 0.5
    asr_confidence_threshold: float = 0.6

@dataclass
class ProcessingResult:
    """Processing result for a single chunk"""
    session_id: str
    chunk_id: int
    stage: ProcessingStage
    success: bool
    data: Dict[str, Any]
    processing_time: float
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_chunks_processed: int = 0
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    stage_times: Dict[ProcessingStage, List[float]] = field(default_factory=dict)
    error_count: int = 0
    last_processed_time: float = 0.0

class StreamProcessor:
    """
    Real-time audio processing pipeline orchestrator.
    
    Features:
    - Unified pipeline for all processing stages
    - Real-time incremental processing
    - Parallel execution for performance
    - Error handling and recovery
    - Integration with buffer and session managers
    """
    
    def __init__(self, 
                 buffer_manager: AudioBufferManager,
                 session_manager: SessionManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize StreamProcessor.
        
        Args:
            buffer_manager: AudioBufferManager instance
            session_manager: SessionManager instance
            config: Configuration dictionary
        """
        self.buffer_manager = buffer_manager
        self.session_manager = session_manager
        self.config = self._load_config(config)
        
        # Processing pipelines
        self.pipelines: Dict[str, ProcessingConfig] = {}
        self.pipeline_states: Dict[str, ProcessingState] = {}
        self.pipeline_metrics: Dict[str, PipelineMetrics] = {}
        
        # Core processing modules
        self.audio_preprocessor: Optional[AudioPreprocessor] = None
        self.voice_separator: Optional[VoiceSeparator] = None
        self.audio_restorer: Optional[AudioRestorer] = None
        self.diarization_engine: Optional[DiarizationEngine] = None
        self.speaker_recognizer: Optional[SpeakerRecognizer] = None
        self.streaming_asr: Optional[StreamingASR] = None
        self.punctuation_restorer: Optional[PunctuationRestorer] = None
        self.output_manager: Optional[OutputManager] = None
        
        # Threading and parallelism
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config["max_workers"])
        self.process_pool = ProcessPoolExecutor(max_workers=self.config["max_workers"])
        self.processing_lock = threading.RLock()
        
        # Processing queues
        self.processing_queues: Dict[str, Queue] = {}
        self.result_queues: Dict[str, Queue] = {}
        
        # Background tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Initialize processing modules
        self._initialize_processing_modules()
        
        logger.info("StreamProcessor initialized")
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            "max_concurrent_pipelines": 10,
            "max_chunk_queue_size": 100,
            "processing_timeout": 30.0,
            "monitoring_interval": 5.0,
            "enable_parallel_processing": True,
            "max_workers": 4,
            "gpu_memory_limit": None,
            "enable_metrics": True,
            "log_level": "INFO"
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _initialize_processing_modules(self):
        """Initialize all core processing modules"""
        try:
            # Initialize with GPU support if available
            gpu_config = {
                "use_gpu": self.config.get("use_gpu", True),
                "gpu_memory_limit": self.config.get("gpu_memory_limit")
            }
            
            self.audio_preprocessor = AudioPreprocessor(config=gpu_config)
            self.voice_separator = VoiceSeparator(config=gpu_config)
            self.audio_restorer = AudioRestorer(config=gpu_config)
            self.diarization_engine = DiarizationEngine(config=gpu_config)
            self.speaker_recognizer = SpeakerRecognizer(config=gpu_config)
            self.streaming_asr = StreamingASR(config=gpu_config)
            self.punctuation_restorer = PunctuationRestorer(config=gpu_config)
            self.output_manager = OutputManager()
            
            logger.info("All processing modules initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize processing modules: %s", str(e))
            raise
    
    async def start(self):
        """Start the stream processor and background tasks"""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_worker())
        logger.info("StreamProcessor started")
    
    async def stop(self):
        """Stop the stream processor and cleanup"""
        self.is_running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("StreamProcessor stopped")
    
    async def create_processing_pipeline(self, 
                                       session_id: str,
                                       config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a processing pipeline for a session.
        
        Args:
            session_id: Session identifier
            config: Pipeline configuration
            
        Returns:
            bool: True if pipeline created successfully
        """
        if session_id in self.pipelines:
            logger.warning("Processing pipeline already exists for session %s", session_id)
            return False
        
        if len(self.pipelines) >= self.config["max_concurrent_pipelines"]:
            logger.error("Maximum concurrent pipelines reached")
            return False
        
        try:
            # Create processing configuration
            processing_config = ProcessingConfig(session_id=session_id)
            
            # Apply configuration overrides
            if config:
                for key, value in config.items():
                    if hasattr(processing_config, key):
                        setattr(processing_config, key, value)
            
            # Store pipeline configuration
            self.pipelines[session_id] = processing_config
            self.pipeline_states[session_id] = ProcessingState.IDLE
            self.pipeline_metrics[session_id] = PipelineMetrics()
            
            # Create processing queues
            self.processing_queues[session_id] = Queue(maxsize=self.config["max_chunk_queue_size"])
            self.result_queues[session_id] = Queue()
            
            # Start processing task
            self.processing_tasks[session_id] = asyncio.create_task(
                self._process_pipeline_worker(session_id)
            )
            
            logger.info("Created processing pipeline for session %s", session_id)
            return True
            
        except Exception as e:
            logger.error("Failed to create processing pipeline for session %s: %s", session_id, str(e))
            return False
    
    async def process_audio_chunk(self, session_id: str, audio_chunk: AudioChunk) -> bool:
        """
        Submit audio chunk for processing.
        
        Args:
            session_id: Session identifier
            audio_chunk: Audio chunk to process
            
        Returns:
            bool: True if chunk submitted successfully
        """
        if session_id not in self.pipelines:
            logger.error("No processing pipeline found for session %s", session_id)
            return False
        
        try:
            # Add chunk to processing queue
            processing_queue = self.processing_queues[session_id]
            
            # Use non-blocking put with timeout
            try:
                processing_queue.put(audio_chunk, timeout=1.0)
                logger.debug("Submitted chunk %d for processing in session %s", 
                           audio_chunk.chunk_id, session_id)
                return True
            except Exception:
                logger.warning("Processing queue full for session %s", session_id)
                return False
                
        except Exception as e:
            logger.error("Error submitting chunk for processing in session %s: %s", session_id, str(e))
            return False
    
    async def _process_pipeline_worker(self, session_id: str):
        """Background worker for processing pipeline"""
        logger.info("Starting processing pipeline worker for session %s", session_id)
        
        processing_config = self.pipelines[session_id]
        processing_queue = self.processing_queues[session_id]
        result_queue = self.result_queues[session_id]
        metrics = self.pipeline_metrics[session_id]
        
        self.pipeline_states[session_id] = ProcessingState.PROCESSING
        
        try:
            while self.is_running and session_id in self.pipelines:
                try:
                    # Get next chunk from queue with timeout
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: processing_queue.get(timeout=1.0)
                    )
                    
                    if audio_chunk is None:  # Sentinel value for shutdown
                        break
                    
                    # Process the chunk through the pipeline
                    start_time = time.time()
                    result = await self._process_single_chunk(session_id, audio_chunk)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    metrics.total_chunks_processed += 1
                    metrics.last_processed_time = time.time()
                    metrics.average_processing_time = (
                        (metrics.average_processing_time * (metrics.total_chunks_processed - 1) + processing_time) 
                        / metrics.total_chunks_processed
                    )
                    metrics.max_processing_time = max(metrics.max_processing_time, processing_time)
                    
                    # Send result to session manager
                    if result.success:
                        await self._send_processing_result(session_id, result)
                    
                    # Check if processing time meets latency requirements
                    if processing_time > 0.5:  # 500ms threshold
                        logger.warning("Processing latency high for session %s: %.3fs", 
                                     session_id, processing_time)
                    
                except Empty:
                    # Queue empty, continue waiting
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in processing pipeline worker for session %s: %s", session_id, str(e))
                    metrics.error_count += 1
                    await asyncio.sleep(0.1)  # Prevent tight loop on errors
        
        except Exception as e:
            logger.error("Processing pipeline worker failed for session %s: %s", session_id, str(e))
            self.pipeline_states[session_id] = ProcessingState.ERROR
        finally:
            if session_id in self.pipeline_states:
                self.pipeline_states[session_id] = ProcessingState.COMPLETED
            logger.info("Processing pipeline worker stopped for session %s", session_id)
    
    async def _process_single_chunk(self, session_id: str, audio_chunk: AudioChunk) -> ProcessingResult:
        """
        Process a single audio chunk through the complete pipeline.
        
        Args:
            session_id: Session identifier
            audio_chunk: Audio chunk to process
            
        Returns:
            ProcessingResult: Processing result
        """
        processing_config = self.pipelines[session_id]
        results = {}
        
        try:
            # Stage 1: Audio Preprocessing
            if processing_config.enable_preprocessing and self.audio_preprocessor:
                start_time = time.time()
                preprocessed_audio = await self._run_processing_stage(
                    self.audio_preprocessor.process, 
                    audio_chunk.audio_data, 
                    audio_chunk.sample_rate
                )
                preprocessing_time = time.time() - start_time
                results["preprocessing"] = {
                    "audio": preprocessed_audio,
                    "processing_time": preprocessing_time
                }
            
            # Stage 2: Voice Separation
            if processing_config.enable_separation and self.voice_separator:
                start_time = time.time()
                input_audio = results.get("preprocessing", {}).get("audio", audio_chunk.audio_data)
                separated_speakers = await self._run_processing_stage(
                    self.voice_separator.separate_speakers,
                    input_audio,
                    audio_chunk.sample_rate
                )
                separation_time = time.time() - start_time
                results["separation"] = {
                    "speakers": separated_speakers,
                    "processing_time": separation_time
                }
            
            # Stage 3: Target Speaker Isolation
            if (processing_config.enable_speaker_recognition and 
                self.speaker_recognizer and 
                processing_config.target_speaker_embedding is not None):
                
                start_time = time.time()
                speakers = results.get("separation", {}).get("speakers", [])
                target_speaker = await self._run_processing_stage(
                    self.speaker_recognizer.identify_target_speaker,
                    speakers,
                    processing_config.target_speaker_embedding,
                    processing_config.separation_threshold
                )
                speaker_recog_time = time.time() - start_time
                results["speaker_recognition"] = {
                    "target_speaker": target_speaker,
                    "processing_time": speaker_recog_time
                }
            
            # Stage 4: Audio Restoration
            if processing_config.enable_restoration and self.audio_restorer:
                start_time = time.time()
                target_audio = results.get("speaker_recognition", {}).get("target_speaker")
                if target_audio is not None:
                    restored_audio = await self._run_processing_stage(
                        self.audio_restorer.enhance_audio,
                        target_audio,
                        audio_chunk.sample_rate
                    )
                    restoration_time = time.time() - start_time
                    results["restoration"] = {
                        "audio": restored_audio,
                        "processing_time": restoration_time
                    }
            
            # Stage 5: Diarization
            if processing_config.enable_diarization and self.diarization_engine:
                start_time = time.time()
                input_audio = results.get("restoration", {}).get("audio", audio_chunk.audio_data)
                diarization_result = await self._run_processing_stage(
                    self.diarization_engine.process,
                    input_audio,
                    audio_chunk.sample_rate,
                    processing_config.max_speakers
                )
                diarization_time = time.time() - start_time
                results["diarization"] = {
                    "result": diarization_result,
                    "processing_time": diarization_time
                }
            
            # Stage 6: ASR
            if processing_config.enable_asr and self.streaming_asr:
                start_time = time.time()
                input_audio = results.get("restoration", {}).get("audio", audio_chunk.audio_data)
                asr_result = await self._run_processing_stage(
                    self.streaming_asr.transcribe,
                    input_audio,
                    audio_chunk.sample_rate,
                    processing_config.language
                )
                asr_time = time.time() - start_time
                results["asr"] = {
                    "transcript": asr_result,
                    "processing_time": asr_time
                }
            
            # Stage 7: Punctuation Restoration
            if (processing_config.enable_punctuation and 
                self.punctuation_restorer and 
                results.get("asr")):
                
                start_time = time.time()
                transcript = results["asr"]["transcript"]
                punctuated_text = await self._run_processing_stage(
                    self.punctuation_restorer.restore_punctuation,
                    transcript
                )
                punctuation_time = time.time() - start_time
                results["punctuation"] = {
                    "text": punctuated_text,
                    "processing_time": punctuation_time
                }
            
            # Stage 8: Output Generation
            if self.output_manager:
                start_time = time.time()
                final_result = await self._run_processing_stage(
                    self.output_manager.generate_incremental_output,
                    session_id,
                    audio_chunk.chunk_id,
                    results,
                    audio_chunk.timestamp,
                    audio_chunk.duration
                )
                output_time = time.time() - start_time
                results["output"] = {
                    "result": final_result,
                    "processing_time": output_time
                }
            
            return ProcessingResult(
                session_id=session_id,
                chunk_id=audio_chunk.chunk_id,
                stage=ProcessingStage.OUTPUT,
                success=True,
                data=results,
                processing_time=time.time() - audio_chunk.timestamp
            )
            
        except Exception as e:
            logger.error("Error processing chunk %d for session %s: %s", 
                       audio_chunk.chunk_id, session_id, str(e))
            return ProcessingResult(
                session_id=session_id,
                chunk_id=audio_chunk.chunk_id,
                stage=ProcessingStage.OUTPUT,
                success=False,
                data={},
                processing_time=0.0,
                error=str(e)
            )
    
    async def _run_processing_stage(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a processing stage with error handling and timeout.
        
        Args:
            func: Processing function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Processing result
        """
        try:
            # Run in thread pool for CPU-bound tasks
            if self.config["enable_parallel_processing"]:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: func(*args, **kwargs)
                )
            else:
                result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error("Error in processing stage %s: %s", func.__name__, str(e))
            raise
    
    async def _send_processing_result(self, session_id: str, result: ProcessingResult):
        """Send processing result to session manager"""
        try:
            # Prepare incremental result message
            incremental_result = {
                "chunk_id": result.chunk_id,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp,
                "stages_completed": list(result.data.keys()),
                "final_output": result.data.get("output", {}).get("result", {})
            }
            
            # Send via session manager
            await self.session_manager.send_message(
                session_id,
                incremental_result,
                "processing_result"
            )
            
            logger.debug("Sent processing result for chunk %d in session %s", 
                       result.chunk_id, session_id)
            
        except Exception as e:
            logger.error("Error sending processing result for session %s: %s", session_id, str(e))
    
    async def _monitoring_worker(self):
        """Background worker for monitoring pipeline performance"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["monitoring_interval"])
                
                # Monitor active pipelines
                active_pipelines = [
                    session_id for session_id, state in self.pipeline_states.items()
                    if state == ProcessingState.PROCESSING
                ]
                
                if active_pipelines:
                    logger.info("Active processing pipelines: %s", active_pipelines)
                    
                    # Log performance metrics
                    for session_id in active_pipelines:
                        metrics = self.pipeline_metrics[session_id]
                        if metrics.total_chunks_processed > 0:
                            logger.debug(
                                "Session %s metrics: %d chunks, avg time: %.3fs, max time: %.3fs, errors: %d",
                                session_id,
                                metrics.total_chunks_processed,
                                metrics.average_processing_time,
                                metrics.max_processing_time,
                                metrics.error_count
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring worker: %s", str(e))
    
    async def stop_processing_pipeline(self, session_id: str) -> bool:
        """
        Stop processing pipeline for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if pipeline stopped successfully
        """
        if session_id not in self.pipelines:
            return False
        
        try:
            # Cancel processing task
            if session_id in self.processing_tasks:
                self.processing_tasks[session_id].cancel()
                del self.processing_tasks[session_id]
            
            # Cleanup queues
            if session_id in self.processing_queues:
                del self.processing_queues[session_id]
            if session_id in self.result_queues:
                del self.result_queues[session_id]
            
            # Remove pipeline data
            del self.pipelines[session_id]
            del self.pipeline_states[session_id]
            del self.pipeline_metrics[session_id]
            
            logger.info("Stopped processing pipeline for session %s", session_id)
            return True
            
        except Exception as e:
            logger.error("Error stopping processing pipeline for session %s: %s", session_id, str(e))
            return False
    
    def get_pipeline_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline metrics for a session"""
        if session_id not in self.pipeline_metrics:
            return None
        
        metrics = self.pipeline_metrics[session_id]
        return {
            "total_chunks_processed": metrics.total_chunks_processed,
            "average_processing_time": metrics.average_processing_time,
            "max_processing_time": metrics.max_processing_time,
            "error_count": metrics.error_count,
            "last_processed_time": metrics.last_processed_time
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        active_pipelines = len([
            state for state in self.pipeline_states.values()
            if state == ProcessingState.PROCESSING
        ])
        
        total_chunks = sum(
            metrics.total_chunks_processed 
            for metrics in self.pipeline_metrics.values()
        )
        
        return {
            "total_pipelines": len(self.pipelines),
            "active_pipelines": active_pipelines,
            "total_chunks_processed": total_chunks,
            "pipeline_metrics": {
                session_id: self.get_pipeline_metrics(session_id)
                for session_id in self.pipelines.keys()
            }
        }

# Factory function for easy creation
async def create_stream_processor(buffer_manager: AudioBufferManager,
                                session_manager: SessionManager,
                                config: Optional[Dict[str, Any]] = None) -> StreamProcessor:
    """
    Create and start a StreamProcessor instance.
    
    Args:
        buffer_manager: AudioBufferManager instance
        session_manager: SessionManager instance
        config: Configuration dictionary
        
    Returns:
        StreamProcessor: Started stream processor instance
    """
    processor = StreamProcessor(buffer_manager, session_manager, config)
    await processor.start()
    return processor

# Example usage and testing
async def example_usage():
    """Example demonstrating stream processor usage"""
    
    # Create buffer manager and session manager first
    buffer_manager = await create_buffer_manager()
    session_manager = await create_session_manager(buffer_manager)
    
    # Create stream processor
    stream_processor = await create_stream_processor(buffer_manager, session_manager)
    
    try:
        # Create a session and processing pipeline
        session_id = await session_manager.create_session(
            user_id="test_user_123",
            config={
                "audio_config": {
                    "chunk_size": 2.0,
                    "overlap_size": 0.5
                }
            }
        )
        
        # Create processing pipeline
        await stream_processor.create_processing_pipeline(
            session_id,
            config={
                "enable_preprocessing": True,
                "enable_separation": True,
                "enable_asr": True,
                "language": "en"
            }
        )
        
        print(f"Created processing pipeline for session: {session_id}")
        
        # Simulate processing some audio chunks
        import numpy as np
        
        sample_rate = 16000
        for i in range(3):
            # Generate test audio
            audio_data = np.random.randn(int(2.0 * sample_rate)).astype(np.float32)
            
            # Create audio chunk
            audio_chunk = AudioChunk(
                session_id=session_id,
                chunk_id=i,
                audio_data=audio_data,
                timestamp=time.time(),
                sample_rate=sample_rate,
                duration=2.0
            )
            
            # Submit for processing
            await stream_processor.process_audio_chunk(session_id, audio_chunk)
            print(f"Submitted chunk {i} for processing")
            
            await asyncio.sleep(0.1)  # Small delay between chunks
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Get metrics
        metrics = stream_processor.get_all_metrics()
        print(f"Stream processor metrics: {metrics}")
        
    finally:
        await stream_processor.stop()
        await session_manager.stop()
        await buffer_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
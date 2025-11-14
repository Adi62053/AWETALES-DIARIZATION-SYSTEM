"""
Circuit Breaker for Awetales Diarization System

Implements circuit breaker pattern for model failure detection,
automatic fallback mechanisms, and graceful degradation.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker state enumeration"""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Circuit open, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered

class FailureType(Enum):
    """Failure type enumeration"""
    TIMEOUT = "timeout"
    ERROR = "error"
    PERFORMANCE = "performance"
    RESOURCE = "resource"

class ModelType(Enum):
    """Model type enumeration"""
    AUDIO_PREPROCESS = "audio_preprocess"
    VOICE_SEPARATION = "voice_separation"
    AUDIO_RESTORATION = "audio_restoration"
    DIARIZATION = "diarization"
    SPEAKER_RECOGNITION = "speaker_recognition"
    STREAMING_ASR = "streaming_asr"
    OFFLINE_ASR = "offline_asr"
    PUNCTUATION = "punctuation"

@dataclass
class FailureMetrics:
    """Failure metrics for a model"""
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    failure_types: Dict[FailureType, int] = field(default_factory=lambda: {
        FailureType.TIMEOUT: 0,
        FailureType.ERROR: 0,
        FailureType.PERFORMANCE: 0,
        FailureType.RESOURCE: 0
    })

@dataclass
class PerformanceMetrics:
    """Performance metrics for a model"""
    average_latency: float = 0.0
    max_latency: float = 0.0
    success_rate: float = 100.0
    total_requests: int = 0
    successful_requests: int = 0
    last_processing_time: float = 0.0

@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    model_type: ModelType
    failure_threshold: int = 5          # Consecutive failures to open circuit
    success_threshold: int = 3          # Consecutive successes to close circuit
    timeout_seconds: float = 30.0       # Time before attempting recovery
    timeout_duration: float = 60.0      # Duration to keep circuit open
    performance_threshold: float = 2.0  # Latency threshold in seconds
    accuracy_threshold: float = 0.8     # Minimum accuracy threshold
    monitoring_interval: float = 10.0   # Health check interval
    enable_fallback: bool = True        # Enable fallback to alternative models

@dataclass
class FallbackConfig:
    """Fallback configuration"""
    primary_model: ModelType
    fallback_models: List[ModelType]    # Ordered by preference
    fallback_strategy: str = "sequential"  # sequential, parallel, hybrid

class CircuitBreaker:
    """
    Circuit breaker implementation for model failure detection and fallback.
    
    Features:
    - Model health monitoring and failure detection
    - Automatic circuit state management
    - Performance threshold monitoring
    - Graceful fallback to alternative models
    - Recovery and retry mechanisms
    """
    
    def __init__(self, model_type: ModelType, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CircuitBreaker for a specific model.
        
        Args:
            model_type: Type of model to monitor
            config: Circuit breaker configuration
        """
        self.model_type = model_type
        self.config = self._load_config(config)
        
        # Circuit state
        self.state: CircuitState = CircuitState.CLOSED
        self.state_changed_time: float = time.time()
        
        # Metrics
        self.failure_metrics = FailureMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Fallback management
        self.fallback_config: Optional[FallbackConfig] = None
        self.active_fallback: Optional[ModelType] = None
        self.fallback_performance: Dict[ModelType, PerformanceMetrics] = {}
        
        # Threading and synchronization
        self.circuit_lock = threading.RLock()
        self.metrics_lock = threading.RLock()
        
        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        self.fallback_callbacks: List[Callable] = []
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("CircuitBreaker initialized for %s with config: %s", 
                   model_type.value, self.config)
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> CircuitConfig:
        """Load and validate configuration"""
        default_config = {
            "failure_threshold": 5,
            "success_threshold": 3,
            "timeout_seconds": 30.0,
            "timeout_duration": 60.0,
            "performance_threshold": 2.0,
            "accuracy_threshold": 0.8,
            "monitoring_interval": 10.0,
            "enable_fallback": True
        }
        
        if config:
            default_config.update(config)
        
        return CircuitConfig(
            model_type=self.model_type,
            **default_config
        )
    
    async def start(self):
        """Start circuit breaker monitoring"""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_worker())
        logger.info("CircuitBreaker started for %s", self.model_type.value)
    
    async def stop(self):
        """Stop circuit breaker monitoring"""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("CircuitBreaker stopped for %s", self.model_type.value)
    
    def set_fallback_config(self, fallback_config: FallbackConfig):
        """Set fallback configuration"""
        self.fallback_config = fallback_config
        logger.info("Fallback config set for %s: %s", 
                   self.model_type.value, [m.value for m in fallback_config.fallback_models])
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            operation: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Operation result
            
        Raises:
            Exception: If circuit is open or operation fails
        """
        # Check circuit state
        if not self._allow_request():
            if self.fallback_config and self.config.enable_fallback:
                return await self._execute_fallback(operation, *args, **kwargs)
            else:
                raise CircuitOpenError(
                    f"Circuit open for {self.model_type.value}. "
                    f"State: {self.state.value}, "
                    f"Last failure: {self.failure_metrics.last_failure_time}"
                )
        
        try:
            # Execute the operation
            start_time = time.time()
            result = await self._execute_operation(operation, *args, **kwargs)
            processing_time = time.time() - start_time
            
            # Record success
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            # Record failure
            failure_type = self._classify_failure(e)
            self._record_failure(failure_type)
            
            # Attempt fallback if configured
            if self.fallback_config and self.config.enable_fallback:
                logger.warning("Primary model %s failed, attempting fallback: %s", 
                             self.model_type.value, str(e))
                return await self._execute_fallback(operation, *args, **kwargs)
            else:
                raise
    
    def _allow_request(self) -> bool:
        """
        Check if request should be allowed based on circuit state.
        
        Returns:
            bool: True if request should be allowed
        """
        with self.circuit_lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if current_time - self.state_changed_time >= self.config.timeout_seconds:
                    # Transition to half-open state
                    self._transition_state(CircuitState.HALF_OPEN)
                    return True
                else:
                    return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True
            
            return False
    
    async def _execute_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with timeout and error handling"""
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.config.performance_threshold
                )
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, operation, *args, **kwargs),
                    timeout=self.config.performance_threshold
                )
            
            return result
            
        except asyncio.TimeoutError:
            raise OperationTimeoutError(
                f"Operation timeout for {self.model_type.value}: "
                f"exceeded {self.config.performance_threshold}s"
            )
        except Exception as e:
            raise
    
    async def _execute_fallback(self, original_operation: Callable, *args, **kwargs) -> Any:
        """
        Execute fallback operation when primary model fails.
        
        Args:
            original_operation: Original operation that failed
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Any: Fallback operation result
            
        Raises:
            Exception: If all fallbacks fail
        """
        if not self.fallback_config:
            raise NoFallbackAvailableError(
                f"No fallback configured for {self.model_type.value}"
            )
        
        fallback_errors = []
        
        for fallback_model in self.fallback_config.fallback_models:
            try:
                logger.info("Attempting fallback to %s for %s", 
                           fallback_model.value, self.model_type.value)
                
                # In a real implementation, this would call the fallback model
                # For now, we'll simulate a fallback operation
                result = await self._execute_fallback_operation(
                    fallback_model, original_operation, *args, **kwargs
                )
                
                # Record successful fallback
                self.active_fallback = fallback_model
                self._record_fallback_success(fallback_model)
                
                # Notify fallback callbacks
                for callback in self.fallback_callbacks:
                    try:
                        await callback(self.model_type, fallback_model, True)
                    except Exception as e:
                        logger.error("Error in fallback callback: %s", str(e))
                
                return result
                
            except Exception as e:
                fallback_errors.append(f"{fallback_model.value}: {str(e)}")
                self._record_fallback_failure(fallback_model)
                
                # Notify fallback callbacks
                for callback in self.fallback_callbacks:
                    try:
                        await callback(self.model_type, fallback_model, False)
                    except Exception as e:
                        logger.error("Error in fallback callback: %s", str(e))
        
        # All fallbacks failed
        raise AllFallbacksFailedError(
            f"All fallbacks failed for {self.model_type.value}. Errors: {fallback_errors}"
        )
    
    async def _execute_fallback_operation(self, fallback_model: ModelType, 
                                        original_operation: Callable, 
                                        *args, **kwargs) -> Any:
        """
        Execute operation using fallback model.
        
        Args:
            fallback_model: Fallback model to use
            original_operation: Original operation
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Any: Fallback operation result
        """
        # In a real implementation, this would:
        # 1. Load the appropriate fallback model
        # 2. Execute the operation with fallback model
        # 3. Return the result
        
        # For simulation purposes, we'll use the original operation
        # but with different behavior based on the fallback model
        start_time = time.time()
        
        try:
            # Simulate fallback operation (usually lighter/faster)
            if asyncio.iscoroutinefunction(original_operation):
                result = await original_operation(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, original_operation, *args, **kwargs
                )
            
            processing_time = time.time() - start_time
            
            # Update fallback performance metrics
            self._update_fallback_performance(fallback_model, processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_fallback_performance(fallback_model, processing_time, False)
            raise
    
    def _record_success(self, processing_time: float):
        """Record successful operation"""
        with self.metrics_lock:
            # Update performance metrics
            self.performance_metrics.total_requests += 1
            self.performance_metrics.successful_requests += 1
            self.performance_metrics.last_processing_time = processing_time
            
            # Update average latency
            total_requests = self.performance_metrics.total_requests
            current_avg = self.performance_metrics.average_latency
            self.performance_metrics.average_latency = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
            
            self.performance_metrics.max_latency = max(
                self.performance_metrics.max_latency, processing_time
            )
            
            self.performance_metrics.success_rate = (
                self.performance_metrics.successful_requests / 
                self.performance_metrics.total_requests * 100
            )
            
            # Update failure metrics
            self.failure_metrics.consecutive_successes += 1
            self.failure_metrics.consecutive_failures = 0
            self.failure_metrics.last_success_time = time.time()
            
            # Check if we should close the circuit
            if (self.state == CircuitState.HALF_OPEN and 
                self.failure_metrics.consecutive_successes >= self.config.success_threshold):
                self._transition_state(CircuitState.CLOSED)
    
    def _record_failure(self, failure_type: FailureType):
        """Record failed operation"""
        with self.metrics_lock:
            # Update failure metrics
            self.failure_metrics.total_failures += 1
            self.failure_metrics.consecutive_failures += 1
            self.failure_metrics.consecutive_successes = 0
            self.failure_metrics.last_failure_time = time.time()
            self.failure_metrics.failure_types[failure_type] += 1
            
            # Update performance metrics
            self.performance_metrics.total_requests += 1
            self.performance_metrics.success_rate = (
                self.performance_metrics.successful_requests / 
                self.performance_metrics.total_requests * 100
            )
            
            # Check if we should open the circuit
            if (self.state != CircuitState.OPEN and 
                self.failure_metrics.consecutive_failures >= self.config.failure_threshold):
                self._transition_state(CircuitState.OPEN)
    
    def _record_fallback_success(self, fallback_model: ModelType):
        """Record successful fallback operation"""
        if fallback_model not in self.fallback_performance:
            self.fallback_performance[fallback_model] = PerformanceMetrics()
        
        metrics = self.fallback_performance[fallback_model]
        metrics.successful_requests += 1
        metrics.total_requests += 1
        metrics.success_rate = metrics.successful_requests / metrics.total_requests * 100
    
    def _record_fallback_failure(self, fallback_model: ModelType):
        """Record failed fallback operation"""
        if fallback_model not in self.fallback_performance:
            self.fallback_performance[fallback_model] = PerformanceMetrics()
        
        metrics = self.fallback_performance[fallback_model]
        metrics.total_requests += 1
        metrics.success_rate = metrics.successful_requests / metrics.total_requests * 100
    
    def _update_fallback_performance(self, fallback_model: ModelType, 
                                   processing_time: float, success: bool):
        """Update fallback performance metrics"""
        if fallback_model not in self.fallback_performance:
            self.fallback_performance[fallback_model] = PerformanceMetrics()
        
        metrics = self.fallback_performance[fallback_model]
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
            metrics.last_processing_time = processing_time
            
            # Update average latency
            current_avg = metrics.average_latency
            metrics.average_latency = (
                (current_avg * (metrics.total_requests - 1) + processing_time) / metrics.total_requests
            )
            metrics.max_latency = max(metrics.max_latency, processing_time)
        
        metrics.success_rate = metrics.successful_requests / metrics.total_requests * 100
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify failure type based on exception"""
        if isinstance(error, OperationTimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(error, ResourceExhaustionError):
            return FailureType.RESOURCE
        elif isinstance(error, PerformanceDegradationError):
            return FailureType.PERFORMANCE
        else:
            return FailureType.ERROR
    
    def _transition_state(self, new_state: CircuitState):
        """Transition to new circuit state"""
        with self.circuit_lock:
            old_state = self.state
            self.state = new_state
            self.state_changed_time = time.time()
            
            # Reset appropriate counters
            if new_state == CircuitState.HALF_OPEN:
                self.failure_metrics.consecutive_failures = 0
                self.failure_metrics.consecutive_successes = 0
            elif new_state == CircuitState.CLOSED:
                self.failure_metrics.consecutive_failures = 0
                self.failure_metrics.consecutive_successes = 0
                self.active_fallback = None
            
            logger.info("Circuit state changed for %s: %s -> %s", 
                       self.model_type.value, old_state.value, new_state.value)
            
            # Notify state change callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(self.model_type, new_state, old_state)
                except Exception as e:
                    logger.error("Error in state change callback: %s", str(e))
    
    async def _monitoring_worker(self):
        """Background worker for circuit monitoring"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Perform health checks
                await self._perform_health_check()
                
                # Log circuit status periodically
                logger.debug(
                    "Circuit %s: state=%s, failures=%d, success_rate=%.1f%%, latency=%.3fs",
                    self.model_type.value,
                    self.state.value,
                    self.failure_metrics.consecutive_failures,
                    self.performance_metrics.success_rate,
                    self.performance_metrics.average_latency
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring worker for %s: %s", 
                           self.model_type.value, str(e))
    
    async def _perform_health_check(self):
        """Perform health check on the circuit"""
        # Check if circuit has been open too long
        if (self.state == CircuitState.OPEN and 
            time.time() - self.state_changed_time > self.config.timeout_duration):
            self._transition_state(CircuitState.HALF_OPEN)
        
        # Check performance thresholds
        if (self.state == CircuitState.CLOSED and 
            self.performance_metrics.average_latency > self.config.performance_threshold):
            logger.warning(
                "Performance degradation detected for %s: latency=%.3fs, threshold=%.3fs",
                self.model_type.value,
                self.performance_metrics.average_latency,
                self.config.performance_threshold
            )
    
    def register_state_change_callback(self, callback: Callable):
        """Register callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def register_fallback_callback(self, callback: Callable):
        """Register callback for fallback operations"""
        self.fallback_callbacks.append(callback)
    
    def get_circuit_state(self) -> Dict[str, Any]:
        """Get current circuit state and metrics"""
        return {
            "model_type": self.model_type.value,
            "state": self.state.value,
            "state_changed_time": self.state_changed_time,
            "failure_metrics": {
                "total_failures": self.failure_metrics.total_failures,
                "consecutive_failures": self.failure_metrics.consecutive_failures,
                "consecutive_successes": self.failure_metrics.consecutive_successes,
                "last_failure_time": self.failure_metrics.last_failure_time,
                "failure_types": {ft.value: count for ft, count in self.failure_metrics.failure_types.items()}
            },
            "performance_metrics": {
                "average_latency": self.performance_metrics.average_latency,
                "max_latency": self.performance_metrics.max_latency,
                "success_rate": self.performance_metrics.success_rate,
                "total_requests": self.performance_metrics.total_requests,
                "successful_requests": self.performance_metrics.successful_requests
            },
            "active_fallback": self.active_fallback.value if self.active_fallback else None,
            "fallback_performance": {
                model.value: {
                    "average_latency": metrics.average_latency,
                    "success_rate": metrics.success_rate,
                    "total_requests": metrics.total_requests
                }
                for model, metrics in self.fallback_performance.items()
            }
        }

class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for all system models.
    """
    
    def __init__(self, config: Optional[Dict[ModelType, Dict[str, Any]]] = None):
        """
        Initialize CircuitBreakerManager.
        
        Args:
            config: Configuration for each model type
        """
        self.config = config or {}
        self.circuit_breakers: Dict[ModelType, CircuitBreaker] = {}
        self.fallback_strategies: Dict[ModelType, FallbackConfig] = {}
        
        # Initialize circuit breakers for all model types
        self._initialize_circuit_breakers()
        
        logger.info("CircuitBreakerManager initialized with %d circuit breakers", 
                   len(self.circuit_breakers))
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all model types"""
        for model_type in ModelType:
            model_config = self.config.get(model_type, {})
            circuit_breaker = CircuitBreaker(model_type, model_config)
            self.circuit_breakers[model_type] = circuit_breaker
    
    async def start(self):
        """Start all circuit breakers"""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.start()
        
        logger.info("CircuitBreakerManager started")
    
    async def stop(self):
        """Stop all circuit breakers"""
        for circuit_breaker in self.circuit_breakers.values():
            await circuit_breaker.stop()
        
        logger.info("CircuitBreakerManager stopped")
    
    def set_fallback_strategy(self, model_type: ModelType, fallback_config: FallbackConfig):
        """Set fallback strategy for a model type"""
        if model_type in self.circuit_breakers:
            self.circuit_breakers[model_type].set_fallback_config(fallback_config)
            self.fallback_strategies[model_type] = fallback_config
    
    def get_circuit_breaker(self, model_type: ModelType) -> CircuitBreaker:
        """Get circuit breaker for a specific model type"""
        return self.circuit_breakers[model_type]
    
    def get_all_states(self) -> Dict[str, Any]:
        """Get states of all circuit breakers"""
        return {
            model_type.value: circuit_breaker.get_circuit_state()
            for model_type, circuit_breaker in self.circuit_breakers.items()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on circuit states"""
        total_circuits = len(self.circuit_breakers)
        open_circuits = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.state == CircuitState.OPEN
        )
        half_open_circuits = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.state == CircuitState.HALF_OPEN
        )
        
        health_score = (
            (total_circuits - open_circuits - 0.5 * half_open_circuits) / 
            total_circuits * 100
        )
        
        return {
            "health_score": health_score,
            "total_circuits": total_circuits,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "closed_circuits": total_circuits - open_circuits - half_open_circuits,
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical"
        }

# Custom exceptions
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open"""
    pass

class OperationTimeoutError(CircuitBreakerError):
    """Raised when operation times out"""
    pass

class NoFallbackAvailableError(CircuitBreakerError):
    """Raised when no fallback is available"""
    pass

class AllFallbacksFailedError(CircuitBreakerError):
    """Raised when all fallbacks fail"""
    pass

class ResourceExhaustionError(CircuitBreakerError):
    """Raised when resources are exhausted"""
    pass

class PerformanceDegradationError(CircuitBreakerError):
    """Raised when performance degrades below threshold"""
    pass

# Factory function for easy creation
async def create_circuit_breaker_manager(
    config: Optional[Dict[ModelType, Dict[str, Any]]] = None
) -> CircuitBreakerManager:
    """
    Create and start a CircuitBreakerManager instance.
    
    Args:
        config: Configuration for circuit breakers
        
    Returns:
        CircuitBreakerManager: Started circuit breaker manager instance
    """
    manager = CircuitBreakerManager(config)
    await manager.start()
    return manager

# Example usage and testing
async def example_usage():
    """Example demonstrating circuit breaker usage"""
    
    # Configure circuit breakers
    config = {
        ModelType.STREAMING_ASR: {
            "failure_threshold": 3,
            "timeout_seconds": 20.0,
            "performance_threshold": 1.5
        },
        ModelType.DIARIZATION: {
            "failure_threshold": 5,
            "timeout_seconds": 30.0,
            "performance_threshold": 2.0
        }
    }
    
    # Create circuit breaker manager
    manager = await create_circuit_breaker_manager(config)
    
    try:
        # Set fallback strategies
        asr_fallback = FallbackConfig(
            primary_model=ModelType.STREAMING_ASR,
            fallback_models=[ModelType.OFFLINE_ASR],
            fallback_strategy="sequential"
        )
        manager.set_fallback_strategy(ModelType.STREAMING_ASR, asr_fallback)
        
        # Get a circuit breaker
        asr_circuit = manager.get_circuit_breaker(ModelType.STREAMING_ASR)
        
        # Simulate an operation
        async def mock_asr_operation(audio_data):
            # Simulate processing
            await asyncio.sleep(0.1)
            return {"transcript": "Hello world", "confidence": 0.9}
        
        try:
            result = await asr_circuit.execute(mock_asr_operation, "audio_data")
            print(f"ASR operation successful: {result}")
        except Exception as e:
            print(f"ASR operation failed: {e}")
        
        # Check system health
        health = manager.get_system_health()
        print(f"System health: {health}")
        
        # Get all circuit states
        states = manager.get_all_states()
        print(f"Circuit states: {list(states.keys())}")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
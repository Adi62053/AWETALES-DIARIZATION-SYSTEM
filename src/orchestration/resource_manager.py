"""
Resource Manager for Awetales Diarization System

Manages GPU/CPU resources, task prioritization, and load balancing
for optimal performance in real-time audio processing.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

# Optional GPU imports with fallback
try:
    import GPUtil
    import torch
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None
    torch = None
    logger.warning("GPU monitoring not available. Using CPU-only mode.")

class ResourceLevel(Enum):
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class ResourceConfig:
    monitoring_interval: float = 1.0
    scheduling_interval: float = 0.5
    gpu_memory_threshold: float = 0.8
    cpu_threshold: float = 0.75
    max_concurrent_tasks: int = 20
    enable_gpu: bool = False
    enable_prediction: bool = True
    prediction_window: float = 5.0
    log_level: str = "INFO"

class ResourceManager:
    """Manages system resources for the diarization pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = ResourceConfig(**(config or {}))
        self.logger = logging.getLogger(__name__)
        self._is_running = False
        self._monitor_task = None
        self.resource_usage = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_used': 0.0,
            'active_tasks': 0
        }
        
        # Try to initialize GPU monitoring
        self.gpu_available = False
        try:
            if torch and torch.cuda.is_available():
                self.gpu_available = True
                logger.info(f"GPU monitoring enabled: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("GPU monitoring not available. Using CPU-only mode.")
        except Exception:
            logger.warning("GPU monitoring not available. Using CPU-only mode.")
    
    async def initialize(self):
        """Initialize the resource manager - required by FastAPI startup"""
        self.logger.info("ResourceManager initialized")
        self._is_running = True
        # Start monitoring in background
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        return True
    
    async def shutdown(self):
        """Shutdown the resource manager"""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_resources(self):
        """Monitor system resources in background"""
        while self._is_running:
            try:
                # Monitor CPU
                self.resource_usage['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                
                # Monitor memory
                memory = psutil.virtual_memory()
                self.resource_usage['memory_percent'] = memory.percent
                
                # Monitor GPU if available
                if self.gpu_available:
                    try:
                        if torch.cuda.is_available():
                            self.resource_usage['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    except Exception as e:
                        logger.debug(f"GPU monitoring error: {e}")
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    def get_resource_level(self) -> ResourceLevel:
        """Get current resource utilization level"""
        cpu_usage = self.resource_usage['cpu_percent']
        memory_usage = self.resource_usage['memory_percent']
        
        if cpu_usage > 90 or memory_usage > 90:
            return ResourceLevel.CRITICAL
        elif cpu_usage > 75 or memory_usage > 80:
            return ResourceLevel.DEGRADED
        elif cpu_usage > 50 or memory_usage > 70:
            return ResourceLevel.ACCEPTABLE
        else:
            return ResourceLevel.OPTIMAL
    
    def can_accept_task(self, estimated_load: float = 1.0) -> bool:
        """Check if system can accept new task"""
        resource_level = self.get_resource_level()
        return resource_level in [ResourceLevel.OPTIMAL, ResourceLevel.ACCEPTABLE]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        return {
            'resource_level': self.get_resource_level().value,
            'cpu_percent': self.resource_usage['cpu_percent'],
            'memory_percent': self.resource_usage['memory_percent'],
            'gpu_memory_used_gb': self.resource_usage['gpu_memory_used'],
            'gpu_available': self.gpu_available,
            'active_tasks': self.resource_usage['active_tasks'],
            'can_accept_tasks': self.can_accept_task()
        }

# Global resource manager instance
resource_manager = ResourceManager()

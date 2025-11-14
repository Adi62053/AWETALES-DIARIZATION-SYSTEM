from typing import Callable, Dict, List, Optional, Any
import time
import threading
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    audio_duration: float
    processing_time: float
    real_time_factor: float
    memory_usage_mb: float
    cpu_usage_percent: float
    quality_metrics: Dict[str, float]

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_callbacks: List[Callable] = []
        self.lock = threading.Lock()
        
    def record_metrics(self, metrics: PerformanceMetrics):
        with self.lock:
            self.metrics_history.append(metrics)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {"status": "no_data"}
        return {"status": "operational", "processed": len(self.metrics_history)}
    
    def register_alert_callback(self, callback: Callable):
        self.alert_callbacks.append(callback)

performance_monitor = PerformanceMonitor()

"""Real-time monitoring system for neuromorphic edge processors."""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import threading
import json


class RealtimeMonitor:
    """Comprehensive real-time monitoring for neuromorphic systems."""
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize real-time monitor.
        
        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size
        self.logger = self._setup_logger()
        
        # Monitoring data structures
        self.metrics_history = deque(maxlen=max_history_size)
        self.performance_history = deque(maxlen=max_history_size)
        self.security_events = deque(maxlen=max_history_size)
        self.health_status = {"status": "healthy", "last_check": time.time()}
        
        # Thresholds for alerts
        self.thresholds = {
            "spike_rate": {"min": 0.1, "max": 100.0},
            "latency": {"max": 100.0},  # ms
            "memory_usage": {"max": 80.0},  # percentage
            "error_rate": {"max": 0.05},  # 5%
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        self.monitoring_thread = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up monitoring logger."""
        logger = logging.getLogger('neuromorphic_monitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - MONITOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record system metrics with timestamp.
        
        Args:
            metrics: Dictionary of metrics to record
        """
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "metrics": metrics.copy()
        }
        
        self.metrics_history.append(record)
        
        # Check for threshold violations
        self._check_thresholds(metrics)
        
        # Log critical metrics
        if "spike_rate" in metrics:
            self.logger.debug(f"Spike rate: {metrics['spike_rate']:.2f} Hz")
    
    def record_performance(self, operation: str, latency: float, success: bool = True) -> None:
        """Record performance metrics for operations.
        
        Args:
            operation: Name of the operation
            latency: Latency in milliseconds
            success: Whether the operation succeeded
        """
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "operation": operation,
            "latency": latency,
            "success": success
        }
        
        self.performance_history.append(record)
        
        # Alert on high latency
        if latency > self.thresholds["latency"]["max"]:
            self._trigger_alert(f"High latency detected: {latency:.2f}ms for {operation}")
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against defined thresholds."""
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                if "min" in threshold and value < threshold["min"]:
                    self._trigger_alert(f"{metric_name} below threshold: {value:.2f} < {threshold['min']}")
                
                if "max" in threshold and value > threshold["max"]:
                    self._trigger_alert(f"{metric_name} above threshold: {value:.2f} > {threshold['max']}")
    
    def _trigger_alert(self, message: str, severity: str = "warning") -> None:
        """Trigger alert with callbacks.
        
        Args:
            message: Alert message
            severity: Alert severity level
        """
        alert = {
            "timestamp": time.time(),
            "message": message,
            "severity": severity
        }
        
        # Log alert
        if severity == "critical":
            self.logger.critical(message)
        elif severity == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status.
        
        Returns:
            Dictionary containing health metrics
        """
        current_time = time.time()
        
        # Calculate recent error rate
        recent_performance = [
            record for record in self.performance_history
            if current_time - record["timestamp"] < 300  # Last 5 minutes
        ]
        
        if recent_performance:
            error_count = sum(1 for record in recent_performance if not record["success"])
            error_rate = error_count / len(recent_performance)
        else:
            error_rate = 0.0
        
        # Calculate average latency
        if recent_performance:
            avg_latency = np.mean([record["latency"] for record in recent_performance])
        else:
            avg_latency = 0.0
        
        # Determine overall health status
        health_score = 100.0
        
        if error_rate > 0.1:  # More than 10% error rate
            health_score -= 30
        elif error_rate > 0.05:  # More than 5% error rate
            health_score -= 15
        
        if avg_latency > 50:  # More than 50ms average latency
            health_score -= 20
        
        # Determine status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        else:
            status = "unhealthy"
        
        self.health_status = {
            "status": status,
            "health_score": health_score,
            "error_rate": error_rate,
            "avg_latency": avg_latency,
            "last_check": current_time
        }
        
        return self.health_status


# Global monitor instance
realtime_monitor = RealtimeMonitor()
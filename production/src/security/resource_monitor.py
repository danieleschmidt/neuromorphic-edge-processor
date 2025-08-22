"""Resource monitoring for security and performance."""

import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from .security_config import SecurityConfig


@dataclass
class ResourceAlert:
    """Resource usage alert."""
    resource_type: str
    current_value: float
    threshold: float
    timestamp: float
    severity: str  # 'warning', 'critical'
    message: str


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize resource monitor.
        
        Args:
            config: Security configuration with resource limits
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resource usage tracking
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self.alerts = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_callbacks = []
        
        # Thresholds
        self.memory_warning_threshold = config.max_memory_mb * 0.8
        self.memory_critical_threshold = config.max_memory_mb * 0.95
        self.cpu_warning_threshold = config.max_cpu_percent * 0.8
        self.cpu_critical_threshold = config.max_cpu_percent * 0.95
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get current resource usage
                memory_mb = self.get_memory_usage_mb()
                cpu_percent = self.get_cpu_usage_percent()
                
                # Store history
                timestamp = time.time()
                self.memory_usage_history.append((timestamp, memory_mb))
                self.cpu_usage_history.append((timestamp, cpu_percent))
                
                # Trim history (keep last hour)
                cutoff_time = timestamp - 3600
                self.memory_usage_history = [
                    (t, v) for t, v in self.memory_usage_history if t > cutoff_time
                ]
                self.cpu_usage_history = [
                    (t, v) for t, v in self.cpu_usage_history if t > cutoff_time
                ]
                
                # Check thresholds
                self._check_memory_thresholds(memory_mb, timestamp)
                self._check_cpu_thresholds(cpu_percent, timestamp)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except psutil.Error:
            return 0.0
    
    def get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except psutil.Error:
            return 0.0
    
    def get_system_memory_usage(self) -> Dict[str, float]:
        """Get system-wide memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'percent': memory.percent
            }
        except psutil.Error:
            return {'total_mb': 0, 'available_mb': 0, 'used_mb': 0, 'percent': 0}
    
    def check_resource_limits(self) -> Dict[str, bool]:
        """Check if current usage is within limits.
        
        Returns:
            Dictionary with limit check results
        """
        memory_mb = self.get_memory_usage_mb()
        cpu_percent = self.get_cpu_usage_percent()
        
        results = {
            'memory_ok': memory_mb <= self.config.max_memory_mb,
            'cpu_ok': cpu_percent <= self.config.max_cpu_percent,
            'memory_usage_mb': memory_mb,
            'cpu_usage_percent': cpu_percent,
            'memory_limit_mb': self.config.max_memory_mb,
            'cpu_limit_percent': self.config.max_cpu_percent
        }
        
        results['all_ok'] = results['memory_ok'] and results['cpu_ok']
        return results
    
    def _check_memory_thresholds(self, memory_mb: float, timestamp: float):
        """Check memory usage against thresholds."""
        if memory_mb >= self.memory_critical_threshold:
            alert = ResourceAlert(
                resource_type='memory',
                current_value=memory_mb,
                threshold=self.memory_critical_threshold,
                timestamp=timestamp,
                severity='critical',
                message=f'Critical memory usage: {memory_mb:.1f}MB (limit: {self.config.max_memory_mb}MB)'
            )
            self._handle_alert(alert)
            
        elif memory_mb >= self.memory_warning_threshold:
            alert = ResourceAlert(
                resource_type='memory',
                current_value=memory_mb,
                threshold=self.memory_warning_threshold,
                timestamp=timestamp,
                severity='warning',
                message=f'High memory usage: {memory_mb:.1f}MB (limit: {self.config.max_memory_mb}MB)'
            )
            self._handle_alert(alert)
    
    def _check_cpu_thresholds(self, cpu_percent: float, timestamp: float):
        """Check CPU usage against thresholds."""
        if cpu_percent >= self.cpu_critical_threshold:
            alert = ResourceAlert(
                resource_type='cpu',
                current_value=cpu_percent,
                threshold=self.cpu_critical_threshold,
                timestamp=timestamp,
                severity='critical',
                message=f'Critical CPU usage: {cpu_percent:.1f}% (limit: {self.config.max_cpu_percent}%)'
            )
            self._handle_alert(alert)
            
        elif cpu_percent >= self.cpu_warning_threshold:
            alert = ResourceAlert(
                resource_type='cpu',
                current_value=cpu_percent,
                threshold=self.cpu_warning_threshold,
                timestamp=timestamp,
                severity='warning',
                message=f'High CPU usage: {cpu_percent:.1f}% (limit: {self.config.max_cpu_percent}%)'
            )
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: ResourceAlert):
        """Handle resource usage alert."""
        self.alerts.append(alert)
        
        # Trim alerts (keep last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        if alert.severity == 'critical':
            self.logger.critical(alert.message)
        else:
            self.logger.warning(alert.message)
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Register callback for resource alerts.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, minutes: int = 60) -> List[ResourceAlert]:
        """Get alerts from recent time period.
        
        Args:
            minutes: Time period in minutes
            
        Returns:
            List of recent alerts
        """
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def get_usage_statistics(self) -> Dict[str, any]:
        """Get resource usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.memory_usage_history or not self.cpu_usage_history:
            return {
                'memory': {'current_mb': self.get_memory_usage_mb()},
                'cpu': {'current_percent': self.get_cpu_usage_percent()},
                'alerts': len(self.alerts)
            }
        
        # Calculate statistics
        recent_memory = [usage for _, usage in self.memory_usage_history[-60:]]  # Last minute
        recent_cpu = [usage for _, usage in self.cpu_usage_history[-60:]]
        
        memory_stats = {
            'current_mb': recent_memory[-1] if recent_memory else 0,
            'avg_mb': sum(recent_memory) / len(recent_memory) if recent_memory else 0,
            'max_mb': max(recent_memory) if recent_memory else 0,
            'min_mb': min(recent_memory) if recent_memory else 0
        }
        
        cpu_stats = {
            'current_percent': recent_cpu[-1] if recent_cpu else 0,
            'avg_percent': sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0,
            'max_percent': max(recent_cpu) if recent_cpu else 0,
            'min_percent': min(recent_cpu) if recent_cpu else 0
        }
        
        return {
            'memory': memory_stats,
            'cpu': cpu_stats,
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.severity == 'critical']),
                'warning': len([a for a in self.alerts if a.severity == 'warning'])
            },
            'monitoring_active': self.monitoring_active
        }
    
    def force_garbage_collection(self):
        """Force Python garbage collection to free memory."""
        import gc
        gc.collect()
        self.logger.info("Forced garbage collection")
    
    def estimate_operation_cost(self, operation_type: str, **kwargs) -> Dict[str, float]:
        """Estimate resource cost of an operation.
        
        Args:
            operation_type: Type of operation ('inference', 'training', etc.)
            **kwargs: Operation parameters
            
        Returns:
            Estimated resource costs
        """
        estimates = {
            'memory_mb': 0,
            'cpu_seconds': 0,
            'confidence': 0.5  # How confident we are in the estimate
        }
        
        if operation_type == 'inference':
            neurons = kwargs.get('neurons', 1000)
            time_steps = kwargs.get('time_steps', 100)
            
            # Simple heuristic estimates
            estimates['memory_mb'] = neurons * time_steps * 0.001  # 1 KB per neuron-timestep
            estimates['cpu_seconds'] = neurons * time_steps * 0.00001  # 10 μs per neuron-timestep
            estimates['confidence'] = 0.7
            
        elif operation_type == 'training':
            neurons = kwargs.get('neurons', 1000)
            time_steps = kwargs.get('time_steps', 100)
            epochs = kwargs.get('epochs', 10)
            
            # Training is more expensive
            base_memory = neurons * time_steps * 0.002
            estimates['memory_mb'] = base_memory * epochs * 1.5  # Include gradients
            estimates['cpu_seconds'] = neurons * time_steps * epochs * 0.0001
            estimates['confidence'] = 0.6
        
        return estimates
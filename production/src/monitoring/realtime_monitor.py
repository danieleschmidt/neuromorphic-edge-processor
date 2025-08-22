"""Real-time monitoring system for neuromorphic processors."""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque, defaultdict
import psutil
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    temperature: Optional[float] = None
    power_consumption: Optional[float] = None


@dataclass
class NeuromorphicMetrics:
    """Neuromorphic-specific metrics."""
    timestamp: datetime
    total_spikes: int
    firing_rate: float
    network_synchrony: float
    energy_consumption: float
    inference_latency: float
    throughput: float
    accuracy: Optional[float] = None


@dataclass
class Alert:
    """System alert information."""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'error', 'critical'
    category: str  # 'performance', 'security', 'hardware', 'network'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class RealtimeMonitor:
    """Real-time monitoring system for neuromorphic edge processors."""
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_gpu_monitoring: bool = True,
        enable_thermal_monitoring: bool = True
    ):
        """Initialize real-time monitor.
        
        Args:
            monitoring_interval: Monitoring update interval (seconds)
            history_size: Number of historical data points to keep
            alert_thresholds: Dictionary of alert thresholds
            enable_gpu_monitoring: Enable GPU monitoring if available
            enable_thermal_monitoring: Enable temperature monitoring
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_thermal_monitoring = enable_thermal_monitoring
        
        # Set default alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 90.0,          # %
            'memory_usage': 90.0,       # %
            'gpu_memory_usage': 90.0,   # %
            'temperature': 85.0,        # °C
            'inference_latency': 1000.0, # ms
            'firing_rate_low': 0.1,     # Hz
            'firing_rate_high': 1000.0, # Hz
            'energy_per_inference': 1.0, # J
        }
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # Data storage
        self.system_metrics_history = deque(maxlen=history_size)
        self.neuromorphic_metrics_history = deque(maxlen=history_size)
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = defaultdict(list)
        
        # Performance counters
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.total_energy_consumed = 0.0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available() and self.enable_gpu_monitoring
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml = pynvml
            except ImportError:
                self.gpu_available = False
                self.logger.warning("pynvml not available, GPU monitoring disabled")
    
    def start_monitoring(self):
        """Start real-time monitoring in a separate thread."""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Real-time monitoring started")
        self._create_alert("info", "monitoring", "Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("Real-time monitoring stopped")
        self._create_alert("info", "monitoring", "Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                sys_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(sys_metrics)
                
                # Check for system alerts
                self._check_system_alerts(sys_metrics)
                
                # Trigger callbacks
                self._trigger_callbacks("system_metrics", sys_metrics)
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._create_alert("error", "monitoring", f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_memory_usage = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                # GPU memory
                mem_info = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_usage = (mem_info.used / mem_info.total) * 100
                
                # GPU utilization
                util_rates = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_utilization = util_rates.gpu
                
            except Exception as e:
                self.logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # Temperature monitoring
        temperature = None
        if self.enable_thermal_monitoring:
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get first available temperature sensor
                        for sensor_name, sensors in temps.items():
                            if sensors:
                                temperature = sensors[0].current
                                break
            except Exception as e:
                self.logger.warning(f"Failed to collect temperature: {e}")
        
        # Power consumption (placeholder - would need hardware-specific implementation)
        power_consumption = None
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization,
            temperature=temperature,
            power_consumption=power_consumption
        )
    
    def record_inference_metrics(
        self,
        spikes_generated: int,
        firing_rate: float,
        network_synchrony: float,
        energy_consumed: float,
        latency: float,
        accuracy: Optional[float] = None
    ):
        """Record neuromorphic inference metrics.
        
        Args:
            spikes_generated: Total number of spikes generated
            firing_rate: Average firing rate (Hz)
            network_synchrony: Network synchrony measure
            energy_consumed: Energy consumed (J)
            latency: Inference latency (ms)
            accuracy: Model accuracy if available
        """
        # Update counters
        self.inference_count += 1
        self.total_inference_time += latency
        self.total_energy_consumed += energy_consumed
        
        # Calculate throughput
        if self.total_inference_time > 0:
            throughput = self.inference_count * 1000 / self.total_inference_time  # inferences/sec
        else:
            throughput = 0.0
        
        # Create metrics record
        metrics = NeuromorphicMetrics(
            timestamp=datetime.now(),
            total_spikes=spikes_generated,
            firing_rate=firing_rate,
            network_synchrony=network_synchrony,
            energy_consumption=energy_consumed,
            inference_latency=latency,
            throughput=throughput,
            accuracy=accuracy
        )
        
        self.neuromorphic_metrics_history.append(metrics)
        
        # Check for neuromorphic-specific alerts
        self._check_neuromorphic_alerts(metrics)
        
        # Trigger callbacks
        self._trigger_callbacks("neuromorphic_metrics", metrics)
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against alert thresholds."""
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            self._create_alert(
                "warning", "performance",
                f"High CPU usage: {metrics.cpu_usage:.1f}%",
                {"cpu_usage": metrics.cpu_usage, "threshold": self.alert_thresholds['cpu_usage']}
            )
        
        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            self._create_alert(
                "warning", "performance", 
                f"High memory usage: {metrics.memory_usage:.1f}%",
                {"memory_usage": metrics.memory_usage, "threshold": self.alert_thresholds['memory_usage']}
            )
        
        # GPU memory alert
        if (metrics.gpu_memory_usage is not None and 
            metrics.gpu_memory_usage > self.alert_thresholds['gpu_memory_usage']):
            self._create_alert(
                "warning", "performance",
                f"High GPU memory usage: {metrics.gpu_memory_usage:.1f}%",
                {"gpu_memory_usage": metrics.gpu_memory_usage, "threshold": self.alert_thresholds['gpu_memory_usage']}
            )
        
        # Temperature alert
        if (metrics.temperature is not None and 
            metrics.temperature > self.alert_thresholds['temperature']):
            severity = "critical" if metrics.temperature > 90 else "warning"
            self._create_alert(
                severity, "hardware",
                f"High temperature: {metrics.temperature:.1f}°C",
                {"temperature": metrics.temperature, "threshold": self.alert_thresholds['temperature']}
            )
    
    def _check_neuromorphic_alerts(self, metrics: NeuromorphicMetrics):
        """Check neuromorphic metrics against alert thresholds."""
        # Latency alert
        if metrics.inference_latency > self.alert_thresholds['inference_latency']:
            self._create_alert(
                "warning", "performance",
                f"High inference latency: {metrics.inference_latency:.1f}ms",
                {"latency": metrics.inference_latency, "threshold": self.alert_thresholds['inference_latency']}
            )
        
        # Firing rate alerts
        if metrics.firing_rate < self.alert_thresholds['firing_rate_low']:
            self._create_alert(
                "warning", "network",
                f"Low firing rate: {metrics.firing_rate:.1f}Hz",
                {"firing_rate": metrics.firing_rate, "threshold": self.alert_thresholds['firing_rate_low']}
            )
        elif metrics.firing_rate > self.alert_thresholds['firing_rate_high']:
            self._create_alert(
                "warning", "network",
                f"High firing rate: {metrics.firing_rate:.1f}Hz",
                {"firing_rate": metrics.firing_rate, "threshold": self.alert_thresholds['firing_rate_high']}
            )
        
        # Energy consumption alert
        if metrics.energy_consumption > self.alert_thresholds['energy_per_inference']:
            self._create_alert(
                "warning", "performance",
                f"High energy consumption: {metrics.energy_consumption:.3f}J",
                {"energy": metrics.energy_consumption, "threshold": self.alert_thresholds['energy_per_inference']}
            )
    
    def _create_alert(self, severity: str, category: str, message: str, details: Optional[Dict] = None):
        """Create and store an alert."""
        alert = Alert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING, 
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"[{category.upper()}] {message}")
        
        # Trigger alert callbacks
        self._trigger_callbacks("alert", alert)
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for specific events.
        
        Args:
            event_type: Type of event ('system_metrics', 'neuromorphic_metrics', 'alert')
            callback: Callback function to register
        """
        self.callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger registered callbacks for an event type."""
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback for {event_type}: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        status = {
            "monitoring_active": self.is_monitoring,
            "inference_count": self.inference_count,
            "total_runtime_hours": self.total_inference_time / 3600000,  # Convert ms to hours
            "total_energy_consumed_j": self.total_energy_consumed,
            "average_latency_ms": self.total_inference_time / max(1, self.inference_count),
            "recent_alerts": len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)])
        }
        
        # Add latest metrics if available
        if self.system_metrics_history:
            latest_sys = self.system_metrics_history[-1]
            status.update({
                "cpu_usage": latest_sys.cpu_usage,
                "memory_usage": latest_sys.memory_usage,
                "gpu_memory_usage": latest_sys.gpu_memory_usage,
                "temperature": latest_sys.temperature
            })
        
        if self.neuromorphic_metrics_history:
            latest_neuro = self.neuromorphic_metrics_history[-1]
            status.update({
                "current_throughput": latest_neuro.throughput,
                "latest_firing_rate": latest_neuro.firing_rate,
                "network_synchrony": latest_neuro.network_synchrony
            })
        
        return status
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics within time window
        recent_sys_metrics = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
        recent_neuro_metrics = [m for m in self.neuromorphic_metrics_history if m.timestamp > cutoff_time]
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        summary = {
            "time_window_hours": hours,
            "data_points": len(recent_sys_metrics),
            "inference_count": len(recent_neuro_metrics),
            "alert_count": len(recent_alerts)
        }
        
        # System metrics summary
        if recent_sys_metrics:
            cpu_values = [m.cpu_usage for m in recent_sys_metrics]
            memory_values = [m.memory_usage for m in recent_sys_metrics]
            
            summary["system"] = {
                "avg_cpu_usage": np.mean(cpu_values),
                "max_cpu_usage": np.max(cpu_values),
                "avg_memory_usage": np.mean(memory_values),
                "max_memory_usage": np.max(memory_values)
            }
            
            # GPU metrics if available
            gpu_memory_values = [m.gpu_memory_usage for m in recent_sys_metrics if m.gpu_memory_usage is not None]
            if gpu_memory_values:
                summary["system"]["avg_gpu_memory"] = np.mean(gpu_memory_values)
                summary["system"]["max_gpu_memory"] = np.max(gpu_memory_values)
        
        # Neuromorphic metrics summary
        if recent_neuro_metrics:
            latencies = [m.inference_latency for m in recent_neuro_metrics]
            energy_values = [m.energy_consumption for m in recent_neuro_metrics]
            throughputs = [m.throughput for m in recent_neuro_metrics]
            
            summary["neuromorphic"] = {
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "avg_energy_j": np.mean(energy_values),
                "total_energy_j": np.sum(energy_values),
                "avg_throughput": np.mean(throughputs),
                "max_throughput": np.max(throughputs)
            }
        
        # Alert breakdown
        if recent_alerts:
            alert_counts = defaultdict(int)
            for alert in recent_alerts:
                alert_counts[alert.severity] += 1
            
            summary["alerts"] = dict(alert_counts)
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export collected metrics to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        if format == "json":
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_usage": m.cpu_usage,
                        "memory_usage": m.memory_usage,
                        "gpu_memory_usage": m.gpu_memory_usage,
                        "gpu_utilization": m.gpu_utilization,
                        "temperature": m.temperature,
                        "power_consumption": m.power_consumption
                    }
                    for m in self.system_metrics_history
                ],
                "neuromorphic_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "total_spikes": m.total_spikes,
                        "firing_rate": m.firing_rate,
                        "network_synchrony": m.network_synchrony,
                        "energy_consumption": m.energy_consumption,
                        "inference_latency": m.inference_latency,
                        "throughput": m.throughput,
                        "accuracy": m.accuracy
                    }
                    for m in self.neuromorphic_metrics_history
                ],
                "alerts": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "severity": a.severity,
                        "category": a.category,
                        "message": a.message,
                        "details": a.details,
                        "resolved": a.resolved
                    }
                    for a in self.alerts
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == "csv":
            import pandas as pd
            
            # Convert metrics to DataFrames and save
            if self.system_metrics_history:
                sys_df = pd.DataFrame([
                    {
                        "timestamp": m.timestamp,
                        "cpu_usage": m.cpu_usage,
                        "memory_usage": m.memory_usage,
                        "gpu_memory_usage": m.gpu_memory_usage,
                        "temperature": m.temperature
                    }
                    for m in self.system_metrics_history
                ])
                sys_df.to_csv(f"{filepath}_system.csv", index=False)
            
            if self.neuromorphic_metrics_history:
                neuro_df = pd.DataFrame([
                    {
                        "timestamp": m.timestamp,
                        "total_spikes": m.total_spikes,
                        "firing_rate": m.firing_rate,
                        "energy_consumption": m.energy_consumption,
                        "inference_latency": m.inference_latency,
                        "throughput": m.throughput
                    }
                    for m in self.neuromorphic_metrics_history
                ])
                neuro_df.to_csv(f"{filepath}_neuromorphic.csv", index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {filepath} in {format} format")
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        self.stop_monitoring()
        
        # Clear data structures
        self.system_metrics_history.clear()
        self.neuromorphic_metrics_history.clear()
        self.alerts.clear()
        self.callbacks.clear()
        
        self.logger.info("Monitor cleanup completed")
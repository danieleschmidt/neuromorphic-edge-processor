"""Advanced monitoring system for neuromorphic processing."""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import psutil
import logging
from datetime import datetime, timedelta


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    power_watts: float = 0.0


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: float
    inference_time_ms: float
    throughput_samples_sec: float
    accuracy: Optional[float] = None
    energy_consumption_j: float = 0.0
    sparsity: float = 0.0
    spike_rate: float = 0.0
    memory_usage_mb: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class AdvancedMonitor:
    """Advanced monitoring system for neuromorphic edge processing.
    
    Provides real-time monitoring of system resources, model performance,
    neuromorphic-specific metrics, and automated alerting/optimization.
    """
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 10000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_gpu_monitoring: bool = True,
        enable_power_monitoring: bool = False,
        log_level: str = "INFO"
    ):
        """Initialize advanced monitor.
        
        Args:
            monitoring_interval: Time between monitoring samples (seconds)
            history_size: Number of historical samples to keep
            alert_thresholds: Threshold values for alerts
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_power_monitoring: Whether to monitor power consumption
            log_level: Logging level
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_power_monitoring = enable_power_monitoring
        
        # Set default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "gpu_memory_mb": 8000.0,
            "temperature": 85.0,
            "inference_time_ms": 100.0,
            "spike_rate": 1000.0,
        }
        
        # Initialize data storage
        self.system_history: deque = deque(maxlen=history_size)
        self.model_history: deque = deque(maxlen=history_size)
        self.alert_history: deque = deque(maxlen=1000)
        
        # Threading for continuous monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Event callbacks
        self.alert_callbacks: List[Callable] = []
        self.metric_callbacks: List[Callable] = []
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Performance tracking
        self.start_time = time.time()
        self.total_inferences = 0
        self.total_errors = 0
        
        # GPU monitoring setup
        self.gpu_available = False
        if self.enable_gpu_monitoring:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except:
                self.logger.warning("GPU monitoring unavailable")
        
        # Power monitoring setup (Linux-specific)
        self.power_available = False
        if self.enable_power_monitoring:
            try:
                import py3nvml.py3nvml as nvml
                nvml.nvmlInit()
                self.power_available = True
            except:
                self.logger.warning("Power monitoring unavailable")
    
    def _setup_logger(self, level: str) -> logging.Logger:
        """Setup monitoring logger."""
        logger = logging.getLogger(f'neuromorphic_monitor_{id(self)}')
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Advanced monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Advanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.system_history.append(system_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics)
                
                # Call metric callbacks
                for callback in self.metric_callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        self.logger.error(f"Metric callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Brief pause before retry
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_memory_mb = 0.0
            gpu_utilization = 0.0
            
            if self.gpu_available:
                try:
                    import pynvml
                    
                    # Memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_mb = mem_info.used / 1024**2
                    
                    # Utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_utilization = utilization.gpu
                    
                except Exception as e:
                    self.logger.debug(f"GPU monitoring error: {e}")
            
            # Temperature monitoring
            temperature = 0.0
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps and temps['coretemp']:
                        temperature = max(temp.current for temp in temps['coretemp'])
            except:
                pass
            
            # Power monitoring
            power_watts = 0.0
            if self.power_available:
                try:
                    import py3nvml.py3nvml as nvml
                    power_mw = nvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_watts = power_mw / 1000.0
                except:
                    pass
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory.used / 1024**2,
                memory_percent=memory.percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization,
                temperature=temperature,
                power_watts=power_watts
            )
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0
            )
    
    def record_inference(
        self,
        inference_time_ms: float,
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        accuracy: Optional[float] = None,
        energy_consumption_j: float = 0.0,
        spike_data: Optional[torch.Tensor] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Record inference performance metrics.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
            accuracy: Model accuracy (if available)
            energy_consumption_j: Energy consumption in joules
            spike_data: Spike train data for neuromorphic metrics
            additional_metrics: Additional custom metrics
        """
        try:
            self.total_inferences += 1
            
            # Calculate throughput
            batch_size = input_shape[0] if input_shape else 1
            throughput = (batch_size * 1000.0) / inference_time_ms  # samples/sec
            
            # Calculate neuromorphic metrics
            sparsity = 0.0
            spike_rate = 0.0
            
            if spike_data is not None:
                sparsity = (spike_data == 0).float().mean().item()
                if spike_data.numel() > 0:
                    total_spikes = spike_data.sum().item()
                    duration_sec = spike_data.shape[-1] * 0.001  # Assume 1ms time steps
                    spike_rate = total_spikes / duration_sec if duration_sec > 0 else 0.0
            
            # Estimate memory usage
            memory_usage_mb = 0.0
            if torch.cuda.is_available():
                memory_usage_mb = torch.cuda.memory_allocated() / 1024**2
            
            # Create model metrics
            model_metrics = ModelMetrics(
                timestamp=time.time(),
                inference_time_ms=inference_time_ms,
                throughput_samples_sec=throughput,
                accuracy=accuracy,
                energy_consumption_j=energy_consumption_j,
                sparsity=sparsity,
                spike_rate=spike_rate,
                memory_usage_mb=memory_usage_mb,
                additional_metrics=additional_metrics or {}
            )
            
            with self.lock:
                self.model_history.append(model_metrics)
            
            # Check for performance alerts
            self._check_performance_alerts(model_metrics)
            
            # Call metric callbacks
            for callback in self.metric_callbacks:
                try:
                    callback(model_metrics)
                except Exception as e:
                    self.logger.error(f"Model metric callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Inference recording error: {e}")
            self.total_errors += 1
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check system metrics against alert thresholds."""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.alert_thresholds.get("cpu_percent", 90):
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory usage alert
        if metrics.memory_percent > self.alert_thresholds.get("memory_percent", 85):
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # GPU memory alert
        if metrics.gpu_memory_mb > self.alert_thresholds.get("gpu_memory_mb", 8000):
            alerts.append(f"High GPU memory: {metrics.gpu_memory_mb:.1f}MB")
        
        # Temperature alert
        if metrics.temperature > self.alert_thresholds.get("temperature", 85):
            alerts.append(f"High temperature: {metrics.temperature:.1f}°C")
        
        # Trigger alerts
        for alert_msg in alerts:
            self._trigger_alert("system", alert_msg, metrics.timestamp)
    
    def _check_performance_alerts(self, metrics: ModelMetrics):
        """Check model performance metrics against thresholds."""
        alerts = []
        
        # Inference time alert
        if metrics.inference_time_ms > self.alert_thresholds.get("inference_time_ms", 100):
            alerts.append(f"Slow inference: {metrics.inference_time_ms:.1f}ms")
        
        # Spike rate alert
        if metrics.spike_rate > self.alert_thresholds.get("spike_rate", 1000):
            alerts.append(f"High spike rate: {metrics.spike_rate:.1f} Hz")
        
        # Low sparsity alert (might indicate issues)
        if metrics.sparsity < 0.1:  # Less than 10% sparsity
            alerts.append(f"Low sparsity: {metrics.sparsity:.2f}")
        
        # Trigger alerts
        for alert_msg in alerts:
            self._trigger_alert("performance", alert_msg, metrics.timestamp)
    
    def _trigger_alert(self, alert_type: str, message: str, timestamp: float):
        """Trigger an alert and notify callbacks."""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat()
        }
        
        with self.lock:
            self.alert_history.append(alert)
        
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def get_system_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get system performance summary over time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            System summary statistics
        """
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self.lock:
            recent_metrics = [
                m for m in self.system_history 
                if m.timestamp > cutoff_time
            ]
        
        if not recent_metrics:
            return {"error": "No recent system metrics available"}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_memory_values = [m.gpu_memory_mb for m in recent_metrics if m.gpu_memory_mb > 0]
        temp_values = [m.temperature for m in recent_metrics if m.temperature > 0]
        
        return {
            "window_minutes": window_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "std": np.std(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "mean_percent": np.mean(memory_values),
                "max_percent": np.max(memory_values),
                "current_percent": memory_values[-1] if memory_values else 0,
                "current_mb": recent_metrics[-1].memory_mb if recent_metrics else 0
            },
            "gpu": {
                "mean_memory_mb": np.mean(gpu_memory_values) if gpu_memory_values else 0,
                "max_memory_mb": np.max(gpu_memory_values) if gpu_memory_values else 0,
                "current_memory_mb": recent_metrics[-1].gpu_memory_mb if recent_metrics else 0
            },
            "temperature": {
                "mean": np.mean(temp_values) if temp_values else 0,
                "max": np.max(temp_values) if temp_values else 0,
                "current": temp_values[-1] if temp_values else 0
            }
        }
    
    def get_performance_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get model performance summary over time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self.lock:
            recent_metrics = [
                m for m in self.model_history 
                if m.timestamp > cutoff_time
            ]
        
        if not recent_metrics:
            return {"error": "No recent performance metrics available"}
        
        inference_times = [m.inference_time_ms for m in recent_metrics]
        throughputs = [m.throughput_samples_sec for m in recent_metrics]
        sparsities = [m.sparsity for m in recent_metrics if m.sparsity > 0]
        spike_rates = [m.spike_rate for m in recent_metrics if m.spike_rate > 0]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        
        return {
            "window_minutes": window_minutes,
            "inference_count": len(recent_metrics),
            "inference_time_ms": {
                "mean": np.mean(inference_times),
                "median": np.median(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
                "min": np.min(inference_times),
                "max": np.max(inference_times)
            },
            "throughput_samples_sec": {
                "mean": np.mean(throughputs),
                "max": np.max(throughputs),
                "current": throughputs[-1] if throughputs else 0
            },
            "neuromorphic": {
                "mean_sparsity": np.mean(sparsities) if sparsities else 0,
                "mean_spike_rate": np.mean(spike_rates) if spike_rates else 0,
                "max_spike_rate": np.max(spike_rates) if spike_rates else 0
            },
            "accuracy": {
                "mean": np.mean(accuracies) if accuracies else None,
                "latest": accuracies[-1] if accuracies else None
            }
        }
    
    def get_alert_summary(self, window_hours: int = 1) -> Dict[str, Any]:
        """Get alert summary over time window."""
        cutoff_time = time.time() - (window_hours * 3600)
        
        with self.lock:
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert["timestamp"] > cutoff_time
            ]
        
        # Group by type
        alert_counts = {}
        for alert in recent_alerts:
            alert_type = alert["type"]
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        return {
            "window_hours": window_hours,
            "total_alerts": len(recent_alerts),
            "alert_counts_by_type": alert_counts,
            "recent_alerts": recent_alerts[-10:] if recent_alerts else []
        }
    
    def export_metrics(self, filename: str, format: str = "json"):
        """Export collected metrics to file.
        
        Args:
            filename: Output filename
            format: Export format ("json", "csv")
        """
        try:
            with self.lock:
                system_data = [
                    {
                        "timestamp": m.timestamp,
                        "cpu_percent": m.cpu_percent,
                        "memory_mb": m.memory_mb,
                        "memory_percent": m.memory_percent,
                        "gpu_memory_mb": m.gpu_memory_mb,
                        "gpu_utilization": m.gpu_utilization,
                        "temperature": m.temperature,
                        "power_watts": m.power_watts
                    }
                    for m in self.system_history
                ]
                
                model_data = [
                    {
                        "timestamp": m.timestamp,
                        "inference_time_ms": m.inference_time_ms,
                        "throughput_samples_sec": m.throughput_samples_sec,
                        "accuracy": m.accuracy,
                        "energy_consumption_j": m.energy_consumption_j,
                        "sparsity": m.sparsity,
                        "spike_rate": m.spike_rate,
                        "memory_usage_mb": m.memory_usage_mb,
                        **m.additional_metrics
                    }
                    for m in self.model_history
                ]
            
            export_data = {
                "export_timestamp": time.time(),
                "system_metrics": system_data,
                "model_metrics": model_data,
                "alerts": list(self.alert_history),
                "summary": {
                    "total_inferences": self.total_inferences,
                    "total_errors": self.total_errors,
                    "uptime_seconds": time.time() - self.start_time
                }
            }
            
            if format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == "csv":
                import pandas as pd
                
                # Export system metrics
                if system_data:
                    pd.DataFrame(system_data).to_csv(
                        filename.replace('.csv', '_system.csv'), 
                        index=False
                    )
                
                # Export model metrics
                if model_data:
                    pd.DataFrame(model_data).to_csv(
                        filename.replace('.csv', '_model.csv'), 
                        index=False
                    )
            
            self.logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable):
        """Add callback function for metrics."""
        self.metric_callbacks.append(callback)
    
    def reset_statistics(self):
        """Reset all collected statistics."""
        with self.lock:
            self.system_history.clear()
            self.model_history.clear()
            self.alert_history.clear()
            
        self.total_inferences = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        self.logger.info("Monitoring statistics reset")
    
    def get_realtime_status(self) -> Dict[str, Any]:
        """Get current real-time system and model status."""
        current_system = self._collect_system_metrics()
        
        with self.lock:
            latest_model = self.model_history[-1] if self.model_history else None
        
        return {
            "monitoring_active": self.monitoring_active,
            "uptime_seconds": time.time() - self.start_time,
            "total_inferences": self.total_inferences,
            "total_errors": self.total_errors,
            "current_system": {
                "cpu_percent": current_system.cpu_percent,
                "memory_percent": current_system.memory_percent,
                "gpu_memory_mb": current_system.gpu_memory_mb,
                "temperature": current_system.temperature
            },
            "latest_inference": {
                "inference_time_ms": latest_model.inference_time_ms if latest_model else None,
                "throughput": latest_model.throughput_samples_sec if latest_model else None,
                "sparsity": latest_model.sparsity if latest_model else None
            } if latest_model else None
        }
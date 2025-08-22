"""Edge-specific monitoring system for neuromorphic processors.

Provides real-time monitoring, telemetry collection, and edge-optimized
alerting for neuromorphic computing operations in production environments.
"""

import time
import threading
import queue
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'type': self.metric_type.value
        }


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    metric_value: float
    threshold: float
    tags: Dict[str, str]
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: str
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float]
    active_alerts: int
    error_rate: float
    throughput: float
    latency_p95: float
    uptime: float


class MetricsCollector:
    """Efficient metrics collection system for edge deployment."""
    
    def __init__(self, 
                 buffer_size: int = 10000,
                 flush_interval: float = 10.0,
                 enable_compression: bool = True):
        """Initialize metrics collector.
        
        Args:
            buffer_size: Maximum metrics to buffer before flush
            flush_interval: Interval to flush metrics (seconds)
            enable_compression: Enable metric value compression
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.enable_compression = enable_compression
        
        # Metrics storage
        self.metrics_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Aggregated metrics for edge efficiency
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Timing
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background flushing
        self.flush_thread = None
        self.running = False
        
    def start(self):
        """Start background metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()
    
    def stop(self):
        """Stop background metrics collection."""
        self.running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self.lock:
            self.counters[name] += value
        
        self._add_metric(name, value, MetricType.COUNTER, tags or {})
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self.lock:
            self.gauges[name] = value
        
        self._add_metric(name, value, MetricType.GAUGE, tags or {})
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only recent values for edge efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
        
        self._add_metric(name, value, MetricType.HISTOGRAM, tags or {})
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        with self.lock:
            self.timers[name].append(duration)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
        
        self._add_metric(name, duration, MetricType.TIMER, tags or {})
    
    def _add_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str]):
        """Add metric to buffer."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags,
                metric_type=metric_type
            )
            
            # Add to buffer (non-blocking)
            if not self.metrics_buffer.full():
                self.metrics_buffer.put_nowait(metric)
            
            # Add to history
            with self.lock:
                self.metrics_history[name].append((metric.timestamp, value))
                
        except queue.Full:
            # Drop oldest metrics if buffer is full (edge behavior)
            pass
    
    def _background_flush(self):
        """Background thread for flushing metrics."""
        while self.running:
            try:
                self._flush_metrics()
                time.sleep(self.flush_interval)
            except Exception as e:
                logging.error(f"Error in metrics flush: {e}")
    
    def _flush_metrics(self):
        """Flush buffered metrics."""
        flushed_metrics = []
        
        # Drain buffer
        while not self.metrics_buffer.empty():
            try:
                metric = self.metrics_buffer.get_nowait()
                flushed_metrics.append(metric)
            except queue.Empty:
                break
        
        # In a production system, this would send to monitoring backend
        # For edge deployment, we just log summary statistics
        if flushed_metrics:
            self._log_metrics_summary(flushed_metrics)
    
    def _log_metrics_summary(self, metrics: List[Metric]):
        """Log summary of flushed metrics."""
        by_type = defaultdict(int)
        for metric in metrics:
            by_type[metric.metric_type.value] += 1
        
        logging.info(f"Flushed {len(metrics)} metrics: {dict(by_type)}")
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics for monitoring dashboard."""
        with self.lock:
            # Counter totals
            counter_data = dict(self.counters)
            
            # Current gauge values
            gauge_data = dict(self.gauges)
            
            # Histogram statistics
            histogram_stats = {}
            for name, values in self.histograms.items():
                if values:
                    histogram_stats[name] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'p50': np.percentile(values, 50),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            # Timer statistics
            timer_stats = {}
            for name, values in self.timers.items():
                if values:
                    timer_stats[name] = {
                        'count': len(values),
                        'mean_ms': np.mean(values) * 1000,
                        'p95_ms': np.percentile(values, 95) * 1000,
                        'p99_ms': np.percentile(values, 99) * 1000,
                        'min_ms': np.min(values) * 1000,
                        'max_ms': np.max(values) * 1000
                    }
            
            return {
                'counters': counter_data,
                'gauges': gauge_data,
                'histograms': histogram_stats,
                'timers': timer_stats,
                'buffer_size': self.metrics_buffer.qsize(),
                'buffer_capacity': self.buffer_size
            }


class EdgeMonitor:
    """Comprehensive edge monitoring system for neuromorphic processors."""
    
    def __init__(self,
                 collection_interval: float = 1.0,
                 alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
                 enable_system_metrics: bool = True,
                 enable_gpu_metrics: bool = True):
        """Initialize edge monitor.
        
        Args:
            collection_interval: System metrics collection interval (seconds)
            alert_thresholds: Custom alert thresholds
            enable_system_metrics: Enable system resource monitoring
            enable_gpu_metrics: Enable GPU metrics (if available)
        """
        self.collection_interval = collection_interval
        self.enable_system_metrics = enable_system_metrics
        self.enable_gpu_metrics = enable_gpu_metrics and TORCH_AVAILABLE
        
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # Alerting
        self.alert_thresholds = alert_thresholds or self._default_alert_thresholds()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # System monitoring
        self.system_monitor_thread = None
        self.neuromorphic_monitor_thread = None
        self.running = False
        
        # Health tracking
        self.start_time = time.time()
        self.last_health_check = time.time()
        
        # Performance tracking
        self.operation_timers: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _default_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds for edge deployment."""
        return {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'gpu_memory_usage': {'warning': 90.0, 'critical': 98.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'latency_p95': {'warning': 100.0, 'critical': 500.0},  # milliseconds
            'throughput': {'warning': 10.0, 'critical': 5.0},  # operations/second
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'network_errors': {'warning': 10, 'critical': 50}
        }
    
    def start(self):
        """Start monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Start metrics collection
        self.metrics.start()
        
        # Start system monitoring
        if self.enable_system_metrics:
            self.system_monitor_thread = threading.Thread(
                target=self._system_monitoring_loop, daemon=True
            )
            self.system_monitor_thread.start()
        
        # Start neuromorphic-specific monitoring
        self.neuromorphic_monitor_thread = threading.Thread(
            target=self._neuromorphic_monitoring_loop, daemon=True
        )
        self.neuromorphic_monitor_thread.start()
        
        self.logger.info("Edge monitoring system started")
    
    def stop(self):
        """Stop monitoring system."""
        self.running = False
        
        # Stop threads
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5.0)
        
        if self.neuromorphic_monitor_thread:
            self.neuromorphic_monitor_thread.join(timeout=5.0)
        
        # Stop metrics collection
        self.metrics.stop()
        
        self.logger.info("Edge monitoring system stopped")
    
    def _system_monitoring_loop(self):
        """Background system resource monitoring."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.collection_interval)
    
    def _neuromorphic_monitoring_loop(self):
        """Background neuromorphic-specific monitoring."""
        while self.running:
            try:
                self._collect_neuromorphic_metrics()
                time.sleep(self.collection_interval * 2)  # Less frequent
            except Exception as e:
                self.logger.error(f"Error in neuromorphic monitoring: {e}")
                time.sleep(self.collection_interval * 2)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.record_gauge('cpu_usage', cpu_percent)
            self._check_alert('cpu_usage', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics.record_gauge('memory_usage', memory_percent)
            self.metrics.record_gauge('memory_available_mb', memory.available / 1024**2)
            self._check_alert('memory_usage', memory_percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.metrics.record_gauge('disk_usage', disk_percent)
            self._check_alert('disk_usage', disk_percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics.record_counter('network_bytes_sent', network.bytes_sent)
            self.metrics.record_counter('network_bytes_recv', network.bytes_recv)
            self.metrics.record_counter('network_errors', network.errin + network.errout)
            
            # GPU metrics
            if self.enable_gpu_metrics and torch.cuda.is_available():
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    
                    gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                    
                    self.metrics.record_gauge('gpu_memory_allocated_gb', gpu_memory_allocated)
                    self.metrics.record_gauge('gpu_memory_reserved_gb', gpu_memory_reserved)
                    self.metrics.record_gauge('gpu_memory_usage', gpu_memory_percent)
                    
                    self._check_alert('gpu_memory_usage', gpu_memory_percent)
                    
                except Exception as e:
                    self.logger.debug(f"GPU metrics collection failed: {e}")
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    def _collect_neuromorphic_metrics(self):
        """Collect neuromorphic-specific metrics."""
        try:
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            total_operations = sum(len(timers) for timers in self.operation_timers.values())
            error_rate = total_errors / max(1, total_operations)
            
            self.metrics.record_gauge('error_rate', error_rate)
            self._check_alert('error_rate', error_rate)
            
            # Calculate throughput (operations per second)
            recent_operations = 0
            current_time = time.time()
            
            for operation_times in self.operation_timers.values():
                recent_operations += len([
                    t for t in operation_times[-100:] 
                    if current_time - t < 60  # Last minute
                ])
            
            throughput = recent_operations / 60.0  # ops/second
            self.metrics.record_gauge('throughput', throughput)
            
            # Check throughput alert (reverse logic - lower is worse)
            if 'throughput' in self.alert_thresholds:
                thresholds = self.alert_thresholds['throughput']
                if throughput < thresholds.get('critical', 0):
                    self._create_alert('throughput', AlertSeverity.CRITICAL, throughput, thresholds['critical'])
                elif throughput < thresholds.get('warning', 0):
                    self._create_alert('throughput', AlertSeverity.WARNING, throughput, thresholds['warning'])
            
            # Calculate latency percentiles
            all_latencies = []
            for operation_times in self.operation_timers.values():
                all_latencies.extend(operation_times[-1000:])  # Recent latencies
            
            if all_latencies:
                p95_latency = np.percentile(all_latencies, 95) * 1000  # Convert to ms
                self.metrics.record_gauge('latency_p95', p95_latency)
                self._check_alert('latency_p95', p95_latency)
                
                mean_latency = np.mean(all_latencies) * 1000
                self.metrics.record_gauge('latency_mean', mean_latency)
            
        except Exception as e:
            self.logger.error(f"Neuromorphic metrics collection failed: {e}")
    
    def _check_alert(self, metric_name: str, value: float):
        """Check if metric value triggers an alert."""
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        alert_key = f"{metric_name}_alert"
        
        # Check for critical threshold
        if 'critical' in thresholds and value >= thresholds['critical']:
            if alert_key not in self.active_alerts:
                self._create_alert(metric_name, AlertSeverity.CRITICAL, value, thresholds['critical'])
        
        # Check for warning threshold
        elif 'warning' in thresholds and value >= thresholds['warning']:
            if alert_key not in self.active_alerts:
                self._create_alert(metric_name, AlertSeverity.WARNING, value, thresholds['warning'])
        
        # Resolve alert if value is back to normal
        else:
            if alert_key in self.active_alerts:
                self._resolve_alert(alert_key)
    
    def _create_alert(self, metric_name: str, severity: AlertSeverity, value: float, threshold: float):
        """Create a new alert."""
        alert_id = f"{metric_name}_{severity.value}_{int(time.time())}"
        alert_key = f"{metric_name}_alert"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            message=f"{metric_name} {severity.value}: {value:.2f} >= {threshold}",
            timestamp=time.time(),
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold,
            tags={'metric': metric_name, 'severity': severity.value}
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Keep alert history manageable
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-5000:]
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        self.logger.warning(f"ALERT: {alert.message}")
    
    def _resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_timestamp = time.time()
            
            del self.active_alerts[alert_key]
            
            self.logger.info(f"RESOLVED: {alert.message}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def record_operation(self, operation_name: str, duration: float):
        """Record operation performance metrics."""
        self.operation_timers[operation_name].append(duration)
        self.metrics.record_timer(f'operation_{operation_name}', duration)
        
        # Keep operation history manageable
        if len(self.operation_timers[operation_name]) > 10000:
            self.operation_timers[operation_name] = self.operation_timers[operation_name][-5000:]
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        self.error_counts[error_type] += 1
        self.metrics.record_counter(f'error_{error_type}')
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        try:
            # Get current metrics
            metrics = self.metrics.get_aggregated_metrics()
            gauges = metrics.get('gauges', {})
            
            # Determine overall status
            if len(self.active_alerts) == 0:
                status = "healthy"
            elif any(alert.severity == AlertSeverity.CRITICAL for alert in self.active_alerts.values()):
                status = "critical"
            elif any(alert.severity == AlertSeverity.ERROR for alert in self.active_alerts.values()):
                status = "error"
            elif any(alert.severity == AlertSeverity.WARNING for alert in self.active_alerts.values()):
                status = "warning"
            else:
                status = "unknown"
            
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            total_operations = sum(len(timers) for timers in self.operation_timers.values())
            error_rate = total_errors / max(1, total_operations)
            
            # Calculate throughput
            current_time = time.time()
            recent_operations = 0
            for operation_times in self.operation_timers.values():
                recent_operations += len([
                    t for t in operation_times[-100:] 
                    if current_time - t < 60
                ])
            throughput = recent_operations / 60.0
            
            # Calculate P95 latency
            all_latencies = []
            for operation_times in self.operation_timers.values():
                all_latencies.extend(operation_times[-1000:])
            
            latency_p95 = np.percentile(all_latencies, 95) * 1000 if all_latencies else 0.0
            
            return SystemHealth(
                overall_status=status,
                cpu_usage=gauges.get('cpu_usage', 0.0),
                memory_usage=gauges.get('memory_usage', 0.0),
                gpu_memory_usage=gauges.get('gpu_memory_usage'),
                active_alerts=len(self.active_alerts),
                error_rate=error_rate,
                throughput=throughput,
                latency_p95=latency_p95,
                uptime=time.time() - self.start_time
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return SystemHealth(
                overall_status="unknown",
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_memory_usage=None,
                active_alerts=0,
                error_rate=0.0,
                throughput=0.0,
                latency_p95=0.0,
                uptime=0.0
            )
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        health = self.get_system_health()
        metrics = self.metrics.get_aggregated_metrics()
        
        return {
            'health': asdict(health),
            'metrics': metrics,
            'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'recent_alerts': [asdict(alert) for alert in self.alert_history[-10:]],
            'error_counts': dict(self.error_counts),
            'operation_stats': {
                name: {
                    'count': len(times),
                    'avg_duration_ms': np.mean(times) * 1000 if times else 0,
                    'p95_duration_ms': np.percentile(times, 95) * 1000 if times else 0
                }
                for name, times in self.operation_timers.items()
            }
        }


# Global monitor instance
_global_monitor: Optional[EdgeMonitor] = None


def get_edge_monitor() -> EdgeMonitor:
    """Get or create global edge monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EdgeMonitor()
    return _global_monitor


def monitor_operation(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_edge_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration)
                monitor.record_error(type(e).__name__)
                raise
        
        return wrapper
    return decorator
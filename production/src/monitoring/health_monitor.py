"""Health monitoring system for neuromorphic edge processors."""

import time
import threading
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import torch

from ..utils.logging import get_logger


@dataclass
class HealthStatus:
    """Health status information."""
    timestamp: str
    overall_status: str  # "healthy", "degraded", "critical", "unknown"
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    active_processes: int = 0
    uptime_seconds: float = 0
    error_count: int = 0
    warning_count: int = 0
    details: Dict[str, Any] = None


@dataclass
class ComponentHealth:
    """Health status for individual components."""
    component_name: str
    status: str  # "healthy", "degraded", "critical", "offline"
    last_check: str
    response_time_ms: float
    error_count: int
    metrics: Dict[str, float]
    details: str = ""


class HealthMonitor:
    """Comprehensive health monitoring for neuromorphic systems."""
    
    def __init__(
        self,
        check_interval: int = 30,
        history_retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        auto_start: bool = True
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            history_retention_hours: Hours to retain health history
            alert_thresholds: Threshold values for alerts
            auto_start: Whether to start monitoring automatically
        """
        self.check_interval = check_interval
        self.history_retention = timedelta(hours=history_retention_hours)
        self.logger = get_logger("monitoring.health")
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'gpu_usage_percent': 80.0,
            'error_rate_per_hour': 10.0,
            'response_time_ms': 1000.0
        }
        
        # Health history and state
        self.health_history: List[HealthStatus] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.start_time = time.time()
        self.error_counts = {}
        self.warning_counts = {}
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Health check functions
        self._health_checks: Dict[str, Callable] = {}
        
        # Register built-in health checks
        self._register_default_checks()
        
        if auto_start:
            self.start_monitoring()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_health", self._check_disk_health)
        
        # GPU checks if available
        if torch.cuda.is_available():
            self.register_health_check("gpu_health", self._check_gpu_health)
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check function.
        
        Args:
            name: Name of the health check
            check_function: Function that returns ComponentHealth
        """
        self._health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Health monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="HealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Perform health check
                health_status = self.perform_health_check()
                
                # Store in history
                with self._lock:
                    self.health_history.append(health_status)
                    self._cleanup_old_history()
                
                # Check for alerts
                self._check_alerts(health_status)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                self._record_error("health_monitor", str(e))
            
            # Wait for next check
            self._stop_monitoring.wait(self.check_interval)
    
    def perform_health_check(self) -> HealthStatus:
        """Perform comprehensive health check."""
        check_start = time.time()
        
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            gpu_usage = None
            gpu_memory = None
            
            # GPU monitoring if available
            if torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_usage()
                except:
                    pass
            
            # Run component health checks
            component_results = {}
            for name, check_func in self._health_checks.items():
                try:
                    component_health = check_func()
                    self.component_health[name] = component_health
                    component_results[name] = component_health.status
                except Exception as e:
                    self.logger.error(f"Health check {name} failed: {e}")
                    self._record_error(name, str(e))
                    component_results[name] = "critical"
            
            # Determine overall health status
            overall_status = self._determine_overall_status(
                cpu_percent, memory.percent, disk.percent,
                gpu_usage, component_results
            )
            
            # Get current error/warning counts
            current_errors = sum(self.error_counts.values())
            current_warnings = sum(self.warning_counts.values())
            
            health_status = HealthStatus(
                timestamp=datetime.now().isoformat(),
                overall_status=overall_status,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory,
                active_processes=len(psutil.pids()),
                uptime_seconds=time.time() - self.start_time,
                error_count=current_errors,
                warning_count=current_warnings,
                details={
                    'check_duration_ms': (time.time() - check_start) * 1000,
                    'component_health': component_results,
                    'memory_available_gb': memory.available / 1024**3,
                    'disk_free_gb': disk.free / 1024**3
                }
            )
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to perform health check: {e}")
            return HealthStatus(
                timestamp=datetime.now().isoformat(),
                overall_status="unknown",
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                error_count=1,
                warning_count=0,
                details={'error': str(e)}
            )
    
    def _check_system_resources(self) -> ComponentHealth:
        """Check system resource health."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Determine status based on usage
            if cpu_percent > 90 or memory_percent > 95:
                status = "critical"
            elif cpu_percent > 70 or memory_percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return ComponentHealth(
                component_name="system_resources",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=0,  # Instant check
                error_count=0,
                metrics={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="system_resources",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_count=1,
                metrics={},
                details=f"Error checking system resources: {e}"
            )
    
    def _check_disk_health(self) -> ComponentHealth:
        """Check disk health and space."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            
            # Check disk I/O
            disk_io = psutil.disk_io_counters()
            
            if disk_percent > 95:
                status = "critical"
            elif disk_percent > 85:
                status = "degraded"
            else:
                status = "healthy"
            
            return ComponentHealth(
                component_name="disk_health",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_count=0,
                metrics={
                    'disk_usage_percent': disk_percent,
                    'free_space_gb': disk_usage.free / 1024**3,
                    'total_space_gb': disk_usage.total / 1024**3,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="disk_health",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_count=1,
                metrics={},
                details=f"Error checking disk health: {e}"
            )
    
    def _check_gpu_health(self) -> ComponentHealth:
        """Check GPU health if available."""
        try:
            if not torch.cuda.is_available():
                return ComponentHealth(
                    component_name="gpu_health",
                    status="offline",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=0,
                    error_count=0,
                    metrics={},
                    details="CUDA not available"
                )
            
            gpu_count = torch.cuda.device_count()
            gpu_metrics = {}
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                gpu_metrics[f'gpu_{i}_memory_allocated_gb'] = memory_allocated / 1024**3
                gpu_metrics[f'gpu_{i}_memory_reserved_gb'] = memory_reserved / 1024**3
                gpu_metrics[f'gpu_{i}_memory_total_gb'] = props.total_memory / 1024**3
                gpu_metrics[f'gpu_{i}_utilization'] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            
            # Determine status
            max_memory_usage = max([
                gpu_metrics.get(f'gpu_{i}_memory_reserved_gb', 0) / gpu_metrics.get(f'gpu_{i}_memory_total_gb', 1)
                for i in range(gpu_count)
            ]) * 100
            
            if max_memory_usage > 95:
                status = "critical"
            elif max_memory_usage > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return ComponentHealth(
                component_name="gpu_health",
                status=status,
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_count=0,
                metrics=gpu_metrics
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="gpu_health",
                status="critical",
                last_check=datetime.now().isoformat(),
                response_time_ms=0,
                error_count=1,
                metrics={},
                details=f"Error checking GPU health: {e}"
            )
    
    def _determine_overall_status(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: float,
        gpu_usage: Optional[float],
        component_results: Dict[str, str]
    ) -> str:
        """Determine overall system health status."""
        
        # Check for critical conditions
        critical_conditions = [
            cpu_percent > self.alert_thresholds['cpu_usage_percent'],
            memory_percent > self.alert_thresholds['memory_usage_percent'],
            disk_percent > self.alert_thresholds['disk_usage_percent'],
            any(status == "critical" for status in component_results.values())
        ]
        
        if gpu_usage is not None:
            critical_conditions.append(gpu_usage > self.alert_thresholds.get('gpu_usage_percent', 80))
        
        if any(critical_conditions):
            return "critical"
        
        # Check for degraded conditions
        degraded_conditions = [
            cpu_percent > self.alert_thresholds['cpu_usage_percent'] * 0.8,
            memory_percent > self.alert_thresholds['memory_usage_percent'] * 0.8,
            disk_percent > self.alert_thresholds['disk_usage_percent'] * 0.8,
            any(status == "degraded" for status in component_results.values())
        ]
        
        if any(degraded_conditions):
            return "degraded"
        
        return "healthy"
    
    def _check_alerts(self, health_status: HealthStatus):
        """Check if alerts should be triggered."""
        alerts = []
        
        if health_status.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append(f"High CPU usage: {health_status.cpu_usage_percent:.1f}%")
        
        if health_status.memory_usage_percent > self.alert_thresholds['memory_usage_percent']:
            alerts.append(f"High memory usage: {health_status.memory_usage_percent:.1f}%")
        
        if health_status.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"High disk usage: {health_status.disk_usage_percent:.1f}%")
        
        if health_status.overall_status == "critical":
            alerts.append("System health is CRITICAL")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"HEALTH ALERT: {alert}")
    
    def _record_error(self, component: str, error: str):
        """Record error for component."""
        with self._lock:
            if component not in self.error_counts:
                self.error_counts[component] = 0
            self.error_counts[component] += 1
    
    def _cleanup_old_history(self):
        """Remove old health history entries."""
        cutoff_time = datetime.now() - self.history_retention
        cutoff_iso = cutoff_time.isoformat()
        
        self.health_history = [
            entry for entry in self.health_history
            if entry.timestamp > cutoff_iso
        ]
    
    def get_current_status(self) -> Optional[HealthStatus]:
        """Get current health status."""
        with self._lock:
            return self.health_history[-1] if self.health_history else None
    
    def get_health_history(self, hours: int = 1) -> List[HealthStatus]:
        """Get health history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        with self._lock:
            return [
                entry for entry in self.health_history
                if entry.timestamp > cutoff_iso
            ]
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for specific component."""
        return self.component_health.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        return self.component_health.copy()
    
    def export_health_report(self, filename: Optional[str] = None) -> str:
        """Export comprehensive health report.
        
        Args:
            filename: Optional filename to save report
            
        Returns:
            JSON string of health report
        """
        current_status = self.get_current_status()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "current_status": asdict(current_status) if current_status else None,
            "component_health": {
                name: asdict(health) for name, health in self.component_health.items()
            },
            "alert_thresholds": self.alert_thresholds,
            "error_counts": self.error_counts,
            "warning_counts": self.warning_counts,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "monitoring_config": {
                "check_interval": self.check_interval,
                "history_retention_hours": self.history_retention.total_seconds() / 3600
            }
        }
        
        report_json = json.dumps(report, indent=2, default=str)
        
        if filename:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_json)
            
            self.logger.info(f"Health report exported to {filename}")
        
        return report_json
    
    def reset_error_counts(self):
        """Reset all error counters."""
        with self._lock:
            self.error_counts.clear()
            self.warning_counts.clear()
        
        self.logger.info("Error counts reset")
    
    def update_thresholds(self, **thresholds):
        """Update alert thresholds.
        
        Args:
            **thresholds: Threshold values to update
        """
        for key, value in thresholds.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
                self.logger.info(f"Updated threshold {key} to {value}")
            else:
                self.logger.warning(f"Unknown threshold: {key}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
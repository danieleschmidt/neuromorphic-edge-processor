#!/usr/bin/env python3
"""
Neuromorphic Health Monitoring Demo
Demonstrates system health monitoring without external dependencies
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    status: str  # "healthy", "warning", "critical"
    timestamp: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: str
    timestamp: float
    uptime_seconds: float
    metrics: Dict[str, HealthMetric]
    alerts: List[str]
    summary: str


class SimpleHealthMonitor:
    """Simplified health monitoring system."""
    
    def __init__(
        self,
        check_interval: float = 5.0,
        enable_auto_monitoring: bool = True
    ):
        """Initialize health monitor."""
        self.check_interval = check_interval
        self.start_time = time.time()
        self.monitoring_active = False
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[str] = []
        self.metric_history: Dict[str, List[float]] = {}
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "error_rate": {"warning": 0.01, "critical": 0.05},
            "response_time": {"warning": 1.0, "critical": 3.0},
            "temperature": {"warning": 65.0, "critical": 80.0},
        }
        
        if enable_auto_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self.monitor_thread.start()
        print("🏥 Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        print("🏥 Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self.check_system_health()
                self.stop_event.wait(self.check_interval)
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                self.stop_event.wait(1.0)
    
    def check_system_health(self) -> SystemHealth:
        """Perform system health check."""
        current_time = time.time()
        self.alerts.clear()
        
        # Collect metrics
        self._collect_system_metrics(current_time)
        self._collect_neuromorphic_metrics(current_time)
        self._collect_application_metrics(current_time)
        
        # Analyze overall health
        overall_status = self._analyze_health()
        
        # Create health summary
        health = SystemHealth(
            status=overall_status,
            timestamp=current_time,
            uptime_seconds=current_time - self.start_time,
            metrics=dict(self.current_metrics),
            alerts=list(self.alerts),
            summary=self._generate_summary()
        )
        
        return health
    
    def _collect_system_metrics(self, timestamp: float):
        """Collect system metrics (simulated)."""
        # Simulate system metrics
        import random
        
        # CPU usage (simulated with some variability)
        base_cpu = 30.0
        cpu_usage = base_cpu + random.uniform(-10, 20)
        cpu_usage = max(0, min(100, cpu_usage))
        self._add_metric("cpu_usage", cpu_usage, "%", timestamp)
        
        # Memory usage (gradually increasing)
        base_memory = 45.0
        time_factor = (timestamp - self.start_time) / 100.0  # Slow increase
        memory_usage = base_memory + time_factor + random.uniform(-5, 10)
        memory_usage = max(0, min(100, memory_usage))
        self._add_metric("memory_usage", memory_usage, "%", timestamp)
        
        # Disk usage (mostly stable)
        disk_usage = 55.0 + random.uniform(-2, 5)
        disk_usage = max(0, min(100, disk_usage))
        self._add_metric("disk_usage", disk_usage, "%", timestamp)
        
        # Temperature (varies with CPU usage)
        temperature = 35 + (cpu_usage / 100) * 30 + random.uniform(-3, 5)
        self._add_metric("temperature", temperature, "°C", timestamp)
    
    def _collect_neuromorphic_metrics(self, timestamp: float):
        """Collect neuromorphic-specific metrics."""
        import random
        import math
        
        # Spike processing rate
        base_rate = 1000.0
        rate_variation = math.sin(timestamp / 10) * 200
        spike_rate = base_rate + rate_variation + random.uniform(-50, 50)
        spike_rate = max(0, spike_rate)
        self._add_metric("spike_processing_rate", spike_rate, "spikes/sec", timestamp)
        
        # Neuron utilization
        neuron_util = 65.0 + random.uniform(-10, 15)
        neuron_util = max(0, min(100, neuron_util))
        self._add_metric("neuron_utilization", neuron_util, "%", timestamp)
        
        # Energy consumption
        energy = 150.0 + random.uniform(-20, 30)
        energy = max(50, energy)
        self._add_metric("energy_consumption", energy, "mW", timestamp)
        
        # Model accuracy (slowly degrading with some recovery)
        time_hours = (timestamp - self.start_time) / 3600
        base_accuracy = 0.92 - (time_hours * 0.001)  # Slow degradation
        accuracy_noise = random.uniform(-0.02, 0.03)
        accuracy = max(0.5, min(1.0, base_accuracy + accuracy_noise))
        self._add_metric("model_accuracy", accuracy, "accuracy", timestamp)
    
    def _collect_application_metrics(self, timestamp: float):
        """Collect application metrics."""
        import random
        
        # Error rate (occasional spikes)
        if random.random() < 0.1:  # 10% chance of error spike
            error_rate = random.uniform(0.02, 0.08)
        else:
            error_rate = random.uniform(0.001, 0.005)
        self._add_metric("error_rate", error_rate, "rate", timestamp)
        
        # Response time (varies with load)
        response_time = 0.1 + random.uniform(0, 0.5)
        if random.random() < 0.05:  # 5% chance of slow response
            response_time += random.uniform(1.0, 3.0)
        self._add_metric("response_time", response_time, "seconds", timestamp)
        
        # Queue size
        queue_size = max(0, random.randint(0, 20))
        self._add_metric("queue_size", queue_size, "items", timestamp)
    
    def _add_metric(self, name: str, value: float, unit: str, timestamp: float):
        """Add metric with threshold checking."""
        # Get thresholds
        thresholds = self.thresholds.get(name, {})
        warning_threshold = thresholds.get("warning")
        critical_threshold = thresholds.get("critical")
        
        # Determine status
        status = "healthy"
        if critical_threshold is not None and value >= critical_threshold:
            status = "critical"
        elif warning_threshold is not None and value >= warning_threshold:
            status = "warning"
        
        # Create metric
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            status=status,
            timestamp=timestamp,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold
        )
        
        # Store metric
        self.current_metrics[name] = metric
        
        # Add to history
        if name not in self.metric_history:
            self.metric_history[name] = []
        self.metric_history[name].append(value)
        
        # Keep history manageable
        if len(self.metric_history[name]) > 100:
            self.metric_history[name] = self.metric_history[name][-50:]
        
        # Generate alert
        if status in ["warning", "critical"]:
            alert = f"{status.upper()}: {name} = {value:.2f}{unit}"
            if warning_threshold:
                alert += f" (threshold: {warning_threshold}{unit})"
            self.alerts.append(alert)
    
    def _analyze_health(self) -> str:
        """Analyze overall system health."""
        if not self.current_metrics:
            return "unknown"
        
        status_counts = {"healthy": 0, "warning": 0, "critical": 0}
        
        for metric in self.current_metrics.values():
            status_counts[metric.status] += 1
        
        total_metrics = len(self.current_metrics)
        
        if status_counts["critical"] > 0:
            return "critical"
        elif status_counts["warning"] > total_metrics * 0.3:  # >30% warnings
            return "warning"
        else:
            return "healthy"
    
    def _generate_summary(self) -> str:
        """Generate health summary."""
        if not self.current_metrics:
            return "No metrics available"
        
        status_counts = {"healthy": 0, "warning": 0, "critical": 0}
        
        for metric in self.current_metrics.values():
            status_counts[metric.status] += 1
        
        summary_parts = []
        
        if status_counts["critical"] > 0:
            summary_parts.append(f"{status_counts['critical']} critical")
        
        if status_counts["warning"] > 0:
            summary_parts.append(f"{status_counts['warning']} warnings")
        
        summary_parts.append(f"{status_counts['healthy']} healthy")
        
        uptime_hours = (time.time() - self.start_time) / 3600
        summary_parts.append(f"uptime: {uptime_hours:.1f}h")
        
        return ", ".join(summary_parts)
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metric_history:
            return {}
        
        values = self.metric_history[metric_name]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
    
    def export_health_report(self) -> str:
        """Export health report as JSON."""
        health = self.check_system_health()
        
        # Convert to serializable format
        report = {
            "timestamp": datetime.fromtimestamp(health.timestamp).isoformat(),
            "status": health.status,
            "uptime_seconds": health.uptime_seconds,
            "summary": health.summary,
            "alerts": health.alerts,
            "metrics": {
                name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                    "statistics": self.get_metric_statistics(name)
                }
                for name, metric in health.metrics.items()
            }
        }
        
        return json.dumps(report, indent=2)


def demonstrate_health_monitoring():
    """Demonstrate health monitoring system."""
    print("🏥 Neuromorphic Health Monitoring System Demo")
    print("=" * 50)
    
    # Create health monitor
    monitor = SimpleHealthMonitor(
        check_interval=2.0,  # Check every 2 seconds for demo
        enable_auto_monitoring=True
    )
    
    print("✓ Health monitor initialized and started")
    print("✓ Collecting system, neuromorphic, and application metrics")
    print("\n⏳ Running health checks for 10 seconds...")
    
    # Let it collect some data
    time.sleep(10)
    
    # Get current health status
    health = monitor.check_system_health()
    
    print(f"\n📊 Current Health Status: {health.status.upper()}")
    print(f"📊 System Uptime: {health.uptime_seconds:.1f} seconds")
    print(f"📊 Summary: {health.summary}")
    
    # Show alerts if any
    if health.alerts:
        print(f"\n⚠️ Active Alerts ({len(health.alerts)}):")
        for alert in health.alerts:
            print(f"  • {alert}")
    else:
        print(f"\n✅ No active alerts")
    
    # Show key metrics
    print(f"\n📈 Key Metrics:")
    key_metrics = [
        "cpu_usage", "memory_usage", "temperature", 
        "spike_processing_rate", "energy_consumption", "model_accuracy"
    ]
    
    for metric_name in key_metrics:
        if metric_name in health.metrics:
            metric = health.metrics[metric_name]
            status_emoji = {
                "healthy": "🟢",
                "warning": "🟡", 
                "critical": "🔴"
            }[metric.status]
            
            print(f"  {status_emoji} {metric_name:20}: {metric.value:8.2f} {metric.unit}")
    
    # Show statistics for one metric
    cpu_stats = monitor.get_metric_statistics("cpu_usage")
    if cpu_stats:
        print(f"\n📊 CPU Usage Statistics:")
        print(f"  Min: {cpu_stats['min']:.1f}%")
        print(f"  Max: {cpu_stats['max']:.1f}%")
        print(f"  Avg: {cpu_stats['avg']:.1f}%")
        print(f"  Samples: {cpu_stats['count']}")
    
    # Export health report
    print(f"\n📤 Exporting health report...")
    report_json = monitor.export_health_report()
    
    # Save to file
    report_file = "/tmp/health_report.json"
    try:
        with open(report_file, 'w') as f:
            f.write(report_json)
        print(f"✓ Health report saved to {report_file}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    # Show sample of report
    print(f"\n📋 Sample Health Report (first 300 chars):")
    print(report_json[:300] + "..." if len(report_json) > 300 else report_json)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print(f"\n✅ Health monitoring demo completed!")
    print(f"🔍 Monitored {len(health.metrics)} metrics")
    print(f"⏱️ Total runtime: {health.uptime_seconds:.1f} seconds")
    print(f"🏥 Final status: {health.status}")


class HealthDashboard:
    """Simple text-based health dashboard."""
    
    def __init__(self, monitor: SimpleHealthMonitor):
        self.monitor = monitor
    
    def display_dashboard(self, refresh_interval: float = 3.0, duration: float = 15.0):
        """Display live health dashboard."""
        print("\n🖥️ Live Health Dashboard")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Clear screen (simple version)
                print("\033[2J\033[H", end="")
                
                # Get current health
                health = self.monitor.check_system_health()
                
                # Header
                print("🏥 Neuromorphic Health Dashboard")
                print("=" * 60)
                print(f"Status: {health.status.upper():8} | Uptime: {health.uptime_seconds:.0f}s | Time: {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 60)
                
                # Key metrics in columns
                metrics_display = [
                    ("CPU Usage", "cpu_usage", "%"),
                    ("Memory", "memory_usage", "%"), 
                    ("Temperature", "temperature", "°C"),
                    ("Spike Rate", "spike_processing_rate", "spikes/s"),
                    ("Energy", "energy_consumption", "mW"),
                    ("Accuracy", "model_accuracy", "")
                ]
                
                for i in range(0, len(metrics_display), 2):
                    left = metrics_display[i] if i < len(metrics_display) else None
                    right = metrics_display[i+1] if i+1 < len(metrics_display) else None
                    
                    left_str = ""
                    right_str = ""
                    
                    if left and left[1] in health.metrics:
                        metric = health.metrics[left[1]]
                        status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}[metric.status]
                        left_str = f"{status_icon} {left[0]:12}: {metric.value:8.2f} {left[2]}"
                    
                    if right and right[1] in health.metrics:
                        metric = health.metrics[right[1]]
                        status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}[metric.status]
                        right_str = f"{status_icon} {right[0]:12}: {metric.value:8.2f} {right[2]}"
                    
                    print(f"{left_str:30} {right_str}")
                
                # Alerts
                if health.alerts:
                    print(f"\n⚠️ Alerts ({len(health.alerts)}):")
                    for alert in health.alerts[:3]:  # Show max 3 alerts
                        print(f"  • {alert}")
                else:
                    print(f"\n✅ No active alerts")
                
                # Progress bar for demo
                progress = min(1.0, (time.time() - start_time) / duration)
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                print(f"\nDemo Progress: [{bar}] {progress*100:.0f}%")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user")


def demonstrate_live_dashboard():
    """Demonstrate live health dashboard."""
    print("🖥️ Starting Live Health Dashboard Demo...")
    
    # Create monitor
    monitor = SimpleHealthMonitor(check_interval=1.0, enable_auto_monitoring=True)
    
    # Create and run dashboard
    dashboard = HealthDashboard(monitor)
    dashboard.display_dashboard(refresh_interval=2.0, duration=20.0)
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("✅ Live dashboard demo completed!")


if __name__ == "__main__":
    # Run basic demo
    demonstrate_health_monitoring()
    
    print("\n" + "="*50)
    print("Would you like to see the live dashboard? (requires terminal)")
    print("Uncomment the next line to run live dashboard demo:")
    # demonstrate_live_dashboard()
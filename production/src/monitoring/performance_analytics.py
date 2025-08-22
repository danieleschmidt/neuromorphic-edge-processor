"""Advanced performance analytics for neuromorphic systems."""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque, defaultdict
import statistics
import json
from pathlib import Path
from ..security.resource_monitor import ResourceMonitor
from ..security.security_config import security_config


class PerformanceProfiler:
    """Advanced performance profiler for neuromorphic operations."""
    
    def __init__(self, 
                 enable_detailed_profiling: bool = True,
                 max_history_size: int = 10000,
                 enable_real_time_alerts: bool = True):
        """Initialize performance profiler.
        
        Args:
            enable_detailed_profiling: Enable detailed timing and memory profiling
            max_history_size: Maximum number of historical records to keep
            enable_real_time_alerts: Enable real-time performance alerts
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.max_history_size = max_history_size
        self.enable_real_time_alerts = enable_real_time_alerts
        
        # Performance data storage
        self.operation_timings = defaultdict(lambda: deque(maxlen=max_history_size))
        self.memory_usage = deque(maxlen=max_history_size)
        self.throughput_data = defaultdict(lambda: deque(maxlen=max_history_size))
        self.error_counts = defaultdict(int)
        self.alert_history = deque(maxlen=1000)
        
        # Real-time monitoring
        self.active_operations = {}
        self.operation_counter = 0
        self.profiler_lock = threading.Lock()
        
        # Performance thresholds (configurable)
        self.performance_thresholds = {
            'max_operation_time': 10.0,  # seconds
            'max_memory_mb': 1000.0,
            'min_throughput': 1.0,  # operations per second
            'max_error_rate': 0.05  # 5%
        }
        
        # Resource monitor integration
        if security_config.log_security_violations:
            self.resource_monitor = ResourceMonitor(security_config)
        else:
            self.resource_monitor = None
    
    def start_operation(self, operation_name: str, **metadata) -> str:
        """Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata about the operation
            
        Returns:
            Operation ID for tracking
        """
        with self.profiler_lock:
            self.operation_counter += 1
            operation_id = f"{operation_name}_{self.operation_counter}_{int(time.time())}"
            
            operation_info = {
                'name': operation_name,
                'start_time': time.time(),
                'metadata': metadata,
                'memory_start': self._get_memory_usage() if self.enable_detailed_profiling else 0
            }
            
            self.active_operations[operation_id] = operation_info
            return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, **result_metadata) -> Dict[str, Any]:
        """End timing an operation and record results.
        
        Args:
            operation_id: Operation ID returned by start_operation
            success: Whether the operation succeeded
            **result_metadata: Additional metadata about the results
            
        Returns:
            Performance metrics for the operation
        """
        end_time = time.time()
        
        with self.profiler_lock:
            if operation_id not in self.active_operations:
                raise ValueError(f"Unknown operation ID: {operation_id}")
            
            operation_info = self.active_operations.pop(operation_id)
            operation_name = operation_info['name']
            
            # Calculate metrics
            duration = end_time - operation_info['start_time']
            memory_end = self._get_memory_usage() if self.enable_detailed_profiling else 0
            memory_delta = memory_end - operation_info['memory_start']
            
            # Store performance data
            performance_record = {
                'operation_id': operation_id,
                'operation_name': operation_name,
                'duration': duration,
                'success': success,
                'memory_delta_mb': memory_delta,
                'timestamp': end_time,
                'metadata': operation_info['metadata'],
                'result_metadata': result_metadata
            }
            
            self.operation_timings[operation_name].append(performance_record)
            
            if not success:
                self.error_counts[operation_name] += 1
            
            # Check performance thresholds and generate alerts
            if self.enable_real_time_alerts:
                self._check_performance_thresholds(performance_record)
            
            return performance_record
    
    def profile_function(self, operation_name: str):
        """Decorator for profiling function calls.
        
        Args:
            operation_name: Name for the operation
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Extract metadata from function signature
                metadata = {
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                operation_id = self.start_operation(operation_name, **metadata)
                
                try:
                    result = func(*args, **kwargs)
                    self.end_operation(operation_id, success=True, result_type=type(result).__name__)
                    return result
                    
                except Exception as e:
                    self.end_operation(operation_id, success=False, error=str(e))
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _check_performance_thresholds(self, record: Dict[str, Any]):
        """Check performance thresholds and generate alerts."""
        alerts = []
        
        # Check operation duration
        if record['duration'] > self.performance_thresholds['max_operation_time']:
            alerts.append({
                'type': 'slow_operation',
                'operation': record['operation_name'],
                'duration': record['duration'],
                'threshold': self.performance_thresholds['max_operation_time'],
                'timestamp': record['timestamp']
            })
        
        # Check memory usage
        if record['memory_delta_mb'] > self.performance_thresholds['max_memory_mb']:
            alerts.append({
                'type': 'high_memory_usage',
                'operation': record['operation_name'],
                'memory_delta': record['memory_delta_mb'],
                'threshold': self.performance_thresholds['max_memory_mb'],
                'timestamp': record['timestamp']
            })
        
        # Check error rate
        operation_name = record['operation_name']
        recent_operations = list(self.operation_timings[operation_name])[-100:]  # Last 100
        
        if len(recent_operations) >= 10:  # Need sufficient data
            error_count = sum(1 for op in recent_operations if not op['success'])
            error_rate = error_count / len(recent_operations)
            
            if error_rate > self.performance_thresholds['max_error_rate']:
                alerts.append({
                    'type': 'high_error_rate',
                    'operation': operation_name,
                    'error_rate': error_rate,
                    'threshold': self.performance_thresholds['max_error_rate'],
                    'timestamp': record['timestamp']
                })
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            print(f"⚠️ Performance Alert: {alert['type']} for {alert['operation']}")
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """Get detailed statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Statistics dictionary
        """
        if operation_name not in self.operation_timings:
            return {"error": f"No data for operation: {operation_name}"}
        
        records = list(self.operation_timings[operation_name])
        
        if not records:
            return {"error": f"No records for operation: {operation_name}"}
        
        # Extract timing data
        durations = [r['duration'] for r in records]
        memory_deltas = [r['memory_delta_mb'] for r in records]
        success_count = sum(1 for r in records if r['success'])
        
        # Calculate statistics
        stats = {
            'operation_name': operation_name,
            'total_calls': len(records),
            'success_count': success_count,
            'error_count': len(records) - success_count,
            'success_rate': success_count / len(records),
            'timing_stats': {
                'mean_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0
            },
            'memory_stats': {
                'mean_delta_mb': statistics.mean(memory_deltas),
                'median_delta_mb': statistics.median(memory_deltas),
                'max_delta_mb': max(memory_deltas),
                'min_delta_mb': min(memory_deltas)
            } if self.enable_detailed_profiling else {},
            'recent_performance': self._calculate_recent_performance(records),
            'trend_analysis': self._analyze_performance_trends(records)
        }
        
        return stats
    
    def _calculate_recent_performance(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate recent performance metrics."""
        if len(records) < 10:
            return {"insufficient_data": True}
        
        # Get last 50 operations
        recent_records = records[-50:]
        
        recent_durations = [r['duration'] for r in recent_records]
        recent_success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records)
        
        return {
            'recent_mean_duration': statistics.mean(recent_durations),
            'recent_success_rate': recent_success_rate,
            'recent_throughput': len(recent_records) / (recent_records[-1]['timestamp'] - recent_records[0]['timestamp'])
        }
    
    def _analyze_performance_trends(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(records) < 20:
            return {"insufficient_data": True}
        
        # Split into first and second half
        mid_point = len(records) // 2
        first_half = records[:mid_point]
        second_half = records[mid_point:]
        
        first_half_duration = statistics.mean([r['duration'] for r in first_half])
        second_half_duration = statistics.mean([r['duration'] for r in second_half])
        
        first_half_success = sum(1 for r in first_half if r['success']) / len(first_half)
        second_half_success = sum(1 for r in second_half if r['success']) / len(second_half)
        
        return {
            'duration_trend': 'improving' if second_half_duration < first_half_duration else 'degrading',
            'duration_change_percent': ((second_half_duration - first_half_duration) / first_half_duration) * 100,
            'success_rate_trend': 'improving' if second_half_success > first_half_success else 'degrading',
            'success_rate_change_percent': ((second_half_success - first_half_success) / first_half_success) * 100 if first_half_success > 0 else 0
        }
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive system performance summary."""
        summary = {
            'profiler_status': {
                'detailed_profiling_enabled': self.enable_detailed_profiling,
                'real_time_alerts_enabled': self.enable_real_time_alerts,
                'active_operations': len(self.active_operations),
                'total_operations_tracked': sum(len(timings) for timings in self.operation_timings.values())
            },
            'operation_summary': {},
            'recent_alerts': list(self.alert_history)[-10:],  # Last 10 alerts
            'performance_thresholds': self.performance_thresholds
        }
        
        # Add summary for each operation type
        for operation_name in self.operation_timings:
            records = list(self.operation_timings[operation_name])
            if records:
                recent_records = records[-10:]  # Last 10 operations
                summary['operation_summary'][operation_name] = {
                    'total_calls': len(records),
                    'recent_average_duration': statistics.mean([r['duration'] for r in recent_records]),
                    'recent_success_rate': sum(1 for r in recent_records if r['success']) / len(recent_records),
                    'error_count': self.error_counts[operation_name]
                }
        
        return summary
    
    def export_performance_data(self, output_path: Path, 
                              operation_filter: Optional[List[str]] = None) -> None:
        """Export performance data to JSON file.
        
        Args:
            output_path: Path to output file
            operation_filter: List of operations to include (None for all)
        """
        export_data = {
            'export_timestamp': time.time(),
            'profiler_config': {
                'detailed_profiling_enabled': self.enable_detailed_profiling,
                'max_history_size': self.max_history_size,
                'performance_thresholds': self.performance_thresholds
            },
            'operations': {}
        }
        
        for operation_name, records in self.operation_timings.items():
            if operation_filter and operation_name not in operation_filter:
                continue
                
            export_data['operations'][operation_name] = {
                'records': list(records),
                'statistics': self.get_operation_statistics(operation_name)
            }
        
        export_data['alerts'] = list(self.alert_history)
        
        # Write to file securely
        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Performance data exported to {output_path}")
        except Exception as e:
            print(f"Failed to export performance data: {e}")
    
    def set_performance_thresholds(self, **thresholds):
        """Update performance thresholds.
        
        Args:
            **thresholds: Threshold values to update
        """
        for key, value in thresholds.items():
            if key in self.performance_thresholds:
                if isinstance(value, (int, float)) and value > 0:
                    self.performance_thresholds[key] = float(value)
                else:
                    raise ValueError(f"Invalid threshold value for {key}: {value}")
            else:
                raise ValueError(f"Unknown threshold: {key}")
    
    def reset_statistics(self, operation_name: Optional[str] = None):
        """Reset performance statistics.
        
        Args:
            operation_name: Specific operation to reset (None for all)
        """
        with self.profiler_lock:
            if operation_name:
                if operation_name in self.operation_timings:
                    self.operation_timings[operation_name].clear()
                    self.error_counts[operation_name] = 0
            else:
                self.operation_timings.clear()
                self.error_counts.clear()
                self.alert_history.clear()
                self.memory_usage.clear()
    
    def get_live_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        current_time = time.time()
        
        # Calculate recent throughput (last 60 seconds)
        recent_operations = []
        for records in self.operation_timings.values():
            recent_ops = [r for r in records if current_time - r['timestamp'] <= 60]
            recent_operations.extend(recent_ops)
        
        return {
            'timestamp': current_time,
            'active_operations': len(self.active_operations),
            'recent_throughput': len(recent_operations),  # operations in last 60 seconds
            'total_operations': sum(len(records) for records in self.operation_timings.values()),
            'total_errors': sum(self.error_counts.values()),
            'recent_alerts': len([a for a in self.alert_history if current_time - a['timestamp'] <= 300]),  # last 5 minutes
            'memory_usage_mb': self._get_memory_usage()
        }


# Global profiler instance
performance_profiler = PerformanceProfiler()
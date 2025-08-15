"""Advanced structured logging system for neuromorphic operations."""

import logging
import json
import time
import threading
import traceback
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys
import os
from contextlib import contextmanager


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, Any]] = None
    exception_info: Optional[Dict[str, str]] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Get caller information
        frame_info = self._get_caller_info()
        
        # Build structured event
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            correlation_id=getattr(record, 'correlation_id', None),
            user_id=getattr(record, 'user_id', None),
            session_id=getattr(record, 'session_id', None),
            trace_id=getattr(record, 'trace_id', None),
            span_id=getattr(record, 'span_id', None),
            tags=getattr(record, 'tags', None),
            metrics=getattr(record, 'metrics', None),
            context=getattr(record, 'context', None) if self.include_context else None
        )
        
        # Add exception information if present
        if record.exc_info:
            event.exception_info = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
        
        return json.dumps(asdict(event), default=str, separators=(',', ':'))
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get caller frame information."""
        frame = sys._getframe()
        while frame:
            if frame.f_code.co_filename != __file__:
                return {
                    'filename': frame.f_code.co_filename,
                    'function': frame.f_code.co_name,
                    'line_number': frame.f_lineno
                }
            frame = frame.f_back
        return {}


class MetricsCollector:
    """Collect and aggregate logging metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[Dict]] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, List[float]] = {}
        self.gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] = self.counters.get(key, 0) + value
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self.timers:
                self.timers[key] = []
            self.timers[key].append(duration_ms)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and tags."""
        if not tags:
            return name
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {}
            }
            
            for name, values in self.timers.items():
                if values:
                    summary['timers'][name] = {
                        'count': len(values),
                        'min_ms': min(values),
                        'max_ms': max(values),
                        'avg_ms': sum(values) / len(values),
                        'p50_ms': sorted(values)[len(values) // 2],
                        'p95_ms': sorted(values)[int(len(values) * 0.95)],
                        'p99_ms': sorted(values)[int(len(values) * 0.99)]
                    }
            
            return summary


class NeuromorphicLogger:
    """Advanced logger with neuromorphic-specific features."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        structured: bool = True,
        enable_metrics: bool = True,
        log_file: Optional[str] = None,
        max_file_size_mb: int = 100,
        backup_count: int = 5
    ):
        """Initialize neuromorphic logger.
        
        Args:
            name: Logger name
            level: Logging level
            structured: Enable structured JSON logging
            enable_metrics: Enable metrics collection
            log_file: Optional log file path
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.structured = structured
        self.enable_metrics = enable_metrics
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Setup handlers
        self._setup_handlers(log_file, max_file_size_mb, backup_count)
        
        # Metrics collector
        self.metrics = MetricsCollector() if enable_metrics else None
        
        # Context storage for request/session tracking
        self._context = threading.local()
    
    def _setup_handlers(self, log_file: Optional[str], max_file_size_mb: int, backup_count: int):
        """Setup logging handlers."""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            from logging.handlers import RotatingFileHandler
            
            # Ensure directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            
            if self.structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            
            self.logger.addHandler(file_handler)
    
    def set_context(self, **context):
        """Set logging context for current thread."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(context)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        return getattr(self._context, 'data', {}).copy()
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context and metrics."""
        
        # Get context
        context = self.get_context()
        context.update(kwargs.pop('context', {}))
        
        # Create log record with extra data
        extra = {
            'context': context,
            'tags': kwargs.pop('tags', None),
            'metrics': kwargs.pop('metrics', None),
            **{k: v for k, v in context.items() if k in ['correlation_id', 'user_id', 'session_id', 'trace_id', 'span_id']}
        }
        
        # Log the message
        getattr(self.logger, level.lower())(message, extra=extra, **kwargs)
        
        # Record metrics if enabled
        if self.enable_metrics and self.metrics:
            self.metrics.record_counter(f"log.{level.lower()}")
            if 'duration_ms' in kwargs:
                self.metrics.record_timer(f"operation.duration", kwargs['duration_ms'])
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context('CRITICAL', message, **kwargs)
    
    @contextmanager
    def operation_timer(self, operation_name: str, **context):
        """Context manager for timing operations."""
        start_time = time.time()
        operation_context = {'operation': operation_name, **context}
        
        # Set operation context
        original_context = self.get_context()
        self.set_context(**operation_context)
        
        try:
            self.info(f"Starting operation: {operation_name}", context=operation_context)
            yield
            
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Completed operation: {operation_name}",
                context=operation_context,
                metrics={'duration_ms': duration_ms}
            )
            
            # Record metrics
            if self.enable_metrics and self.metrics:
                self.metrics.record_timer(f"operation.{operation_name}", duration_ms)
                self.metrics.record_counter(f"operation.{operation_name}.success")
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(
                f"Failed operation: {operation_name}",
                context=operation_context,
                metrics={'duration_ms': duration_ms},
                exc_info=True
            )
            
            # Record failure metrics
            if self.enable_metrics and self.metrics:
                self.metrics.record_timer(f"operation.{operation_name}", duration_ms)
                self.metrics.record_counter(f"operation.{operation_name}.error")
            
            raise
        
        finally:
            # Restore original context
            self.clear_context()
            self.set_context(**original_context)
    
    @contextmanager
    def request_context(self, request_id: str = None, user_id: str = None, session_id: str = None):
        """Context manager for request-scoped logging."""
        
        import uuid
        request_id = request_id or str(uuid.uuid4())
        
        context = {
            'correlation_id': request_id,
            'user_id': user_id,
            'session_id': session_id
        }
        
        # Set request context
        original_context = self.get_context()
        self.set_context(**context)
        
        try:
            self.info("Request started", context={'request_id': request_id})
            yield request_id
            self.info("Request completed", context={'request_id': request_id})
        
        except Exception as e:
            self.error("Request failed", context={'request_id': request_id}, exc_info=True)
            raise
        
        finally:
            # Restore original context
            self.clear_context()
            self.set_context(**original_context)
    
    def log_model_performance(
        self,
        model_name: str,
        accuracy: float,
        inference_time_ms: float,
        memory_usage_mb: float,
        **additional_metrics
    ):
        """Log model performance metrics."""
        
        metrics = {
            'accuracy': accuracy,
            'inference_time_ms': inference_time_ms,
            'memory_usage_mb': memory_usage_mb,
            **additional_metrics
        }
        
        self.info(
            f"Model performance: {model_name}",
            context={'model_name': model_name},
            metrics=metrics,
            tags={'event_type': 'model_performance'}
        )
        
        # Record detailed metrics
        if self.enable_metrics and self.metrics:
            for metric_name, value in metrics.items():
                self.metrics.record_gauge(f"model.{model_name}.{metric_name}", value)
    
    def log_spike_processing(
        self,
        num_spikes: int,
        processing_time_ms: float,
        spike_rate_hz: float,
        sparsity: float
    ):
        """Log spike processing metrics."""
        
        metrics = {
            'num_spikes': num_spikes,
            'processing_time_ms': processing_time_ms,
            'spike_rate_hz': spike_rate_hz,
            'sparsity': sparsity
        }
        
        self.info(
            "Spike processing completed",
            metrics=metrics,
            tags={'event_type': 'spike_processing'}
        )
        
        # Record metrics
        if self.enable_metrics and self.metrics:
            self.metrics.record_counter('spikes.processed', num_spikes)
            self.metrics.record_timer('spikes.processing_time', processing_time_ms)
            self.metrics.record_gauge('spikes.rate_hz', spike_rate_hz)
            self.metrics.record_gauge('spikes.sparsity', sparsity)
    
    def log_learning_progress(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        convergence_metric: Optional[float] = None
    ):
        """Log learning progress."""
        
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate
        }
        
        if convergence_metric is not None:
            metrics['convergence_metric'] = convergence_metric
        
        self.info(
            f"Learning progress - Epoch {epoch}",
            metrics=metrics,
            tags={'event_type': 'learning_progress'}
        )
        
        # Record metrics
        if self.enable_metrics and self.metrics:
            self.metrics.record_gauge('learning.loss', loss)
            self.metrics.record_gauge('learning.accuracy', accuracy)
            self.metrics.record_gauge('learning.learning_rate', learning_rate)
            if convergence_metric is not None:
                self.metrics.record_gauge('learning.convergence', convergence_metric)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if self.metrics:
            return self.metrics.get_metrics_summary()
        return {}
    
    def export_metrics(self, filename: str):
        """Export metrics to file."""
        if not self.metrics:
            return
        
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'logger_name': self.name,
            'metrics': self.get_metrics_summary()
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.info(f"Metrics exported to {filename}")


# Global logger registry
_loggers: Dict[str, NeuromorphicLogger] = {}
_default_config = {
    'level': 'INFO',
    'structured': True,
    'enable_metrics': True
}


def get_logger(
    name: str,
    **config
) -> NeuromorphicLogger:
    """Get or create a neuromorphic logger.
    
    Args:
        name: Logger name
        **config: Logger configuration overrides
        
    Returns:
        NeuromorphicLogger instance
    """
    global _loggers
    
    if name not in _loggers:
        logger_config = {**_default_config, **config}
        _loggers[name] = NeuromorphicLogger(name, **logger_config)
    
    return _loggers[name]


def configure_logging(**config):
    """Configure global logging settings."""
    global _default_config
    _default_config.update(config)


def get_all_loggers() -> Dict[str, NeuromorphicLogger]:
    """Get all active loggers."""
    return _loggers.copy()


def export_all_metrics(directory: str):
    """Export metrics from all loggers."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, logger in _loggers.items():
        filename = output_dir / f"{name}_metrics.json"
        logger.export_metrics(str(filename))
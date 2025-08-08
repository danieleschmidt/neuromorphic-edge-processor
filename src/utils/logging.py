"""Advanced logging and monitoring system for neuromorphic computing."""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import traceback
import psutil
import threading
from contextlib import contextmanager
from functools import wraps

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class NeuromorphicFormatter(logging.Formatter):
    """Custom formatter for neuromorphic computing logs."""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Add system information to log record
        if not hasattr(record, 'memory_usage_mb'):
            record.memory_usage_mb = psutil.virtual_memory().used / 1024 / 1024
            
        if not hasattr(record, 'cpu_usage_percent'):
            record.cpu_usage_percent = psutil.cpu_percent()
        
        # Custom format with neuromorphic context
        log_format = (
            "[{asctime}] {levelname:8s} | "
            "MEM:{memory_usage_mb:6.1f}MB CPU:{cpu_usage_percent:5.1f}% | "
            "{name} | {message}"
        )
        
        formatter = logging.Formatter(log_format, style='{')
        return formatter.format(record)


class MetricsCollector:
    """Collects and aggregates metrics for monitoring."""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None, tags: Optional[Dict] = None):
        """Log a metric value."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            metric_entry = {
                'value': value,
                'timestamp': time.time(),
                'step': step,
                'tags': tags or {}
            }
            
            self.metrics[name].append(metric_entry)
    
    def log_metrics_dict(self, metrics_dict: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log multiple metrics at once."""
        for name, value in metrics_dict.items():
            metric_name = f"{prefix}{name}" if prefix else name
            self.log_metric(metric_name, value, step)
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            values = [entry['value'] for entry in self.metrics[name]]
            
            return {
                'count': len(values),
                'mean': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'latest': values[-1] if values else 0
            }
    
    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """Get all logged metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()


class ExperimentLogger:
    """Advanced experiment logging with multiple backends."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        wandb_project: str = "neuromorphic-edge",
        log_level: str = "INFO"
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            enable_wandb: Enable Weights & Biases logging
            enable_tensorboard: Enable TensorBoard logging
            wandb_project: W&B project name
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create experiment-specific subdirectory
        self.experiment_dir = self.log_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics collector
        self.metrics = MetricsCollector()
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize external logging services
        self.wandb_run = None
        self.tensorboard_writer = None
        
        if enable_wandb and WANDB_AVAILABLE:
            self._setup_wandb(wandb_project)
        elif enable_wandb:
            self.logger.warning("W&B logging requested but wandb not available")
        
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard()
        elif enable_tensorboard:
            self.logger.warning("TensorBoard logging requested but tensorboard not available")
        
        # Log system information
        self._log_system_info()
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"neuromorphic.{self.experiment_name}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(NeuromorphicFormatter())
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(NeuromorphicFormatter())
        logger.addHandler(console_handler)
        
        # Error file handler
        error_file = self.experiment_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(NeuromorphicFormatter())
        logger.addHandler(error_handler)
        
        return logger
    
    def _setup_wandb(self, project: str):
        """Setup Weights & Biases logging."""
        try:
            self.wandb_run = wandb.init(
                project=project,
                name=self.experiment_name,
                dir=str(self.experiment_dir)
            )
            self.logger.info("W&B logging initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            tb_dir = self.experiment_dir / "tensorboard"
            self.tensorboard_writer = SummaryWriter(str(tb_dir))
            self.logger.info(f"TensorBoard logging initialized: {tb_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard: {e}")
    
    def _log_system_info(self):
        """Log system and environment information."""
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": sys.version,
            "platform": sys.platform,
        }
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                system_info["gpu_count"] = torch.cuda.device_count()
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
                system_info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        
        self.logger.info(f"System Information: {json.dumps(system_info, indent=2)}")
        
        # Save system info to file
        with open(self.experiment_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_file = self.experiment_dir / "config.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
            # Log to external services
            if self.wandb_run:
                wandb.config.update(config)
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to all configured backends."""
        
        # Log to internal collector
        self.metrics.log_metrics_dict(metrics, step, prefix)
        
        # Log to external services
        if self.wandb_run:
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
        
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{prefix}{name}", value, step or 0)
        
        # Log summary to file
        metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics (step {step}): {metrics_str}")
    
    def log_model_summary(self, model, input_shape: tuple):
        """Log model architecture summary."""
        try:
            # Count parameters
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info = {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "input_shape": input_shape,
                    "model_type": type(model).__name__
                }
                
                self.logger.info(f"Model Summary: {json.dumps(model_info, indent=2)}")
                
                # Save model info
                with open(self.experiment_dir / "model_summary.json", 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                if self.wandb_run:
                    wandb.config.update({"model": model_info})
                    
        except Exception as e:
            self.logger.error(f"Failed to log model summary: {e}")
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file", description: str = ""):
        """Log file artifact."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"Artifact file not found: {file_path}")
            return
        
        # Copy to experiment directory
        artifact_dir = self.experiment_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        dest_path = artifact_dir / file_path.name
        try:
            import shutil
            shutil.copy2(file_path, dest_path)
            
            self.logger.info(f"Artifact saved: {dest_path} ({artifact_type})")
            
            if self.wandb_run:
                artifact = wandb.Artifact(
                    name=file_path.stem,
                    type=artifact_type,
                    description=description
                )
                artifact.add_file(str(dest_path))
                wandb.log_artifact(artifact)
                
        except Exception as e:
            self.logger.error(f"Failed to log artifact {file_path}: {e}")
    
    @contextmanager
    def log_execution_time(self, operation_name: str):
        """Context manager to log execution time of operations."""
        start_time = time.time()
        self.logger.info(f"Starting {operation_name}")
        
        try:
            yield
            execution_time = time.time() - start_time
            self.logger.info(f"Completed {operation_name} in {execution_time:.3f}s")
            self.log_metrics({f"{operation_name}_duration_s": execution_time})
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed {operation_name} after {execution_time:.3f}s: {e}")
            self.log_metrics({f"{operation_name}_duration_s": execution_time})
            raise
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with full traceback."""
        error_info = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.error(f"Exception in {context}: {json.dumps(error_info, indent=2)}")
        
        # Save detailed error report
        error_file = self.experiment_dir / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
    
    def finalize(self):
        """Finalize logging and cleanup resources."""
        try:
            # Save final metrics summary
            metrics_summary = {}
            for name in self.metrics.metrics:
                metrics_summary[name] = self.metrics.get_metric_summary(name)
            
            summary_file = self.experiment_dir / "metrics_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            
            # Close external services
            if self.wandb_run:
                wandb.finish()
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            self.logger.info(f"Experiment logging finalized. Results in {self.experiment_dir}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing logging: {e}")


def setup_logging(config: Dict[str, Any]) -> ExperimentLogger:
    """Setup logging system from configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured ExperimentLogger instance
    """
    experiment_name = config.get('experiment_name', f'neuromorphic_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=config.get('log_dir', 'logs'),
        enable_wandb=config.get('enable_wandb', False),
        enable_tensorboard=config.get('enable_tensorboard', True),
        wandb_project=config.get('wandb_project', 'neuromorphic-edge'),
        log_level=config.get('log_level', 'INFO')
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with neuromorphic formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(NeuromorphicFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_performance(func):
    """Decorator to automatically log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(f"performance.{func.__module__}.{func.__name__}")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024**2
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024**2
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Function executed successfully | "
                f"Time: {execution_time:.3f}s | "
                f"Memory: {memory_delta:+.1f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(
                f"Function failed after {execution_time:.3f}s: {e} | "
                f"Args: {len(args)} | Kwargs: {list(kwargs.keys())}"
            )
            raise
    
    return wrapper


class SecurityLogger:
    """Specialized logger for security events and monitoring."""
    
    def __init__(self, log_dir: str = "security_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("neuromorphic.security")
        self.logger.setLevel(logging.INFO)
        
        # Security log file with strict permissions
        security_log = self.log_dir / "security.log"
        handler = logging.handlers.RotatingFileHandler(
            security_log, maxBytes=10*1024*1024, backupCount=10
        )
        
        # Enhanced format for security logs
        formatter = logging.Formatter(
            "[{asctime}] SECURITY {levelname} | "
            "PID:{process} | TID:{thread} | "
            "{name} | {message}",
            style='{'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Set restrictive permissions on log file
        try:
            import stat
            security_log.chmod(stat.S_IRUSR | stat.S_IWUSR)  # Owner read/write only
        except Exception:
            pass
    
    def log_access_attempt(self, source: str, resource: str, allowed: bool, details: Dict = None):
        """Log access attempt to system resources."""
        log_data = {
            "event": "access_attempt",
            "source": source,
            "resource": resource,
            "allowed": allowed,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        level = logging.INFO if allowed else logging.WARNING
        self.logger.log(level, f"Access attempt: {json.dumps(log_data)}")
    
    def log_input_validation(self, input_type: str, validation_result: bool, details: Dict = None):
        """Log input validation events."""
        log_data = {
            "event": "input_validation",
            "input_type": input_type,
            "valid": validation_result,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        level = logging.INFO if validation_result else logging.WARNING
        self.logger.log(level, f"Input validation: {json.dumps(log_data)}")
    
    def log_security_violation(self, violation_type: str, severity: str, details: Dict = None):
        """Log security violations."""
        log_data = {
            "event": "security_violation",
            "violation_type": violation_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        level = logging.CRITICAL if severity == "high" else logging.ERROR
        self.logger.log(level, f"SECURITY VIOLATION: {json.dumps(log_data)}")
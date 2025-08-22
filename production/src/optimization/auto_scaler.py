"""Auto-scaling system for neuromorphic edge processors."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np
from collections import deque


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    CUSTOM = "custom"


@dataclass
class ScalingMetric:
    """Metric for scaling decisions."""
    name: str
    value: float
    timestamp: float
    threshold_low: float
    threshold_high: float
    weight: float = 1.0


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    cooldown_seconds: float = 300.0
    window_size: int = 5
    confidence_threshold: float = 0.8
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    direction: ScalingDirection
    factor: float  # Scaling factor (e.g., 1.5 for 50% increase)
    reason: str
    confidence: float
    timestamp: float
    parameters: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collects and maintains metrics for scaling decisions."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            history_size: Maximum number of metrics to keep in history
        """
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            if name not in self.metrics_history:
                self.metrics_history[name] = deque(maxlen=self.history_size)
            
            self.metrics_history[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def get_recent_metrics(self, name: str, window_seconds: float = 300.0) -> List[Dict]:
        """Get recent metrics within time window.
        
        Args:
            name: Metric name
            window_seconds: Time window in seconds
            
        Returns:
            List of metric entries
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.lock:
            if name not in self.metrics_history:
                return []
            
            return [
                entry for entry in self.metrics_history[name]
                if entry['timestamp'] > cutoff_time
            ]
    
    def get_metric_statistics(self, name: str, window_seconds: float = 300.0) -> Dict[str, float]:
        """Get statistical summary of metric.
        
        Args:
            name: Metric name
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with statistics
        """
        recent_metrics = self.get_recent_metrics(name, window_seconds)
        
        if not recent_metrics:
            return {}
        
        values = [m['value'] for m in recent_metrics]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Normalize to [-1, 1] based on value range
        value_range = np.max(values) - np.min(values)
        if value_range == 0:
            return 0.0
        
        normalized_slope = slope / value_range * len(values)
        return np.clip(normalized_slope, -1, 1)


class AutoScaler:
    """Auto-scaling system for neuromorphic processors."""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        scaling_factor: float = 1.5,
        cooldown_seconds: float = 300.0,
        evaluation_interval: float = 60.0
    ):
        """Initialize auto-scaler.
        
        Args:
            min_capacity: Minimum capacity units
            max_capacity: Maximum capacity units
            scaling_factor: Default scaling factor
            cooldown_seconds: Cooldown period after scaling
            evaluation_interval: How often to evaluate scaling (seconds)
        """
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scaling_factor = scaling_factor
        self.cooldown_seconds = cooldown_seconds
        self.evaluation_interval = evaluation_interval
        
        # Current state
        self.current_capacity = min_capacity
        self.last_scaling_time = 0.0
        self.scaling_history: List[ScalingAction] = []
        
        # Components
        self.metrics_collector = MetricsCollector()
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_callbacks: List[Callable[[ScalingAction], None]] = []
        
        # Monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default scaling rules."""
        
        # CPU utilization rule
        self.add_scaling_rule(
            "cpu_utilization",
            ScalingRule(
                trigger=ScalingTrigger.CPU_UTILIZATION,
                threshold_up=75.0,
                threshold_down=25.0,
                cooldown_seconds=300.0,
                window_size=5,
                weight=1.0
            )
        )
        
        # Memory usage rule
        self.add_scaling_rule(
            "memory_usage",
            ScalingRule(
                trigger=ScalingTrigger.MEMORY_USAGE,
                threshold_up=80.0,
                threshold_down=30.0,
                cooldown_seconds=180.0,
                window_size=3,
                weight=1.2
            )
        )
        
        # Throughput rule
        self.add_scaling_rule(
            "throughput",
            ScalingRule(
                trigger=ScalingTrigger.THROUGHPUT,
                threshold_up=0.8,  # 80% of target throughput
                threshold_down=0.3,  # 30% of target throughput
                cooldown_seconds=240.0,
                window_size=6,
                weight=0.8
            )
        )
        
        # Latency rule
        self.add_scaling_rule(
            "latency",
            ScalingRule(
                trigger=ScalingTrigger.LATENCY,
                threshold_up=100.0,  # 100ms
                threshold_down=20.0,   # 20ms
                cooldown_seconds=120.0,
                window_size=4,
                weight=1.5
            )
        )
    
    def add_scaling_rule(self, name: str, rule: ScalingRule):
        """Add scaling rule.
        
        Args:
            name: Rule name
            rule: Scaling rule configuration
        """
        with self._lock:
            self.scaling_rules[name] = rule
    
    def remove_scaling_rule(self, name: str):
        """Remove scaling rule.
        
        Args:
            name: Rule name
        """
        with self._lock:
            self.scaling_rules.pop(name, None)
    
    def record_metric(self, name: str, value: float):
        """Record metric for scaling decisions.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics_collector.record_metric(name, value)
    
    def add_scaling_callback(self, callback: Callable[[ScalingAction], None]):
        """Add callback for scaling actions.
        
        Args:
            callback: Function to call when scaling occurs
        """
        self.scaling_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AutoScaler",
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Evaluate scaling decisions
                scaling_action = self.evaluate_scaling()
                
                if scaling_action and scaling_action.direction != ScalingDirection.NONE:
                    self._execute_scaling(scaling_action)
                
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in auto-scaler monitoring loop: {e}")
            
            # Wait for next evaluation
            self._stop_monitoring.wait(self.evaluation_interval)
    
    def evaluate_scaling(self) -> Optional[ScalingAction]:
        """Evaluate whether scaling is needed.
        
        Returns:
            Scaling action if needed, None otherwise
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.cooldown_seconds:
            return None
        
        # Collect scaling signals from all rules
        scaling_signals = []
        
        with self._lock:
            for rule_name, rule in self.scaling_rules.items():
                if not rule.enabled:
                    continue
                
                signal = self._evaluate_rule(rule)
                if signal:
                    scaling_signals.append(signal)
        
        if not scaling_signals:
            return None
        
        # Aggregate signals
        return self._aggregate_scaling_signals(scaling_signals)
    
    def _evaluate_rule(self, rule: ScalingRule) -> Optional[Tuple[ScalingDirection, float, str]]:
        """Evaluate a single scaling rule.
        
        Args:
            rule: Scaling rule to evaluate
            
        Returns:
            Tuple of (direction, confidence, reason) or None
        """
        metric_name = rule.trigger.value
        
        # Get recent metrics
        window_seconds = rule.window_size * self.evaluation_interval
        stats = self.metrics_collector.get_metric_statistics(metric_name, window_seconds)
        
        if not stats or stats['count'] < rule.window_size:
            return None
        
        mean_value = stats['mean']
        trend = stats['trend']
        
        # Determine scaling direction
        direction = ScalingDirection.NONE
        confidence = 0.0
        
        if mean_value > rule.threshold_up:
            direction = ScalingDirection.UP
            # Higher confidence if trending up and further above threshold
            confidence = min(1.0, (mean_value - rule.threshold_up) / rule.threshold_up)
            confidence = confidence * (1 + max(0, trend) * 0.5)
            
        elif mean_value < rule.threshold_down:
            direction = ScalingDirection.DOWN
            # Higher confidence if trending down and further below threshold
            confidence = min(1.0, (rule.threshold_down - mean_value) / rule.threshold_down)
            confidence = confidence * (1 + max(0, -trend) * 0.5)
        
        if direction == ScalingDirection.NONE or confidence < rule.confidence_threshold:
            return None
        
        reason = f"{metric_name}: {mean_value:.2f} ({'>' if direction == ScalingDirection.UP else '<'}) threshold {rule.threshold_up if direction == ScalingDirection.UP else rule.threshold_down}"
        
        return direction, confidence * rule.weight, reason
    
    def _aggregate_scaling_signals(self, signals: List[Tuple[ScalingDirection, float, str]]) -> Optional[ScalingAction]:
        """Aggregate multiple scaling signals.
        
        Args:
            signals: List of (direction, confidence, reason) tuples
            
        Returns:
            Aggregated scaling action
        """
        if not signals:
            return None
        
        # Separate up and down signals
        up_signals = [(conf, reason) for direction, conf, reason in signals if direction == ScalingDirection.UP]
        down_signals = [(conf, reason) for direction, conf, reason in signals if direction == ScalingDirection.DOWN]
        
        # Calculate weighted votes
        up_vote = sum(conf for conf, _ in up_signals)
        down_vote = sum(conf for conf, _ in down_signals)
        
        # Determine final direction
        if up_vote > down_vote and up_vote > 0.5:
            direction = ScalingDirection.UP
            confidence = min(1.0, up_vote)
            reasons = [reason for _, reason in up_signals]
        elif down_vote > up_vote and down_vote > 0.5:
            direction = ScalingDirection.DOWN
            confidence = min(1.0, down_vote)
            reasons = [reason for _, reason in down_signals]
        else:
            return None
        
        # Check capacity limits
        if direction == ScalingDirection.UP and self.current_capacity >= self.max_capacity:
            return None
        
        if direction == ScalingDirection.DOWN and self.current_capacity <= self.min_capacity:
            return None
        
        return ScalingAction(
            direction=direction,
            factor=self.scaling_factor,
            reason="; ".join(reasons),
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _execute_scaling(self, action: ScalingAction):
        """Execute scaling action.
        
        Args:
            action: Scaling action to execute
        """
        with self._lock:
            # Calculate new capacity
            if action.direction == ScalingDirection.UP:
                new_capacity = min(self.max_capacity, int(self.current_capacity * action.factor))
            else:
                new_capacity = max(self.min_capacity, int(self.current_capacity / action.factor))
            
            if new_capacity == self.current_capacity:
                return
            
            # Update state
            old_capacity = self.current_capacity
            self.current_capacity = new_capacity
            self.last_scaling_time = time.time()
            
            # Add to history
            action.parameters = {
                'old_capacity': old_capacity,
                'new_capacity': new_capacity
            }
            self.scaling_history.append(action)
            
            # Keep history limited
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]
        
        # Execute callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(action)
            except Exception as e:
                print(f"Error in scaling callback: {e}")
        
        print(f"Auto-scaled {action.direction.value}: {old_capacity} -> {new_capacity} ({action.reason})")
    
    def manual_scale(self, target_capacity: int, reason: str = "Manual scaling"):
        """Manually scale to target capacity.
        
        Args:
            target_capacity: Target capacity
            reason: Reason for scaling
        """
        target_capacity = max(self.min_capacity, min(self.max_capacity, target_capacity))
        
        if target_capacity == self.current_capacity:
            return
        
        direction = ScalingDirection.UP if target_capacity > self.current_capacity else ScalingDirection.DOWN
        factor = target_capacity / self.current_capacity
        
        action = ScalingAction(
            direction=direction,
            factor=factor,
            reason=reason,
            confidence=1.0,
            timestamp=time.time()
        )
        
        self._execute_scaling(action)
    
    def get_current_capacity(self) -> int:
        """Get current capacity."""
        return self.current_capacity
    
    def get_scaling_history(self, hours: float = 24.0) -> List[ScalingAction]:
        """Get recent scaling history.
        
        Args:
            hours: Number of hours to include
            
        Returns:
            List of recent scaling actions
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [
                action for action in self.scaling_history
                if action.timestamp > cutoff_time
            ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring."""
        current_time = time.time()
        
        # Get metrics for all rules
        metrics = {}
        for rule_name, rule in self.scaling_rules.items():
            metric_name = rule.trigger.value
            stats = self.metrics_collector.get_metric_statistics(metric_name, 300.0)
            metrics[metric_name] = stats
        
        # Recent scaling actions
        recent_actions = self.get_scaling_history(1.0)  # Last hour
        
        return {
            'current_capacity': self.current_capacity,
            'capacity_limits': {
                'min': self.min_capacity,
                'max': self.max_capacity
            },
            'last_scaling': {
                'timestamp': self.last_scaling_time,
                'seconds_ago': current_time - self.last_scaling_time
            },
            'recent_actions_count': len(recent_actions),
            'metrics': metrics,
            'rules_enabled': sum(1 for rule in self.scaling_rules.values() if rule.enabled),
            'cooldown_remaining': max(0, self.cooldown_seconds - (current_time - self.last_scaling_time))
        }
    
    def configure_rule(self, rule_name: str, **kwargs):
        """Configure scaling rule parameters.
        
        Args:
            rule_name: Name of the rule to configure
            **kwargs: Rule parameters to update
        """
        with self._lock:
            if rule_name in self.scaling_rules:
                rule = self.scaling_rules[rule_name]
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
    
    def enable_rule(self, rule_name: str):
        """Enable scaling rule."""
        self.configure_rule(rule_name, enabled=True)
    
    def disable_rule(self, rule_name: str):
        """Disable scaling rule."""
        self.configure_rule(rule_name, enabled=False)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


class NeuromorphicAutoScaler(AutoScaler):
    """Specialized auto-scaler for neuromorphic workloads."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add neuromorphic-specific rules
        self._register_neuromorphic_rules()
    
    def _register_neuromorphic_rules(self):
        """Register neuromorphic-specific scaling rules."""
        
        # Spike processing rate
        self.add_scaling_rule(
            "spike_rate",
            ScalingRule(
                trigger=ScalingTrigger.CUSTOM,
                threshold_up=1000.0,  # spikes/second
                threshold_down=200.0,
                cooldown_seconds=120.0,
                window_size=4,
                weight=1.3
            )
        )
        
        # Neural network accuracy
        self.add_scaling_rule(
            "accuracy_degradation",
            ScalingRule(
                trigger=ScalingTrigger.CUSTOM,
                threshold_up=0.95,  # Scale up if accuracy drops below 95%
                threshold_down=0.99,  # Scale down if accuracy above 99%
                cooldown_seconds=600.0,  # Longer cooldown for accuracy-based scaling
                window_size=10,
                weight=2.0  # High priority for accuracy
            )
        )
        
        # Inference queue length
        self.add_scaling_rule(
            "inference_queue",
            ScalingRule(
                trigger=ScalingTrigger.QUEUE_LENGTH,
                threshold_up=50,   # requests in queue
                threshold_down=5,
                cooldown_seconds=90.0,
                window_size=3,
                weight=1.1
            )
        )
        
        # Energy efficiency
        self.add_scaling_rule(
            "energy_efficiency",
            ScalingRule(
                trigger=ScalingTrigger.CUSTOM,
                threshold_up=0.7,  # Scale up if efficiency drops below 70%
                threshold_down=0.9,  # Scale down if efficiency above 90%
                cooldown_seconds=300.0,
                window_size=8,
                weight=0.8
            )
        )
    
    def record_spike_metrics(self, spike_rate: float, processing_latency: float, sparsity: float):
        """Record spike processing metrics.
        
        Args:
            spike_rate: Spikes processed per second
            processing_latency: Average processing latency (ms)
            sparsity: Spike sparsity (0-1)
        """
        self.record_metric("spike_rate", spike_rate)
        self.record_metric("latency", processing_latency)
        self.record_metric("sparsity", sparsity)
    
    def record_inference_metrics(self, accuracy: float, queue_length: int, energy_per_inference: float):
        """Record inference metrics.
        
        Args:
            accuracy: Model accuracy (0-1)
            queue_length: Number of requests in queue
            energy_per_inference: Energy consumption per inference (mJ)
        """
        # For accuracy-based scaling, we invert the metric (scale up when accuracy is LOW)
        self.record_metric("accuracy_degradation", 1.0 - accuracy)
        self.record_metric("queue_length", queue_length)
        
        # Calculate energy efficiency (higher is better)
        if energy_per_inference > 0:
            efficiency = min(1.0, 10.0 / energy_per_inference)  # Normalize to 0-1
            self.record_metric("energy_efficiency", efficiency)
    
    def get_neuromorphic_recommendations(self) -> List[str]:
        """Get scaling recommendations specific to neuromorphic workloads.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        current_time = time.time()
        
        # Analyze recent metrics
        spike_stats = self.metrics_collector.get_metric_statistics("spike_rate", 300.0)
        latency_stats = self.metrics_collector.get_metric_statistics("latency", 300.0)
        accuracy_stats = self.metrics_collector.get_metric_statistics("accuracy_degradation", 600.0)
        
        if spike_stats and spike_stats['mean'] > 800:
            recommendations.append("High spike processing load detected - consider scaling up or optimizing spike processing pipeline")
        
        if latency_stats and latency_stats['p95'] > 100:
            recommendations.append("High latency detected - scaling up could improve response times")
        
        if accuracy_stats and accuracy_stats['mean'] > 0.1:  # Accuracy drop > 10%
            recommendations.append("Accuracy degradation detected - scale up to maintain model performance")
        
        if self.current_capacity > self.min_capacity:
            recent_actions = self.get_scaling_history(1.0)
            up_actions = [a for a in recent_actions if a.direction == ScalingDirection.UP]
            if len(up_actions) > 2:
                recommendations.append("Frequent scaling up - consider increasing base capacity or optimizing workload")
        
        return recommendations if recommendations else ["System operating within normal parameters"]
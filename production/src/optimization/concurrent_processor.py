"""Concurrent processing system for neuromorphic computations."""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import threading
import queue
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
from collections import deque
import weakref


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ProcessingMode(Enum):
    """Processing modes for different workload types."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"


@dataclass
class Task:
    """Task representation for concurrent processing."""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    callback: Optional[Callable] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    memory_used: float = 0.0
    completed_at: float = field(default_factory=time.time)


class WorkerPool:
    """Pool of workers for concurrent task execution."""
    
    def __init__(
        self,
        num_workers: int,
        worker_type: str = "thread",
        max_queue_size: int = 1000,
        worker_timeout: float = 300.0
    ):
        """Initialize worker pool.
        
        Args:
            num_workers: Number of worker threads/processes
            worker_type: Type of workers ('thread' or 'process')
            max_queue_size: Maximum task queue size
            worker_timeout: Worker timeout in seconds
        """
        self.num_workers = num_workers
        self.worker_type = worker_type
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(), 
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue()
        }
        
        # Results and worker management
        self.result_queue = queue.Queue()
        self.workers = []
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'throughput_tasks_per_second': 0.0
        }
        
        # Control flags
        self.shutdown_flag = threading.Event()
        self.paused = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize workers
        self._initialize_workers()
        
        # Start result collector
        self.result_collector = threading.Thread(target=self._collect_results, daemon=True)
        self.result_collector.start()
    
    def _initialize_workers(self):
        """Initialize worker threads or processes."""
        if self.worker_type == "thread":
            for i in range(self.num_workers):
                worker = threading.Thread(
                    target=self._worker_thread,
                    args=(f"worker_{i}",),
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
        
        elif self.worker_type == "process":
            # For process-based workers, use multiprocessing
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            )
        
        else:
            raise ValueError(f"Unknown worker type: {self.worker_type}")
    
    def _worker_thread(self, worker_id: str):
        """Worker thread main loop."""
        while not self.shutdown_flag.is_set():
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Get next task from highest priority queue
                task = self._get_next_task()
                
                if task is None:
                    time.sleep(0.01)  # Brief sleep if no tasks
                    continue
                
                # Execute task
                self._execute_task(task, worker_id)
                
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def _get_next_task(self) -> Optional[Task]:
        """Get the next task from priority queues."""
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            try:
                # Use timeout to avoid blocking indefinitely
                _, task = self.task_queues[priority].get(timeout=0.01)
                return task
            except queue.Empty:
                continue
        
        return None
    
    def _execute_task(self, task: Task, worker_id: str):
        """Execute a task and handle results."""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Mark task as active
            with self.lock:
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'worker_id': worker_id,
                    'start_time': start_time
                }
            
            # Execute with timeout if specified
            if task.timeout:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - initial_memory
            
            # Create success result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id,
                memory_used=memory_used
            )
            
            # Execute callback if provided
            if task.callback:
                try:
                    task.callback(task_result)
                except Exception as callback_error:
                    logging.warning(f"Callback error for task {task.task_id}: {callback_error}")
            
            # Queue result
            self.result_queue.put(task_result)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logging.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                
                # Re-queue with exponential backoff
                delay = min(2 ** task.retry_count, 60)  # Max 60 second delay
                threading.Timer(delay, lambda: self.submit_task(task)).start()
            else:
                # Create failure result
                task_result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=e,
                    execution_time=execution_time,
                    worker_id=worker_id
                )
                
                self.result_queue.put(task_result)
        
        finally:
            # Remove from active tasks
            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    def _collect_results(self):
        """Collect and process task results."""
        while not self.shutdown_flag.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                
                with self.lock:
                    # Store result
                    self.completed_tasks[result.task_id] = result
                    
                    # Update statistics
                    if result.success:
                        self.stats['tasks_completed'] += 1
                    else:
                        self.stats['tasks_failed'] += 1
                    
                    self.stats['total_execution_time'] += result.execution_time
                    
                    total_tasks = self.stats['tasks_completed'] + self.stats['tasks_failed']
                    if total_tasks > 0:
                        self.stats['average_execution_time'] = (
                            self.stats['total_execution_time'] / total_tasks
                        )
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Result collector error: {e}")
    
    def submit_task(
        self,
        function: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution.
        
        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            priority: Task priority
            timeout: Execution timeout
            callback: Completion callback
            context: Additional context
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}_{id(function)}"
        
        task = Task(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            callback=callback,
            context=context
        )
        
        # Add to appropriate priority queue
        priority_value = priority.value
        self.task_queues[priority].put((priority_value, task))
        
        with self.lock:
            self.stats['tasks_submitted'] += 1
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result for a specific task.
        
        Args:
            task_id: Task ID
            timeout: Wait timeout
            
        Returns:
            TaskResult or None if not found/timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs
            timeout: Wait timeout
            
        Returns:
            List of TaskResults
        """
        start_time = time.time()
        results = {}
        
        while len(results) < len(task_ids):
            with self.lock:
                for task_id in task_ids:
                    if task_id in self.completed_tasks and task_id not in results:
                        results[task_id] = self.completed_tasks[task_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            time.sleep(0.01)
        
        return [results.get(task_id) for task_id in task_ids]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self.lock:
            stats = self.stats.copy()
            
            # Add current state information
            stats.update({
                'active_tasks': len(self.active_tasks),
                'queued_tasks': sum(q.qsize() for q in self.task_queues.values()),
                'num_workers': self.num_workers,
                'worker_type': self.worker_type,
                'paused': self.paused
            })
            
            # Calculate throughput
            if stats['total_execution_time'] > 0:
                stats['throughput_tasks_per_second'] = (
                    stats['tasks_completed'] / stats['total_execution_time']
                )
        
        return stats
    
    def pause(self):
        """Pause task execution."""
        self.paused = True
    
    def resume(self):
        """Resume task execution."""
        self.paused = False
    
    def shutdown(self, wait_for_completion: bool = True, timeout: float = 30.0):
        """Shutdown the worker pool.
        
        Args:
            wait_for_completion: Wait for active tasks to complete
            timeout: Shutdown timeout
        """
        self.shutdown_flag.set()
        
        if wait_for_completion:
            # Wait for active tasks to complete
            start_time = time.time()
            while self.active_tasks and (time.time() - start_time) < timeout:
                time.sleep(0.1)
        
        # Cleanup process pool if using processes
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)


class ConcurrentNeuromorphicProcessor:
    """High-level concurrent processor for neuromorphic computations."""
    
    def __init__(
        self,
        num_cpu_workers: Optional[int] = None,
        num_gpu_workers: int = 1,
        enable_gpu: bool = True,
        adaptive_scaling: bool = True,
        max_memory_usage_gb: float = 8.0
    ):
        """Initialize concurrent neuromorphic processor.
        
        Args:
            num_cpu_workers: Number of CPU workers (auto-detect if None)
            num_gpu_workers: Number of GPU workers
            enable_gpu: Enable GPU processing
            adaptive_scaling: Enable adaptive worker scaling
            max_memory_usage_gb: Maximum memory usage limit
        """
        # Auto-detect optimal number of CPU workers
        if num_cpu_workers is None:
            num_cpu_workers = max(1, psutil.cpu_count(logical=False))
        
        self.num_cpu_workers = num_cpu_workers
        self.num_gpu_workers = num_gpu_workers if enable_gpu and torch.cuda.is_available() else 0
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.adaptive_scaling = adaptive_scaling
        self.max_memory_usage_bytes = max_memory_usage_gb * 1024 * 1024 * 1024
        
        # Worker pools
        self.cpu_pool = WorkerPool(
            num_workers=num_cpu_workers,
            worker_type="thread"
        )
        
        if self.enable_gpu:
            self.gpu_pool = WorkerPool(
                num_workers=num_gpu_workers,
                worker_type="thread"  # GPU tasks often benefit from thread-based coordination
            )
        
        # Task routing and load balancing
        self.task_router = TaskRouter(self)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(self)
        
        # Adaptive scaling
        if self.adaptive_scaling:
            self.scaler = AdaptiveScaler(self)
    
    def process_spike_batch(
        self,
        spike_data: torch.Tensor,
        model: torch.nn.Module,
        batch_size: int = 32,
        use_gpu: bool = True,
        parallel_batches: bool = True
    ) -> List[torch.Tensor]:
        """Process spike data in parallel batches.
        
        Args:
            spike_data: Input spike data [total_samples, neurons, time_steps]
            model: Neural network model
            batch_size: Batch size for processing
            use_gpu: Use GPU if available
            parallel_batches: Process batches in parallel
            
        Returns:
            List of output tensors
        """
        total_samples = spike_data.size(0)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        if parallel_batches and num_batches > 1:
            # Submit parallel batch processing tasks
            task_ids = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                batch_data = spike_data[start_idx:end_idx]
                
                pool = self.gpu_pool if use_gpu and self.enable_gpu else self.cpu_pool
                
                task_id = pool.submit_task(
                    self._process_single_batch,
                    batch_data,
                    model,
                    use_gpu,
                    task_id=f"spike_batch_{i}",
                    priority=TaskPriority.HIGH
                )
                
                task_ids.append(task_id)
            
            # Collect results
            pool = self.gpu_pool if use_gpu and self.enable_gpu else self.cpu_pool
            results = pool.wait_for_completion(task_ids, timeout=300.0)
            
            # Extract successful results
            outputs = []
            for result in results:
                if result and result.success:
                    outputs.append(result.result)
            
            return outputs
        
        else:
            # Sequential processing
            outputs = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                batch_data = spike_data[start_idx:end_idx]
                
                output = self._process_single_batch(batch_data, model, use_gpu)
                outputs.append(output)
            
            return outputs
    
    def _process_single_batch(
        self,
        batch_data: torch.Tensor,
        model: torch.nn.Module,
        use_gpu: bool
    ) -> torch.Tensor:
        """Process a single batch of spike data."""
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Move data and model to device
        batch_data = batch_data.to(device)
        model = model.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(batch_data)
        
        return output.cpu()  # Move back to CPU for collection
    
    def parallel_parameter_search(
        self,
        search_function: Callable,
        parameter_grid: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """Perform parallel hyperparameter search.
        
        Args:
            search_function: Function to evaluate parameters
            parameter_grid: List of parameter combinations
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            List of (parameters, result) tuples
        """
        if max_concurrent is None:
            max_concurrent = self.num_cpu_workers + self.num_gpu_workers
        
        # Submit parameter evaluation tasks
        task_ids = []
        parameter_map = {}
        
        for i, params in enumerate(parameter_grid[:max_concurrent]):
            task_id = self.cpu_pool.submit_task(
                search_function,
                params,
                task_id=f"param_search_{i}",
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
            parameter_map[task_id] = params
        
        # Collect results
        results = self.cpu_pool.wait_for_completion(task_ids, timeout=1800.0)  # 30 minute timeout
        
        # Combine parameters and results
        search_results = []
        for task_id, result in zip(task_ids, results):
            if result and result.success:
                params = parameter_map[task_id]
                search_results.append((params, result.result))
        
        return search_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'cpu_workers': self.num_cpu_workers,
            'gpu_workers': self.num_gpu_workers,
            'gpu_available': self.enable_gpu,
            'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
        }
        
        # Add worker pool statistics
        status['cpu_pool_stats'] = self.cpu_pool.get_statistics()
        
        if self.enable_gpu:
            status['gpu_pool_stats'] = self.gpu_pool.get_statistics()
            
            # GPU memory info if available
            if torch.cuda.is_available():
                status['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                status['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return status
    
    def optimize_performance(self):
        """Optimize processor performance based on current workload."""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.optimize()
        
        if hasattr(self, 'scaler') and self.adaptive_scaling:
            self.scaler.scale_workers()
    
    def shutdown(self):
        """Shutdown all worker pools and cleanup."""
        self.cpu_pool.shutdown()
        
        if self.enable_gpu:
            self.gpu_pool.shutdown()
        
        if hasattr(self, 'scaler'):
            self.scaler.stop()
        
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop()


class TaskRouter:
    """Routes tasks to appropriate worker pools based on characteristics."""
    
    def __init__(self, processor: ConcurrentNeuromorphicProcessor):
        self.processor = processor
    
    def route_task(self, task: Task) -> WorkerPool:
        """Route task to appropriate worker pool."""
        # Simple routing logic - can be enhanced
        if hasattr(task, 'use_gpu') and task.use_gpu and self.processor.enable_gpu:
            return self.processor.gpu_pool
        else:
            return self.processor.cpu_pool


class PerformanceMonitor:
    """Monitor and optimize performance of concurrent processor."""
    
    def __init__(self, processor: ConcurrentNeuromorphicProcessor):
        self.processor = processor
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Monitor system resources and performance
                self._check_memory_usage()
                self._check_cpu_usage()
                self._analyze_task_patterns()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logging.error(f"Performance monitor error: {e}")
                time.sleep(5.0)
    
    def _check_memory_usage(self):
        """Check memory usage and take action if necessary."""
        memory_usage = psutil.virtual_memory().used
        
        if memory_usage > self.processor.max_memory_usage_bytes * 0.9:
            logging.warning("High memory usage detected, optimizing...")
            # Could trigger garbage collection, cache cleanup, etc.
    
    def _check_cpu_usage(self):
        """Check CPU usage and adjust worker allocation."""
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_usage > 90:
            logging.info("High CPU usage detected")
            # Could reduce concurrent tasks or adjust priorities
    
    def _analyze_task_patterns(self):
        """Analyze task execution patterns for optimization."""
        # Placeholder for pattern analysis
        pass
    
    def optimize(self):
        """Perform performance optimizations."""
        # Placeholder for optimization logic
        pass
    
    def stop(self):
        """Stop performance monitoring."""
        self.monitoring = False


class AdaptiveScaler:
    """Adaptive scaling of worker pools based on workload."""
    
    def __init__(self, processor: ConcurrentNeuromorphicProcessor):
        self.processor = processor
        self.scaling = True
        self.scale_thread = threading.Thread(target=self._scale_loop, daemon=True)
        self.scale_thread.start()
    
    def _scale_loop(self):
        """Main scaling loop."""
        while self.scaling:
            try:
                self.scale_workers()
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Adaptive scaler error: {e}")
                time.sleep(30.0)
    
    def scale_workers(self):
        """Scale worker pools based on current demand."""
        # Get current statistics
        cpu_stats = self.processor.cpu_pool.get_statistics()
        
        # Simple scaling logic
        if cpu_stats['queued_tasks'] > cpu_stats['active_tasks'] * 2:
            # High queue, consider scaling up
            logging.info("High task queue detected")
        elif cpu_stats['queued_tasks'] == 0 and cpu_stats['active_tasks'] < self.processor.num_cpu_workers // 2:
            # Low utilization, consider scaling down
            logging.info("Low utilization detected")
    
    def stop(self):
        """Stop adaptive scaling."""
        self.scaling = False
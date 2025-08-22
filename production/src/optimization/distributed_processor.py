"""Distributed processing system for neuromorphic computations."""

import threading
import multiprocessing as mp
import time
import queue
import pickle
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import torch
import numpy as np
from pathlib import Path
import logging
import socket
import json


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Distributed processing task."""
    task_id: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other):
        """Compare tasks for priority queue."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    retries_used: int = 0


class WorkerProcess:
    """Worker process for distributed computation."""
    
    def __init__(self, worker_id: str, task_queue: mp.Queue, result_queue: mp.Queue):
        """Initialize worker process.
        
        Args:
            worker_id: Unique worker identifier
            task_queue: Queue for receiving tasks
            result_queue: Queue for sending results
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.running = False
        
        # Function registry for distributed execution
        self.function_registry: Dict[str, Callable] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in neuromorphic functions."""
        
        # Basic tensor operations
        self.function_registry['tensor_multiply'] = self._tensor_multiply
        self.function_registry['spike_processing'] = self._spike_processing
        self.function_registry['neural_inference'] = self._neural_inference
        self.function_registry['stdp_learning'] = self._stdp_learning
        self.function_registry['pattern_matching'] = self._pattern_matching
    
    def register_function(self, name: str, function: Callable):
        """Register function for distributed execution.
        
        Args:
            name: Function name
            function: Function to register
        """
        self.function_registry[name] = function
    
    def run(self):
        """Main worker loop."""
        self.running = True
        
        while self.running:
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Execute task
                result = self._execute_task(task)
                
                # Send result
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error and continue
                print(f"Worker {self.worker_id} error: {e}")
                continue
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        start_time = time.time()
        
        try:
            # Check if function is registered
            if task.function_name not in self.function_registry:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            function = self.function_registry[task.function_name]
            
            # Execute with timeout if specified
            if task.timeout_seconds:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout_seconds)
            else:
                result = function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
        
        except concurrent.futures.TimeoutError:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="Task timeout",
                execution_time=time.time() - start_time,
                worker_id=self.worker_id
            )
        
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                worker_id=self.worker_id
            )
    
    def stop(self):
        """Stop worker."""
        self.running = False
    
    # Built-in neuromorphic functions
    def _tensor_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply two tensors."""
        return torch.matmul(a, b)
    
    def _spike_processing(self, spikes: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
        """Process spike data."""
        # Simulate spike processing
        spike_count = np.sum(spikes)
        weighted_spikes = np.dot(spikes.flatten(), weights.flatten()) if weights.size == spikes.size else 0
        
        return {
            'spike_count': int(spike_count),
            'weighted_sum': float(weighted_spikes),
            'sparsity': float(1.0 - spike_count / spikes.size),
            'processing_time_ms': time.time() * 1000 % 100  # Simulated processing time
        }
    
    def _neural_inference(self, input_data: np.ndarray, model_params: Dict) -> Dict[str, Any]:
        """Perform neural network inference."""
        # Simulate inference
        batch_size = input_data.shape[0] if len(input_data.shape) > 0 else 1
        num_classes = model_params.get('num_classes', 10)
        
        # Generate fake predictions
        predictions = np.random.rand(batch_size, num_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        return {
            'predictions': predictions.tolist(),
            'confidence': float(np.max(predictions, axis=1).mean()),
            'batch_size': batch_size,
            'inference_time_ms': time.time() * 1000 % 50
        }
    
    def _stdp_learning(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """STDP learning rule."""
        # Simplified STDP
        tau_plus, tau_minus = 20.0, 20.0
        a_plus, a_minus = 0.01, 0.01
        
        # Calculate weight changes (simplified)
        pre_sum = np.sum(pre_spikes, axis=-1, keepdims=True)
        post_sum = np.sum(post_spikes, axis=-1, keepdims=True)
        
        # Outer product for weight updates
        dw = a_plus * np.outer(pre_sum, post_sum) - a_minus * np.outer(post_sum, pre_sum)
        
        # Apply updates
        new_weights = weights + dw
        return np.clip(new_weights, -1.0, 1.0)
    
    def _pattern_matching(self, input_pattern: np.ndarray, templates: List[np.ndarray]) -> Dict[str, Any]:
        """Pattern matching against templates."""
        similarities = []
        
        for i, template in enumerate(templates):
            if template.shape == input_pattern.shape:
                # Calculate normalized cross-correlation
                correlation = np.corrcoef(input_pattern.flatten(), template.flatten())[0, 1]
                similarities.append(correlation if not np.isnan(correlation) else 0.0)
            else:
                similarities.append(0.0)
        
        best_match = int(np.argmax(similarities))
        
        return {
            'best_match_index': best_match,
            'best_similarity': float(similarities[best_match]),
            'all_similarities': similarities,
            'num_templates': len(templates)
        }


class DistributedProcessor:
    """Main distributed processing coordinator."""
    
    def __init__(
        self,
        num_workers: int = None,
        max_queue_size: int = 1000,
        result_timeout: float = 300.0,
        enable_persistence: bool = False,
        persistence_dir: Optional[str] = None
    ):
        """Initialize distributed processor.
        
        Args:
            num_workers: Number of worker processes (defaults to CPU count)
            max_queue_size: Maximum queue size
            result_timeout: Timeout for result collection
            enable_persistence: Enable task persistence
            persistence_dir: Directory for persistent storage
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.max_queue_size = max_queue_size
        self.result_timeout = result_timeout
        self.enable_persistence = enable_persistence
        self.persistence_dir = Path(persistence_dir) if persistence_dir else Path('./distributed_tasks')
        
        # Queues and processes
        self.task_queue = mp.Queue(maxsize=max_queue_size)
        self.result_queue = mp.Queue()
        self.workers: List[mp.Process] = []
        
        # Task management
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_counter = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Result collection thread
        self.result_collector_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Function registry
        self.function_registry: Dict[str, Callable] = {}
        
        # Setup persistence
        if self.enable_persistence:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the distributed processor."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker processes
        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            worker = mp.Process(
                target=self._worker_main,
                args=(worker_id, self.task_queue, self.result_queue),
                name=worker_id
            )
            worker.start()
            self.workers.append(worker)
        
        # Start result collector
        self.result_collector_thread = threading.Thread(
            target=self._result_collector,
            name="ResultCollector",
            daemon=True
        )
        self.result_collector_thread.start()
        
        print(f"Distributed processor started with {self.num_workers} workers")
    
    def stop(self):
        """Stop the distributed processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Send shutdown signal to workers
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()
        print("Distributed processor stopped")
    
    def _worker_main(self, worker_id: str, task_queue: mp.Queue, result_queue: mp.Queue):
        """Main function for worker processes."""
        worker = WorkerProcess(worker_id, task_queue, result_queue)
        
        # Register functions from main process
        for name, func in self.function_registry.items():
            worker.register_function(name, func)
        
        worker.run()
    
    def _result_collector(self):
        """Collect results from workers."""
        while self.running:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                with self.lock:
                    # Update task tracking
                    if result.task_id in self.pending_tasks:
                        del self.pending_tasks[result.task_id]
                    
                    self.completed_tasks[result.task_id] = result
                    
                    # Update statistics
                    self.stats['tasks_completed'] += 1
                    self.stats['total_execution_time'] += result.execution_time
                    
                    if result.status == TaskStatus.FAILED:
                        self.stats['tasks_failed'] += 1
                    
                    # Calculate average execution time
                    if self.stats['tasks_completed'] > 0:
                        self.stats['average_execution_time'] = (
                            self.stats['total_execution_time'] / self.stats['tasks_completed']
                        )
                    
                    # Persist result if enabled
                    if self.enable_persistence:
                        self._persist_result(result)
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in result collector: {e}")
    
    def register_function(self, name: str, function: Callable):
        """Register function for distributed execution.
        
        Args:
            name: Function name
            function: Function to register
        """
        self.function_registry[name] = function
    
    def submit_task(
        self,
        function_name: str,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Submit task for distributed execution.
        
        Args:
            function_name: Name of function to execute
            *args: Function arguments
            priority: Task priority
            timeout_seconds: Task timeout
            max_retries: Maximum retry attempts
            metadata: Optional task metadata
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Distributed processor not started")
        
        with self.lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time() * 1000)}"
        
        task = Task(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata
        )
        
        with self.lock:
            self.pending_tasks[task_id] = task
            self.stats['tasks_submitted'] += 1
        
        # Submit to queue
        try:
            self.task_queue.put(task, timeout=5.0)
        except queue.Full:
            with self.lock:
                del self.pending_tasks[task_id]
                self.stats['tasks_submitted'] -= 1
            raise RuntimeError("Task queue is full")
        
        # Persist task if enabled
        if self.enable_persistence:
            self._persist_task(task)
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get task result.
        
        Args:
            task_id: Task ID
            timeout: Timeout for waiting
            
        Returns:
            Task result or None if not available
        """
        timeout = timeout or self.result_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
            
            time.sleep(0.1)
        
        return None
    
    def wait_for_results(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for multiple task results.
        
        Args:
            task_ids: List of task IDs
            timeout: Timeout for waiting
            
        Returns:
            Dictionary of task results
        """
        timeout = timeout or self.result_timeout
        start_time = time.time()
        results = {}
        
        while len(results) < len(task_ids) and time.time() - start_time < timeout:
            with self.lock:
                for task_id in task_ids:
                    if task_id not in results and task_id in self.completed_tasks:
                        results[task_id] = self.completed_tasks[task_id]
            
            if len(results) < len(task_ids):
                time.sleep(0.1)
        
        return results
    
    def submit_batch(
        self,
        function_name: str,
        args_list: List[tuple],
        kwargs_list: Optional[List[dict]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **common_kwargs
    ) -> List[str]:
        """Submit batch of tasks.
        
        Args:
            function_name: Function name
            args_list: List of argument tuples
            kwargs_list: Optional list of keyword arguments
            priority: Task priority
            **common_kwargs: Common keyword arguments for all tasks
            
        Returns:
            List of task IDs
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(args_list)
        
        if len(args_list) != len(kwargs_list):
            raise ValueError("args_list and kwargs_list must have same length")
        
        task_ids = []
        
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
            # Merge common kwargs with task-specific kwargs
            merged_kwargs = {**common_kwargs, **kwargs}
            
            task_id = self.submit_task(
                function_name,
                *args,
                priority=priority,
                metadata={'batch_index': i},
                **merged_kwargs
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_batch_results(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Get results for batch of tasks in order.
        
        Args:
            task_ids: List of task IDs
            timeout: Timeout for waiting
            
        Returns:
            List of task results in same order as task_ids
        """
        results_dict = self.wait_for_results(task_ids, timeout)
        
        # Return results in original order
        return [results_dict.get(task_id) for task_id in task_ids]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled
        """
        with self.lock:
            if task_id in self.pending_tasks:
                # Mark as cancelled (workers will see this)
                self.completed_tasks[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                    timestamp=time.time()
                )
                del self.pending_tasks[task_id]
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status.
        
        Returns:
            Status information
        """
        with self.lock:
            return {
                'running': self.running,
                'num_workers': len([w for w in self.workers if w.is_alive()]),
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
                'statistics': self.stats.copy(),
                'registered_functions': list(self.function_registry.keys())
            }
    
    def clear_completed_tasks(self, older_than_hours: float = 24.0):
        """Clear old completed tasks.
        
        Args:
            older_than_hours: Clear tasks older than this many hours
        """
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        with self.lock:
            tasks_to_remove = [
                task_id for task_id, result in self.completed_tasks.items()
                if result.timestamp < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
        
        print(f"Cleared {len(tasks_to_remove)} old completed tasks")
    
    def _persist_task(self, task: Task):
        """Persist task to storage."""
        if not self.enable_persistence:
            return
        
        try:
            task_file = self.persistence_dir / f"task_{task.task_id}.pkl"
            with open(task_file, 'wb') as f:
                pickle.dump(task, f)
        except Exception as e:
            print(f"Failed to persist task {task.task_id}: {e}")
    
    def _persist_result(self, result: TaskResult):
        """Persist result to storage."""
        if not self.enable_persistence:
            return
        
        try:
            result_file = self.persistence_dir / f"result_{result.task_id}.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Failed to persist result {result.task_id}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# High-level convenience functions
def distribute_spike_processing(
    spike_data: List[np.ndarray],
    weights: List[np.ndarray],
    processor: DistributedProcessor
) -> List[Dict[str, Any]]:
    """Distribute spike processing across workers.
    
    Args:
        spike_data: List of spike arrays
        weights: List of weight arrays
        processor: Distributed processor instance
        
    Returns:
        List of processing results
    """
    # Prepare arguments
    args_list = [(spikes, weights) for spikes, weights in zip(spike_data, weights)]
    
    # Submit batch
    task_ids = processor.submit_batch('spike_processing', args_list, priority=TaskPriority.HIGH)
    
    # Get results
    results = processor.get_batch_results(task_ids, timeout=30.0)
    
    # Extract successful results
    return [result.result for result in results if result and result.status == TaskStatus.COMPLETED]


def distribute_neural_inference(
    input_batches: List[np.ndarray],
    model_params: Dict,
    processor: DistributedProcessor
) -> List[Dict[str, Any]]:
    """Distribute neural inference across workers.
    
    Args:
        input_batches: List of input arrays
        model_params: Model parameters
        processor: Distributed processor instance
        
    Returns:
        List of inference results
    """
    # Prepare arguments
    args_list = [(batch, model_params) for batch in input_batches]
    
    # Submit batch
    task_ids = processor.submit_batch('neural_inference', args_list, priority=TaskPriority.HIGH)
    
    # Get results
    results = processor.get_batch_results(task_ids, timeout=60.0)
    
    # Extract successful results
    return [result.result for result in results if result and result.status == TaskStatus.COMPLETED]
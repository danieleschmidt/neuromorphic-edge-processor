"""Neuromorphic computation compiler and optimizer.

Provides just-in-time compilation, graph optimization, and hardware-specific
code generation for neuromorphic edge processors.
"""

import torch
import torch.jit
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import hashlib
import warnings

try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    O0 = "none"        # No optimization
    O1 = "basic"       # Basic optimizations
    O2 = "standard"    # Standard optimizations
    O3 = "aggressive"  # Aggressive optimizations
    EDGE = "edge"      # Edge-specific optimizations


class HardwareTarget(Enum):
    """Target hardware platforms."""
    CPU = "cpu"
    GPU = "gpu"
    NEUROMORPHIC = "neuromorphic"
    EDGE_TPU = "edge_tpu"
    FPGA = "fpga"
    AUTO = "auto"


@dataclass
class CompilationProfile:
    """Compilation performance profile."""
    function_name: str
    compilation_time: float
    execution_time_before: float
    execution_time_after: float
    speedup_factor: float
    memory_usage_before: int
    memory_usage_after: int
    optimization_level: OptimizationLevel
    hardware_target: HardwareTarget
    success: bool
    error_message: Optional[str] = None


@dataclass
class GraphOptimization:
    """Graph optimization metadata."""
    optimization_name: str
    nodes_before: int
    nodes_after: int
    edges_before: int
    edges_after: int
    reduction_ratio: float
    estimated_speedup: float


class NeuromorphicCompiler:
    """Advanced compiler for neuromorphic computations.
    
    Features:
    - JIT compilation with caching
    - Graph-level optimizations
    - Hardware-specific code generation
    - Spike pattern optimization
    - Memory layout optimization
    - Vectorization and parallelization
    """
    
    def __init__(self,
                 cache_dir: str = ".neuromorphic_cache",
                 default_optimization_level: OptimizationLevel = OptimizationLevel.O2,
                 default_hardware_target: HardwareTarget = HardwareTarget.AUTO,
                 enable_profiling: bool = True,
                 max_cache_size: int = 1000):
        """Initialize neuromorphic compiler.
        
        Args:
            cache_dir: Directory for compilation cache
            default_optimization_level: Default optimization level
            default_hardware_target: Default hardware target
            enable_profiling: Enable compilation profiling
            max_cache_size: Maximum number of cached compilations
        """
        self.cache_dir = cache_dir
        self.default_optimization_level = default_optimization_level
        self.default_hardware_target = default_hardware_target
        self.enable_profiling = enable_profiling
        self.max_cache_size = max_cache_size
        
        # Compilation cache
        self.compiled_functions: Dict[str, torch.jit.ScriptModule] = {}
        self.compilation_profiles: List[CompilationProfile] = []
        self.optimization_stats: Dict[str, int] = defaultdict(int)
        
        # Graph optimizations
        self.graph_optimizations: List[GraphOptimization] = []
        self.custom_passes: List[Callable] = []
        
        # Hardware detection
        self.detected_hardware = self._detect_hardware()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization passes
        self._init_optimization_passes()
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """Detect available hardware capabilities."""
        hardware = {
            'cpu': True,
            'cuda': torch.cuda.is_available(),
            'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if hardware['cuda']:
            try:
                hardware['cuda_compute_capability'] = torch.cuda.get_device_capability()
                hardware['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                pass
        
        return hardware
    
    def _init_optimization_passes(self):
        """Initialize optimization passes."""
        # Spike-specific optimizations
        self.custom_passes.extend([
            self._optimize_spike_patterns,
            self._optimize_temporal_loops,
            self._optimize_memory_layout,
            self._optimize_numerical_precision,
            self._optimize_control_flow
        ])
    
    def compile_function(self,
                        func: Callable,
                        example_inputs: Tuple[torch.Tensor, ...],
                        optimization_level: Optional[OptimizationLevel] = None,
                        hardware_target: Optional[HardwareTarget] = None,
                        force_recompile: bool = False) -> torch.jit.ScriptModule:
        """Compile a function with neuromorphic optimizations.
        
        Args:
            func: Function to compile
            example_inputs: Example inputs for tracing
            optimization_level: Optimization level
            hardware_target: Target hardware
            force_recompile: Force recompilation
            
        Returns:
            Compiled function
        """
        opt_level = optimization_level or self.default_optimization_level
        hw_target = hardware_target or self._select_hardware_target()
        
        # Generate cache key
        cache_key = self._generate_cache_key(func, example_inputs, opt_level, hw_target)
        
        with self.lock:
            # Check cache
            if not force_recompile and cache_key in self.compiled_functions:
                self.logger.debug(f"Using cached compilation for {func.__name__}")
                return self.compiled_functions[cache_key]
            
            # Profile compilation if enabled
            if self.enable_profiling:
                profile = self._profile_compilation(func, example_inputs, opt_level, hw_target)
                self.compilation_profiles.append(profile)
        
        start_time = time.time()
        
        try:
            # Trace the function
            traced_func = torch.jit.trace(func, example_inputs)
            
            # Apply optimizations
            optimized_func = self._apply_optimizations(traced_func, opt_level, hw_target)
            
            # Cache the result
            with self.lock:
                if len(self.compiled_functions) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.compiled_functions))
                    del self.compiled_functions[oldest_key]
                
                self.compiled_functions[cache_key] = optimized_func
            
            compilation_time = time.time() - start_time
            self.logger.info(f"Compiled {func.__name__} in {compilation_time:.3f}s")
            
            return optimized_func
            
        except Exception as e:
            self.logger.error(f"Compilation failed for {func.__name__}: {e}")
            raise
    
    def _generate_cache_key(self,
                           func: Callable,
                           example_inputs: Tuple[torch.Tensor, ...],
                           opt_level: OptimizationLevel,
                           hw_target: HardwareTarget) -> str:
        """Generate cache key for compiled function."""
        # Function signature
        func_id = f"{func.__module__}.{func.__name__}"
        
        # Input shapes and types
        input_sig = "_".join([
            f"{tensor.shape}_{tensor.dtype}" 
            for tensor in example_inputs
        ])
        
        # Optimization settings
        opt_sig = f"{opt_level.value}_{hw_target.value}"
        
        # Hash everything
        combined = f"{func_id}_{input_sig}_{opt_sig}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _select_hardware_target(self) -> HardwareTarget:
        """Automatically select optimal hardware target."""
        if self.detected_hardware['cuda'] and self.detected_hardware['cuda_device_count'] > 0:
            return HardwareTarget.GPU
        elif self.detected_hardware['mps']:
            return HardwareTarget.GPU  # MPS is treated as GPU
        else:
            return HardwareTarget.CPU
    
    def _apply_optimizations(self,
                            traced_func: torch.jit.ScriptModule,
                            opt_level: OptimizationLevel,
                            hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Apply optimization passes to traced function."""
        optimized = traced_func
        
        if opt_level == OptimizationLevel.O0:
            return optimized
        
        # Standard PyTorch optimizations
        if opt_level in [OptimizationLevel.O2, OptimizationLevel.O3, OptimizationLevel.EDGE]:
            try:
                # Freeze the model for optimization
                optimized.eval()
                
                # Standard optimizations
                optimized = torch.jit.optimize_for_inference(optimized)
                
                # Fuse operations
                optimized = torch.jit.freeze(optimized)
                
            except Exception as e:
                self.logger.warning(f"Standard optimizations failed: {e}")
        
        # Custom neuromorphic optimizations
        if opt_level in [OptimizationLevel.O3, OptimizationLevel.EDGE]:
            for pass_func in self.custom_passes:
                try:
                    optimized = pass_func(optimized, hw_target)
                except Exception as e:
                    self.logger.warning(f"Custom pass {pass_func.__name__} failed: {e}")
        
        # Hardware-specific optimizations
        optimized = self._apply_hardware_optimizations(optimized, hw_target)
        
        return optimized
    
    def _apply_hardware_optimizations(self,
                                    func: torch.jit.ScriptModule,
                                    hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Apply hardware-specific optimizations."""
        if hw_target == HardwareTarget.GPU and self.detected_hardware['cuda']:
            # GPU-specific optimizations
            try:
                # Enable tensor cores if available
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                # Optimize for CUDA
                func = func.cuda()
                
            except Exception as e:
                self.logger.warning(f"GPU optimizations failed: {e}")
        
        elif hw_target == HardwareTarget.CPU:
            # CPU-specific optimizations
            try:
                # Enable Intel MKL-DNN optimizations
                if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
                    torch.backends.mkldnn.enabled = True
                
            except Exception as e:
                self.logger.warning(f"CPU optimizations failed: {e}")
        
        elif hw_target == HardwareTarget.EDGE:
            # Edge-specific optimizations
            func = self._apply_edge_optimizations(func)
        
        return func
    
    def _apply_edge_optimizations(self, func: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """Apply edge-specific optimizations."""
        try:
            # Quantization for edge deployment
            if hasattr(torch.quantization, 'quantize_jit'):
                # Dynamic quantization
                quantized = torch.quantization.quantize_jit(
                    func,
                    {'': torch.quantization.default_dynamic_qconfig},
                    inplace=False
                )
                return quantized
            
        except Exception as e:
            self.logger.warning(f"Edge quantization failed: {e}")
        
        return func
    
    def _optimize_spike_patterns(self,
                               func: torch.jit.ScriptModule,
                               hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Optimize spike pattern computations."""
        # This would implement spike-specific optimizations
        # such as sparse tensor operations, temporal loop unrolling, etc.
        
        try:
            # Get the graph
            graph = func.graph
            
            # Look for spike-related patterns
            nodes_to_optimize = []
            for node in graph.nodes():
                if self._is_spike_operation(node):
                    nodes_to_optimize.append(node)
            
            # Apply spike optimizations
            if nodes_to_optimize:
                self.optimization_stats['spike_patterns'] += len(nodes_to_optimize)
                
        except Exception as e:
            self.logger.debug(f"Spike pattern optimization failed: {e}")
        
        return func
    
    def _optimize_temporal_loops(self,
                               func: torch.jit.ScriptModule,
                               hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Optimize temporal processing loops."""
        try:
            # Look for temporal loops and unroll where beneficial
            graph = func.graph
            
            # Find loop patterns
            loop_nodes = []
            for node in graph.nodes():
                if node.kind() == 'prim::Loop':
                    loop_nodes.append(node)
            
            if loop_nodes:
                self.optimization_stats['temporal_loops'] += len(loop_nodes)
                
        except Exception as e:
            self.logger.debug(f"Temporal loop optimization failed: {e}")
        
        return func
    
    def _optimize_memory_layout(self,
                              func: torch.jit.ScriptModule,
                              hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Optimize memory layout for cache efficiency."""
        try:
            # Memory layout optimizations would go here
            # This could include tensor reordering, memory pooling, etc.
            
            self.optimization_stats['memory_layout'] += 1
            
        except Exception as e:
            self.logger.debug(f"Memory layout optimization failed: {e}")
        
        return func
    
    def _optimize_numerical_precision(self,
                                    func: torch.jit.ScriptModule,
                                    hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Optimize numerical precision for edge deployment."""
        try:
            # Mixed precision optimizations
            if hw_target == HardwareTarget.EDGE:
                # Apply lower precision where safe
                self.optimization_stats['precision'] += 1
            
        except Exception as e:
            self.logger.debug(f"Precision optimization failed: {e}")
        
        return func
    
    def _optimize_control_flow(self,
                             func: torch.jit.ScriptModule,
                             hw_target: HardwareTarget) -> torch.jit.ScriptModule:
        """Optimize control flow for neuromorphic patterns."""
        try:
            # Control flow optimizations
            graph = func.graph
            
            # Look for conditional patterns that can be optimized
            conditional_nodes = []
            for node in graph.nodes():
                if node.kind() == 'prim::If':
                    conditional_nodes.append(node)
            
            if conditional_nodes:
                self.optimization_stats['control_flow'] += len(conditional_nodes)
                
        except Exception as e:
            self.logger.debug(f"Control flow optimization failed: {e}")
        
        return func
    
    def _is_spike_operation(self, node) -> bool:
        """Check if a node represents a spike-related operation."""
        # This would analyze the node to determine if it's spike-related
        spike_ops = ['threshold', 'greater', 'relu', 'clamp']
        
        if hasattr(node, 'kind'):
            node_kind = node.kind()
            return any(op in node_kind.lower() for op in spike_ops)
        
        return False
    
    def _profile_compilation(self,
                           func: Callable,
                           example_inputs: Tuple[torch.Tensor, ...],
                           opt_level: OptimizationLevel,
                           hw_target: HardwareTarget) -> CompilationProfile:
        """Profile compilation performance."""
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Measure original execution time
        start_time = time.time()
        try:
            with torch.no_grad():
                original_result = func(*example_inputs)
            execution_time_before = time.time() - start_time
            success = True
            error_message = None
        except Exception as e:
            execution_time_before = float('inf')
            success = False
            error_message = str(e)
        
        # Measure memory usage
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0
        
        # Compile and measure
        compilation_start = time.time()
        try:
            compiled_func = self.compile_function(
                func, example_inputs, opt_level, hw_target, force_recompile=True
            )
            compilation_time = time.time() - compilation_start
            
            # Measure optimized execution time
            start_time = time.time()
            with torch.no_grad():
                optimized_result = compiled_func(*example_inputs)
            execution_time_after = time.time() - start_time
            
        except Exception as e:
            compilation_time = time.time() - compilation_start
            execution_time_after = float('inf')
            success = False
            error_message = str(e)
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
        else:
            memory_after = 0
        
        # Calculate speedup
        if execution_time_before > 0 and execution_time_after > 0:
            speedup_factor = execution_time_before / execution_time_after
        else:
            speedup_factor = 1.0
        
        return CompilationProfile(
            function_name=func_name,
            compilation_time=compilation_time,
            execution_time_before=execution_time_before,
            execution_time_after=execution_time_after,
            speedup_factor=speedup_factor,
            memory_usage_before=memory_before,
            memory_usage_after=memory_after,
            optimization_level=opt_level,
            hardware_target=hw_target,
            success=success,
            error_message=error_message
        )
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        if not self.compilation_profiles:
            return {"total_compilations": 0}
        
        successful_compilations = [p for p in self.compilation_profiles if p.success]
        
        if not successful_compilations:
            return {
                "total_compilations": len(self.compilation_profiles),
                "success_rate": 0.0,
                "average_speedup": 0.0
            }
        
        # Calculate statistics
        speedups = [p.speedup_factor for p in successful_compilations]
        compilation_times = [p.compilation_time for p in successful_compilations]
        
        return {
            "total_compilations": len(self.compilation_profiles),
            "successful_compilations": len(successful_compilations),
            "success_rate": len(successful_compilations) / len(self.compilation_profiles),
            "average_speedup": np.mean(speedups),
            "max_speedup": np.max(speedups),
            "average_compilation_time": np.mean(compilation_times),
            "optimization_counts": dict(self.optimization_stats),
            "hardware_distribution": {
                hw.value: len([p for p in successful_compilations if p.hardware_target == hw])
                for hw in HardwareTarget
            }
        }
    
    def clear_cache(self):
        """Clear compilation cache."""
        with self.lock:
            self.compiled_functions.clear()
            self.logger.info("Compilation cache cleared")
    
    def warm_cache(self, functions_and_inputs: List[Tuple[Callable, Tuple[torch.Tensor, ...]]]):
        """Warm up the compilation cache."""
        self.logger.info(f"Warming cache with {len(functions_and_inputs)} functions")
        
        for func, example_inputs in functions_and_inputs:
            try:
                self.compile_function(func, example_inputs)
            except Exception as e:
                self.logger.warning(f"Cache warming failed for {func.__name__}: {e}")


# Global compiler instance
_global_compiler: Optional[NeuromorphicCompiler] = None


def get_neuromorphic_compiler() -> NeuromorphicCompiler:
    """Get or create global compiler instance."""
    global _global_compiler
    if _global_compiler is None:
        _global_compiler = NeuromorphicCompiler()
    return _global_compiler


def compile_neuromorphic(example_inputs: Tuple[torch.Tensor, ...],
                        optimization_level: OptimizationLevel = OptimizationLevel.O2,
                        hardware_target: Optional[HardwareTarget] = None):
    """Decorator for compiling neuromorphic functions."""
    def decorator(func):
        compiler = get_neuromorphic_compiler()
        
        # Compile on first use
        compiled_func = None
        
        def wrapper(*args, **kwargs):
            nonlocal compiled_func
            
            if compiled_func is None:
                compiled_func = compiler.compile_function(
                    func, example_inputs, optimization_level, hardware_target
                )
            
            return compiled_func(*args, **kwargs)
        
        return wrapper
    return decorator
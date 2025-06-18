# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Performance optimization and circuit optimization for Qiskit backend."""

import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import hashlib

try:
    from qiskit import transpile, QuantumCircuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CXCancellation, Collect2qBlocks
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CircuitCache:
    """Advanced circuit caching with intelligent invalidation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.hit_counts = defaultdict(int)
        self.lock = threading.RLock()
    
    def _circuit_hash(self, circuit: QuantumCircuit, backend_config: Dict) -> str:
        """Create a unique hash for circuit and backend configuration."""
        # Hash circuit structure
        circuit_str = str(circuit)
        
        # Hash relevant backend configuration
        config_items = [
            str(backend_config.get('name', '')),
            str(backend_config.get('coupling_map', '')),
            str(backend_config.get('basis_gates', '')),
            str(backend_config.get('optimization_level', 1))
        ]
        config_str = '|'.join(config_items)
        
        # Create combined hash
        combined = f"{circuit_str}|{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, circuit: QuantumCircuit, backend_config: Dict) -> Optional[QuantumCircuit]:
        """Get transpiled circuit from cache."""
        with self.lock:
            cache_key = self._circuit_hash(circuit, backend_config)
            
            if cache_key in self.cache:
                self.hit_counts[cache_key] += 1
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key].copy()
            
            return None
    
    def put(self, circuit: QuantumCircuit, transpiled_circuit: QuantumCircuit, 
            backend_config: Dict):
        """Store transpiled circuit in cache."""
        with self.lock:
            cache_key = self._circuit_hash(circuit, backend_config)
            
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[cache_key] = transpiled_circuit.copy()
            self.access_times[cache_key] = time.time()
            self.hit_counts[cache_key] = 0
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.access_times, key=self.access_times.get)
        
        # Remove from all structures
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.hit_counts[lru_key]
    
    def clear(self):
        """Clear all cached circuits."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_counts.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_hits = sum(self.hit_counts.values())
            total_requests = len(self.hit_counts) + total_hits
            hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_hits': total_hits,
                'total_requests': total_requests
            }


class OptimizedTranspiler:
    """Enhanced transpiler with circuit optimization."""
    
    def __init__(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for OptimizedTranspiler")
        
        self.optimization_passes = {
            0: [],  # No optimization
            1: [Optimize1qGatesDecomposition()],  # Basic single-qubit optimization
            2: [Optimize1qGatesDecomposition(), CXCancellation()],  # + two-qubit optimization
            3: [Optimize1qGatesDecomposition(), CXCancellation(), Collect2qBlocks()]  # Advanced
        }
    
    def create_pass_manager(self, backend, optimization_level: int = 1) -> PassManager:
        """Create optimized pass manager for backend."""
        pm = PassManager()
        
        # Add optimization passes based on level
        if optimization_level in self.optimization_passes:
            for pass_obj in self.optimization_passes[optimization_level]:
                pm.append(pass_obj)
        
        return pm
    
    def transpile_optimized(self, circuit: QuantumCircuit, backend, 
                          optimization_level: int = 1, **kwargs) -> QuantumCircuit:
        """Transpile circuit with optimizations."""
        # Basic transpilation with optimization
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            **kwargs
        )
        
        # Apply additional custom optimizations
        if optimization_level >= 2:
            transpiled = self._apply_custom_optimizations(transpiled, backend)
        
        return transpiled
    
    def _apply_custom_optimizations(self, circuit: QuantumCircuit, backend) -> QuantumCircuit:
        """Apply custom optimization passes."""
        # Custom optimization logic can be added here
        # For now, return the circuit as-is
        return circuit


class BatchProcessor:
    """Efficient batch processing for multiple circuits."""
    
    def __init__(self, max_batch_size: int = 100):
        self.max_batch_size = max_batch_size
    
    def process_batch(self, circuits: List[QuantumCircuit], backend, 
                     optimization_level: int = 1) -> List[QuantumCircuit]:
        """Process multiple circuits efficiently."""
        if not circuits:
            return []
        
        # Split into manageable batches
        batches = self._create_batches(circuits)
        transpiled_circuits = []
        
        for batch in batches:
            # Transpile batch together for efficiency
            batch_transpiled = transpile(
                batch,
                backend=backend,
                optimization_level=optimization_level
            )
            
            # Handle single circuit vs list return
            if isinstance(batch_transpiled, list):
                transpiled_circuits.extend(batch_transpiled)
            else:
                transpiled_circuits.append(batch_transpiled)
        
        return transpiled_circuits
    
    def _create_batches(self, circuits: List[QuantumCircuit]) -> List[List[QuantumCircuit]]:
        """Split circuits into batches."""
        batches = []
        for i in range(0, len(circuits), self.max_batch_size):
            batch = circuits[i:i + self.max_batch_size]
            batches.append(batch)
        return batches


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and record duration."""
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.metrics[f"{name}_duration"].append(duration)
        del self.timers[name]
        return duration
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter metric."""
        self.counters[name] += amount
    
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        self.metrics[name].append(value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'counters': dict(self.counters),
            'metrics': {}
        }
        
        # Calculate statistics for metrics
        for name, values in self.metrics.items():
            if values:
                stats['metrics'][name] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()


class ResourceOptimizer:
    """Optimize resource usage for different execution scenarios."""
    
    def __init__(self):
        self.shot_recommendations = {
            'quick_test': 1024,
            'development': 4096, 
            'production': 8192,
            'high_precision': 16384
        }
    
    def recommend_shots(self, scenario: str, n_qubits: int, 
                       error_tolerance: float = 0.01) -> int:
        """Recommend optimal shot count for scenario."""
        base_shots = self.shot_recommendations.get(scenario, 4096)
        
        # Adjust based on number of qubits
        qubit_factor = min(2.0, 1.0 + (n_qubits - 5) * 0.1)
        
        # Adjust based on error tolerance
        error_factor = max(0.5, 0.01 / error_tolerance)
        
        recommended_shots = int(base_shots * qubit_factor * error_factor)
        
        # Reasonable bounds
        return max(512, min(50000, recommended_shots))
    
    def optimize_circuit_depth(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """Analyze and suggest circuit depth optimizations."""
        depth = circuit.depth()
        gate_count = len(circuit.data)
        cx_count = sum(1 for instr, _, _ in circuit.data if instr.name in ['cx', 'cnot'])
        
        analysis = {
            'current_depth': depth,
            'gate_count': gate_count,
            'cx_count': cx_count,
            'recommendations': []
        }
        
        # Provide recommendations
        if depth > 100:
            analysis['recommendations'].append("Circuit depth is high - consider circuit decomposition")
        
        if cx_count > gate_count * 0.5:
            analysis['recommendations'].append("High CNOT ratio - consider gate optimization")
        
        if gate_count > 1000:
            analysis['recommendations'].append("Large circuit - consider parallelization")
        
        return analysis


class AdaptiveExecution:
    """Adaptive execution strategies based on circuit and backend characteristics."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.resource_optimizer = ResourceOptimizer()
    
    def choose_execution_strategy(self, circuit: QuantumCircuit, backend_info: Dict[str, Any],
                                measurement_type: str) -> Dict[str, Any]:
        """Choose optimal execution strategy."""
        n_qubits = circuit.num_qubits
        depth = circuit.depth()
        is_simulator = backend_info.get('simulator', True)
        
        strategy = {
            'optimization_level': 1,
            'shots': 4096,
            'parallel_execution': False,
            'cache_strategy': 'standard'
        }
        
        # Adjust for circuit size
        if n_qubits <= 5 and depth <= 20:
            strategy.update({
                'optimization_level': 0,
                'shots': 1024,
                'cache_strategy': 'aggressive'
            })
        elif n_qubits >= 15 or depth >= 100:
            strategy.update({
                'optimization_level': 3,
                'shots': 8192,
                'parallel_execution': True,
                'cache_strategy': 'conservative'
            })
        
        # Adjust for measurement type
        if measurement_type == 'expectation':
            # Expectation values need more shots for accuracy
            strategy['shots'] = max(strategy['shots'], 4096)
        elif measurement_type == 'sampling':
            # Sampling can use fewer shots
            strategy['shots'] = max(strategy['shots'] // 2, 1024)
        
        # Adjust for backend type
        if not is_simulator:
            # Real hardware needs more conservative settings
            strategy.update({
                'optimization_level': max(strategy['optimization_level'], 2),
                'shots': min(strategy['shots'], 8192),  # Hardware shot limits
                'parallel_execution': False  # Hardware usually doesn't support parallel
            })
        
        return strategy 
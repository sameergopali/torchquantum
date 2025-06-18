# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import warnings
from typing import List, Union, Dict, Optional, Any

try:
    from qiskit import execute, transpile
    from qiskit_aer import AerSimulator
    from qiskit.providers import Backend as QiskitBackendBase
    from qiskit_aer.noise import NoiseModel
    from qiskit.circuit import QuantumCircuit, ClassicalRegister
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QiskitBackendBase = object
    NoiseModel = object
    AerSimulator = object

from ..abstract_backend import QuantumBackend
from ..core.circuit import ParameterizedQuantumCircuit
from .utils import convert_tq_circuit_to_qiskit, create_parameter_binds

# Import advanced features (with graceful fallback)
try:
    from .noise import create_depolarizing_noise_model, create_thermal_noise_model, NoiseModelBuilder
    from .hardware import HardwareManager, setup_hardware_backend, JobMonitor
    from .optimization import CircuitCache, OptimizedTranspiler, PerformanceMonitor, AdaptiveExecution
    from .error_handling import SafeExecutor, RetryConfig, CircuitValidator
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False


class QiskitBackend(QuantumBackend):
    """Qiskit backend for quantum circuit simulation and execution.
    
    This backend provides shot-based quantum simulation using Qiskit's
    simulators and real quantum hardware. It supports noise models,
    hardware constraints, and statistical sampling.
    """
    
    def __init__(
        self,
        device: Union[str, QiskitBackendBase] = 'qasm_simulator',
        shots: int = 8192,
        seed: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
        coupling_map: Optional[List[List[int]]] = None,
        basis_gates: Optional[List[str]] = None,
        optimization_level: int = 1,
        initial_layout: Optional[List[int]] = None,
        max_parallel_experiments: int = 1,
        warn_large_shots: bool = True,
        large_shots_threshold: int = 100000,
        enable_advanced_features: bool = True,
        enable_circuit_caching: bool = True,
        enable_error_recovery: bool = True,
        enable_performance_monitoring: bool = False,
        cache_size: int = 1000
    ):
        """Initialize the Qiskit backend.
        
        Args:
            device: Qiskit backend name or backend instance
            shots: Number of measurement shots
            seed: Random seed for reproducibility
            noise_model: Noise model for simulation
            coupling_map: Device coupling map for transpilation
            basis_gates: Available basis gates
            optimization_level: Transpilation optimization level (0-3)
            initial_layout: Initial qubit layout
            max_parallel_experiments: Maximum parallel experiments
            warn_large_shots: Whether to warn about large shot counts
            large_shots_threshold: Shot count threshold for warnings
            enable_advanced_features: Enable advanced features (caching, error handling, etc.)
            enable_circuit_caching: Enable intelligent circuit caching
            enable_error_recovery: Enable automatic error recovery
            enable_performance_monitoring: Enable performance monitoring
            cache_size: Maximum number of circuits to cache
        """
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is not installed. Please install it with: pip install qiskit"
            )
        
        self.shots = shots
        self.seed = seed
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.optimization_level = optimization_level
        self.initial_layout = initial_layout
        self.max_parallel_experiments = max_parallel_experiments
        self.warn_large_shots = warn_large_shots
        self.large_shots_threshold = large_shots_threshold
        
        # Advanced features configuration
        self.enable_advanced_features = enable_advanced_features and ADVANCED_FEATURES_AVAILABLE
        self.enable_circuit_caching = enable_circuit_caching
        self.enable_error_recovery = enable_error_recovery
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Initialize backend
        self._setup_backend(device)
        
        # Initialize advanced features
        self._setup_advanced_features(cache_size)
        
        # Warn about large shot counts
        if self.warn_large_shots and self.shots > self.large_shots_threshold:
            warnings.warn(
                f"Using {self.shots} shots may result in long execution times. "
                f"Consider reducing shots or setting warn_large_shots=False.",
                UserWarning
            )
        
        # Warn if advanced features are disabled
        if not self.enable_advanced_features and ADVANCED_FEATURES_AVAILABLE:
            warnings.warn("Advanced features are disabled. Some functionality may be limited.")
        elif not ADVANCED_FEATURES_AVAILABLE:
            warnings.warn("Advanced features not available. Install additional dependencies for full functionality.")
        
    def _setup_backend(self, device: Union[str, QiskitBackendBase]):
        """Setup the Qiskit backend."""
        if isinstance(device, str):
            if device in ['qasm_simulator', 'aer_simulator']:
                # Use AerSimulator with appropriate method
                if device == 'qasm_simulator':
                    self.backend = AerSimulator(method='automatic')
                else:
                    self.backend = AerSimulator()
            elif device == 'statevector_simulator':
                self.backend = AerSimulator(method='statevector')
            elif device == 'unitary_simulator':
                self.backend = AerSimulator(method='unitary')
            else:
                # Try to create AerSimulator with custom method or assume it's a provider backend
                try:
                    self.backend = AerSimulator(method=device)
                except:
                    # Create a temporary simulator to get available methods
                    temp_sim = AerSimulator()
                    available_methods = temp_sim.available_methods()
                    raise ValueError(f"Backend {device} not supported. Available methods: {available_methods}")
        else:
            # Backend instance provided
            self.backend = device
        
        # Set up backend-specific parameters
        self.backend_name = self.backend.name
        
        # Use backend's coupling map and basis gates if not provided
        if hasattr(self.backend, 'configuration'):
            config = self.backend.configuration()
            if self.coupling_map is None and hasattr(config, 'coupling_map'):
                self.coupling_map = config.coupling_map
            if self.basis_gates is None and hasattr(config, 'basis_gates'):
                self.basis_gates = config.basis_gates
    
    def _setup_advanced_features(self, cache_size: int):
        """Setup advanced features like caching, error handling, and monitoring."""
        # Initialize simple circuit cache (fallback)
        self._circuit_cache = {}
        
        if not self.enable_advanced_features:
            return
        
        # Initialize advanced circuit cache
        if self.enable_circuit_caching:
            self.circuit_cache = CircuitCache(max_size=cache_size)
        
        # Initialize optimized transpiler
        self.optimized_transpiler = OptimizedTranspiler()
        
        # Initialize error handling
        if self.enable_error_recovery:
            self.safe_executor = SafeExecutor()
            self.circuit_validator = CircuitValidator()
        
        # Initialize performance monitoring
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        # Initialize adaptive execution
        self.adaptive_execution = AdaptiveExecution()
        
        # Initialize hardware manager (for future use)
        self.hardware_manager = HardwareManager()
        
        # Initialize job monitor
        self.job_monitor = JobMonitor()
    
    def _create_expectation_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        pauli_ops: Union[List[str], Dict[str, float]]
    ) -> 'QiskitExpectation':
        """Create a module for computing expectation values of Pauli operators."""
        from .expectation import QiskitExpectation
        return QiskitExpectation(circuit, self, pauli_ops)
    
    def _create_amplitude_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        bitstrings: List[str]
    ) -> 'QiskitAmplitude':
        """Create a module for computing state amplitudes."""
        from .amplitude import QiskitAmplitude
        return QiskitAmplitude(circuit, self, bitstrings)
    
    def _create_sampling_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        n_samples: int, 
        wires: Optional[List[int]] = None
    ) -> 'QiskitSampling':
        """Create a module for sampling from the quantum state."""
        from .sampling import QiskitSampling
        return QiskitSampling(circuit, self, n_samples, wires)
    
    def execute_circuit(
        self,
        circuit: ParameterizedQuantumCircuit,
        input_params: Optional[torch.Tensor] = None,
        measurements: Optional[List[int]] = None
    ) -> List[Dict[str, int]]:
        """Execute a quantum circuit and return measurement counts.
        
        Args:
            circuit: The quantum circuit to execute
            input_params: Input parameters [batch_size, n_params]
            measurements: List of qubits to measure (all if None)
            
        Returns:
            List of count dictionaries from Qiskit execution
        """
        # Convert to Qiskit circuit
        qiskit_circuit, qiskit_params = convert_tq_circuit_to_qiskit(circuit)
        
        # Add measurements
        if measurements is None:
            measurements = list(range(circuit.n_wires))
        
        # Add classical register and measurements
        if len(qiskit_circuit.cregs) == 0:
            creg = ClassicalRegister(len(measurements), 'c')
            qiskit_circuit.add_register(creg)
        
        for i, qubit in enumerate(measurements):
            qiskit_circuit.measure(qubit, i)
        
        # Create parameter bindings
        parameter_binds = create_parameter_binds(qiskit_params, input_params)
        
        # Transpile circuit
        transpiled_circuit = self._transpile_circuit(qiskit_circuit)
        
        # Execute
        job = execute(
            experiments=transpiled_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=parameter_binds,
            seed_simulator=self.seed,
            noise_model=self.noise_model,
            optimization_level=0,  # Already transpiled
            max_parallel_experiments=self.max_parallel_experiments
        )
        
        result = job.result()
        counts = result.get_counts()
        
        # Ensure counts is a list
        if not isinstance(counts, list):
            counts = [counts]
        
        return counts
    
    def _transpile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile a Qiskit circuit for the target backend."""
        # Create backend configuration for caching
        backend_config = {
            'name': self.backend_name,
            'coupling_map': self.coupling_map,
            'basis_gates': self.basis_gates,
            'optimization_level': self.optimization_level
        }
        
        # Use advanced caching if available
        if self.enable_advanced_features and hasattr(self, 'circuit_cache'):
            cached_circuit = self.circuit_cache.get(circuit, backend_config)
            if cached_circuit is not None:
                return cached_circuit
        else:
            # Fallback to simple caching
            cache_key = (
                str(circuit), 
                self.backend_name, 
                self.optimization_level,
                str(self.coupling_map),
                str(self.basis_gates)
            )
            
            if cache_key in self._circuit_cache:
                return self._circuit_cache[cache_key]
        
        # Start performance monitoring if enabled
        if self.enable_performance_monitoring and hasattr(self, 'performance_monitor'):
            self.performance_monitor.start_timer('transpilation')
        
        # Use optimized transpiler if available
        if self.enable_advanced_features and hasattr(self, 'optimized_transpiler'):
            transpiled = self.optimized_transpiler.transpile_optimized(
                circuit,
                backend=self.backend,
                optimization_level=self.optimization_level,
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                initial_layout=self.initial_layout,
                seed_transpiler=self.seed
            )
        else:
            # Fallback to standard transpilation
            transpiled = transpile(
                circuit,
                backend=self.backend,
                optimization_level=self.optimization_level,
                coupling_map=self.coupling_map,
                basis_gates=self.basis_gates,
                initial_layout=self.initial_layout,
                seed_transpiler=self.seed
            )
        
        # End performance monitoring
        if self.enable_performance_monitoring and hasattr(self, 'performance_monitor'):
            duration = self.performance_monitor.end_timer('transpilation')
            self.performance_monitor.record_metric('circuit_depth', transpiled.depth())
            self.performance_monitor.record_metric('gate_count', len(transpiled.data))
        
        # Cache the result
        if self.enable_advanced_features and hasattr(self, 'circuit_cache'):
            self.circuit_cache.put(circuit, transpiled, backend_config)
        else:
            # Fallback caching
            cache_key = (
                str(circuit), 
                self.backend_name, 
                self.optimization_level,
                str(self.coupling_map),
                str(self.basis_gates)
            )
            self._circuit_cache[cache_key] = transpiled
        
        return transpiled
    
    def clear_cache(self):
        """Clear the circuit transpilation cache."""
        self._circuit_cache.clear()
        if self.enable_advanced_features and hasattr(self, 'circuit_cache'):
            self.circuit_cache.clear()
    
    def set_shots(self, shots: int):
        """Update the number of shots."""
        self.shots = shots
        if self.warn_large_shots and self.shots > self.large_shots_threshold:
            warnings.warn(
                f"Using {self.shots} shots may result in long execution times.",
                UserWarning
            )
    
    def set_noise_model(self, noise_model: Optional[NoiseModel]):
        """Update the noise model."""
        self.noise_model = noise_model
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        info = {
            'name': self.backend_name,
            'shots': self.shots,
            'seed': self.seed,
            'optimization_level': self.optimization_level,
            'max_parallel_experiments': self.max_parallel_experiments,
            'advanced_features_enabled': self.enable_advanced_features,
            'circuit_caching_enabled': self.enable_circuit_caching,
            'error_recovery_enabled': self.enable_error_recovery,
            'performance_monitoring_enabled': self.enable_performance_monitoring
        }
        
        if hasattr(self.backend, 'configuration'):
            config = self.backend.configuration()
            info.update({
                'n_qubits': getattr(config, 'n_qubits', None),
                'coupling_map': getattr(config, 'coupling_map', None),
                'basis_gates': getattr(config, 'basis_gates', None),
                'simulator': getattr(config, 'simulator', None),
                'local': getattr(config, 'local', None)
            })
        
        # Add cache statistics if available
        if self.enable_advanced_features and hasattr(self, 'circuit_cache'):
            info['cache_stats'] = self.circuit_cache.stats()
        
        return info
    
    # Advanced Features Methods
    
    def create_noise_model(self, noise_type: str = 'depolarizing', **kwargs) -> Optional[NoiseModel]:
        """Create a noise model for simulation.
        
        Args:
            noise_type: Type of noise ('depolarizing', 'thermal', 'device')
            **kwargs: Noise parameters
            
        Returns:
            NoiseModel or None if advanced features disabled
        """
        if not self.enable_advanced_features:
            warnings.warn("Advanced features disabled. Cannot create noise model.")
            return None
        
        if noise_type == 'depolarizing':
            return create_depolarizing_noise_model(**kwargs)
        elif noise_type == 'thermal':
            return create_thermal_noise_model(**kwargs)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def apply_noise_model(self, noise_model: NoiseModel):
        """Apply a noise model to this backend."""
        self.set_noise_model(noise_model)
    
    def setup_hardware_execution(self, device_name: str, **kwargs) -> Dict[str, Any]:
        """Setup backend for hardware execution.
        
        Args:
            device_name: Name of the quantum device
            **kwargs: Additional setup parameters
            
        Returns:
            Setup result dictionary
        """
        if not self.enable_advanced_features:
            return {'success': False, 'error': 'Advanced features disabled'}
        
        return setup_hardware_backend(self, device_name, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics."""
        if (self.enable_performance_monitoring and 
            hasattr(self, 'performance_monitor')):
            return self.performance_monitor.get_stats()
        else:
            return {'error': 'Performance monitoring not enabled'}
    
    def reset_performance_monitor(self):
        """Reset performance monitoring statistics."""
        if (self.enable_performance_monitoring and 
            hasattr(self, 'performance_monitor')):
            self.performance_monitor.reset()
    
    def validate_circuit(self, circuit: QuantumCircuit) -> List[str]:
        """Validate a circuit against backend constraints.
        
        Args:
            circuit: Circuit to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        if not self.enable_advanced_features or not hasattr(self, 'circuit_validator'):
            return []  # Skip validation if advanced features disabled
        
        backend_config = {
            'n_qubits': getattr(self.backend.configuration(), 'n_qubits', float('inf')),
            'basis_gates': self.basis_gates or [],
            'coupling_map': self.coupling_map
        }
        
        return self.circuit_validator.validate_circuit(circuit, backend_config)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get circuit cache statistics."""
        if self.enable_advanced_features and hasattr(self, 'circuit_cache'):
            return self.circuit_cache.stats()
        else:
            return {'error': 'Advanced caching not enabled'}
    
    def optimize_for_execution(self, circuit: QuantumCircuit, 
                              measurement_type: str = 'expectation') -> Dict[str, Any]:
        """Get optimization recommendations for circuit execution.
        
        Args:
            circuit: Circuit to analyze
            measurement_type: Type of measurement ('expectation', 'sampling', 'amplitude')
            
        Returns:
            Optimization strategy dictionary
        """
        if not self.enable_advanced_features or not hasattr(self, 'adaptive_execution'):
            return {'error': 'Advanced features not enabled'}
        
        backend_info = self.get_backend_info()
        return self.adaptive_execution.choose_execution_strategy(
            circuit, backend_info, measurement_type
        ) 
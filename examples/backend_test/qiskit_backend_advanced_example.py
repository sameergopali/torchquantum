"""Comprehensive example demonstrating the advanced Qiskit backend features."""

import torch
import numpy as np
from torchquantum.backend import (
    ParameterizedQuantumCircuit,
    QuantumExpectation,
    QuantumSampling
)
from torchquantum.backend.qiskit_backend import (
    QiskitBackend,
    create_depolarizing_noise_model,
    create_thermal_noise_model,
    NoiseModelBuilder,
    HardwareManager,
    CircuitCache,
    PerformanceMonitor
)
from torchquantum.operator.standard_gates import Hadamard, RX, RY, RZ, CNOT


def create_variational_circuit(n_qubits=4, n_layers=3):
    """Create a variational quantum circuit for testing."""
    n_params = n_qubits * n_layers * 2
    circuit = ParameterizedQuantumCircuit(n_wires=n_qubits, n_trainable_params=n_params)
    
    # Initialize parameters
    circuit.set_trainable_params(torch.randn(n_params) * 0.1)
    
    param_idx = 0
    for layer in range(n_layers):
        # Parameterized rotation layer
        for q in range(n_qubits):
            circuit.append_gate(RY, wires=q, trainable_idx=param_idx)
            param_idx += 1
            circuit.append_gate(RZ, wires=q, trainable_idx=param_idx)
            param_idx += 1
        
        # Entangling layer
        for q in range(n_qubits - 1):
            circuit.append_gate(CNOT, wires=[q, q + 1])
    
    return circuit


def demonstrate_basic_features():
    """Demonstrate basic Qiskit backend functionality."""
    print("=" * 60)
    print("BASIC QISKIT BACKEND FEATURES")
    print("=" * 60)
    
    # Create backend with advanced features enabled
    backend = QiskitBackend(
        device='qasm_simulator',
        shots=4096,
        enable_performance_monitoring=True,
        enable_circuit_caching=True,
        enable_error_recovery=True
    )
    
    print(f"Backend info: {backend.get_backend_info()}")
    
    # Create a simple Bell state circuit
    bell_circuit = ParameterizedQuantumCircuit(n_wires=2, n_trainable_params=0)
    bell_circuit.append_gate(Hadamard, wires=0)
    bell_circuit.append_gate(CNOT, wires=[0, 1])
    
    # Test expectation values
    observables = ['ZZ', 'XX', 'YY']
    expectation = QuantumExpectation(bell_circuit, backend, observables)
    
    print("\nBell State Expectation Values:")
    exp_vals = expectation()
    for i, obs in enumerate(observables):
        print(f"  <{obs}> = {exp_vals[0, i].item():.4f}")
    
    # Test sampling
    sampler = QuantumSampling(bell_circuit, backend, n_samples=1000)
    samples = sampler()
    
    print("\nBell State Sampling Results:")
    from collections import Counter
    
    # Convert tensor samples to bitstrings
    bitstrings = []
    for sample in samples[0]:  # samples[0] is [n_samples, n_wires]
        bitstring = ''.join([str(bit.item()) for bit in sample])
        bitstrings.append(bitstring)
    
    counts = Counter(bitstrings)
    for bitstring, count in counts.most_common():
        print(f"  |{bitstring}⟩: {count/1000:.3f}")


def demonstrate_noise_models():
    """Demonstrate noise model functionality."""
    print("\n" + "=" * 60)
    print("NOISE MODEL FEATURES")
    print("=" * 60)
    
    # Create depolarizing noise model
    backend_noisy = QiskitBackend(device='qasm_simulator', shots=8192)
    noise_model = backend_noisy.create_noise_model(
        'depolarizing',
        single_qubit_error=0.01,
        two_qubit_error=0.05,
        readout_error=0.03
    )
    backend_noisy.apply_noise_model(noise_model)
    
    print("Created depolarizing noise model")
    
    # Test with Bell state
    bell_circuit = ParameterizedQuantumCircuit(n_wires=2, n_trainable_params=0)
    bell_circuit.append_gate(Hadamard, wires=0)
    bell_circuit.append_gate(CNOT, wires=[0, 1])
    
    observables = ['ZZ', 'XX', 'YY']
    expectation_noisy = QuantumExpectation(bell_circuit, backend_noisy, observables)
    
    print("\nNoisy Bell State Expectation Values:")
    exp_vals_noisy = expectation_noisy()
    for i, obs in enumerate(observables):
        print(f"  <{obs}> = {exp_vals_noisy[0, i].item():.4f}")
    
    # Create thermal noise model
    thermal_noise = create_thermal_noise_model(
        t1_time=50e-6,
        t2_time=70e-6,
        gate_time=0.1e-6,
        readout_error=0.02
    )
    backend_noisy.apply_noise_model(thermal_noise)
    
    print("\nApplied thermal relaxation noise model")
    
    # Create custom noise model using builder
    builder = NoiseModelBuilder()
    custom_noise = (builder
                   .add_depolarizing_error(0.005, ['h', 'x', 'y', 'z'], 1)
                   .add_depolarizing_error(0.02, ['cx', 'cnot'], 2)
                   .add_readout_error(0.01)
                   .build())
    
    print("Created custom noise model using builder pattern")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING")
    print("=" * 60)
    
    # Create backend with performance monitoring
    backend = QiskitBackend(
        device='qasm_simulator',
        shots=4096,
        enable_performance_monitoring=True,
        optimization_level=2
    )
    
    # Create a larger circuit for meaningful performance metrics
    vqe_circuit = create_variational_circuit(n_qubits=6, n_layers=4)
    
    # Define Hamiltonian
    hamiltonian = {
        'ZIIIII': 0.5, 'IZIIII': 0.5, 'IIZIII': 0.5,
        'IIIZII': 0.5, 'IIIIZI': 0.5, 'IIIIIZ': 0.5,
        'XXIIII': 0.25, 'IIXXII': 0.25, 'IIIIXX': 0.25
    }
    
    # Test performance with multiple executions
    expectation = QuantumExpectation(vqe_circuit, backend, hamiltonian)
    
    print("Executing circuit multiple times to gather performance metrics...")
    for i in range(5):
        energies = expectation()
        print(f"  Execution {i+1}: Energy = {energies.sum().item():.4f}")
    
    # Get performance statistics
    perf_stats = backend.get_performance_stats()
    print("\nPerformance Statistics:")
    for metric_name, stats in perf_stats.get('metrics', {}).items():
        print(f"  {metric_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    Max: {stats['max']:.4f}")
        print(f"    Count: {stats['count']}")


def demonstrate_circuit_caching():
    """Demonstrate circuit caching functionality."""
    print("\n" + "=" * 60)
    print("CIRCUIT CACHING")
    print("=" * 60)
    
    # Create backend with caching enabled
    backend = QiskitBackend(
        device='qasm_simulator',
        shots=2048,
        enable_circuit_caching=True,
        cache_size=100
    )
    
    # Create circuit
    circuit = create_variational_circuit(n_qubits=4, n_layers=2)
    observables = ['ZIII', 'IZII', 'IIZI', 'IIIZ']
    expectation = QuantumExpectation(circuit, backend, observables)
    
    print("First execution (cache miss):")
    import time
    start_time = time.time()
    result1 = expectation()
    time1 = time.time() - start_time
    print(f"  Time: {time1:.3f}s")
    
    print("Second execution (cache hit):")
    start_time = time.time()
    result2 = expectation()
    time2 = time.time() - start_time
    print(f"  Time: {time2:.3f}s")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Get cache statistics
    cache_stats = backend.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Total Hits: {cache_stats['total_hits']}")


def demonstrate_circuit_optimization():
    """Demonstrate circuit optimization features."""
    print("\n" + "=" * 60)
    print("CIRCUIT OPTIMIZATION")
    print("=" * 60)
    
    backend = QiskitBackend(
        device='qasm_simulator',
        shots=4096,
        optimization_level=3
    )
    
    # Create a deep circuit for optimization testing
    deep_circuit = create_variational_circuit(n_qubits=5, n_layers=8)
    
    # Get optimization recommendations
    from qiskit.circuit import QuantumCircuit
    from torchquantum.backend.qiskit_backend.utils import convert_tq_circuit_to_qiskit
    
    qiskit_circuit, _ = convert_tq_circuit_to_qiskit(deep_circuit)
    
    print(f"Original circuit:")
    print(f"  Depth: {qiskit_circuit.depth()}")
    print(f"  Gates: {len(qiskit_circuit.data)}")
    
    # Get optimization strategy
    strategy = backend.optimize_for_execution(qiskit_circuit, 'expectation')
    print(f"\nOptimization Strategy:")
    print(f"  Optimization Level: {strategy.get('optimization_level', 'N/A')}")
    print(f"  Recommended Shots: {strategy.get('shots', 'N/A')}")
    print(f"  Cache Strategy: {strategy.get('cache_strategy', 'N/A')}")


def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING AND RECOVERY")
    print("=" * 60)
    
    backend = QiskitBackend(
        device='qasm_simulator',
        shots=1000000,  # Very large shot count to potentially trigger warnings
        enable_error_recovery=True
    )
    
    # Create a circuit that might have validation issues
    large_circuit = create_variational_circuit(n_qubits=25, n_layers=5)  # Large circuit
    
    # Test circuit validation
    from torchquantum.backend.qiskit_backend.utils import convert_tq_circuit_to_qiskit
    qiskit_circuit, _ = convert_tq_circuit_to_qiskit(large_circuit)
    
    validation_errors = backend.validate_circuit(qiskit_circuit)
    if validation_errors:
        print("Circuit validation errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Circuit passed validation")
    
    # Demonstrate automatic shot reduction for large circuits
    print(f"\nOriginal shot count: {backend.shots}")
    if backend.shots > 50000:
        print("Large shot count detected - backend will handle this automatically")




def main():
    """Run all demonstrations."""
    print("TorchQuantum Qiskit Backend - Advanced Features Demo")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_basic_features()
    demonstrate_noise_models()
    demonstrate_performance_monitoring()
    demonstrate_circuit_caching()
    demonstrate_circuit_optimization()
    demonstrate_error_handling()

    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("The Qiskit backend provides:")
    print("✓ Shot-based quantum simulation")
    print("✓ Realistic noise models")
    print("✓ Performance monitoring")
    print("✓ Intelligent circuit caching")
    print("✓ Circuit optimization")
    print("✓ Error handling and recovery")
    print("✓ Hardware integration capabilities")



if __name__ == "__main__":
    main() 
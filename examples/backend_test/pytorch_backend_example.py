"""Example of using the PyTorch backend with the new architecture."""

import torch
from torchquantum.backend import (
    ParameterizedQuantumCircuit,
    PyTorchBackend,
    QuantumExpectation,
    QuantumSampling
)
from torchquantum.operator.standard_gates import Hadamard, RX, CNOT, RZ


def create_bell_circuit():
    """Create a simple Bell state preparation circuit."""
    circuit = ParameterizedQuantumCircuit(n_wires=2, n_trainable_params=0)
    circuit.append_gate(Hadamard, wires=0)
    circuit.append_gate(CNOT, wires=[0, 1])
    return circuit


def create_vqe_circuit(n_qubits=4, n_layers=2):
    """Create a simple VQE ansatz circuit."""
    n_params = n_qubits * n_layers * 2  # RX and RZ for each qubit in each layer
    circuit = ParameterizedQuantumCircuit(n_wires=n_qubits, n_trainable_params=n_params)
    
    # Initialize with random parameters
    circuit.set_trainable_params(torch.randn(n_params) * 0.1)
    
    param_idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            circuit.append_gate(RX, wires=q, trainable_idx=param_idx)
            param_idx += 1
            circuit.append_gate(RZ, wires=q, trainable_idx=param_idx)
            param_idx += 1
            
        # Entangling layer
        for q in range(0, n_qubits - 1, 2):
            circuit.append_gate(CNOT, wires=[q, q + 1])
        for q in range(1, n_qubits - 1, 2):
            circuit.append_gate(CNOT, wires=[q, q + 1])
            
    return circuit


def main():
    # Example 1: Bell state with expectation values
    print("=== Example 1: Bell State ===")
    bell_circuit = create_bell_circuit()
    
    # Create backend
    backend = PyTorchBackend(device='cpu')
    
    # Define observables
    observables = ['ZZ', 'XX', 'YY']  # Bell state correlations
    
    # Create expectation module
    expectation = QuantumExpectation(bell_circuit, backend, observables)
    
    # Compute expectations (no input params for Bell state)
    exp_vals = expectation()
    print(f"Expectation values: {exp_vals}")
    print(f"<ZZ> = {exp_vals[0, 0].item():.4f}, <XX> = {exp_vals[0, 1].item():.4f}, <YY> = {exp_vals[0, 2].item():.4f}")
    
    # Example 2: VQE circuit with optimization
    print("\n=== Example 2: VQE Circuit ===")
    vqe_circuit = create_vqe_circuit(n_qubits=4, n_layers=2)
    
    # Define Hamiltonian as linear combination
    hamiltonian = [
        {'ZIII': 0.5, 'IZII': 0.5, 'IIZI': 0.5, 'IIIZ': 0.5},  # Sum of Z operators
        {'XXII': 0.25, 'IIXX': 0.25}  # Nearest neighbor interactions
    ]
    
    # Create model
    model = QuantumExpectation(vqe_circuit, backend, hamiltonian)
    
    # Optimize
    optimizer = torch.optim.Adam([vqe_circuit.trainable_params], lr=0.1)
    
    print("Optimizing...")
    for step in range(50):
        optimizer.zero_grad()
        energies = model()  # Shape: [1, 2] for 2 Hamiltonians
        total_energy = energies.sum()
        total_energy.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Energy = {total_energy.item():.4f}")
    
    # Example 3: Sampling
    print("\n=== Example 3: Sampling ===")
    sampler = QuantumSampling(vqe_circuit, backend, n_samples=1000, wires=None)
    samples = sampler()  # Returns list of bitstrings
    
    # Count occurrences
    from collections import Counter
    counts = Counter(samples[0])  # First (and only) batch
    print("Top 5 measurement outcomes:")
    for bitstring, count in counts.most_common(5):
        print(f"  |{bitstring}⟩: {count/1000:.3f}")
    
    # Example 4: GPU support (if available)
    if torch.cuda.is_available():
        print("\n=== Example 4: GPU Acceleration ===")
        
        # Create a simple Bell circuit for GPU test
        simple_circuit = ParameterizedQuantumCircuit(n_wires=2, n_trainable_params=0)
        simple_circuit.append_gate(Hadamard, wires=0)
        simple_circuit.append_gate(CNOT, wires=[0, 1])
        
        backend_gpu = PyTorchBackend(device='cuda')
        simple_observables = ['ZZ']
        
        expectation_gpu = QuantumExpectation(simple_circuit, backend_gpu, simple_observables)
        
        print("Testing GPU computation...")
        energies_gpu = expectation_gpu()
        print(f"GPU Bell state <ZZ> expectation: {energies_gpu.item():.4f}")
        print(f"GPU computation successful!")
    else:
        print("\n=== Example 4: GPU Acceleration ===")
        print("CUDA not available, skipping GPU example")


if __name__ == "__main__":
    main() 
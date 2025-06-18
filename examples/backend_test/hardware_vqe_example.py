#!/usr/bin/env python3
"""Example running VQE algorithm on IBM Quantum hardware."""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from torchquantum.backend import ParameterizedQuantumCircuit, QuantumExpectation
from torchquantum.backend.qiskit_backend import QiskitBackend, HardwareManager
from torchquantum.operator.standard_gates import RY, RZ, CNOT


def create_vqe_ansatz(n_qubits=2, n_layers=2):
    """Create a hardware-efficient VQE ansatz."""
    n_params = n_qubits * n_layers * 2
    circuit = ParameterizedQuantumCircuit(n_wires=n_qubits, n_trainable_params=n_params)
    
    # Initialize parameters near ground state
    circuit.set_trainable_params(torch.randn(n_params) * 0.1)
    
    param_idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            circuit.append_gate(RY, wires=q, trainable_idx=param_idx)
            param_idx += 1
            circuit.append_gate(RZ, wires=q, trainable_idx=param_idx)
            param_idx += 1
        
        # Entangling layer
        for q in range(n_qubits - 1):
            circuit.append_gate(CNOT, wires=[q, q + 1])
    
    return circuit


def select_backend():
    """Select an appropriate backend for VQE."""
    print("🔍 Finding suitable quantum backend...")
    
    try:
        # Try to connect to IBM Quantum
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print(f"✅ Connected to IBM Quantum Runtime")
        print(f"📋 Found {len(backends)} available backends")
        
        # Prefer simulators for reliable results, but show real hardware options
        simulators = []
        real_devices = []
        
        for backend in backends:
            if backend.num_qubits >= 2:  # Need at least 2 qubits for our VQE
                if backend.simulator:
                    simulators.append(backend)
                else:
                    try:
                        status = backend.status()
                        if status.operational:
                            real_devices.append((backend, status.pending_jobs))
                    except:
                        pass
        
        print("\n🎯 Available options:")
        
        # Show simulators
        if simulators:
            print("\n🖥️  Simulators (recommended for VQE):")
            for i, sim in enumerate(simulators[:3]):
                print(f"   {i+1}. {sim.name}: {sim.num_qubits} qubits")
        
        # Show real devices
        if real_devices:
            real_devices.sort(key=lambda x: x[1])  # Sort by queue length
            print("\n🔬 Real Quantum Devices:")
            for i, (device, queue) in enumerate(real_devices[:3]):
                print(f"   {i+1+len(simulators)}. {device.name}: {device.num_qubits} qubits (Queue: {queue})")
        
        # Let user choose
        total_options = len(simulators) + len(real_devices)
        if total_options == 0:
            print("❌ No suitable backends found")
            return None
        
        print(f"\n🔢 Select backend (1-{total_options}), or 0 for local simulator:")
        choice = input("Choice: ").strip()
        
        try:
            choice = int(choice)
            if choice == 0:
                return "local"
            elif 1 <= choice <= len(simulators):
                return simulators[choice - 1]
            elif len(simulators) < choice <= total_options:
                device, _ = real_devices[choice - len(simulators) - 1]
                return device
            else:
                print("❌ Invalid choice, using local simulator")
                return "local"
        except ValueError:
            print("❌ Invalid choice, using local simulator")
            return "local"
            
    except Exception as e:
        print(f"⚠️ Could not connect to IBM Quantum: {e}")
        print("Using local simulator instead")
        return "local"


def run_vqe(backend_choice, max_iterations=50):
    """Run VQE algorithm on the selected backend."""
    print(f"\n🚀 Running VQE Algorithm")
    print("=" * 40)
    
    # Create VQE circuit for H2 molecule (simplified)
    circuit = create_vqe_ansatz(n_qubits=2, n_layers=2)
    
    # H2 Hamiltonian (simplified, 2-qubit version)
    hamiltonian = {
        'ZZ': -1.0523732,  # Main interaction
        'ZI': -0.39793742,  # Single qubit terms
        'IZ': -0.39793742,
        'XX': -0.01128010,  # Exchange terms
        'YY': 0.01128010
    }
    
    print(f"🧬 Optimizing H2 molecule ground state")
    print(f"🔬 Hamiltonian: {len(hamiltonian)} terms")
    
    # Create backend
    if backend_choice == "local":
        backend = QiskitBackend(device='qasm_simulator', shots=8192)
        print(f"🖥️  Using local QASM simulator")
    else:
        backend = QiskitBackend(device=backend_choice, shots=4096)  # Lower shots for hardware
        print(f"🔬 Using {backend_choice.name}: {backend_choice.num_qubits} qubits")
    
    # Create VQE model
    vqe_model = QuantumExpectation(circuit, backend, hamiltonian)
    
    # Optimizer
    optimizer = torch.optim.Adam([circuit.trainable_params], lr=0.1)
    
    print(f"\n⚙️ Starting optimization ({max_iterations} iterations)...")
    
    best_energy = float('inf')
    energies = []
    
    try:
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute energy
            energy_tensor = vqe_model()
            total_energy = energy_tensor.sum()
            
            # Backward pass
            total_energy.backward()
            optimizer.step()
            
            current_energy = total_energy.item()
            energies.append(current_energy)
            
            if current_energy < best_energy:
                best_energy = current_energy
            
            # Print progress
            if iteration % 10 == 0 or iteration == max_iterations - 1:
                print(f"   Iter {iteration:3d}: Energy = {current_energy:.6f} Ha")
        
        print(f"\n✅ Optimization complete!")
        print(f"🎯 Best energy: {best_energy:.6f} Ha")
        print(f"📊 Theoretical H2 ground state: ≈ -1.857 Ha")
        
        error = abs(best_energy - (-1.857))
        if error < 0.5:
            print(f"✅ Good agreement! Error: {error:.3f} Ha")
        else:
            print(f"⚠️ Large error: {error:.3f} Ha (hardware noise expected)")
        
        return best_energy, energies
        
    except Exception as e:
        print(f"❌ VQE optimization failed: {e}")
        return None, []


def plot_convergence(energies):
    """Plot VQE convergence (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(energies, 'b-', linewidth=2, label='VQE Energy')
        plt.axhline(y=-1.857, color='r', linestyle='--', label='Theoretical Ground State')
        plt.xlabel('Iteration')
        plt.ylabel('Energy (Ha)')
        plt.title('VQE Convergence on Quantum Hardware')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('vqe_convergence.png', dpi=150, bbox_inches='tight')
        print("📊 Convergence plot saved as 'vqe_convergence.png'")
        
    except ImportError:
        print("⚠️ Matplotlib not available, skipping plot")


def main():
    """Main VQE hardware demo."""
    print("🧪 VQE on IBM Quantum Hardware")
    print("=" * 50)
    
    print("This example demonstrates running the Variational Quantum Eigensolver")
    print("algorithm to find the ground state of the H2 molecule using real")
    print("quantum hardware or high-fidelity simulators.")
    
    # Select backend
    backend_choice = select_backend()
    if backend_choice is None:
        print("❌ No backend available")
        return False
    
    # Ask for number of iterations
    print(f"\n⏱️ How many optimization iterations?")
    print("   Simulators: 50-100 iterations recommended")
    print("   Real hardware: 20-30 iterations (due to queue time)")
    
    try:
        max_iter = int(input("Iterations (press Enter for 30): ").strip() or "30")
        max_iter = max(1, min(max_iter, 200))  # Reasonable bounds
    except ValueError:
        max_iter = 30
    
    # Run VQE
    best_energy, energies = run_vqe(backend_choice, max_iter)
    
    if best_energy is not None:
        # Plot results if we have data
        if len(energies) > 1:
            plot_convergence(energies)
        
        print("\n🎉 VQE Demo Complete!")
        print("\n📋 Summary:")
        print(f"   Backend: {backend_choice if isinstance(backend_choice, str) else backend_choice.name}")
        print(f"   Iterations: {len(energies)}")
        print(f"   Final energy: {best_energy:.6f} Ha")
        print(f"   Target energy: -1.857 Ha")
        print(f"   Error: {abs(best_energy - (-1.857)):.3f} Ha")
        
        return True
    else:
        print("❌ VQE demo failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
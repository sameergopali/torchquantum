#!/usr/bin/env python3
"""
Quick test to verify updated Qiskit imports work correctly.
"""

import torch
import sys

try:
    print("Testing updated Qiskit imports...")
    
    # Test the new imports
    from qiskit import execute, transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    print("✓ Successfully imported qiskit_aer components")
    
    # Test AerSimulator creation
    simulator = AerSimulator()
    print(f"✓ Created AerSimulator: {simulator.name}")
    
    # Test available methods
    methods = simulator.available_methods()
    print(f"✓ Available simulation methods: {methods}")
    
    # Test backend creation with different methods
    qasm_sim = AerSimulator(method='automatic')
    sv_sim = AerSimulator(method='statevector')
    print(f"✓ Created simulators: QASM={qasm_sim.name}, Statevector={sv_sim.name}")
    
    # Test TorchQuantum backend
    from torchquantum.backend import get_backend
    backend = get_backend('qiskit', shots=1024, seed=42)
    print(f"✓ Created TorchQuantum Qiskit backend: {backend.get_backend_info()['name']}")
    
    # Test simple circuit execution
    from torchquantum.backend.core import ParameterizedQuantumCircuit
    from torchquantum.operator.standard_gates import Hadamard
    
    circuit = ParameterizedQuantumCircuit(n_wires=1, n_input_params=0)
    circuit.append_gate(Hadamard, wires=0)
    
    # Test expectation computation
    from torchquantum.backend.core import QuantumExpectation
    exp_module = QuantumExpectation(circuit, backend, ['Z'])
    result = exp_module()
    print(f"✓ Expectation computation works: <Z> = {result[0, 0].item():.4f}")
    
    print("\n🎉 All Qiskit import tests PASSED!")
    print("✓ qiskit_aer imports work correctly")
    print("✓ AerSimulator creation works")
    print("✓ TorchQuantum integration works")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure to install the latest Qiskit and qiskit-aer:")
    print("  pip install qiskit qiskit-aer")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 
#!/usr/bin/env python3
"""
Basic test for Phase 1 of Qiskit backend implementation.

This test verifies that the core infrastructure is working:
- Backend initialization
- Circuit conversion
- Basic module creation
- Parameter handling
"""

import torch
import sys
import traceback

try:
    # Test imports
    from torchquantum.backend import get_backend, list_backends
    from torchquantum.backend.core import ParameterizedQuantumCircuit
    print("✓ Successfully imported TorchQuantum backend components")
    
    # Check available backends
    backends = list_backends()
    print(f"✓ Available backends: {backends}")
    
    # Test Qiskit backend availability
    if 'qiskit' not in backends:
        print("⚠ Qiskit backend not available. This is expected if Qiskit is not installed.")
        print("To test Qiskit backend, install Qiskit: pip install qiskit")
        sys.exit(0)
    
    print("✓ Qiskit backend is available")
    
    # Test backend creation
    try:
        backend = get_backend('qiskit', shots=1024, seed=42)
        print(f"✓ Created Qiskit backend: {backend.get_backend_info()['name']}")
    except Exception as e:
        print(f"✗ Failed to create Qiskit backend: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test circuit creation
    try:
        from torchquantum.operator.standard_gates import Hadamard, CNOT, RX
        
        circuit = ParameterizedQuantumCircuit(n_wires=2, n_input_params=1)
        circuit.append_gate(Hadamard, wires=0)
        circuit.append_gate(CNOT, wires=[0, 1])
        circuit.append_gate(RX, wires=0, input_idx=0)  # Input parameterized gate
        print(f"✓ Created circuit with {circuit.n_wires} qubits and {circuit.n_input_params} parameters")
    except Exception as e:
        print(f"✗ Failed to create circuit: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test expectation module creation
    try:
        exp_module = backend._create_expectation_module(circuit, ['ZZ', 'XX'])
        print("✓ Created expectation module")
        
        # Test forward pass with placeholder
        params = torch.randn(2, 1)  # batch_size=2, n_params=1
        result = exp_module(params)
        print(f"✓ Expectation module forward pass: shape {result.shape}")
    except Exception as e:
        print(f"✗ Failed expectation module test: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test amplitude module creation
    try:
        amp_module = backend._create_amplitude_module(circuit, ['00', '01', '10', '11'])
        print("✓ Created amplitude module")
        
        # Test forward pass with placeholder
        result = amp_module(params)
        print(f"✓ Amplitude module forward pass: shape {result.shape}, dtype {result.dtype}")
    except Exception as e:
        print(f"✗ Failed amplitude module test: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test sampling module creation
    try:
        samp_module = backend._create_sampling_module(circuit, n_samples=100)
        print("✓ Created sampling module")
        
        # Test forward pass with placeholder
        result = samp_module(params)
        print(f"✓ Sampling module forward pass: shape {result.shape}, dtype {result.dtype}")
    except Exception as e:
        print(f"✗ Failed sampling module test: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test circuit conversion (basic test)
    try:
        from torchquantum.backend.qiskit_backend.utils import convert_tq_circuit_to_qiskit
        qiskit_circuit, qiskit_params = convert_tq_circuit_to_qiskit(circuit)
        print(f"✓ Circuit conversion: {qiskit_circuit.num_qubits} qubits, {len(qiskit_params)} parameters")
        print(f"  Qiskit circuit depth: {qiskit_circuit.depth()}")
    except Exception as e:
        print(f"✗ Failed circuit conversion test: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test parameter binding
    try:
        from torchquantum.backend.qiskit_backend.utils import create_parameter_binds
        params_tensor = torch.tensor([[0.5], [1.0]])  # 2 batches, 1 param each
        binds = create_parameter_binds(qiskit_params, params_tensor)
        print(f"✓ Parameter binding: {len(binds)} bindings created")
    except Exception as e:
        print(f"✗ Failed parameter binding test: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 Phase 1 implementation test PASSED!")
    print("✓ Backend initialization works")
    print("✓ Circuit conversion works") 
    print("✓ Module creation works")
    print("✓ Parameter handling works")
    print("\nReady for Phase 2 implementation (full measurement functionality)")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure TorchQuantum is properly installed")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1) 
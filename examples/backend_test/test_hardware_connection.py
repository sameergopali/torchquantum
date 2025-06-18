#!/usr/bin/env python3
"""Test script for IBM Quantum Runtime hardware connection."""

import os
import sys
import warnings
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import qiskit
        print(f"✅ qiskit: {qiskit.__version__}")
    except ImportError:
        print("❌ qiskit not found. Install with: pip install qiskit")
        return False
    
    try:
        import qiskit_ibm_runtime
        print(f"✅ qiskit-ibm-runtime: {qiskit_ibm_runtime.__version__}")
    except ImportError:
        print("❌ qiskit-ibm-runtime not found. Install with: pip install qiskit-ibm-runtime")
        return False
    
    try:
        from torchquantum.backend.qiskit_backend import QiskitBackend, HardwareManager
        print("✅ TorchQuantum Qiskit backend available")
    except ImportError as e:
        print(f"❌ TorchQuantum backend import failed: {e}")
        return False
    
    return True


def test_hardware_manager_creation():
    """Test hardware manager creation."""
    print("\n🔧 Testing Hardware Manager Creation...")
    
    try:
        from torchquantum.backend.qiskit_backend import HardwareManager
        
        # Test with default parameters
        manager = HardwareManager()
        print("✅ Hardware manager created successfully")
        
        # Test with custom parameters
        manager_custom = HardwareManager(
            channel='ibm_quantum',
            instance='ibm-q/open/main'
        )
        print("✅ Hardware manager with custom parameters created")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware manager creation failed: {e}")
        return False


def test_runtime_service_connection():
    """Test connection to IBM Quantum Runtime service."""
    print("\n🌐 Testing IBM Quantum Runtime Connection...")
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        # Try to initialize service (will use saved credentials if available)
        try:
            service = QiskitRuntimeService()
            print("✅ Connected to IBM Quantum Runtime service")
            
            # List available backends
            backends = service.backends()
            print(f"✅ Found {len(backends)} available backends")
            
            return service, backends
            
        except Exception as e:
            print(f"⚠️ Connection failed: {e}")
            print("\n📋 To set up IBM Quantum access:")
            print("1. Create account at: https://quantum-computing.ibm.com/")
            print("2. Get your API token from the account dashboard")
            print("3. Save credentials:")
            print("   from qiskit_ibm_runtime import QiskitRuntimeService")
            print("   QiskitRuntimeService.save_account(token='YOUR_TOKEN')")
            print("4. Re-run this test")
            
            return None, []
            
    except ImportError:
        print("❌ qiskit-ibm-runtime not available")
        return None, []


def list_available_backends(service, backends):
    """List and categorize available backends."""
    if not service or not backends:
        return
    
    print("\n📋 Available Quantum Backends:")
    print("-" * 60)
    
    simulators = []
    real_devices = []
    
    for backend in backends:
        try:
            info = {
                'name': backend.name,
                'n_qubits': backend.num_qubits,
                'simulator': backend.simulator,
                'operational': True
            }
            
            # Check if backend is operational
            try:
                status = backend.status()
                info['operational'] = status.operational
                info['pending_jobs'] = getattr(status, 'pending_jobs', 'N/A')
            except:
                pass
            
            if info['simulator']:
                simulators.append(info)
            else:
                real_devices.append(info)
                
        except Exception as e:
            print(f"⚠️ Error getting info for {backend.name}: {e}")
    
    # Display simulators
    if simulators:
        print("\n🖥️  Simulators:")
        for sim in simulators:
            print(f"  • {sim['name']}: {sim['n_qubits']} qubits")
    
    # Display real devices
    if real_devices:
        print("\n🔬 Real Quantum Devices:")
        for device in real_devices:
            status_icon = "🟢" if device['operational'] else "🔴"
            pending = device.get('pending_jobs', 'N/A')
            print(f"  {status_icon} {device['name']}: {device['n_qubits']} qubits (Queue: {pending})")
    else:
        print("\n⚠️ No real quantum devices available (may require premium access)")
    
    return real_devices


def test_torchquantum_integration(service, backends):
    """Test TorchQuantum integration with real hardware."""
    print("\n🔗 Testing TorchQuantum Hardware Integration...")
    
    if not service or not backends:
        print("⚠️ Skipping integration test - no service connection")
        return False
    
    try:
        from torchquantum.backend.qiskit_backend import HardwareManager, setup_hardware_backend
        from torchquantum.backend.qiskit_backend import QiskitBackend
        
        # Create hardware manager and connect
        manager = HardwareManager()
        
        # Mock the service connection (since we already have it)
        manager.service = service
        manager._available_backends = backends
        
        print("✅ TorchQuantum hardware manager connected")
        
        # List backends through TorchQuantum
        available_backends = manager.list_available_backends()
        print(f"✅ TorchQuantum found {len(available_backends)} backends")
        
        # Test backend info retrieval
        if available_backends:
            test_backend_name = available_backends[0]
            backend_info = manager.get_backend_info(test_backend_name)
            print(f"✅ Retrieved info for {test_backend_name}")
            print(f"   Qubits: {backend_info.get('n_qubits', 'N/A')}")
            print(f"   Simulator: {backend_info.get('simulator', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ TorchQuantum integration test failed: {e}")
        return False


def test_simple_circuit_execution(service, backends):
    """Test executing a simple circuit on hardware/simulator."""
    print("\n⚙️ Testing Circuit Execution...")
    
    if not service or not backends:
        print("⚠️ Skipping circuit execution - no service connection")
        return False
    
    try:
        from torchquantum.backend import ParameterizedQuantumCircuit, QuantumExpectation
        from torchquantum.backend.qiskit_backend import QiskitBackend
        from torchquantum.operator.standard_gates import Hadamard, CNOT
        
        # Find a suitable backend (prefer simulator for testing)
        test_backend = None
        for backend in backends:
            if backend.simulator and backend.num_qubits >= 2:
                test_backend = backend
                break
        
        if not test_backend:
            # Fall back to first available backend with enough qubits
            for backend in backends:
                if backend.num_qubits >= 2:
                    test_backend = backend
                    break
        
        if not test_backend:
            print("⚠️ No suitable backend found for testing")
            return False
        
        print(f"🎯 Testing with backend: {test_backend.name}")
        
        # Create a simple Bell state circuit
        circuit = ParameterizedQuantumCircuit(n_wires=2, n_trainable_params=0)
        circuit.append_gate(Hadamard, wires=0)
        circuit.append_gate(CNOT, wires=[0, 1])
        
        # Create TorchQuantum backend pointing to the hardware
        tq_backend = QiskitBackend(
            device=test_backend,  # Use the actual hardware backend
            shots=1024,
            enable_advanced_features=True
        )
        
        print(f"✅ Created TorchQuantum backend for {test_backend.name}")
        
        # Test expectation value computation
        observables = ['ZZ']
        expectation = QuantumExpectation(circuit, tq_backend, observables)
        
        print("🚀 Executing Bell state circuit...")
        result = expectation()
        
        expected_value = result[0, 0].item()
        print(f"✅ Circuit executed successfully!")
        print(f"   <ZZ> expectation value: {expected_value:.4f}")
        
        # Validate result makes sense (should be close to 1.0 for perfect Bell state)
        if abs(expected_value - 1.0) < 0.3:  # Allow for noise
            print("✅ Result looks reasonable for Bell state")
        else:
            print(f"⚠️ Unexpected result (expected ~1.0, got {expected_value:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Circuit execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive hardware connection test."""
    print("🚀 IBM Quantum Runtime Hardware Connection Test")
    print("=" * 60)
    
    # Test 1: Dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return False
    
    # Test 2: Hardware Manager Creation
    if not test_hardware_manager_creation():
        print("\n❌ Hardware manager creation failed.")
        return False
    
    # Test 3: Runtime Service Connection
    service, backends = test_runtime_service_connection()
    
    # Test 4: List Available Backends
    real_devices = list_available_backends(service, backends)
    
    # Test 5: TorchQuantum Integration
    integration_success = test_torchquantum_integration(service, backends)
    
    # Test 6: Simple Circuit Execution
    execution_success = test_simple_circuit_execution(service, backends)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Dependencies", True),
        ("Hardware Manager", True),
        ("Runtime Connection", service is not None),
        ("Backend Listing", len(backends) > 0),
        ("TorchQuantum Integration", integration_success),
        ("Circuit Execution", execution_success)
    ]
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<25} {status}")
    
    overall_success = all(success for _, success in tests)
    
    if overall_success:
        print("\n🎉 All tests passed! TorchQuantum is ready for quantum hardware!")
        if real_devices:
            print(f"🔬 {len(real_devices)} real quantum devices available")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
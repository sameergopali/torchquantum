# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Noise model integration for Qiskit backend."""

import warnings
from typing import Optional, List, Dict, Union

try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
    from qiskit_aer.noise import thermal_relaxation_error, ReadoutError, pauli_error
    from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    NoiseModel = object


def _get_gate_qubit_counts() -> Dict[str, List[str]]:
    """Get a mapping of qubit counts to gate names."""
    return {
        1: ['h', 'x', 'y', 'z', 's', 't', 'sx', 'rx', 'ry', 'rz', 'p', 'u1', 'u2', 'u3', 'reset'],
        2: ['cx', 'cnot', 'cz', 'cy', 'swap', 'rxx', 'ryy', 'rzz', 'rzx', 'iswap'],
        3: ['cswap', 'ccx', 'toffoli', 'fredkin']
    }


def create_depolarizing_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    three_qubit_error: Optional[float] = None,
    readout_error: float = 0.02
) -> Optional[NoiseModel]:
    """Create a simple depolarizing noise model.
    
    Args:
        single_qubit_error: Single-qubit depolarizing error probability
        two_qubit_error: Two-qubit depolarizing error probability
        three_qubit_error: Three-qubit depolarizing error probability (auto-calculated if None)
        readout_error: Readout error probability
        
    Returns:
        NoiseModel or None if Qiskit not available
    """
    if not QISKIT_AVAILABLE:
        warnings.warn("Qiskit not available, cannot create noise model")
        return None
    
    # Create noise model
    noise_model = NoiseModel()
    
    # Get gate categorization
    gate_counts = _get_gate_qubit_counts()
    
    # Single-qubit depolarizing errors
    if single_qubit_error > 0:
        single_error = depolarizing_error(single_qubit_error, 1)
        for gate in gate_counts[1]:
            noise_model.add_all_qubit_quantum_error(single_error, gate)
    
    # Two-qubit depolarizing errors
    if two_qubit_error > 0:
        two_error = depolarizing_error(two_qubit_error, 2)
        for gate in gate_counts[2]:
            noise_model.add_all_qubit_quantum_error(two_error, gate)
    
    # Three-qubit depolarizing errors
    if three_qubit_error is None:
        three_qubit_error = two_qubit_error * 1.5 if two_qubit_error > 0 else 0
    
    if three_qubit_error > 0:
        three_error = depolarizing_error(three_qubit_error, 3)
        for gate in gate_counts[3]:
            noise_model.add_all_qubit_quantum_error(three_error, gate)
    
    # Readout errors
    if readout_error > 0:
        readout_err = ReadoutError([[1 - readout_error, readout_error], 
                                   [readout_error, 1 - readout_error]])
        noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model


def create_thermal_noise_model(
    t1_time: float = 50e-6,  # T1 relaxation time (50 μs)
    t2_time: float = 70e-6,  # T2 dephasing time (70 μs)  
    gate_time: float = 0.1e-6,  # Gate time (100 ns)
    readout_error: float = 0.02
) -> Optional[NoiseModel]:
    """Create a thermal relaxation noise model.
    
    Applies thermal relaxation (T1/T2) errors to single-qubit gates and
    depolarizing errors to multi-qubit gates (scaled by gate time).
    
    Args:
        t1_time: T1 relaxation time in seconds
        t2_time: T2 dephasing time in seconds
        gate_time: Gate execution time in seconds
        readout_error: Readout error probability
        
    Returns:
        NoiseModel or None if Qiskit not available
    """
    if not QISKIT_AVAILABLE:
        warnings.warn("Qiskit not available, cannot create noise model")
        return None
    
    # Create noise model
    noise_model = NoiseModel()
    
    # Get gate categorization
    gate_counts = _get_gate_qubit_counts()
    
    # Thermal relaxation error for single-qubit gates only
    # (T1/T2 relaxation is inherently a single-qubit phenomenon)
    single_thermal_error = thermal_relaxation_error(t1_time, t2_time, gate_time)
    for gate in gate_counts[1]:
        noise_model.add_all_qubit_quantum_error(single_thermal_error, gate)
    
    # For multi-qubit gates, use depolarizing errors with rates derived from gate times
    # Two-qubit gate errors (use depolarizing error scaled by gate time)
    two_qubit_error_rate = gate_time * 2 / t1_time * 0.1  # Scale with gate time and T1
    if two_qubit_error_rate > 0:
        two_qubit_depol_error = depolarizing_error(min(two_qubit_error_rate, 0.1), 2)
        for gate in gate_counts[2]:
            noise_model.add_all_qubit_quantum_error(two_qubit_depol_error, gate)
    
    # Three-qubit gate errors (higher error rate due to longer gate time)
    three_qubit_error_rate = gate_time * 3 / t1_time * 0.15  # Scale with gate time and T1
    if three_qubit_error_rate > 0:
        three_qubit_depol_error = depolarizing_error(min(three_qubit_error_rate, 0.15), 3)
        for gate in gate_counts[3]:
            noise_model.add_all_qubit_quantum_error(three_qubit_depol_error, gate)
    
    # Readout errors
    if readout_error > 0:
        readout_err = ReadoutError([[1 - readout_error, readout_error], 
                                   [readout_error, 1 - readout_error]])
        noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model


def create_device_noise_model(device_name: str) -> Optional[NoiseModel]:
    """Create a noise model based on a real device.
    
    Args:
        device_name: Name of the device to simulate
        
    Returns:
        NoiseModel or None if device not found
    """
    if not QISKIT_AVAILABLE:
        warnings.warn("Qiskit not available, cannot create noise model")
        return None
    
    # This would require access to IBM Quantum Network
    # For now, return a representative noise model
    device_configs = {
        'ibmq_qasm_simulator': create_depolarizing_noise_model(0.001, 0.01, 0.02),
        'ibmq_lima': create_thermal_noise_model(100e-6, 150e-6, 0.1e-6, 0.03),
        'ibmq_belem': create_thermal_noise_model(80e-6, 120e-6, 0.1e-6, 0.025),
        'ibmq_quito': create_thermal_noise_model(90e-6, 130e-6, 0.1e-6, 0.028),
    }
    
    if device_name in device_configs:
        return device_configs[device_name]
    else:
        warnings.warn(f"Device {device_name} not found, using default noise model")
        return create_depolarizing_noise_model()


def apply_noise_to_backend(backend, noise_type: str = 'depolarizing', **kwargs):
    """Apply noise model to a Qiskit backend.
    
    Args:
        backend: QiskitBackend instance
        noise_type: Type of noise ('depolarizing', 'thermal', 'device')
        **kwargs: Noise parameters
    """
    if noise_type == 'depolarizing':
        noise_model = create_depolarizing_noise_model(**kwargs)
    elif noise_type == 'thermal':
        noise_model = create_thermal_noise_model(**kwargs)
    elif noise_type == 'device':
        device_name = kwargs.get('device_name', 'ibmq_qasm_simulator')
        noise_model = create_device_noise_model(device_name)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    backend.set_noise_model(noise_model)
    return noise_model


class NoiseModelBuilder:
    """Builder class for creating custom noise models."""
    
    def __init__(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for NoiseModelBuilder")
        self.noise_model = NoiseModel()
    
    def add_depolarizing_error(self, probability: float, gates: List[str], num_qubits: int = 1):
        """Add depolarizing error to specified gates."""
        error = depolarizing_error(probability, num_qubits)
        for gate in gates:
            self.noise_model.add_all_qubit_quantum_error(error, gate)
        return self
    
    def add_thermal_error(self, t1: float, t2: float, gate_time: float, gates: List[str]):
        """Add thermal relaxation error to specified gates."""
        error = thermal_relaxation_error(t1, t2, gate_time)
        for gate in gates:
            self.noise_model.add_all_qubit_quantum_error(error, gate)
        return self
    
    def add_readout_error(self, probability: float):
        """Add readout error to all qubits."""
        error = ReadoutError([[1 - probability, probability], 
                             [probability, 1 - probability]])
        self.noise_model.add_all_qubit_readout_error(error)
        return self
    
    def add_pauli_error(self, pauli_list: List[tuple], gates: List[str]):
        """Add Pauli error to specified gates.
        
        Args:
            pauli_list: List of (Pauli_string, probability) tuples
            gates: List of gate names
        """
        error = pauli_error(pauli_list)
        for gate in gates:
            self.noise_model.add_all_qubit_quantum_error(error, gate)
        return self
    
    def build(self) -> NoiseModel:
        """Build and return the noise model."""
        return self.noise_model 
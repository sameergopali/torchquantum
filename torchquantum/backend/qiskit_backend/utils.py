# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Utility functions for Qiskit backend circuit conversion and processing."""

import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

from ..core.circuit import ParameterizedQuantumCircuit, _ParameterizedQuantumGate


def create_qiskit_circuit(n_qubits: int, n_params: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Create a parameterized Qiskit circuit.
    
    Args:
        n_qubits: Number of qubits
        n_params: Number of parameters
        
    Returns:
        Tuple of (QuantumCircuit, parameter list)
    """
    circuit = QuantumCircuit(n_qubits)
    
    # Create parameters
    params = []
    for i in range(n_params):
        param = Parameter(f'theta_{i}')
        params.append(param)
    
    return circuit, params


def convert_tq_gate_to_qiskit(
    qiskit_circuit: QuantumCircuit, 
    gate: _ParameterizedQuantumGate,
    qiskit_params: List[Parameter],
    param_offset: int = 0
) -> int:
    """Convert a TorchQuantum gate to Qiskit and add to circuit.
    
    Args:
        qiskit_circuit: Target Qiskit circuit
        gate: TorchQuantum gate to convert
        qiskit_params: List of Qiskit parameters
        param_offset: Offset for parameter indexing
        
    Returns:
        Number of parameters consumed
    """
    # Use the stored operator name
    gate_name = gate.op_name.lower()
    
    wires = gate.wires
    n_params_used = 0
    
    # Check if gate has input parameters (parameters that come from circuit input)
    has_input_params = any(idx is not None for idx in gate.input_idx)
    
    # Handle different gate types
    if gate_name == 'hadamard':
        qiskit_circuit.h(wires[0])
    elif gate_name == 'paulix':
        qiskit_circuit.x(wires[0])
    elif gate_name == 'pauliy':
        qiskit_circuit.y(wires[0])
    elif gate_name == 'pauliz':
        qiskit_circuit.z(wires[0])
    elif gate_name == 's':
        qiskit_circuit.s(wires[0])
    elif gate_name == 't':
        qiskit_circuit.t(wires[0])
    elif gate_name == 'sx':
        qiskit_circuit.sx(wires[0])
    elif gate_name == 'cnot':
        qiskit_circuit.cnot(wires[0], wires[1])
    elif gate_name == 'cz':
        qiskit_circuit.cz(wires[0], wires[1])
    elif gate_name == 'cy':
        qiskit_circuit.cy(wires[0], wires[1])
    elif gate_name == 'swap':
        qiskit_circuit.swap(wires[0], wires[1])
    elif gate_name == 'cswap':
        qiskit_circuit.cswap(wires[0], wires[1], wires[2])
    elif gate_name == 'toffoli' or gate_name == 'ccx':
        qiskit_circuit.ccx(wires[0], wires[1], wires[2])
    
    # Parameterized single-qubit gates
    elif 'rx' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.rx(param, wires[0])
    elif 'ry' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.ry(param, wires[0])
    elif 'rz' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.rz(param, wires[0])
    elif 'phaseshift' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.p(param, wires[0])
    
    # Parameterized two-qubit gates
    elif 'rxx' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.rxx(param, wires[0], wires[1])
    elif 'ryy' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.ryy(param, wires[0], wires[1])
    elif 'rzz' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.rzz(param, wires[0], wires[1])
    elif 'rzx' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.rzx(param, wires[0], wires[1])
    
    # Controlled parameterized gates
    elif 'crx' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.crx(param, wires[0], wires[1])
    elif 'cry' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.cry(param, wires[0], wires[1])
    elif 'crz' in gate_name:
        if has_input_params:
            param = qiskit_params[param_offset]
            n_params_used = 1
        else:
            param = gate.params[0].item()
        qiskit_circuit.crz(param, wires[0], wires[1])
    
    # Universal gates
    elif 'u3' in gate_name:
        if has_input_params:
            params_slice = qiskit_params[param_offset:param_offset+3]
            qiskit_circuit.u(*params_slice, wires[0])
            n_params_used = 3
        else:
            theta = gate.params[0].item()
            phi = gate.params[1].item()
            lam = gate.params[2].item()
            qiskit_circuit.u(theta, phi, lam, wires[0])
    
    else:
        raise NotImplementedError(f"Gate with name '{gate_name}' not implemented for Qiskit conversion")
    
    return n_params_used


def convert_tq_circuit_to_qiskit(circuit: ParameterizedQuantumCircuit) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Convert a ParameterizedQuantumCircuit to a Qiskit QuantumCircuit.
    
    Args:
        circuit: TorchQuantum ParameterizedQuantumCircuit
        
    Returns:
        Tuple of (Qiskit QuantumCircuit, parameter list)
    """
    # Count total input parameters needed
    total_input_params = 0
    for gate in circuit.gates:
        input_params_in_gate = sum(1 for idx in gate.input_idx if idx is not None)
        total_input_params += input_params_in_gate
    
    # Create base Qiskit circuit with the actual number of input parameters used
    qiskit_circuit, qiskit_params = create_qiskit_circuit(
        circuit.n_wires, 
        total_input_params
    )
    
    # Convert gates
    param_offset = 0
    for gate in circuit.gates:
        n_params_used = convert_tq_gate_to_qiskit(
            qiskit_circuit, gate, qiskit_params, param_offset
        )
        param_offset += n_params_used
    
    return qiskit_circuit, qiskit_params


def create_parameter_binds(
    qiskit_params: List[Parameter],
    input_params: torch.Tensor
) -> List[Dict[Parameter, float]]:
    """Create parameter binding dictionaries for Qiskit execution.
    
    Args:
        qiskit_params: List of Qiskit parameters
        input_params: Input parameter tensor [batch_size, n_params]
        
    Returns:
        List of parameter binding dictionaries
    """
    if input_params is None:
        return [{}]
    
    # Ensure 2D tensor
    if input_params.dim() == 1:
        input_params = input_params.unsqueeze(0)
    
    binds = []
    for batch_idx in range(input_params.shape[0]):
        bind_dict = {}
        for param_idx, qiskit_param in enumerate(qiskit_params):
            if param_idx < input_params.shape[1]:
                bind_dict[qiskit_param] = input_params[batch_idx, param_idx].item()
        binds.append(bind_dict)
    
    return binds


def get_expectations_from_counts(
    counts_list: List[Dict[str, int]], 
    n_wires: int
) -> List[List[float]]:
    """Extract expectation values from Qiskit measurement counts.
    
    This function converts measurement counts to expectation values for
    Z measurements on each qubit.
    
    Args:
        counts_list: List of count dictionaries from Qiskit
        n_wires: Number of qubits
        
    Returns:
        List of expectation values for each batch and each qubit
    """
    expectations = []
    
    for counts in counts_list:
        if isinstance(counts, list):
            # Handle nested lists from parallel execution
            batch_expectations = []
            for count_dict in counts:
                exp_vals = _compute_z_expectations(count_dict, n_wires)
                batch_expectations.append(exp_vals)
            expectations.extend(batch_expectations)
        else:
            # Single count dictionary
            exp_vals = _compute_z_expectations(counts, n_wires)
            expectations.append(exp_vals)
    
    return expectations


def _compute_z_expectations(counts: Dict[str, int], n_wires: int) -> List[float]:
    """Compute Z expectation values from measurement counts."""
    total_shots = sum(counts.values())
    expectations = []
    
    for qubit_idx in range(n_wires):
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Qiskit uses big-endian, so bit 0 is rightmost
            bit_idx = n_wires - 1 - qubit_idx
            if bit_idx < len(bitstring):
                bit_value = int(bitstring[bit_idx])
                # Z eigenvalue: 0 -> +1, 1 -> -1
                eigenvalue = 1.0 - 2.0 * bit_value
                expectation += eigenvalue * count
        
        expectation /= total_shots
        expectations.append(expectation)
    
    return expectations 
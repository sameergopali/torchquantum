# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Expectation value computation using Qiskit backend."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Union

try:
    from qiskit import QuantumCircuit, ClassicalRegister, execute
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.circuit import ParameterizedQuantumCircuit
from .utils import convert_tq_circuit_to_qiskit, create_parameter_binds


class QiskitExpectation(nn.Module):
    """PyTorch module for computing expectation values using Qiskit backend.
    
    This module uses shot-based sampling to compute expectation values
    of Pauli operators, providing realistic quantum simulation with
    statistical noise.
    """
    
    def __init__(
        self,
        circuit: ParameterizedQuantumCircuit,
        backend: 'QiskitBackend',
        pauli_ops: Union[List[str], Dict[str, float]]
    ):
        super().__init__()
        self.circuit = circuit.copy()
        self.backend = backend
        self.pauli_ops = pauli_ops.copy() if isinstance(pauli_ops, list) else pauli_ops.copy()
        
        # Prepare circuits for each Pauli operator
        self._prepare_measurement_circuits()
        
    def _prepare_measurement_circuits(self):
        """Prepare measurement circuits for each Pauli operator."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitExpectation")
            
        self.measurement_circuits = {}
        self.qiskit_params = None
        
        # Convert base circuit to Qiskit
        base_qiskit_circuit, qiskit_params = convert_tq_circuit_to_qiskit(self.circuit)
        self.qiskit_params = qiskit_params
        
        # Handle different pauli_ops formats
        pauli_strings = set()
        if isinstance(self.pauli_ops, list):
            for item in self.pauli_ops:
                if isinstance(item, str):
                    pauli_strings.add(item)
                elif isinstance(item, dict):
                    pauli_strings.update(item.keys())
        else:
            # Single dict format
            pauli_strings.update(self.pauli_ops.keys())
        
        pauli_strings = list(pauli_strings)
        
        # Create measurement circuits for each unique Pauli string
        for pauli_string in pauli_strings:
            circuit = self._create_pauli_measurement_circuit(base_qiskit_circuit, pauli_string)
            self.measurement_circuits[pauli_string] = circuit
    
    def _create_pauli_measurement_circuit(self, base_circuit: QuantumCircuit, pauli_string: str) -> QuantumCircuit:
        """Create a measurement circuit for a specific Pauli operator.
        
        Args:
            base_circuit: Base quantum circuit
            pauli_string: Pauli string like 'XYZI'
            
        Returns:
            Circuit with basis rotation and measurements
        """
        # Copy the base circuit
        circuit = base_circuit.copy()
        
        # Add classical register for measurements
        n_qubits = len(pauli_string)
        if len(circuit.cregs) == 0:
            creg = ClassicalRegister(n_qubits, 'c')
            circuit.add_register(creg)
        
        # Add basis rotation gates based on Pauli operator
        for qubit_idx, pauli in enumerate(pauli_string):
            if pauli.upper() == 'X':
                # Rotate from X basis to Z basis
                circuit.h(qubit_idx)
            elif pauli.upper() == 'Y':
                # Rotate from Y basis to Z basis
                circuit.sdg(qubit_idx)  # S†
                circuit.h(qubit_idx)
            # Z and I don't need rotation
        
        # Add measurements
        for qubit_idx in range(min(n_qubits, circuit.num_qubits)):
            circuit.measure(qubit_idx, qubit_idx)
        
        return circuit
    
    def _compute_pauli_expectation(self, counts: Dict[str, int], pauli_string: str) -> float:
        """Compute expectation value from measurement counts.
        
        Args:
            counts: Measurement counts from Qiskit
            pauli_string: Pauli string
            
        Returns:
            Expectation value
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Compute parity for non-identity Pauli operators
            parity = 0
            for qubit_idx, pauli in enumerate(pauli_string):
                if pauli.upper() != 'I':
                    # Qiskit uses big-endian, so we need to reverse the index
                    bit_idx = len(bitstring) - 1 - qubit_idx
                    if bit_idx >= 0 and bit_idx < len(bitstring):
                        bit_value = int(bitstring[bit_idx])
                        parity ^= bit_value
            
            # Even parity -> +1, odd parity -> -1
            eigenvalue = 1.0 - 2.0 * parity
            expectation += eigenvalue * count
        
        return expectation / total_shots
        
    def forward(self, input_params=None):
        """Compute expectation values for the specified Pauli operators.
        
        Args:
            input_params: Input parameters tensor [batch_size, n_params]
            
        Returns:
            Tensor of expectation values [batch_size, n_operators]
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitExpectation")
            
        # Determine batch size
        if input_params is None:
            batch_size = 1
        elif isinstance(input_params, torch.Tensor):
            batch_size = input_params.shape[0] if input_params.dim() > 1 else 1
        else:
            batch_size = 1
        
        # Create parameter bindings
        parameter_binds = create_parameter_binds(self.qiskit_params, input_params)
        
        # Execute circuits and collect results
        all_expectations = []
        
        # Process each observable
        for observable in self.pauli_ops:
            if isinstance(observable, str):
                # Simple Pauli string
                circuit = self.measurement_circuits[observable]
                expectations_for_pauli = []
                
                # Execute for each parameter binding
                for bind in parameter_binds:
                    counts = self._execute_single_circuit(circuit, bind)
                    exp_val = self._compute_pauli_expectation(counts, observable)
                    expectations_for_pauli.append(exp_val)
                
                all_expectations.append(expectations_for_pauli)
                
            elif isinstance(observable, dict):
                # Linear combination of Pauli strings
                expectations_for_combo = []
                
                # Execute for each parameter binding
                for bind in parameter_binds:
                    combo_expectation = 0.0
                    
                    # Compute linear combination
                    for pauli_string, coeff in observable.items():
                        circuit = self.measurement_circuits[pauli_string]
                        counts = self._execute_single_circuit(circuit, bind)
                        exp_val = self._compute_pauli_expectation(counts, pauli_string)
                        combo_expectation += coeff * exp_val
                    
                    expectations_for_combo.append(combo_expectation)
                
                all_expectations.append(expectations_for_combo)
        
        # Transpose to get [batch_size, n_operators]
        result = torch.tensor(all_expectations).T
        
        return result
    
    def _execute_single_circuit(self, circuit: QuantumCircuit, parameter_bind: Dict) -> Dict[str, int]:
        """Execute a single circuit with parameter binding.
        
        Args:
            circuit: Qiskit circuit to execute
            parameter_bind: Parameter binding dictionary
            
        Returns:
            Measurement counts
        """
        # Bind parameters directly to the circuit if there are parameters
        if parameter_bind:
            bound_circuit = circuit.assign_parameters(parameter_bind)
        else:
            bound_circuit = circuit
        
        # Transpile circuit
        transpiled_circuit = self.backend._transpile_circuit(bound_circuit)
        
        # Execute without parameter_binds since we already bound them
        job = execute(
            experiments=transpiled_circuit,
            backend=self.backend.backend,
            shots=self.backend.shots,
            seed_simulator=self.backend.seed,
            noise_model=self.backend.noise_model,
            optimization_level=0  # Already transpiled
        )
        
        result = job.result()
        counts = result.get_counts()
        
        # Handle different return formats
        if isinstance(counts, list):
            return counts[0] if counts else {}
        else:
            return counts 
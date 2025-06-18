# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Quantum state sampling using Qiskit backend."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict

try:
    from qiskit import QuantumCircuit, ClassicalRegister, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from ..core.circuit import ParameterizedQuantumCircuit
from .utils import convert_tq_circuit_to_qiskit, create_parameter_binds


class QiskitSampling(nn.Module):
    """PyTorch module for sampling from quantum states using Qiskit backend.
    
    This module provides native quantum sampling using Qiskit's
    measurement capabilities, giving realistic shot-based results.
    """
    
    def __init__(
        self,
        circuit: ParameterizedQuantumCircuit,
        backend: 'QiskitBackend',
        n_samples: int,
        wires: Optional[List[int]] = None
    ):
        super().__init__()
        self.circuit = circuit.copy()
        self.backend = backend
        self.n_samples = n_samples
        self.wires = wires if wires is not None else list(range(circuit.n_wires))
        
        # Prepare the measurement circuit
        self._prepare_sampling_circuit()
        
    def _prepare_sampling_circuit(self):
        """Prepare the circuit with measurements on specified wires."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitSampling")
            
        # Convert to Qiskit circuit
        self.qiskit_circuit, self.qiskit_params = convert_tq_circuit_to_qiskit(self.circuit)
        
        # Add classical register for measurements
        n_measured_qubits = len(self.wires)
        if len(self.qiskit_circuit.cregs) == 0:
            creg = ClassicalRegister(n_measured_qubits, 'c')
            self.qiskit_circuit.add_register(creg)
        
        # Add measurements on specified wires
        for i, wire in enumerate(self.wires):
            if wire < self.qiskit_circuit.num_qubits:
                self.qiskit_circuit.measure(wire, i)
        
    def forward(self, input_params=None):
        """Generate samples from the quantum state.
        
        Args:
            input_params: Input parameters tensor [batch_size, n_params]
            
        Returns:
            Integer tensor of samples [batch_size, n_samples, n_wires]
            Each element is 0 or 1 representing the measurement outcome
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitSampling")
            
        # Determine batch size
        if input_params is None:
            batch_size = 1
        elif isinstance(input_params, torch.Tensor):
            batch_size = input_params.shape[0] if input_params.dim() > 1 else 1
        else:
            batch_size = 1
        
        # Create parameter bindings
        parameter_binds = create_parameter_binds(self.qiskit_params, input_params)
        
        # Execute sampling for each batch
        all_samples = []
        
        for bind in parameter_binds:
            # Execute circuit with current parameters
            counts = self._execute_sampling_circuit(bind)
            
            # Convert counts to samples
            samples = self._counts_to_samples(counts)
            all_samples.append(samples)
        
        # Stack to get [batch_size, n_samples, n_wires]
        result = torch.stack(all_samples, dim=0)
        
        return result
    
    def _execute_sampling_circuit(self, parameter_bind: Dict) -> Dict[str, int]:
        """Execute the sampling circuit with parameter binding.
        
        Args:
            parameter_bind: Parameter binding dictionary
            
        Returns:
            Measurement counts
        """
        # Bind parameters directly to the circuit if there are parameters
        if parameter_bind:
            bound_circuit = self.qiskit_circuit.assign_parameters(parameter_bind)
        else:
            bound_circuit = self.qiskit_circuit
        
        # Transpile circuit
        transpiled_circuit = self.backend._transpile_circuit(bound_circuit)
        
        # Execute with the required number of samples as shots
        job = execute(
            experiments=transpiled_circuit,
            backend=self.backend.backend,
            shots=self.n_samples,
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
    
    def _counts_to_samples(self, counts: Dict[str, int]) -> torch.Tensor:
        """Convert measurement counts to sample tensor.
        
        Args:
            counts: Measurement counts from Qiskit
            
        Returns:
            Tensor of samples [n_samples, n_wires]
        """
        n_wires = len(self.wires)
        samples = []
        
        # Expand counts into individual samples
        for bitstring, count in counts.items():
            # Parse bitstring (Qiskit uses big-endian format)
            bits = []
            for i in range(n_wires):
                if i < len(bitstring):
                    # Qiskit bitstrings are big-endian, so we reverse
                    bit_idx = len(bitstring) - 1 - i
                    bit_value = int(bitstring[bit_idx])
                else:
                    bit_value = 0
                bits.append(bit_value)
            
            # Add this bitstring 'count' times to samples
            for _ in range(count):
                samples.append(bits)
        
        # Convert to tensor and ensure we have exactly n_samples
        if len(samples) < self.n_samples:
            # Pad with zeros if we have fewer samples than expected
            while len(samples) < self.n_samples:
                samples.append([0] * n_wires)
        elif len(samples) > self.n_samples:
            # Truncate if we have more samples than expected
            samples = samples[:self.n_samples]
        
        return torch.tensor(samples, dtype=torch.long) 
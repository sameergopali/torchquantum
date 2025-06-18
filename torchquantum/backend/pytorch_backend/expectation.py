# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List, Dict, Union

from torchquantum.measurement import expval_joint_analytical
from .state import PyTorchState, QuantumDeviceCompat
from ..core.circuit import ParameterizedQuantumCircuit


class PyTorchExpectation(nn.Module):
    """Expectation value computation using existing TorchQuantum measurement functions."""
    
    def __init__(self, circuit: ParameterizedQuantumCircuit, pauli_ops: Union[List[str], List[Dict[str, float]]], backend):
        super().__init__()
        self.circuit = circuit
        self.pauli_ops = pauli_ops
        self.backend = backend
        
    def forward(self, input_params=None):
        # Determine batch size
        if input_params is not None:
            batch_size = input_params.shape[0]
            # Move trainable params to the same device as input_params and backend
            trainable_params = self.circuit.trainable_params.to(self.backend.device)
            input_params = input_params.to(self.backend.device)
            # Combine trainable and input parameters
            all_params = torch.cat([
                trainable_params.unsqueeze(0).expand(batch_size, -1),
                input_params
            ], dim=1)
        else:
            batch_size = 1
            # Move trainable params to backend device
            trainable_params = self.circuit.trainable_params.to(self.backend.device)
            all_params = trainable_params.unsqueeze(0)
            
        # Create state and apply circuit
        state = PyTorchState(
            self.circuit.n_wires,
            batch_size=batch_size,
            device=self.backend.device,
            dtype=self.backend.dtype
        )
        
        # Apply circuit gates
        self.backend.apply_circuit_to_state(self.circuit, state, all_params)
        
        # Create compatibility wrapper for measurement functions
        qdev_compat = QuantumDeviceCompat(self.circuit.n_wires, batch_size, self.backend.device)
        qdev_compat._state = state
        
        # Compute expectation values using existing functions
        expectations = []
        for pauli_op in self.pauli_ops:
            if isinstance(pauli_op, str):
                # Single Pauli string - use existing function directly
                exp_val = expval_joint_analytical(qdev_compat, pauli_op)
            else:
                # Linear combination of Pauli strings
                exp_val = torch.zeros(batch_size, device=self.backend.device)
                for pauli_str, coeff in pauli_op.items():
                    exp_val += coeff * expval_joint_analytical(qdev_compat, pauli_str)
            expectations.append(exp_val)
            
        # Stack expectations: shape [batch_size, n_operators]
        return torch.stack(expectations, dim=-1) 
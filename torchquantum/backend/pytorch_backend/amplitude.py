# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List

from .state import PyTorchState
from ..core.circuit import ParameterizedQuantumCircuit


class PyTorchAmplitude(nn.Module):
    """Amplitude extraction for specific bitstrings."""
    
    def __init__(self, circuit: ParameterizedQuantumCircuit, bitstrings: List[str], backend):
        super().__init__()
        self.circuit = circuit
        self.bitstrings = bitstrings
        self.backend = backend
        
        # Precompute indices for bitstrings
        self.indices = []
        for bitstring in bitstrings:
            # Convert bitstring to index
            idx = int(bitstring, 2)
            self.indices.append(idx)
            
    def forward(self, input_params=None):
        # Determine batch size
        if input_params is not None:
            batch_size = input_params.shape[0]
            # Combine trainable and input parameters
            all_params = torch.cat([
                self.circuit.trainable_params.unsqueeze(0).expand(batch_size, -1),
                input_params
            ], dim=1)
        else:
            batch_size = 1
            all_params = self.circuit.trainable_params.unsqueeze(0)
            
        # Create state and apply circuit
        state = PyTorchState(
            self.circuit.n_wires,
            batch_size=batch_size,
            device=self.backend.device,
            dtype=self.backend.dtype
        )
        
        # Apply circuit gates
        self.backend.apply_circuit_to_state(self.circuit, state, all_params)
        
        # Get amplitudes for specified bitstrings
        state_1d = state.get_states_1d()
        amplitudes = []
        
        for idx in self.indices:
            amp = state_1d[:, idx]
            amplitudes.append(amp)
            
        # Stack amplitudes: shape [batch_size, n_bitstrings]
        return torch.stack(amplitudes, dim=-1) 
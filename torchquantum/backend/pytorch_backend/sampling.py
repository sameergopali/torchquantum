# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List, Optional

from .state import PyTorchState
from ..core.circuit import ParameterizedQuantumCircuit


class PyTorchSampling(nn.Module):
    """Sampling measurement outcomes from quantum states."""
    
    def __init__(self, circuit: ParameterizedQuantumCircuit, n_samples: int, wires: Optional[List[int]], backend):
        super().__init__()
        self.circuit = circuit
        self.n_samples = n_samples
        self.wires = wires if wires is not None else list(range(circuit.n_wires))
        self.backend = backend
        
        # Precompute masks for partial measurements
        self.n_measured_qubits = len(self.wires)
        if self.n_measured_qubits < circuit.n_wires:
            # Create mapping from full state indices to reduced indices
            self._compute_partial_measurement_map()
            
    def _compute_partial_measurement_map(self):
        """Precompute mapping for partial measurements."""
        # This will be implemented if needed for partial measurements
        pass
        
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
        
        # Get probabilities
        state_1d = state.get_states_1d()
        probs = (state_1d.abs() ** 2)
        
        if self.n_measured_qubits < self.circuit.n_wires:
            # Trace out unmeasured qubits
            # For now, we'll implement full measurement
            # TODO: Implement partial measurement tracing
            pass
            
        # Sample using multinomial
        samples = torch.multinomial(probs, self.n_samples, replacement=True)
        
        # Convert indices to bit strings (as list of lists for compatibility)
        all_samples = []
        for b in range(batch_size):
            batch_samples = []
            for s in range(self.n_samples):
                idx = samples[b, s].item()
                # Convert index to bitstring
                bitstring = format(idx, f'0{self.n_measured_qubits}b')
                batch_samples.append(bitstring)
            all_samples.append(batch_samples)
            
        return all_samples 
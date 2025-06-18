# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import Optional, List, Union

from torchquantum.macro import C_DTYPE
from torchquantum.functional.gate_wrapper import apply_unitary_bmm, apply_unitary_einsum


class PyTorchState:
    """State vector management for PyTorch backend, reusing existing TorchQuantum functions."""
    
    def __init__(self, n_qubits: int, batch_size: int = 1, device: Union[str, torch.device] = 'cpu', dtype=C_DTYPE):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
        # Initialize |00...0> state using existing pattern
        _state = torch.zeros(2**self.n_qubits, dtype=dtype, device=self.device)
        _state[0] = 1 + 0j
        _state = torch.reshape(_state, [2] * self.n_qubits)
        
        # Create batch dimension
        repeat_times = [batch_size] + [1] * self.n_qubits
        self.states = _state.repeat(*repeat_times)
        
    def apply_gate_matrix(self, matrix: torch.Tensor, wires: List[int], use_bmm: bool = True):
        """Apply gate matrix using existing TorchQuantum functions."""
        if use_bmm:
            self.states = apply_unitary_bmm(self.states, matrix, wires)
        else:
            self.states = apply_unitary_einsum(self.states, matrix, wires)
            
    def get_states_1d(self) -> torch.Tensor:
        """Return states in 1D format, compatible with existing measurement functions."""
        return torch.reshape(self.states, [self.batch_size, 2**self.n_qubits])
    
    def clone(self) -> 'PyTorchState':
        """Create a copy of the current state."""
        new_state = PyTorchState.__new__(PyTorchState)
        new_state.n_qubits = self.n_qubits
        new_state.batch_size = self.batch_size
        new_state.device = self.device
        new_state.dtype = self.dtype
        new_state.states = self.states.clone()
        return new_state


class QuantumDeviceCompat:
    """Minimal QuantumDevice interface for compatibility with existing TorchQuantum functions."""
    
    def __init__(self, n_wires: int, bsz: int = 1, device: Union[str, torch.device] = 'cpu'):
        self.n_wires = n_wires
        self.bsz = bsz
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Create PyTorchState internally
        self._state = PyTorchState(n_wires, bsz, device)
        
    @property
    def states(self):
        """Get states in the format expected by existing functions."""
        return self._state.states
    
    @states.setter
    def states(self, value):
        """Set states."""
        self._state.states = value
        
    def get_states_1d(self):
        """Compatible with existing measurement functions."""
        return self._state.get_states_1d() 
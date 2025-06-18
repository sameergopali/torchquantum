# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
import warnings
from typing import List, Union, Dict, Optional

from torchquantum.macro import C_DTYPE
from torchquantum.functional import func_name_dict
from torchquantum.operator.standard_gates import all_variables

from ..abstract_backend import QuantumBackend
from ..core.circuit import ParameterizedQuantumCircuit, _ParameterizedQuantumGate
from .state import PyTorchState
from .expectation import PyTorchExpectation
from .amplitude import PyTorchAmplitude
from .sampling import PyTorchSampling


class PyTorchBackend(QuantumBackend):
    """PyTorch backend for quantum circuit simulation using state vectors.
    
    This backend reuses existing TorchQuantum functionality for gate operations
    and measurements while providing the new backend interface.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = 'auto',
        dtype=C_DTYPE,
        use_bmm: bool = True,
        warn_large_circuits: bool = True,
        large_circuit_threshold: int = 20
    ):
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.use_bmm = use_bmm
        self.warn_large_circuits = warn_large_circuits
        self.large_circuit_threshold = large_circuit_threshold
        
        # Cache for gate matrices
        self._gate_cache = {}
        
    def _resolve_device(self, device: Union[str, torch.device]) -> torch.device:
        """Resolve device selection."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            return torch.device(device)
        else:
            return device
            
    def _create_expectation_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        pauli_ops: Union[List[str], Dict[str, float]]
    ) -> torch.nn.Module:
        """Create expectation value computation module."""
        if self.warn_large_circuits and circuit.n_wires > self.large_circuit_threshold:
            warnings.warn(
                f"Circuit has {circuit.n_wires} qubits. "
                f"Consider using CuQuantumBackend for better performance.",
                UserWarning
            )
        return PyTorchExpectation(circuit, pauli_ops, self)
        
    def _create_amplitude_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        bitstrings: List[str]
    ) -> torch.nn.Module:
        """Create amplitude extraction module."""
        if self.warn_large_circuits and circuit.n_wires > self.large_circuit_threshold:
            warnings.warn(
                f"Circuit has {circuit.n_wires} qubits. "
                f"State vector may require {2**(circuit.n_wires - 30):.1f} GB of memory.",
                UserWarning
            )
        return PyTorchAmplitude(circuit, bitstrings, self)
        
    def _create_sampling_module(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        n_samples: int, 
        wires: Optional[List[int]] = None
    ) -> torch.nn.Module:
        """Create sampling module."""
        return PyTorchSampling(circuit, n_samples, wires, self)
        
    def apply_circuit_to_state(
        self, 
        circuit: ParameterizedQuantumCircuit, 
        state: PyTorchState,
        params: torch.Tensor
    ):
        """Apply circuit to state using existing TorchQuantum functions."""
        for gate in circuit.gates:
            # Get gate parameters
            gate_params = self._extract_gate_params(gate, params, circuit)
            
            # Get gate matrix
            matrix = self._get_gate_matrix(gate, gate_params)
            
            # Apply using existing functions
            state.apply_gate_matrix(matrix, gate.wires, use_bmm=self.use_bmm)
            
    def _extract_gate_params(
        self, 
        gate: _ParameterizedQuantumGate, 
        all_params: torch.Tensor,
        circuit: ParameterizedQuantumCircuit
    ) -> Optional[torch.Tensor]:
        """Extract parameters for a specific gate."""
        if gate.matrix_generator is None:
            return None
            
        # Get parameters from the appropriate indices
        batch_size = all_params.shape[0]
        n_params = len(gate.params)
        
        if n_params == 0:
            return None
            
        gate_params = torch.zeros((batch_size, n_params), device=all_params.device)
        
        for i in range(n_params):
            if gate.trainable_idx[i] is not None:
                # Trainable parameter
                gate_params[:, i] = all_params[:, gate.trainable_idx[i]]
            elif gate.input_idx[i] is not None:
                # Input parameter
                param_idx = circuit.n_trainable_params + gate.input_idx[i]
                gate_params[:, i] = all_params[:, param_idx]
            else:
                # Fixed parameter
                gate_params[:, i] = gate.params[i]
                
        return gate_params
        
    def _get_gate_matrix(
        self, 
        gate: _ParameterizedQuantumGate,
        params: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get gate matrix, using cache when possible."""
        # For parameterized gates, compute matrix
        if params is not None:
            # Generate matrix using parameters (params should be [batch_size, n_params])
            if params.dim() == 1:
                params = params.unsqueeze(0)  # Add batch dimension if missing
            matrices = gate.matrix_generator(params)
            
            # Convert tensor form to matrix form if needed
            matrices = self._tensor_to_matrix(matrices, len(gate.wires))
            
            # Ensure matrix is on correct device
            matrices = matrices.to(self.device)
            
            if gate.inverse:
                # Apply conjugate transpose
                matrices = matrices.conj()
                if matrices.dim() == 3:
                    matrices = matrices.permute(0, 2, 1)
                else:
                    matrices = matrices.permute(1, 0)
            return matrices
            
        # For non-parameterized gates, try cache first
        cache_key = (gate.matrix_generator, tuple(gate.wires), gate.inverse)
        if cache_key in self._gate_cache:
            cached_matrix = self._gate_cache[cache_key]
            # Always return with proper batching for bmm compatibility
            if cached_matrix.dim() == 2:
                return cached_matrix.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
            return cached_matrix.to(self.device)
            
        # Compute and cache
        # Create dummy parameters tensor for matrix generation
        dummy_params = torch.empty(1, 0, device=self.device)  # [1, 0] for batch compatibility
        matrix = gate.matrix_generator(dummy_params)
        
        # Convert tensor form to matrix form
        matrix = self._tensor_to_matrix(matrix, len(gate.wires))
        
        # Move to correct device
        matrix = matrix.to(self.device)
        
        # Handle the matrix shape properly
        if matrix.dim() == 3 and matrix.shape[0] == 1:
            # Matrix generator returned [1, n, n] - squeeze to [n, n] for caching
            matrix_2d = matrix.squeeze(0)
        elif matrix.dim() == 2:
            # Matrix generator returned [n, n] directly
            matrix_2d = matrix
        else:
            # Unexpected shape
            raise ValueError(f"Unexpected matrix shape after conversion: {matrix.shape}")
            
        if gate.inverse:
            matrix_2d = matrix_2d.conj().T
            
        # Cache the 2D version (keep on device for cache efficiency)
        self._gate_cache[cache_key] = matrix_2d
        
        # Return with batch dimension for bmm compatibility
        return matrix_2d.unsqueeze(0)  # [n, n] -> [1, n, n]
        
    def _tensor_to_matrix(self, tensor: torch.Tensor, n_qubits: int) -> torch.Tensor:
        """Convert tensor representation to matrix form."""
        expected_matrix_size = 2 ** n_qubits
        
        if tensor.dim() == 2 and tensor.shape == (expected_matrix_size, expected_matrix_size):
            # Already in matrix form
            return tensor
        elif tensor.dim() == 3 and tensor.shape[0] == 1 and tensor.shape[1:] == (expected_matrix_size, expected_matrix_size):
            # Batched matrix form
            return tensor
        elif tensor.dim() == 2 * n_qubits:
            # Tensor form: reshape to matrix form
            # For n_qubits, shape should be [2]*2n, reshape to [2^n, 2^n]
            return tensor.reshape(expected_matrix_size, expected_matrix_size)
        elif tensor.dim() == 2 * n_qubits + 1:
            # Batched tensor form: reshape to batched matrix form
            batch_size = tensor.shape[0]
            return tensor.reshape(batch_size, expected_matrix_size, expected_matrix_size)
        else:
            raise ValueError(f"Cannot convert tensor shape {tensor.shape} to matrix form for {n_qubits} qubits") 
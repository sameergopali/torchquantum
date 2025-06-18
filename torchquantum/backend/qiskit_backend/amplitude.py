# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""State amplitude computation using Qiskit backend."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

try:
    from qiskit import execute
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    AerSimulator = object

from ..core.circuit import ParameterizedQuantumCircuit
from .utils import convert_tq_circuit_to_qiskit, create_parameter_binds


class QiskitAmplitude(nn.Module):
    """PyTorch module for computing state amplitudes using Qiskit backend.
    
    This module computes amplitudes for specified bitstrings using
    Qiskit's statevector simulator. Limited to small circuits due to
    exponential memory requirements.
    """
    
    def __init__(
        self,
        circuit: ParameterizedQuantumCircuit,
        backend: 'QiskitBackend',
        bitstrings: List[str]
    ):
        super().__init__()
        self.circuit = circuit.copy()
        self.backend = backend
        self.bitstrings = bitstrings.copy()
        
        # Warn about large circuits
        if circuit.n_wires > 20:
            import warnings
            warnings.warn(
                f"Circuit has {circuit.n_wires} qubits. Amplitude computation "
                f"may be slow or fail due to memory requirements.",
                UserWarning
            )
        
        # Prepare the amplitude extraction circuit
        self._prepare_amplitude_circuit()
        
    def _prepare_amplitude_circuit(self):
        """Prepare the circuit for amplitude computation."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitAmplitude")
            
        # Convert to Qiskit circuit (no measurements needed for statevector)
        self.qiskit_circuit, self.qiskit_params = convert_tq_circuit_to_qiskit(self.circuit)
        
        # Validate bitstrings
        for bitstring in self.bitstrings:
            if len(bitstring) != self.circuit.n_wires:
                raise ValueError(
                    f"Bitstring '{bitstring}' length ({len(bitstring)}) "
                    f"must match circuit qubits ({self.circuit.n_wires})"
                )
            if not all(bit in '01' for bit in bitstring):
                raise ValueError(f"Bitstring '{bitstring}' must contain only '0' and '1'")
        
    def forward(self, input_params=None):
        """Compute amplitudes for the specified bitstrings.
        
        Args:
            input_params: Input parameters tensor [batch_size, n_params]
            
        Returns:
            Complex tensor of amplitudes [batch_size, n_bitstrings]
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QiskitAmplitude")
            
        # Determine batch size
        if input_params is None:
            batch_size = 1
        elif isinstance(input_params, torch.Tensor):
            batch_size = input_params.shape[0] if input_params.dim() > 1 else 1
        else:
            batch_size = 1
        
        # Create parameter bindings
        parameter_binds = create_parameter_binds(self.qiskit_params, input_params)
        
        # Execute circuits and extract amplitudes
        all_amplitudes = []
        
        for bind in parameter_binds:
            # Get statevector for current parameters
            statevector = self._execute_statevector_circuit(bind)
            
            # Extract amplitudes for specified bitstrings
            amplitudes = self._extract_amplitudes(statevector)
            all_amplitudes.append(amplitudes)
        
        # Stack to get [batch_size, n_bitstrings]
        result = torch.stack(all_amplitudes, dim=0)
        
        return result
    
    def _execute_statevector_circuit(self, parameter_bind: Dict) -> np.ndarray:
        """Execute circuit and return statevector.
        
        Args:
            parameter_bind: Parameter binding dictionary
            
        Returns:
            Complex statevector as numpy array
        """
        # Use statevector simulator
        statevector_backend = AerSimulator(method='statevector')
        
        # Bind parameters directly to the circuit if there are parameters
        if parameter_bind:
            bound_circuit = self.qiskit_circuit.assign_parameters(parameter_bind)
        else:
            bound_circuit = self.qiskit_circuit
        
        # Add save_statevector instruction to get the statevector
        transpiled_circuit = bound_circuit.copy()
        transpiled_circuit.save_statevector()
        
        # Execute circuit
        job = execute(
            experiments=transpiled_circuit,
            backend=statevector_backend,
            seed_simulator=self.backend.seed,
            optimization_level=0
        )
        
        result = job.result()
        
        # Get statevector from saved data
        try:
            statevector = result.get_statevector()
            # Convert to numpy array
            if hasattr(statevector, 'data'):
                return statevector.data
            else:
                return np.array(statevector)
        except:
            # Fallback to data method
            data = result.data(0)
            statevector = data['statevector']
            if hasattr(statevector, 'data'):
                return statevector.data
            else:
                return np.array(statevector)
    
    def _extract_amplitudes(self, statevector: np.ndarray) -> torch.Tensor:
        """Extract amplitudes for specified bitstrings from statevector.
        
        Args:
            statevector: Complex statevector
            
        Returns:
            Complex tensor of amplitudes for each bitstring
        """
        amplitudes = []
        
        for bitstring in self.bitstrings:
            # Convert bitstring to index in statevector
            # Qiskit uses big-endian, so reverse the bitstring
            reversed_bitstring = bitstring[::-1]
            index = int(reversed_bitstring, 2)
            
            # Extract amplitude
            if index < len(statevector):
                amplitude = complex(statevector[index])
            else:
                amplitude = complex(0.0, 0.0)
            
            amplitudes.append(amplitude)
        
        # Convert to torch tensor
        real_parts = [amp.real for amp in amplitudes]
        imag_parts = [amp.imag for amp in amplitudes]
        
        result = torch.complex(
            torch.tensor(real_parts, dtype=torch.float32),
            torch.tensor(imag_parts, dtype=torch.float32)
        )
        
        return result 
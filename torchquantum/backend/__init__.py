# torchquantum/backends/__init__.py

"""
TorchQuantum Backends - New Architecture

This module provides the new backend-based architecture for TorchQuantum.
"""

# Import core components
from .core import (
    ParameterizedQuantumCircuit,
    QuantumBackend,
    QuantumExpectation,
    QuantumAmplitude,
    QuantumSampling,
)

# Import backends
from .pytorch_backend import PyTorchBackend
from .cuquantum_backend import CuTensorNetworkBackend

# Import Qiskit backend with optional dependency handling
try:
    from .qiskit_backend import QiskitBackend
    QISKIT_AVAILABLE = True
except ImportError:
    QiskitBackend = None
    QISKIT_AVAILABLE = False

# Backend registry
_BACKENDS = {
    'pytorch': PyTorchBackend,
    'cuquantum': CuTensorNetworkBackend,
}

# Add Qiskit backend if available
if QISKIT_AVAILABLE:
    _BACKENDS['qiskit'] = QiskitBackend

def register_backend(name: str, backend_class):
    """Register a custom backend"""
    _BACKENDS[name] = backend_class

def get_backend(name: str = 'pytorch', **kwargs):
    """Get a backend instance by name
    
    Args:
        name: Backend name ('pytorch', 'cuquantum', 'qiskit')
        **kwargs: Backend-specific configuration
        
    Returns:
        Backend instance
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. "
            f"Available backends: {list(_BACKENDS.keys())}"
        )
    
    return _BACKENDS[name](**kwargs)

def list_backends():
    """List available backends"""
    return list(_BACKENDS.keys())

__all__ = [
    # Core components
    'ParameterizedQuantumCircuit',
    'QuantumBackend',
    'QuantumExpectation',
    'QuantumAmplitude',
    'QuantumSampling',
    # Backends
    'PyTorchBackend',
    'CuTensorNetworkBackend',
    # Functions
    'get_backend',
    'register_backend',
    'list_backends',
]

# Add QiskitBackend to exports if available
if QISKIT_AVAILABLE:
    __all__.append('QiskitBackend')
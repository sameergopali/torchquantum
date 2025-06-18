# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qiskit backend for TorchQuantum quantum circuit execution."""

from .backend import QiskitBackend
from .expectation import QiskitExpectation
from .amplitude import QiskitAmplitude
from .sampling import QiskitSampling

__all__ = [
    'QiskitBackend',
    'QiskitExpectation', 
    'QiskitAmplitude',
    'QiskitSampling'
]

# Try to import advanced features
try:
    from .noise import (
        create_depolarizing_noise_model,
        create_thermal_noise_model,
        create_device_noise_model,
        NoiseModelBuilder,
        apply_noise_to_backend
    )
    from .hardware import (
        HardwareManager,
        setup_hardware_backend,
        JobMonitor
    )
    from .optimization import (
        CircuitCache,
        OptimizedTranspiler,
        PerformanceMonitor,
        AdaptiveExecution
    )
    from .error_handling import (
        SafeExecutor,
        RetryConfig,
        CircuitValidator,
        ErrorRecovery
    )
    
    __all__.extend([
        # Noise models
        'create_depolarizing_noise_model',
        'create_thermal_noise_model', 
        'create_device_noise_model',
        'NoiseModelBuilder',
        'apply_noise_to_backend',
        # Hardware integration
        'HardwareManager',
        'setup_hardware_backend',
        'JobMonitor',
        # Optimization
        'CircuitCache',
        'OptimizedTranspiler',
        'PerformanceMonitor',
        'AdaptiveExecution',
        # Error handling
        'SafeExecutor',
        'RetryConfig',
        'CircuitValidator',
        'ErrorRecovery'
    ])
    
except ImportError:
    # Advanced features not available
    pass 
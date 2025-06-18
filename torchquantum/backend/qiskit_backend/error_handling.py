# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Error handling and recovery for Qiskit backend."""

import time
import logging
from typing import Optional, Callable, Any, Dict, List
from functools import wraps
import warnings

try:
    from qiskit.providers.exceptions import QiskitBackendNotFoundError, JobError, JobTimeoutError
    from qiskit.exceptions import QiskitError
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QiskitBackendNotFoundError = Exception
    JobError = Exception  
    JobTimeoutError = Exception
    QiskitError = Exception


class QiskitBackendError(Exception):
    """Custom exception for Qiskit backend errors."""
    pass


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor


class ErrorClassifier:
    """Classify errors and determine appropriate recovery strategies."""
    
    TRANSIENT_ERRORS = [
        'network',
        'timeout', 
        'rate_limit',
        'queue_full',
        'service_unavailable'
    ]
    
    PERMANENT_ERRORS = [
        'authentication',
        'permission_denied',
        'invalid_circuit',
        'backend_not_found'
    ]
    
    @classmethod
    def classify_error(cls, error: Exception) -> str:
        """Classify an error as transient, permanent, or unknown."""
        error_msg = str(error).lower()
        
        # Check for transient errors
        for transient_pattern in cls.TRANSIENT_ERRORS:
            if transient_pattern in error_msg:
                return 'transient'
        
        # Check for permanent errors
        for permanent_pattern in cls.PERMANENT_ERRORS:
            if permanent_pattern in error_msg:
                return 'permanent'
        
        # Classify specific exception types
        if isinstance(error, (TimeoutError, JobTimeoutError)):
            return 'transient'
        elif isinstance(error, (QiskitBackendNotFoundError, PermissionError)):
            return 'permanent'
        
        return 'unknown'
    
    @classmethod
    def should_retry(cls, error: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if an error should trigger a retry."""
        if attempt >= max_attempts:
            return False
        
        classification = cls.classify_error(error)
        
        # Never retry permanent errors
        if classification == 'permanent':
            return False
        
        # Always retry transient errors (up to max attempts)
        if classification == 'transient':
            return True
        
        # For unknown errors, retry up to half the max attempts
        return attempt < max_attempts // 2


def with_retry(retry_config: Optional[RetryConfig] = None):
    """Decorator for automatic retry with exponential backoff."""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if not ErrorClassifier.should_retry(e, attempt + 1, retry_config.max_attempts):
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.backoff_factor ** attempt),
                        retry_config.max_delay
                    )
                    
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            # All attempts failed
            raise QiskitBackendError(f"Operation failed after {retry_config.max_attempts} attempts") from last_exception
        
        return wrapper
    return decorator


class ErrorRecovery:
    """Error recovery strategies for different failure scenarios."""
    
    def __init__(self):
        self.fallback_backends = ['aer_simulator', 'qasm_simulator']
        self.recovery_strategies = {
            'backend_unavailable': self._recover_backend_unavailable,
            'circuit_too_large': self._recover_circuit_too_large,
            'shot_limit_exceeded': self._recover_shot_limit_exceeded,
            'timeout': self._recover_timeout,
            'memory_error': self._recover_memory_error
        }
    
    def recover_from_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from an error using appropriate strategy."""
        error_type = self._identify_error_type(error)
        
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error, context)
        else:
            return {'success': False, 'strategy': 'none', 'error': str(error)}
    
    def _identify_error_type(self, error: Exception) -> str:
        """Identify the type of error for recovery purposes."""
        error_msg = str(error).lower()
        
        if 'backend' in error_msg and ('unavailable' in error_msg or 'not found' in error_msg):
            return 'backend_unavailable'
        elif 'too large' in error_msg or 'memory' in error_msg:
            return 'circuit_too_large'
        elif 'shot' in error_msg and 'limit' in error_msg:
            return 'shot_limit_exceeded'
        elif 'timeout' in error_msg:
            return 'timeout'
        elif 'memory' in error_msg:
            return 'memory_error'
        else:
            return 'unknown'
    
    def _recover_backend_unavailable(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from backend unavailability by switching to fallback."""
        current_backend = context.get('backend_name', '')
        
        for fallback in self.fallback_backends:
            if fallback != current_backend:
                return {
                    'success': True,
                    'strategy': 'fallback_backend',
                    'new_backend': fallback,
                    'message': f"Switched to fallback backend: {fallback}"
                }
        
        return {
            'success': False,
            'strategy': 'fallback_backend',
            'message': 'No suitable fallback backend available'
        }
    
    def _recover_circuit_too_large(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from circuit size issues by reducing complexity."""
        current_shots = context.get('shots', 4096)
        current_optimization = context.get('optimization_level', 1)
        
        recovery_actions = []
        
        # Reduce shot count
        if current_shots > 1024:
            new_shots = max(1024, current_shots // 2)
            recovery_actions.append(f"Reduced shots from {current_shots} to {new_shots}")
            
        # Increase optimization level
        if current_optimization < 3:
            new_optimization = min(3, current_optimization + 1)
            recovery_actions.append(f"Increased optimization level to {new_optimization}")
        
        if recovery_actions:
            return {
                'success': True,
                'strategy': 'reduce_complexity',
                'actions': recovery_actions,
                'new_shots': new_shots if 'new_shots' in locals() else current_shots,
                'new_optimization': new_optimization if 'new_optimization' in locals() else current_optimization
            }
        
        return {
            'success': False,
            'strategy': 'reduce_complexity',
            'message': 'No further complexity reduction possible'
        }
    
    def _recover_shot_limit_exceeded(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from shot limit errors by reducing shot count."""
        current_shots = context.get('shots', 4096)
        max_shots = context.get('max_shots', 8192)
        
        if current_shots > max_shots:
            new_shots = max_shots
        else:
            new_shots = max(1024, current_shots // 2)
        
        return {
            'success': True,
            'strategy': 'reduce_shots',
            'new_shots': new_shots,
            'message': f"Reduced shots from {current_shots} to {new_shots}"
        }
    
    def _recover_timeout(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from timeout errors by adjusting parameters."""
        return {
            'success': True,
            'strategy': 'increase_timeout',
            'new_timeout': context.get('timeout', 300) * 2,
            'message': 'Increased timeout for next attempt'
        }
    
    def _recover_memory_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from memory errors by reducing resource usage."""
        return {
            'success': True,
            'strategy': 'reduce_memory',
            'new_shots': max(512, context.get('shots', 4096) // 4),
            'message': 'Reduced memory usage by reducing shot count'
        }


class CircuitValidator:
    """Validate circuits before execution to prevent common errors."""
    
    @staticmethod
    def validate_circuit(circuit, backend_config: Dict[str, Any]) -> List[str]:
        """Validate a circuit against backend constraints."""
        errors = []
        
        # Check qubit count
        max_qubits = backend_config.get('n_qubits', float('inf'))
        if circuit.num_qubits > max_qubits:
            errors.append(f"Circuit requires {circuit.num_qubits} qubits, backend supports {max_qubits}")
        
        # Check basis gates
        basis_gates = backend_config.get('basis_gates', [])
        if basis_gates:
            used_gates = set()
            for instr, _, _ in circuit.data:
                # Handle different Qiskit versions and gate name access
                if hasattr(instr, 'operation'):
                    gate_name = instr.operation.name
                elif hasattr(instr, 'name'):
                    gate_name = instr.name
                else:
                    gate_name = str(type(instr).__name__).lower().replace('gate', '')
                used_gates.add(gate_name)
            
            unsupported_gates = used_gates - set(basis_gates)
            if unsupported_gates:
                errors.append(f"Unsupported gates: {unsupported_gates}")
        
        # Check circuit depth
        if circuit.depth() > 1000:
            errors.append(f"Circuit depth ({circuit.depth()}) is very high and may cause timeouts")
        
        # Check for unconnected qubits in coupling map
        coupling_map = backend_config.get('coupling_map')
        if coupling_map:
            connected_qubits = set()
            for edge in coupling_map:
                connected_qubits.update(edge)
            
            used_qubits = set(range(circuit.num_qubits))
            unconnected = used_qubits - connected_qubits
            if unconnected:
                errors.append(f"Some qubits may not be connected: {unconnected}")
        
        return errors
    
    @staticmethod
    def validate_parameters(shots: int, backend_config: Dict[str, Any]) -> List[str]:
        """Validate execution parameters."""
        errors = []
        
        # Check shot limits
        max_shots = backend_config.get('max_shots', 100000)
        if shots > max_shots:
            errors.append(f"Requested {shots} shots, maximum is {max_shots}")
        
        if shots < 1:
            errors.append("Shot count must be positive")
        
        return errors


class SafeExecutor:
    """Safe execution wrapper with comprehensive error handling."""
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.error_recovery = ErrorRecovery()
        self.validator = CircuitValidator()
    
    @with_retry()
    def safe_execute(self, func: Callable, *args, **kwargs):
        """Execute a function with comprehensive error handling."""
        return func(*args, **kwargs)
    
    def execute_with_recovery(self, func: Callable, context: Dict[str, Any], *args, **kwargs):
        """Execute with automatic error recovery."""
        try:
            return self.safe_execute(func, *args, **kwargs)
        except Exception as e:
            recovery_result = self.error_recovery.recover_from_error(e, context)
            
            if recovery_result['success']:
                # Apply recovery actions and retry
                warnings.warn(f"Recovered from error: {recovery_result['message']}")
                
                # Update context with recovery parameters
                if 'new_shots' in recovery_result:
                    kwargs['shots'] = recovery_result['new_shots']
                if 'new_backend' in recovery_result:
                    kwargs['backend'] = recovery_result['new_backend']
                
                return self.safe_execute(func, *args, **kwargs)
            else:
                raise QiskitBackendError(f"Unrecoverable error: {e}") from e 
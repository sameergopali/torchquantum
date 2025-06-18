# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Hardware integration for real quantum devices."""

import warnings
from typing import Optional, List, Dict, Any

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, EstimatorV2
    from qiskit.providers import Backend
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    QiskitRuntimeService = object
    SamplerV2 = object
    EstimatorV2 = object
    Backend = object


class HardwareManager:
    """Manager for real quantum hardware integration using IBM Quantum Runtime."""
    
    def __init__(self, token: Optional[str] = None, channel: str = 'ibm_quantum',
                 instance: Optional[str] = None):
        """Initialize hardware manager.
        
        Args:
            token: IBM Quantum Network token
            channel: Channel to use ('ibm_quantum' or 'ibm_cloud')
            instance: Instance in format 'hub/group/project' (for ibm_quantum channel)
        """
        self.token = token
        self.channel = channel
        self.instance = instance
        self.service = None
        self._available_backends = []
        
    def connect(self) -> bool:
        """Connect to IBM Quantum Runtime service.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not IBM_RUNTIME_AVAILABLE:
            warnings.warn("IBM Quantum Runtime not available. Install with: pip install qiskit-ibm-runtime")
            return False
        
        try:
            # Initialize the runtime service
            if self.token:
                # Save token for future use
                QiskitRuntimeService.save_account(
                    token=self.token,
                    channel=self.channel,
                    instance=self.instance,
                    overwrite=True
                )
            
            # Create service instance
            self.service = QiskitRuntimeService(
                channel=self.channel,
                instance=self.instance
            )
            
            # Get available backends
            self._available_backends = self.service.backends()
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to connect to IBM Quantum Runtime: {e}")
            return False
    
    def list_available_backends(self) -> List[str]:
        """List available quantum backends.
        
        Returns:
            List of backend names
        """
        if self.service is None:
            return []
        
        return [backend.name for backend in self._available_backends]
    
    def get_backend(self, name: str) -> Optional[Backend]:
        """Get a specific quantum backend.
        
        Args:
            name: Backend name
            
        Returns:
            Backend instance or None if not found
        """
        if self.service is None:
            warnings.warn("Not connected to IBM Quantum Runtime. Call connect() first.")
            return None
        
        try:
            return self.service.backend(name)
        except Exception as e:
            warnings.warn(f"Backend {name} not found: {e}")
            return None
    
    def get_backend_info(self, name: str) -> Dict[str, Any]:
        """Get information about a backend.
        
        Args:
            name: Backend name
            
        Returns:
            Dictionary with backend information
        """
        backend = self.get_backend(name)
        if backend is None:
            return {}
        
        info = {
            'name': backend.name,
            'n_qubits': backend.num_qubits,
            'basis_gates': backend.basis_gates,
            'coupling_map': backend.coupling_map,
            'simulator': backend.simulator,
            'max_shots': getattr(backend, 'max_shots', None),
            'supported_features': getattr(backend, 'supported_features', [])
        }
        
        # Add status information if available
        try:
            status = backend.status()
            info.update({
                'operational': status.operational,
                'pending_jobs': status.pending_jobs,
                'status_msg': getattr(status, 'status_msg', '')
            })
        except:
            pass
        
        # Add target information if available (new backend interface)
        try:
            target = backend.target
            if target:
                info.update({
                    'instruction_durations': dict(target.durations()) if hasattr(target, 'durations') else {},
                    'qubit_properties': self._extract_qubit_properties(target) if hasattr(target, 'qubit_properties') else {}
                })
        except:
            pass
        
        return info
    
    def _extract_qubit_properties(self, target) -> Dict[str, Any]:
        """Extract qubit properties from backend target."""
        qubit_props = {}
        
        try:
            # Extract T1 and T2 times if available
            for qubit in range(target.num_qubits):
                qubit_props[f"qubit_{qubit}"] = {}
                
                # Get qubit properties
                if hasattr(target, 'qubit_properties') and target.qubit_properties:
                    props = target.qubit_properties[qubit]
                    if props:
                        if hasattr(props, 't1') and props.t1:
                            qubit_props[f"qubit_{qubit}"]["t1"] = props.t1
                        if hasattr(props, 't2') and props.t2:
                            qubit_props[f"qubit_{qubit}"]["t2"] = props.t2
                        if hasattr(props, 'frequency') and props.frequency:
                            qubit_props[f"qubit_{qubit}"]["frequency"] = props.frequency
        except:
            pass
        
        return qubit_props
    
    def find_best_backend(self, n_qubits: int, exclude_simulators: bool = True) -> Optional[str]:
        """Find the best available backend for a given number of qubits.
        
        Args:
            n_qubits: Required number of qubits
            exclude_simulators: Whether to exclude simulator backends
            
        Returns:
            Name of best backend or None if none suitable
        """
        if self.service is None:
            return None
        
        suitable_backends = []
        
        for backend in self._available_backends:
            # Check if backend has enough qubits
            if backend.num_qubits < n_qubits:
                continue
            
            # Exclude simulators if requested
            if exclude_simulators and backend.simulator:
                continue
            
            # Check if backend is operational
            try:
                status = backend.status()
                if not status.operational:
                    continue
            except:
                continue
            
            suitable_backends.append((backend.name, backend.num_qubits, 
                                    getattr(status, 'pending_jobs', 0)))
        
        if not suitable_backends:
            return None
        
        # Sort by number of qubits (ascending) and pending jobs (ascending)
        suitable_backends.sort(key=lambda x: (x[1], x[2]))
        return suitable_backends[0][0]


def setup_hardware_backend(backend_instance, device_name: str, 
                          optimization_level: int = 2) -> Dict[str, Any]:
    """Setup a Qiskit backend for hardware execution using IBM Quantum Runtime.
    
    Args:
        backend_instance: QiskitBackend instance
        device_name: Name of the quantum device
        optimization_level: Transpilation optimization level
        
    Returns:
        Dictionary with setup information
    """
    manager = HardwareManager()
    
    if not manager.connect():
        return {'success': False, 'error': 'Could not connect to IBM Quantum Runtime'}
    
    hardware_backend = manager.get_backend(device_name)
    if hardware_backend is None:
        return {'success': False, 'error': f'Backend {device_name} not found'}
    
    # Update backend instance
    backend_instance.backend = hardware_backend
    backend_instance.backend_name = hardware_backend.name
    backend_instance.coupling_map = hardware_backend.coupling_map
    backend_instance.basis_gates = hardware_backend.basis_gates
    backend_instance.optimization_level = optimization_level
    
    # Set reasonable shot count for hardware
    max_shots = getattr(hardware_backend, 'max_shots', 8192)
    if backend_instance.shots > max_shots:
        backend_instance.shots = max_shots
        warnings.warn(f"Reduced shot count to {max_shots} for hardware execution")
    
    # Clear circuit cache (hardware circuits need different transpilation)
    backend_instance.clear_cache()
    
    return {
        'success': True,
        'backend_name': hardware_backend.name,
        'n_qubits': hardware_backend.num_qubits,
        'coupling_map': hardware_backend.coupling_map,
        'basis_gates': hardware_backend.basis_gates,
        'max_shots': max_shots
    }


class JobMonitor:
    """Monitor and manage quantum jobs for IBM Quantum Runtime."""
    
    def __init__(self):
        self.jobs = {}
    
    def submit_job(self, job, job_id: str):
        """Submit a job for monitoring."""
        self.jobs[job_id] = {
            'job': job,
            'submitted_at': getattr(job, 'creation_date', lambda: None)(),
            'status': 'SUBMITTED'
        }
    
    def check_job_status(self, job_id: str) -> str:
        """Check the status of a job."""
        if job_id not in self.jobs:
            return 'NOT_FOUND'
        
        job = self.jobs[job_id]['job']
        try:
            status = job.status()
            status_name = status if isinstance(status, str) else getattr(status, 'name', str(status))
            self.jobs[job_id]['status'] = status_name
            return status_name
        except:
            return 'UNKNOWN'
    
    def wait_for_job(self, job_id: str, timeout: Optional[int] = None):
        """Wait for a job to complete."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]['job']
        return job.result(timeout=timeout)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id not in self.jobs:
            return False
        
        try:
            job = self.jobs[job_id]['job']
            if hasattr(job, 'cancel'):
                job.cancel()
                self.jobs[job_id]['status'] = 'CANCELLED'
                return True
            return False
        except:
            return False
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get queue position for a job."""
        if job_id not in self.jobs:
            return None
        
        try:
            job = self.jobs[job_id]['job']
            if hasattr(job, 'queue_position'):
                return job.queue_position()
            return None
        except:
            return None 
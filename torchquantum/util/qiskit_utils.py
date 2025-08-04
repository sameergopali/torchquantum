"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Qiskit-dependent utilities
# This module isolates all qiskit dependencies to avoid import issues for non-qiskit users

from torchpack.utils.logging import logger

# Optional qiskit imports - only imported when functions are called
def _import_qiskit_runtime():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService
    except ImportError:
        raise ImportError("qiskit_ibm_runtime is required for this functionality. Install with: pip install qiskit-ibm-runtime")

def _import_qiskit_error():
    try:
        from qiskit.exceptions import QiskitError
        return QiskitError
    except ImportError:
        raise ImportError("qiskit is required for this functionality. Install with: pip install qiskit")

def _import_gate_error_values():
    try:
        from qiskit.providers.aer.noise.device.parameters import gate_error_values
        return gate_error_values
    except ImportError:
        raise ImportError("qiskit-aer is required for this functionality. Install with: pip install qiskit-aer")


def get_success_rate(properties, transpiled_circ):
    """
        Estimate the success rate of a transpiled quantum circuit.

        Args:
            properties (list): List of gate error properties.
            transpiled_circ (QuantumCircuit): The transpiled quantum circuit.

        Returns:
            float: The estimated success rate.
        """
    # estimate the success rate according to the error rates of single and
    # two-qubit gates in transpiled circuits

    gate_error_values = _import_gate_error_values()
    gate_errors = gate_error_values(properties)
    # construct the error dict
    gate_error_dict = {}
    for gate_error in gate_errors:
        if gate_error[0] not in gate_error_dict.keys():
            gate_error_dict[gate_error[0]] = {tuple(gate_error[1]): gate_error[2]}
        else:
            gate_error_dict[gate_error[0]][tuple(gate_error[1])] = gate_error[2]

    success_rate = 1

    for instruction, qubits, cbit in transpiled_circ.data:
        if instruction.name in gate_error_dict.keys():
            if tuple(qubits) in gate_error_dict[instruction.name].keys():
                gate_err = gate_error_dict[instruction.name][tuple(qubits)]
                success_rate *= 1 - gate_err
            else:
                logger.warning(
                    f"no error rate found for gate {instruction.name} on qubits {qubits}"
                )

    return success_rate


def get_provider(backend_name, hub=None):
    """
        Get the provider object for a specific backend from IBM Quantum.

        Args:
            backend_name (str): Name of the backend.
            hub (str): Optional hub name.

        Returns:
            IBMQProvider: The provider object.
        """
    QiskitRuntimeService = _import_qiskit_runtime()
    QiskitError = _import_qiskit_error()
    
    # mass-inst-tech-1 or MIT-1
    if backend_name in ["ibmq_casablanca", "ibmq_rome", "ibmq_bogota", "ibmq_jakarta"]:
        if hub == "mass" or hub is None:
            provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q-research/mass-inst-tech-1/main")
        elif hub == "mit":
            provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q-research/MIT-1/main")
        else:
            raise ValueError(f"not supported backend {backend_name} in hub " f"{hub}")
    elif backend_name in [
        "ibmq_paris",
        "ibmq_toronto",
        "ibmq_manhattan",
        "ibmq_guadalupe",
        "ibmq_montreal",
    ]:
        provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q-ornl/anl/csc428")
    else:
        if hub == "mass" or hub is None:
            try:
                provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q-research/mass-inst-tech-1/main")
            except QiskitError:
                # logger.warning(f"Cannot use MIT backend, roll back to open")
                logger.warning(f"Use the open backend")
                provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q/open/main")
        elif hub == "mit":
            provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q-research/MIT-1/main")
        else:
            provider = QiskitRuntimeService(channel = "ibm_quantum", instance = "ibm-q/open/main")

    return provider


def get_provider_hub_group_project(hub="ibm-q", group="open", project="main"):
    """
        Get provider by specifying hub, group, and project.

        Args:
            hub (str): Hub name.
            group (str): Group name.
            project (str): Project name.

        Returns:
            IBMQProvider: The provider object.
        """
    QiskitRuntimeService = _import_qiskit_runtime()
    provider = QiskitRuntimeService(channel = "ibm_quantum", instance = f"{hub}/{group}/{project}")
    return provider

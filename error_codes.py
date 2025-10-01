import torch
from abc import ABC, abstractmethod
from quantum_states import QuantumStates
from utils import kron_multiple, commutes


class ErrorCorrectionCode(ABC):
    def __init__(self):
        self.states = QuantumStates()
        self.n_qubits = None
        self.stabilizers = []
        self.recovery_map = {}

    @abstractmethod
    def create_stabilizers(self):
        pass

    @abstractmethod
    def create_recovery_map(self):
        pass

    def generate_syndromes(self):
        syndrome_map = {}

        for i in range(self.n_qubits):
            error = [self.states.identity] * self.n_qubits
            error[i] = self.states.pauli_X
            error_op = kron_multiple(*error)

            syndrome = tuple(
                1 if not commutes(error_op, stab) else 0
                for stab in self.stabilizers
            )
            syndrome_map[f'X_{i + 1}'] = syndrome

        for i in range(self.n_qubits):
            error = [self.states.identity] * self.n_qubits
            error[i] = self.states.pauli_Z
            error_op = kron_multiple(*error)

            syndrome = tuple(
                1 if not commutes(error_op, stab) else 0
                for stab in self.stabilizers
            )
            syndrome_map[f'Z_{i + 1}'] = syndrome

        return syndrome_map


class ThreeQubitBitFlipCode(ErrorCorrectionCode):
    def __init__(self):
        super().__init__()
        self.n_qubits = 3
        self.stabilizers = self.create_stabilizers()
        self.recovery_map = self.create_recovery_map()

    def create_stabilizers(self):
        return [
            kron_multiple(self.states.identity, self.states.pauli_Z, self.states.pauli_Z),
            kron_multiple(self.states.pauli_Z, self.states.pauli_Z, self.states.identity)
        ]

    def create_recovery_map(self):
        return {
            (0, 0): kron_multiple(self.states.identity, self.states.identity, self.states.identity),
            (0, 1): kron_multiple(self.states.pauli_X, self.states.identity, self.states.identity),
            (1, 0): kron_multiple(self.states.identity, self.states.identity, self.states.pauli_X),
            (1, 1): kron_multiple(self.states.identity, self.states.pauli_X, self.states.identity)
        }


class FiveQubitSurfaceCode(ErrorCorrectionCode):
    def __init__(self):
        super().__init__()
        self.n_qubits = 5
        self.stabilizers = self.create_stabilizers()

    def create_stabilizers(self):
        return [
            kron_multiple(self.states.pauli_X, self.states.pauli_X, self.states.pauli_X, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.pauli_Z, self.states.identity, self.states.pauli_Z, self.states.pauli_Z,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.pauli_Z, self.states.pauli_Z, self.states.identity,
                          self.states.pauli_Z),
            kron_multiple(self.states.identity, self.states.identity, self.states.pauli_X, self.states.pauli_X,
                          self.states.pauli_X)
        ]

    def create_recovery_map(self):
        return {}

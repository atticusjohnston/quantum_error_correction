import torch
from abc import ABC, abstractmethod
from quantum_states import QuantumStates
from utils import kron_multiple, commutes


class ErrorCorrectionCode(ABC):
    def __init__(self, device='cpu'):
        self.device = device
        self.states = QuantumStates(device=self.device)
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

        # Build all error operators
        X_errors = torch.stack([
            kron_multiple(*[self.states.pauli_X if j == i else self.states.identity
                            for j in range(self.n_qubits)])
            for i in range(self.n_qubits)
        ])

        Z_errors = torch.stack([
            kron_multiple(*[self.states.pauli_Z if j == i else self.states.identity
                            for j in range(self.n_qubits)])
            for i in range(self.n_qubits)
        ])

        stabilizers = torch.stack(self.stabilizers)

        def compute_commutators(errors):
            syndromes = []
            for error in errors:
                syndrome = []
                for stab in stabilizers:
                    ES = torch.matmul(error, stab)
                    SE = torch.matmul(stab, error)
                    commutes = (torch.abs(ES - SE).sum() < 1e-7).int().item()
                    syndrome.append(0 if commutes else 1)
                syndromes.append(syndrome)
            return torch.tensor(syndromes, dtype=torch.int)

        X_syndromes = compute_commutators(X_errors)
        Z_syndromes = compute_commutators(Z_errors)

        for i in range(self.n_qubits):
            syndrome_map[f'X_{i + 1}'] = tuple(X_syndromes[i].cpu().tolist())
            syndrome_map[f'Z_{i + 1}'] = tuple(Z_syndromes[i].cpu().tolist())

        return syndrome_map


class ThreeQubitBitFlipCode(ErrorCorrectionCode):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
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
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.n_qubits = 5
        self.stabilizers = self.create_stabilizers()
        self.recovery_map = self.create_recovery_map()

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
        """
        Error-Syndrome map:
        {    'X_1': (0, 1, 0, 0),
             'X_2': (0, 0, 1, 0),
             'X_3': (0, 1, 1, 0),
             'X_4': (0, 1, 0, 0),
             'X_5': (0, 0, 1, 0),
             'Z_1': (1, 0, 0, 0),
             'Z_2': (1, 0, 0, 0),
             'Z_3': (1, 0, 0, 1),
             'Z_4': (0, 0, 0, 1),
             'Z_5': (0, 0, 0, 1)    }
        """
        return {
            (0, 0, 0, 0): kron_multiple(self.states.identity, self.states.identity, self.states.identity,
                                        self.states.identity, self.states.identity),
            (0, 1, 0, 0): kron_multiple(self.states.pauli_X, self.states.identity, self.states.identity,
                                        self.states.identity, self.states.identity),
            (0, 0, 1, 0): kron_multiple(self.states.identity, self.states.pauli_X, self.states.identity,
                                        self.states.identity, self.states.identity),
            (0, 1, 1, 0): kron_multiple(self.states.identity, self.states.identity, self.states.pauli_X,
                                        self.states.identity, self.states.identity),
            (1, 0, 0, 0): kron_multiple(self.states.pauli_Z, self.states.identity, self.states.identity,
                                        self.states.identity, self.states.identity),
            (1, 0, 0, 1): kron_multiple(self.states.identity, self.states.identity, self.states.pauli_Z,
                                        self.states.identity, self.states.identity),
            (0, 0, 0, 1): kron_multiple(self.states.identity, self.states.identity, self.states.identity,
                                        self.states.pauli_Z, self.states.identity),
        }


class ThirteenQubitSurfaceCode(ErrorCorrectionCode):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.n_qubits = 13
        self.stabilizers = self.create_stabilizers()
        # self.recovery_map = self.create_recovery_map()

    def create_stabilizers(self):
        return [
            kron_multiple(self.states.pauli_X, self.states.pauli_X, self.states.identity, self.states.pauli_X,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.pauli_X, self.states.pauli_X, self.states.identity,
                          self.states.pauli_X, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.pauli_Z, self.states.identity, self.states.identity, self.states.pauli_Z,
                          self.states.identity, self.states.pauli_Z, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.pauli_Z, self.states.identity, self.states.pauli_Z,
                          self.states.pauli_Z, self.states.identity, self.states.pauli_Z, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.pauli_Z, self.states.identity,
                          self.states.pauli_Z, self.states.identity, self.states.identity, self.states.pauli_Z,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.pauli_X,
                          self.states.identity, self.states.pauli_X, self.states.pauli_X, self.states.identity,
                          self.states.pauli_X, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.pauli_X, self.states.identity, self.states.pauli_X, self.states.pauli_X,
                          self.states.identity, self.states.pauli_X, self.states.identity, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.pauli_Z, self.states.identity, self.states.identity,
                          self.states.pauli_Z, self.states.identity, self.states.pauli_Z, self.states.identity,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.pauli_Z, self.states.identity,
                          self.states.pauli_Z, self.states.pauli_Z, self.states.identity, self.states.pauli_Z,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.pauli_Z,
                          self.states.identity, self.states.pauli_Z, self.states.identity, self.states.identity,
                          self.states.pauli_Z),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.pauli_X, self.states.identity, self.states.pauli_X, self.states.pauli_X,
                          self.states.identity),
            kron_multiple(self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.identity, self.states.identity, self.states.identity,
                          self.states.identity, self.states.pauli_X, self.states.identity, self.states.pauli_X,
                          self.states.pauli_X)
        ]

    def create_recovery_map(self):
        """
        Error-Syndrome map:
        {    'X_1': (0, 1, 0, 0),
             'X_2': (0, 0, 1, 0),
             'X_3': (0, 1, 1, 0),
             'X_4': (0, 1, 0, 0),
             'X_5': (0, 0, 1, 0),
             'Z_1': (1, 0, 0, 0),
             'Z_2': (1, 0, 0, 0),
             'Z_3': (1, 0, 0, 1),
             'Z_4': (0, 0, 0, 1),
             'Z_5': (0, 0, 0, 1)    }
        """
        pass

# error_codes.py
import torch
import logging
from abc import ABC, abstractmethod
from quantum_states import QuantumStates
from utils import kron_multiple, commutes
from itertools import combinations

logger = logging.getLogger(__name__)


class ErrorCorrectionCode(ABC):
    def __init__(self, device='cpu'):
        self.device = device
        self.states = QuantumStates(device=self.device)
        self.n_qubits = None
        self.stabilizers = []
        self.recovery_map = {}
        logger.debug(f"ErrorCorrectionCode initialized on {device}")

    @abstractmethod
    def create_stabilizers(self):
        pass

    @abstractmethod
    def create_recovery_map(self):
        pass

    def generate_syndromes(self):
        logger.info(f"Generating syndromes for {self.n_qubits}-qubit code")
        syndrome_map = {}

        logger.debug(f"Building {self.n_qubits} X error operators")
        X_errors = torch.stack([
            kron_multiple(*[self.states.pauli_X if j == i else self.states.identity
                            for j in range(self.n_qubits)])
            for i in range(self.n_qubits)
        ])

        logger.debug(f"Building {self.n_qubits} Z error operators")
        Z_errors = torch.stack([
            kron_multiple(*[self.states.pauli_Z if j == i else self.states.identity
                            for j in range(self.n_qubits)])
            for i in range(self.n_qubits)
        ])

        logger.debug(f"Building {self.n_qubits} Y error operators")
        Y_errors = torch.stack([
            kron_multiple(*[self.states.pauli_Y if j == i else self.states.identity
                            for j in range(self.n_qubits)])
            for i in range(self.n_qubits)
        ])

        logger.debug(f"Building 2-qubit error operators")
        two_qubit_errors = []
        paulis = {
            "X": self.states.pauli_X,
            "Y": self.states.pauli_Y,
            "Z": self.states.pauli_Z,
        }

        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                for p1_label, p1 in paulis.items():
                    for p2_label, p2 in paulis.items():
                        label = f"{p1_label}{i + 1}_{p2_label}{j + 1}"
                        op = kron_multiple(*[
                            (p1 if k == i else (p2 if k == j else self.states.identity))
                            for k in range(self.n_qubits)
                        ])
                        two_qubit_errors.append((label, op))

        # separate labels and operators
        multi_labels, multi_ops = zip(*two_qubit_errors)
        multi_errors = torch.stack(multi_ops)

        print(multi_errors)

        stabilizers = torch.stack(self.stabilizers)
        logger.debug(f"Stabilizers tensor: shape={stabilizers.shape}")

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

        logger.debug("Computing X error syndromes")
        X_syndromes = compute_commutators(X_errors)
        logger.debug("Computing Z error syndromes")
        Z_syndromes = compute_commutators(Z_errors)
        logger.debug("Computing Y error syndromes")
        Y_syndromes = compute_commutators(Y_errors)

        for i in range(self.n_qubits):
            syndrome_map[f'X_{i + 1}'] = tuple(X_syndromes[i].cpu().tolist())
            syndrome_map[f'Z_{i + 1}'] = tuple(Z_syndromes[i].cpu().tolist())
            syndrome_map[f'Y_{i + 1}'] = tuple(Y_syndromes[i].cpu().tolist())

        multi_syndromes = compute_commutators(multi_errors)
        for idx, syndrome in enumerate(multi_syndromes):
            syndrome_map[multi_labels[idx]] = tuple(syndrome.cpu().tolist())

        logger.info(f"Generated {len(syndrome_map)} syndrome mappings")
        return syndrome_map


class ThreeQubitBitFlipCode(ErrorCorrectionCode):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.n_qubits = 3
        logger.info("Initializing ThreeQubitBitFlipCode")
        self.stabilizers = self.create_stabilizers()
        self.recovery_map = self.create_recovery_map()
        logger.info(
            f"ThreeQubitBitFlipCode ready: {len(self.stabilizers)} stabilizers, {len(self.recovery_map)} recovery ops")

    def create_stabilizers(self):
        logger.debug("Creating stabilizers for 3-qubit code")
        return [
            kron_multiple(self.states.identity, self.states.pauli_Z, self.states.pauli_Z),
            kron_multiple(self.states.pauli_Z, self.states.pauli_Z, self.states.identity)
        ]

    def create_recovery_map(self):
        logger.debug("Creating recovery map for 3-qubit code")
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
        logger.info("Initializing FiveQubitSurfaceCode")
        self.stabilizers = self.create_stabilizers()
        self.recovery_map = self.create_recovery_map()
        logger.info(
            f"FiveQubitSurfaceCode ready: {len(self.stabilizers)} stabilizers, {len(self.recovery_map)} recovery ops")

    def create_stabilizers(self):
        logger.debug("Creating stabilizers for 5-qubit surface code")
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
        logger.debug("Creating recovery map for 5-qubit surface code")
        I = self.states.identity
        X = self.states.pauli_X
        Y = self.states.pauli_Y
        Z = self.states.pauli_Z

        return {
            (0, 0, 0, 0): kron_multiple(I, I, I, I, I),
            (0, 1, 0, 0): kron_multiple(X, I, I, I, I),
            (0, 0, 1, 0): kron_multiple(I, X, I, I, I),
            (0, 1, 1, 0): kron_multiple(I, I, X, I, I),
            (1, 0, 0, 0): kron_multiple(Z, I, I, I, I),
            (1, 0, 0, 1): kron_multiple(I, I, Z, I, I),
            (0, 0, 0, 1): kron_multiple(I, I, I, Z, I),
            (1, 1, 0, 0): kron_multiple(Y, I, I, I, I),
            (1, 0, 1, 0): kron_multiple(I, Y, I, I, I),
            (1, 1, 1, 1): kron_multiple(I, I, Y, I, I),
            (0, 1, 0, 1): kron_multiple(I, I, I, Y, I),
            (0, 0, 1, 1): kron_multiple(I, I, I, I, Y),
            (1, 0, 1, 1): kron_multiple(X, I, Y, I, I),
            (1, 1, 0, 1): kron_multiple(X, I, Z, I, I),
            (0, 1, 1, 1): kron_multiple(X, I, I, I, Y),
            (1, 1, 1, 0): kron_multiple(X, Y, I, I, I),
        }


class ThirteenQubitSurfaceCode(ErrorCorrectionCode):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.n_qubits = 13
        logger.info("Initializing ThirteenQubitSurfaceCode")
        self.stabilizers = self.create_stabilizers()
        self.recovery_map = self.create_recovery_map()
        logger.info(
            f"ThirteenQubitSurfaceCode ready: {len(self.stabilizers)} stabilizers, {len(self.recovery_map)} recovery ops")

    def create_stabilizers(self):
        logger.debug("Creating stabilizers for 13-qubit surface code")
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
        logger.debug("Creating recovery map for 13-qubit surface code")
        I = self.states.identity
        X = self.states.pauli_X
        Z = self.states.pauli_Z

        def op(pos, pauli):
            ops = [I] * 13
            ops[pos] = pauli
            return kron_multiple(*ops)

        recovery_map = {
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): kron_multiple(*[I] * 13),
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0): op(0, X),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0): op(1, X),
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0): op(2, X),
            (0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0): op(3, X),
            (0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0): op(4, X),
            (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0): op(5, X),
            (0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0): op(6, X),
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0): op(7, X),
            (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0): op(8, X),
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0): op(9, X),
            (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0): op(10, X),
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0): op(11, X),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0): op(12, X),
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): op(0, Z),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): op(1, Z),
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): op(2, Z),
            (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): op(3, Z),
            (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): op(4, Z),
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): op(5, Z),
            (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0): op(6, Z),
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0): op(7, Z),
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0): op(8, Z),
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1): op(9, Z),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0): op(10, Z),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1): op(11, Z),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1): op(12, Z),
        }
        logger.info(f"Recovery map created with {len(recovery_map)} entries")
        return recovery_map

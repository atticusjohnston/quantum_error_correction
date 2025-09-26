import torch
import logging
from error_codes import *
from utils import check_kraus_sum


class QuantumErrorCorrection:
    def __init__(self, code_type, **kwargs):
        self.code_type = code_type
        self.code = self._initialize_code(code_type, **kwargs)
        self.n_qubits = self.code.n_qubits
        self.dim = 2 ** self.n_qubits
        self.stabilizers = self.code.stabilizers
        self.recovery_map = self.code.recovery_map
        logging.info(f"Initialized {code_type} error correction code with {self.n_qubits} qubits")

    def _initialize_code(self, code_type, **kwargs):
        codes = {
            'three_qubit': ThreeQubitBitFlipCode,
        }

        if code_type not in codes:
            raise ValueError(f"Unknown code type: {code_type}")

        return codes[code_type]()

    def compute_syndrome_projector(self, syndrome_bits):
        projector = torch.eye(self.dim, dtype=torch.complex128)
        for i, bit in enumerate(syndrome_bits):
            factor = 0.5 * (torch.eye(self.dim, dtype=torch.complex128) +
                            ((-1) ** bit) * self.stabilizers[i])
            projector = projector @ factor
        return projector

    def build_superoperator(self, tricky=True):
        superoperator = torch.zeros(self.dim ** 2, self.dim ** 2, dtype=torch.complex128)
        kraus_sum = torch.zeros(self.dim, self.dim, dtype=torch.complex128)

        if tricky:
            no_error_syndrome = tuple([0] * len(self.stabilizers))
            all_zero_projector = self.compute_syndrome_projector(no_error_syndrome)

        for syndrome, recovery in self.recovery_map.items():
            projector_m = self.compute_syndrome_projector(syndrome)
            if tricky:
                combined_op = all_zero_projector @ recovery
            else:
                combined_op = recovery @ projector_m

            kraus_sum += combined_op.conj().T @ combined_op
            superoperator += torch.kron(torch.conj(combined_op), combined_op)

        check_kraus_sum(self.dim, kraus_sum)
        return superoperator

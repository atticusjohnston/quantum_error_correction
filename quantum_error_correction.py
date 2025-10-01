import torch
import logging
from error_codes import *
from utils import check_kraus_sum


def _initialize_code(code_type, device, **kwargs):
    codes = {
        'three_qubit': ThreeQubitBitFlipCode,
        'five_qubit_surface': FiveQubitSurfaceCode,
        'thirteen_qubit_surface': ThirteenQubitSurfaceCode,
    }

    if code_type not in codes:
        raise ValueError(f"Unknown code type: {code_type}")

    return codes[code_type](device=device)


class QuantumErrorCorrection:
    def __init__(self, code_type, device='cpu', **kwargs):
        self.device = device
        self.code_type = code_type
        self.code = _initialize_code(code_type, device=self.device, **kwargs)
        self.n_qubits = self.code.n_qubits
        self.dim = 2 ** self.n_qubits
        self.stabilizers = self.code.stabilizers
        self.recovery_map = self.code.recovery_map
        logging.info(f"Initialized {code_type} error correction code on device {self.device} with {self.n_qubits} qubits")

    def compute_syndrome_projector(self, syndrome_bits):
        projector = torch.eye(self.dim, dtype=torch.complex64, device=self.device)
        for i, bit in enumerate(syndrome_bits):
            factor = 0.5 * (torch.eye(self.dim, dtype=torch.complex64, device=self.device) +
                            ((-1) ** bit) * self.stabilizers[i])
            projector = projector @ factor
        return projector

    def build_superoperator(self, tricky=True):
        dim_sq = self.dim ** 2
        superoperator = torch.zeros(dim_sq, dim_sq, dtype=torch.complex64, device=self.device)
        kraus_sum = torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)

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

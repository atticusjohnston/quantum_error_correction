# quantum_error_correction.py
import torch
import logging
from error_codes import *
from utils import check_kraus_sum

logger = logging.getLogger(__name__)


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
        logger.info(f"Initializing {code_type} on {device}")

        self.code = _initialize_code(code_type, device=self.device, **kwargs)
        self.n_qubits = self.code.n_qubits
        self.dim = 2 ** self.n_qubits
        self.stabilizers = self.code.stabilizers
        self.recovery_map = self.code.recovery_map

        logger.info(f"Code initialized: {self.n_qubits} qubits, dim={self.dim}, "
                    f"{len(self.stabilizers)} stabilizers, {len(self.recovery_map)} recovery ops")

    def compute_syndrome_projector(self, syndrome_bits):
        logger.debug(f"Computing projector for syndrome: {syndrome_bits}")
        projector = torch.eye(self.dim, dtype=torch.complex64, device=self.device)
        for i, bit in enumerate(syndrome_bits):
            factor = 0.5 * (torch.eye(self.dim, dtype=torch.complex64, device=self.device) +
                            ((-1) ** bit) * self.stabilizers[i])
            projector = projector @ factor
        logger.debug(f"Projector computed: shape={projector.shape}, norm={torch.norm(projector):.6f}")
        return projector

    def build_superoperator(self, tricky=True):
        logger.info(f"Building superoperator (tricky={tricky})")
        dim_sq = self.dim ** 2
        superoperator = torch.zeros(dim_sq, dim_sq, dtype=torch.complex64, device=self.device)
        kraus_sum = torch.zeros(self.dim, self.dim, dtype=torch.complex64, device=self.device)

        if tricky:
            no_error_syndrome = tuple([0] * len(self.stabilizers))
            logger.debug(f"Computing all-zero projector for syndrome: {no_error_syndrome}")
            all_zero_projector = self.compute_syndrome_projector(no_error_syndrome)

        for idx, (syndrome, recovery) in enumerate(self.recovery_map.items()):
            logger.debug(f"Processing syndrome {idx + 1}/{len(self.recovery_map)}: {syndrome}")
            projector_m = self.compute_syndrome_projector(syndrome)

            if tricky:
                combined_op = all_zero_projector @ recovery
            else:
                combined_op = recovery @ projector_m

            kraus_sum += combined_op.conj().T @ combined_op
            superoperator += torch.kron(torch.conj(combined_op), combined_op)
            logger.debug(f"Syndrome {syndrome}: combined_op norm={torch.norm(combined_op):.6f}")

        logger.info(f"Superoperator built: shape={superoperator.shape}")
        check_kraus_sum(self.dim, kraus_sum)
        return superoperator
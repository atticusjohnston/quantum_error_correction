# main.py
import torch
import logging
import sys
import argparse
from quantum_error_correction import QuantumErrorCorrection
from analysis import QuantumAnalysis
from plotting import QuantumPlotter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_type', type=str, default='three_qubit',
                        choices=['three_qubit', 'five_qubit_surface', 'thirteen_qubit_surface'])
    parser.add_argument('--tricky', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('qec.log')
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    if args.device == 'auto':
        if torch.mps.is_available():
            DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(args.device)

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Code type: {args.code_type}")
    logger.info(f"Tricky mode: {args.tricky}")

    try:
        qec = QuantumErrorCorrection(code_type=args.code_type, device=DEVICE)
        logger.info(f"QEC initialized: {qec.n_qubits} qubits, dim={qec.dim}")

        superoperator = qec.build_superoperator(tricky=args.tricky)
        logger.info(f"Superoperator shape: {superoperator.shape}")

        n = qec.n_qubits
        k = n - len(qec.stabilizers)

        expected_nonzero = 2 ** ((n - k) // 2)
        mult_nonzero = 4 ** k
        mult_zero = (4 ** n) - mult_nonzero

        print("\nExpected Singular Value Spectrum:")
        print(f"  0: {mult_zero}")
        print(f"  {expected_nonzero}: {mult_nonzero}\n")

        results = QuantumAnalysis.compute_eigenvalues_and_singular_values(superoperator)
        analysis = QuantumAnalysis.analyze_spectrum(results)

        print(f"\nEigenvalue Magnitudes:")
        for val, count in zip(analysis['unique_eigenvalue_magnitudes'], analysis['eigenvalue_multiplicities']):
            print(f"  {val:.3f}: {count}")

        print(f"\nSingular Values:")
        for val, count in zip(analysis['unique_singular_values'], analysis['singular_value_multiplicities']):
            print(f"  {val:.3f}: {count}")

        logger.info("Computation completed successfully")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
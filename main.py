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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == 'auto':
        if torch.mps.is_available():
            DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(args.device)

    print(f"Using device: {DEVICE}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    qec = QuantumErrorCorrection(code_type=args.code_type, device=DEVICE)

    print(qec.code.generate_syndromes())

    superoperator = qec.build_superoperator(tricky=args.tricky)

    results = QuantumAnalysis.compute_eigenvalues_and_singular_values(superoperator)
    analysis = QuantumAnalysis.analyze_spectrum(results)

    print(f"\nEigenvalue Magnitudes:")
    for val, count in zip(analysis['unique_eigenvalue_magnitudes'], analysis['eigenvalue_multiplicities']):
        print(f"  {val:.3f}: {count}")

    print(f"\nSingular Values:")
    for val, count in zip(analysis['unique_singular_values'], analysis['singular_value_multiplicities']):
        print(f"  {val:.3f}: {count}")
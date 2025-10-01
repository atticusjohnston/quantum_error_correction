import torch
import logging
import sys

from quantum_error_correction import QuantumErrorCorrection
from analysis import QuantumAnalysis
from plotting import QuantumPlotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

module_path = '/Users/atticusjohnston/Documents/uq-coding-assignments/matrixChernoff/code'
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def print_analysis_results(analysis, title):
    print(f"\n--- {title} ---")
    print("Eigenvalue Magnitudes:")
    for val, count in zip(analysis['unique_eigenvalue_magnitudes'],
                          analysis['eigenvalue_multiplicities']):
        print(f"  Value: {val:.3f}, Multiplicity: {count}")

    print("\nSingular Values:")
    for val, count in zip(analysis['unique_singular_values'],
                          analysis['singular_value_multiplicities']):
        print(f"  Value: {val:.3f}, Multiplicity: {count}")


def compare_with_accepted_result(superoperator, code_type='three_qubit'):
    """Compare calculated superoperator with mentor's accepted result."""
    try:
        from applications import make_syndrome_measurements
        accepted_result = torch.tensor(
            make_syndrome_measurements(code_type)[0],
            dtype=torch.complex128
        )

        if torch.allclose(superoperator, accepted_result, atol=1e-6):
            logging.info("Superoperator check PASSED: Matches mentor's result")
            return True
        else:
            diff = superoperator - accepted_result
            logging.error("Superoperator check FAILED: Does not match mentor's result")

            # Log top differences
            k = 5
            flat_diff = diff.abs().view(-1)
            topk_vals, topk_idx = torch.topk(flat_diff, min(k, len(flat_diff)))
            indices = [divmod(i.item(), diff.shape[1]) for i in topk_idx]

            for (row, col), val in zip(indices, topk_vals):
                logging.debug(f"Diff at ({row}, {col}): {val:.6f}")
            return False

    except ImportError:
        logging.warning("Could not import applications module for comparison with mentor's result")
        return None


def main():
    # code_type = 'three_qubit'
    code_type = 'five_qubit_surface'

    qec = QuantumErrorCorrection(code_type=code_type)

    tricky = True
    logging.info(f"Building superoperator. Tricky = {tricky}")
    superoperator = qec.build_superoperator(tricky=tricky)

    if code_type == 'three_qubit':
        compare_with_accepted_result(superoperator, code_type)

    results = QuantumAnalysis.compute_eigenvalues_and_singular_values(superoperator)
    analysis = QuantumAnalysis.analyze_spectrum(results)
    print_analysis_results(analysis, f"{code_type} Superoperator")

    plotter = QuantumPlotter()
    plotter.plot_eigenvalues_and_singular_values(
        results,
        save_path=f"outputs/figures/{code_type}_spectrum.png",
        plot_title=f"{code_type.replace('_', ' ').title()} Superoperator Spectrum"
    )


if __name__ == "__main__":
    main()
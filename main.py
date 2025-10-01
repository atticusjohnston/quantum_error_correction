import torch
import logging
import sys
from quantum_error_correction import QuantumErrorCorrection
from analysis import QuantumAnalysis
from plotting import QuantumPlotter

if torch.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple Metal Performance Shaders (MPS)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = "NVIDIA CUDA (GPU)"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

print(f"Using compute device: {DEVICE_NAME} ({DEVICE})")

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
            dtype=torch.complex64,
            device=superoperator.device
        )

        if torch.allclose(superoperator, accepted_result, atol=1e-6):
            logging.info("Superoperator check PASSED: Matches mentor's result")
            return True
        else:
            diff = superoperator - accepted_result
            logging.error(
                f"Superoperator check FAILED: Does not match mentor's result on device {superoperator.device}")

            # Log top differences
            k = 5
            flat_diff = diff.abs().view(-1)
            topk_vals, topk_idx = torch.topk(flat_diff, min(k, len(flat_diff)))

            indices = [divmod(i.item(), diff.shape[1]) for i in topk_idx.cpu()]
            topk_vals_cpu = topk_vals.cpu()

            for (row, col), val in zip(indices, topk_vals_cpu):
                logging.debug(f"Diff at ({row}, {col}): {val:.6f}")
            return False

    except ImportError:
        logging.warning("Could not import applications module for comparison with mentor's result")
        return None


def main():
    code_type = 'three_qubit'
    # code_type = 'five_qubit_surface'
    # code_type = 'thirteen_qubit_surface'

    qec = QuantumErrorCorrection(code_type=code_type, device=DEVICE)

    tricky = False
    logging.info(f"Building superoperator. Tricky = {tricky}")
    superoperator = qec.build_superoperator(tricky=tricky)

    logging.info(f"Superoperator built on device: {superoperator.device}")

    if code_type == 'three_qubit':
        compare_with_accepted_result(superoperator, code_type)

    n = qec.n_qubits
    k = n - len(qec.stabilizers)

    expected_nonzero = 2 ** ((n - k) // 2)
    mult_nonzero = 4 ** k
    mult_zero = (4 ** n) - mult_nonzero

    print("\nExpected Singular Value Spectrum:")
    print(f"  Value: 0, Multiplicity: {mult_zero}")
    print(f"  Value: {expected_nonzero}, Multiplicity: {mult_nonzero}")

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

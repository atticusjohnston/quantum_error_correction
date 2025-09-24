import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import logging
import os
import numba
import qiskit
import importlib.util
from pathlib import Path

module_path = '/Users/atticusjohnston/Documents/uq-coding-assignments/matrixChernoff/code'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

try:
    from applications import make_syndrome_measurements
except ImportError as e:
    logging.critical("Failed to import 'make_syndrome_measurements' from 'applications'.")
    logging.critical("Ensure that the 'code' directory contains a file named `__init__.py` and the `applications.py` module.")
    logging.critical(f"Original error: {e}")
    raise SystemExit(1)


# --- LOGGING CONFIGURATION ---
# Set the desired logging level here.
# Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOGGING_LEVEL = logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


# --- UTILITY FUNCTIONS ---
def kron_multiple(*matrices) -> torch.Tensor:
    """Performs the Kronecker product of multiple matrices."""
    result = torch.tensor([1], dtype=torch.complex128)
    for matrix in matrices:
        result = torch.kron(result, matrix)
    return result


def check_kraus_sum_identity(kraus_sum: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Checks if the accumulated Kraus sum equals the identity matrix.
    This condition ensures the channel is trace-preserving.
    """
    dim = kraus_sum.shape[0]
    identity_matrix = torch.eye(dim, dtype=torch.complex128)
    if torch.allclose(kraus_sum, identity_matrix, atol=tol):
        logging.info("Kraus sum condition met: The sum of K^dag * K is the identity matrix.")
        return True
    else:
        logging.error("Kraus sum condition FAILED: The sum is not the identity matrix.")
        logging.debug(f"Computed sum:\n{kraus_sum}")
        logging.debug(f"Expected identity:\n{identity_matrix}")
        return False


# --- QUANTUM CLASSES ---
class QuantumStates:
    def __init__(self):
        logging.debug("Initialising quantum state tensors.")
        self.zero_state = torch.tensor([1, 0], dtype=torch.complex128)
        self.one_state = torch.tensor([0, 1], dtype=torch.complex128)

        logging.debug("Initialising Pauli and Identity matrices.")
        self.identity = torch.eye(2, dtype=torch.complex128)
        self.pauli_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        self.pauli_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        self.pauli_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


class QuantumErrorCorrection:
    def __init__(self, n_qubits=3):
        logging.info(f"Initialising QuantumErrorCorrection for {n_qubits} qubits.")
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.states = QuantumStates()
        self.stabilizers = self._create_stabilizers()
        self.recovery_map = self._create_recovery_map()
        logging.info("QuantumErrorCorrection initialisation complete.")

    def _create_stabilizers(self):
        logging.info("Creating stabilizer operators.")
        stabilizers = [
            kron_multiple(self.states.identity, self.states.pauli_Z, self.states.pauli_Z),
            kron_multiple(self.states.pauli_Z, self.states.pauli_Z, self.states.identity)
        ]
        logging.debug(f"Stabilizers created:\n{stabilizers}")
        return stabilizers

    def _create_recovery_map(self):
        logging.info("Creating recovery map operators.")
        recovery_map = {
            (0, 0): kron_multiple(self.states.identity, self.states.identity, self.states.identity),
            (0, 1): kron_multiple(self.states.pauli_X, self.states.identity, self.states.identity),
            (1, 0): kron_multiple( self.states.identity, self.states.identity, self.states.pauli_X),
            (1, 1): kron_multiple(self.states.identity, self.states.pauli_X, self.states.identity)
        }
        logging.debug(f"Recovery map created:\n{recovery_map}")
        return recovery_map

    def compute_syndrome_projector(self, syndrome_bits):
        logging.debug(f"Computing syndrome projector for bits: {syndrome_bits}")
        projector = torch.eye(self.dim, dtype=torch.complex128)
        for i in range(len(self.stabilizers)):
            factor = (1 / 2) * ((torch.eye(self.dim, dtype=torch.complex128)) +
                                (((-1) ** syndrome_bits[i]) * self.stabilizers[i]))
            logging.debug(f"Projector factor {i}:\n{factor}")
            projector = projector @ factor
        logging.debug(f"Final projector for {syndrome_bits}:\n{projector}")
        return projector

    def build_superoperator(self, tricky=True):
        superoperator = torch.zeros(self.dim ** 2, self.dim ** 2, dtype=torch.complex128)
        kraus_sum = torch.zeros(self.dim, self.dim, dtype=torch.complex128)
        all_zero_projector = self.compute_syndrome_projector((0, 0))

        for syndrome, recovery in self.recovery_map.items():
            projector_m = self.compute_syndrome_projector(syndrome)
            if tricky:
                combined_op = all_zero_projector @ recovery
            else:
                combined_op = recovery @ projector_m
            kraus_sum += combined_op.conj().T @ combined_op
            superoperator += torch.kron(
                torch.conj(combined_op),
                combined_op
            )

        check_kraus_sum_identity(kraus_sum)
        return superoperator


class QuantumAnalysis:
    @staticmethod
    def compute_eigenvalues_and_singular_values(matrix):
        logging.debug("Computing eigenvalues and singular values.")
        eigenvalues = torch.linalg.eigvals(matrix)
        singular_values = torch.linalg.svdvals(matrix)

        results = {
            'eigenvalues': eigenvalues,
            'eigenvalues_real': eigenvalues.real.numpy(),
            'eigenvalues_imag': eigenvalues.imag.numpy(),
            'eigenvalues_magnitude': torch.abs(eigenvalues).numpy(),
            'singular_values': singular_values.numpy()
        }
        logging.debug(f"Eigenvalues: {results['eigenvalues']}")
        logging.debug(f"Singular values: {results['singular_values']}")
        return results

    @staticmethod
    def unique_floats_summed(arr, tol=1e-5):
        logging.debug("Finding unique floats with tolerance: $1 \times 10^{-5}$")
        arr_rounded = np.round(arr / tol) * tol
        unique_vals, inverse = np.unique(arr_rounded, return_inverse=True)
        counts = np.bincount(inverse)
        return unique_vals, counts

    @classmethod
    def analyze_spectrum(cls, results):
        logging.debug("Analyzing spectral properties.")
        unique_eigen, counts_eigen = cls.unique_floats_summed(results['eigenvalues_magnitude'])
        unique_vals, counts = cls.unique_floats_summed(results['singular_values'])

        analysis = {
            'unique_eigenvalue_magnitudes': unique_eigen,
            'eigenvalue_multiplicities': counts_eigen,
            'unique_singular_values': unique_vals,
            'singular_value_multiplicities': counts
        }
        logging.debug(f"Analysis results: {analysis}")
        return analysis


class QuantumPlotter:
    def __init__(self):
        logging.info("Initialising QuantumPlotter with seaborn style.")
        self._setup_style()

    def _setup_style(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "0.3",
            "axes.linewidth": 1,
            "grid.alpha": 0.3,
            "figure.dpi": 100
        })

    def plot_eigenvalues_and_singular_values(self, results, save_path="eval.png"):
        logging.info(f"Generating and saving plots to '{save_path}'")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        self._plot_eigenvalues_complex_plane(axes[0], results)
        self._plot_eigenvalues_vs_singular_values(axes[1], results)

        plt.tight_layout(pad=2.0)

        try:
            plt.savefig(save_path)
            logging.info(f"Successfully saved plot to {os.path.abspath(save_path)}")
        except Exception as e:
            logging.error(f"Failed to save plot to {save_path}: {e}")
            logging.warning("Plot will be displayed instead of saved.")
            plt.show()

        return fig

    def _plot_eigenvalues_complex_plane(self, ax, results):
        logging.debug("Plotting eigenvalues in complex plane.")
        scatter = ax.scatter(
            results['eigenvalues_real'], results['eigenvalues_imag'],
            c=results['eigenvalues_magnitude'], cmap="viridis",
            s=50, alpha=0.8, edgecolor="k", linewidth=0.2
        )
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title("Eigenvalues in Complex Plane")
        ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), "r--", lw=1.2, alpha=0.7, label="Unit Circle")
        ax.legend(frameon=True, loc="upper right")
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Magnitude")

    def _plot_eigenvalues_vs_singular_values(self, ax, results):
        logging.debug("Plotting eigenvalues vs singular values.")
        sorted_eigen_mag = np.sort(results['eigenvalues_magnitude'])[::-1]
        indices = np.arange(len(results['singular_values']))
        ax.plot(indices, sorted_eigen_mag, "-", color="tab:blue", lw=0.8, alpha=0.4)
        ax.scatter(indices, sorted_eigen_mag, color="tab:blue", s=10, alpha=0.8, label=r"$|\lambda|$", zorder=3)
        ax.plot(indices, results['singular_values'], "-", color="tab:red", lw=0.8, alpha=0.4)
        ax.scatter(indices, results['singular_values'], color="tab:red", s=10, alpha=0.8, label="Singular Values",
                   zorder=3)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Eigenvalue Magnitudes vs Singular Values")
        ax.legend(frameon=True, loc="upper right")
        ymax = max(sorted_eigen_mag.max(), results['singular_values'].max()) * 1.05
        ax.set_ylim(bottom=0, top=ymax)


# --- MAIN EXECUTION ---
def main():
    logging.info("Starting quantum analysis script...")

    try:
        qec = QuantumErrorCorrection(n_qubits=3)

        tricky = True
        logging.info(f"Building superoperator from recovery map. Tricky = {tricky}.")
        superoperator = qec.build_superoperator(tricky=tricky)
        logging.debug(f"Superoperator shape: {superoperator.shape}")

        logging.info("Getting accepted result for comparison...")
        accepted_result_superoperator = torch.tensor(make_syndrome_measurements('three_qubit')[0],
                                                     dtype=torch.complex128)
        logging.debug(f"Accepted superoperator shape: {accepted_result_superoperator.shape}")

        logging.info("Computing eigenvalues and singular values for both superoperators...")
        results = QuantumAnalysis.compute_eigenvalues_and_singular_values(superoperator)
        results_accepted = QuantumAnalysis.compute_eigenvalues_and_singular_values(accepted_result_superoperator)

        logging.info("Analyzing spectral properties...")
        analysis = QuantumAnalysis.analyze_spectrum(results)
        analysis_accepted = QuantumAnalysis.analyze_spectrum(results_accepted)

        logging.info(f"Number of eigenvalues: {len(results['eigenvalues'])} | Number of singular values: {len(results['singular_values'])}")

        print_analysis_results(analysis, "Calculated Superoperator")
        print_analysis_results(analysis_accepted, "Accepted Result Superoperator")

        if torch.allclose(superoperator, accepted_result_superoperator, atol=1e-6):
            logging.info("Superoperator check PASSED: The two superoperators are allclose.")
        else:
            diff = superoperator - accepted_result_superoperator
            logging.error("Superoperator check FAILED: The two superoperators are not allclose.")

            # Consolidate top-k differences into a single log message.
            k = 5
            flat_diff = diff.abs().view(-1)
            topk_vals, topk_idx = torch.topk(flat_diff, k)
            indices = [divmod(i.item(), diff.shape[1]) for i in topk_idx]

            diff_message_lines = ["Top 5 largest differences:"]
            for (row, col), val in zip(indices, topk_vals):
                diff_message_lines.append(
                    f"  - ({row}, {col}): calc={superoperator[row, col]:.4f}, "
                    f"acc={accepted_result_superoperator[row, col]:.4f}, "
                    f"diff={val:.4f}"
                )
            logging.debug('\n'.join(diff_message_lines))

        plotter = QuantumPlotter()
        plotter.plot_eigenvalues_and_singular_values(results, save_path="calculated_superoperator_spectrum.png")
        plotter.plot_eigenvalues_and_singular_values(results_accepted, save_path="accepted_superoperator_spectrum.png")

        logging.info("Script execution complete.")

    except Exception as e:
        logging.critical(f"An unrecoverable error occurred during script execution: {e}", exc_info=True)


def print_analysis_results(analysis, title):
    """A helper function to print the spectral analysis results in a readable format."""
    message_lines = [f"\n--- {title} ---", "Eigenvalue Magnitudes:"]

    if len(analysis['unique_eigenvalue_magnitudes']) == 0:
        message_lines.append("No unique eigenvalue magnitudes found.")
    else:
        for val, count in zip(analysis['unique_eigenvalue_magnitudes'], analysis['eigenvalue_multiplicities']):
            message_lines.append(f"Value: {val:.3f}, Multiplicity: {count}")

    message_lines.append("\nSingular Values:")
    if len(analysis['unique_singular_values']) == 0:
        message_lines.append("No unique singular values found.")
    else:
        for val, count in zip(analysis['unique_singular_values'], analysis['singular_value_multiplicities']):
            message_lines.append(f"Value: {val:.3f}, Multiplicity: {count}")

    logging.info('\n'.join(message_lines))


if __name__ == "__main__":
    main()

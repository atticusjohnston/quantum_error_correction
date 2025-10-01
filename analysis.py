import torch
import numpy as np
from utils import unique_floats_summed


class QuantumAnalysis:
    @staticmethod
    def compute_eigenvalues_and_singular_values(matrix):
        eigenvalues = torch.linalg.eigvals(matrix.cpu())
        singular_values = torch.linalg.svdvals(matrix.cpu())

        return {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvalues_real': eigenvalues.real.cpu().numpy(),
            'eigenvalues_imag': eigenvalues.imag.cpu().numpy(),
            'eigenvalues_magnitude': torch.abs(eigenvalues).cpu().numpy(),
            'singular_values': singular_values.cpu().numpy()
        }

    @staticmethod
    def analyze_spectrum(results):
        unique_eigen, counts_eigen = unique_floats_summed(results['eigenvalues_magnitude'])
        unique_vals, counts = unique_floats_summed(results['singular_values'])

        return {
            'unique_eigenvalue_magnitudes': unique_eigen,
            'eigenvalue_multiplicities': counts_eigen,
            'unique_singular_values': unique_vals,
            'singular_value_multiplicities': counts
        }

import torch
import numpy as np
from utils import unique_floats_summed


class QuantumAnalysis:
    @staticmethod
    def compute_eigenvalues_and_singular_values(matrix):
        eigenvalues = torch.linalg.eigvals(matrix)
        singular_values = torch.linalg.svdvals(matrix)

        return {
            'eigenvalues': eigenvalues,
            'eigenvalues_real': eigenvalues.real.numpy(),
            'eigenvalues_imag': eigenvalues.imag.numpy(),
            'eigenvalues_magnitude': torch.abs(eigenvalues).numpy(),
            'singular_values': singular_values.numpy()
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
# analysis.py
import torch
import numpy as np
import logging
from utils import unique_floats_summed

logger = logging.getLogger(__name__)


class QuantumAnalysis:
    @staticmethod
    def compute_eigenvalues_and_singular_values(matrix):
        logger.info(f"Computing eigenvalues and singular values for matrix: {matrix.shape}")
        logger.debug(f"Matrix dtype: {matrix.dtype}, device: {matrix.device}")

        try:
            eigenvalues = torch.linalg.eigvals(matrix.cpu())
            logger.info(f"Eigenvalues computed: {len(eigenvalues)} values")
            logger.debug(f"Eigenvalue range: [{eigenvalues.abs().min():.6f}, {eigenvalues.abs().max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to compute eigenvalues: {e}")
            raise

        try:
            singular_values = torch.linalg.svdvals(matrix.cpu())
            logger.info(f"Singular values computed: {len(singular_values)} values")
            logger.debug(f"Singular value range: [{singular_values.min():.6f}, {singular_values.max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to compute singular values: {e}")
            raise

        return {
            'eigenvalues': eigenvalues.cpu(),
            'eigenvalues_real': eigenvalues.real.cpu().numpy(),
            'eigenvalues_imag': eigenvalues.imag.cpu().numpy(),
            'eigenvalues_magnitude': torch.abs(eigenvalues).cpu().numpy(),
            'singular_values': singular_values.cpu().numpy()
        }

    @staticmethod
    def analyze_spectrum(results):
        logger.info("Analyzing spectrum")

        unique_eigen, counts_eigen = unique_floats_summed(results['eigenvalues_magnitude'])
        logger.info(f"Unique eigenvalue magnitudes: {len(unique_eigen)}")
        logger.debug(f"Eigenvalue multiplicities: {counts_eigen}")

        unique_vals, counts = unique_floats_summed(results['singular_values'])
        logger.info(f"Unique singular values: {len(unique_vals)}")
        logger.debug(f"Singular value multiplicities: {counts}")

        return {
            'unique_eigenvalue_magnitudes': unique_eigen,
            'eigenvalue_multiplicities': counts_eigen,
            'unique_singular_values': unique_vals,
            'singular_value_multiplicities': counts
        }
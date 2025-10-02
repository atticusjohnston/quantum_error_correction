# analysis.py
import torch
import numpy as np
import logging
from utils import unique_floats_summed
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


class QuantumAnalysis:
    @staticmethod
    def compute_eigenvalues_and_singular_values(matrix, sparse=False):
        if sparse:
            return QuantumAnalysis._compute_sparse(matrix)
        else:
            return QuantumAnalysis._compute_dense(matrix)

    @staticmethod
    def _compute_dense(matrix):
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
    def _compute_sparse(matrix):
        logger.info(f"Computing eigenvalues and singular values for sparse matrix: {matrix.shape}, nnz={matrix.nnz}")

        k = min(100, matrix.shape[0] - 2)

        try:
            eigenvalues, _ = spla.eigs(matrix, k=k, which='LM')
            logger.info(f"Eigenvalues computed: {len(eigenvalues)} values")
            logger.debug(f"Eigenvalue range: [{np.abs(eigenvalues).min():.6f}, {np.abs(eigenvalues).max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to compute eigenvalues: {e}")
            raise

        try:
            sv_squared, _ = spla.eigs(matrix @ matrix.conj().T, k=k, which='LM')
            singular_values = np.sqrt(np.abs(sv_squared.real))
            singular_values = np.sort(singular_values)[::-1]
            logger.info(f"Singular values computed: {len(singular_values)} values")
            logger.debug(f"Singular value range: [{singular_values.min():.6f}, {singular_values.max():.6f}]")
        except Exception as e:
            logger.error(f"Failed to compute singular values: {e}")
            raise

        return {
            'eigenvalues': eigenvalues,
            'eigenvalues_real': eigenvalues.real,
            'eigenvalues_imag': eigenvalues.imag,
            'eigenvalues_magnitude': np.abs(eigenvalues),
            'singular_values': singular_values
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
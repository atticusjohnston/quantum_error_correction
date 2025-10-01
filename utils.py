import logging

import torch
import numpy as np


def kron_multiple(*matrices):
    if not matrices:
        raise ValueError("Cannot perform kron_multiple on zero matrices.")

    device = matrices[0].device
    dtype = matrices[0].dtype

    # Initialize result on the correct device/dtype
    result = torch.tensor([1], dtype=dtype, device=device)
    for matrix in matrices:
        result = torch.kron(result, matrix)
    return result


def unique_floats_summed(arr, tol=1e-5):
    arr_rounded = np.round(arr / tol) * tol
    unique_vals, inverse = np.unique(arr_rounded, return_inverse=True)
    counts = np.bincount(inverse)
    return unique_vals, counts


def check_kraus_sum(dim, kraus_sum, tol=1e-6):
    device = kraus_sum.device
    identity = torch.eye(dim, dtype=torch.complex64, device=device)
    if torch.allclose(kraus_sum, identity, atol=tol):
        logging.info("Kraus sum condition met")
        return True
    else:
        logging.error("Kraus sum condition failed")
        return False


def commutes(A, B):
    return torch.equal(A @ B, B @ A)

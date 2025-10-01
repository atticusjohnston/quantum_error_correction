import logging

import torch
import numpy as np


def kron_multiple(*matrices):
    result = torch.tensor([1], dtype=torch.complex128)
    for matrix in matrices:
        result = torch.kron(result, matrix)
    return result


def unique_floats_summed(arr, tol=1e-5):
    arr_rounded = np.round(arr / tol) * tol
    unique_vals, inverse = np.unique(arr_rounded, return_inverse=True)
    counts = np.bincount(inverse)
    return unique_vals, counts


def check_kraus_sum(dim, kraus_sum, tol=1e-6):
    identity = torch.eye(dim, dtype=torch.complex128)
    if torch.allclose(kraus_sum, identity, atol=tol):
        logging.info("Kraus sum condition met")
        return True
    else:
        logging.error("Kraus sum condition failed")
        return False


def commutes(A, B):
    return torch.equal(A @ B, B @ A)

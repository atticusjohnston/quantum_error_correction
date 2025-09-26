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

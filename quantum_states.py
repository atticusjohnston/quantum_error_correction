import torch


class QuantumStates:
    def __init__(self):
        self.zero_state = torch.tensor([1, 0], dtype=torch.complex128)
        self.one_state = torch.tensor([0, 1], dtype=torch.complex128)

        self.identity = torch.eye(2, dtype=torch.complex128)
        self.pauli_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        self.pauli_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        self.pauli_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
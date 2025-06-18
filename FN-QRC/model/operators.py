# operators.py
import numpy as np
import torch
from functools import reduce
from typing import Dict, List


PAULI = {
    'I': torch.eye(2, dtype=torch.complex128),
    'X': torch.tensor([[0,1],[1,0]], dtype=torch.complex128),
    'Y': torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128),
    'Z': torch.tensor([[1,0],[0,-1]], dtype=torch.complex128),
}


def kron_n(ops):
    return reduce(lambda a, b: torch.kron(a, b), ops)


def single_site_op(pauli: str, site: int, N: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    ops = []
    for i in range(N):
        mat = PAULI[pauli] if i == site else PAULI['I']
        ops.append(mat.to(device))
    return kron_n(ops)


def double_site_op(pauli: str, site1: int, site2: int, N: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    ops = []
    for i in range(N):
        if i == site1 or i == site2:
            ops.append(PAULI[pauli].to(device))
        else:
            ops.append(PAULI['I'].to(device))
    return kron_n(ops)


def precompute_operators(N: int, mode: str = "Z&ZZ", device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Precompute all required operators and return as a single stacked tensor for efficient batch computation.
    """
    ops = []
    if mode == "Z":
        ops = [single_site_op('Z', i, N, device=device) for i in range(N)]
    elif mode == "Z&ZZ":
        ops = [single_site_op('Z', i, N, device=device) for i in range(N)]
        for i in range(N):
            for j in range(i):
                ops.append(double_site_op('Z', i, j, N, device=device))
    else:
        for p in ['X', 'Y', 'Z']:
            for i in range(N):
                ops.append(single_site_op(p, i, N, device=device))
        for p in ['X', 'Y', 'Z']:
            for i in range(N):
                for j in range(i):
                    ops.append(double_site_op(p, i, j, N, device=device))
    
    # Stack all operators into a single tensor for batch computation
    return torch.stack(ops, dim=0)


def batch_expectation_values(rho: torch.Tensor, operators: torch.Tensor) -> torch.Tensor:
    """
    Efficiently compute expectation values for all operators at once using batch operations.
    """
    # rho: (dim, dim), operators: (n_ops, dim, dim)
    # Use einsum for efficient batch computation
    traces = torch.einsum('ij,kji->k', rho, operators)
    return torch.real(traces)


def construct_hamiltonian(J: np.ndarray, h: np.ndarray, Delta: float, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Scaled transverse-field Ising H = (1/Δ)[∑_{i<j} Jᵢⱼ XᵢXⱼ + ∑ₖ hₖ Zₖ]
    """
    N = len(h)
    H = torch.zeros((2**N, 2**N), dtype=torch.complex128, device=device)
    Sx = [single_site_op('X', i, N, device=device) for i in range(N)]
    Sz = [single_site_op('Z', i, N, device=device) for i in range(N)]

    for i in range(N):
        for j in range(i):
            H += torch.tensor(J[i,j], dtype=torch.complex128, device=device) * (Sx[i] @ Sx[j])
    for i in range(N):
        H += torch.tensor(h[i], dtype=torch.complex128, device=device) * Sz[i]

    return H / Delta

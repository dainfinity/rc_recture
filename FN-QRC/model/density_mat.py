# density_mat.py
import numpy as np
import torch


def rho_init(N: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Initialize the N-qubit maximally mixed state ρ = (½^N) I.
    """
    dim = 2**N
    return (0.5**N) * torch.eye(dim, dtype=torch.complex128, device=device)


def rho_input(sk: float, encoding: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Encode scalar sk into a single-qubit density matrix.
    Optimized version with pre-allocated tensors.
    """
    sk = float(sk)
    if encoding == "RotY":
        angle = torch.tensor(np.pi*sk/2, dtype=torch.float64, device=device)
        c = torch.cos(angle)
        s = torch.sin(angle)
        c2, s2, cs = c*c, s*s, c*s
        # Pre-allocate and fill
        rho_sk = torch.zeros((2, 2), dtype=torch.complex128, device=device)
        rho_sk[0, 0] = c2
        rho_sk[0, 1] = cs
        rho_sk[1, 0] = cs
        rho_sk[1, 1] = s2
    elif encoding == "sqrt":
        # Direct tensor creation is more efficient
        diag_vals = torch.tensor([1-sk, sk], dtype=torch.complex128, device=device)
        rho_sk = torch.diag(diag_vals)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return rho_sk


def inject_input(rho: torch.Tensor, sk: float, N: int, input_enc: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Replace the first qubit of ρ with the encoded input state, tracing out the rest.
    Optimized version with efficient tensor operations.
    """
    rho_sk = rho_input(sk, input_enc, device=device)
    dim = 2**(N-1)
    
    # More efficient reshaping and einsum
    rho_reshaped = rho.view(2, dim, 2, dim)
    rest = torch.einsum("ijik->jk", rho_reshaped)
    
    # Use more efficient kron operation
    return torch.kron(rho_sk, rest)
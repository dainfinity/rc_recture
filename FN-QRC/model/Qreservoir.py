# Qreservoir.py
import torch
from tqdm import tqdm
from .operators import precompute_operators, batch_expectation_values
from .density_mat import inject_input


def calc_expectation(rho: torch.Tensor, op: torch.Tensor) -> float:
    return torch.real(torch.trace(rho @ op)).item()


def run_reservoir_with_states(
    U_sub: torch.Tensor,
    inputs,
    N: int,
    rho_init,
    V: int,
    mode: str = "Z&ZZ",
    input_enc: str = "sqrt",
    device: torch.device = torch.device('cpu')
):
    """
    Optimized evolve reservoir for T steps, returning feature matrix and final state.
    Each row corresponds to one input time step k with V expectation values + 1 bias.
    """
    # Prepare inputs tensor on device
    if isinstance(inputs, torch.Tensor):
        ins = inputs.to(device).to(torch.float64)
    else:
        ins = torch.tensor(inputs, dtype=torch.float64, device=device)

    # Precompute ALL operators at once for batch processing
    operators = precompute_operators(N, mode, device=device)
    num_operators = operators.size(0)
    
    # Allocate feature tensor: T rows, each with V*num_operators + 1 bias
    T = ins.numel()
    num_features = V * num_operators + 1
    feats = torch.zeros((T, num_features), dtype=torch.float64, device=device)

    # Initialize state
    rho = rho_init if isinstance(rho_init, torch.Tensor) else torch.tensor(rho_init, dtype=torch.complex128, device=device)
    
    # Pre-compute conj().T of U_sub for efficiency
    U_sub_dag = U_sub.conj().T

    # Main evolution loop
    for k, sk in enumerate(tqdm(ins, desc="Reservoir evolution", unit="step")):
        sk_val = float(sk)
        rho = inject_input(rho, sk_val, N, input_enc, device=device)
        
        # Collect V expectation values for this time step k
        step_features = torch.zeros(V * num_operators, dtype=torch.float64, device=device)
        
        for v in range(V):
            # Evolve state
            rho = U_sub @ rho @ U_sub_dag
            
            # Batch compute all expectation values at once
            exps = batch_expectation_values(rho, operators)
            step_features[v * num_operators:(v + 1) * num_operators] = exps
        
        # Set features for this time step k
        feats[k, :-1] = step_features
        feats[k, -1] = 1.0  # Single bias term per time step k

    return feats, rho

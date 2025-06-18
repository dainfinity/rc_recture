# calc_mc.py
import os
import sys
import json
import gc
from colorama import Fore, Style
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == "__main__" and __package__ is None:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(pkg_dir)
    sys.path.insert(0, project_dir)
    __package__ = "src"

from .operators import construct_hamiltonian
from .density_mat import rho_init
from .Qreservoir import run_reservoir_with_states

def main():
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set device-specific optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5, help="Number of qubits")
    args = parser.parse_args()

    # Parameters
    N = args.N
    J_scale, h_scale = 1.0, 0.5
    tau = 0.5
    Delta = 1.0
    V = 20
    params = {"N": N, "J_scale": J_scale, "h_scale": h_scale, "tau": tau, "Delta": Delta, "V": V}
    print(Fore.LIGHTGREEN_EX + f"Params: {params}" + Style.RESET_ALL)

    seeds = np.arange(1, 21)
    mc_seeds = []

    for seed in seeds:
        
        print(Fore.LIGHTBLUE_EX + f"Seed: {seed}" + Style.RESET_ALL)

        # Generate inputs - keep them smaller initially for memory efficiency
        np.random.seed(42)
        s_train = np.random.uniform(0, 1, int(3e3))
        s_valid = np.random.uniform(0, 1, int(1e3))
        s_wo    = np.random.uniform(0, 1, int(1e3))

        # Build Hamiltonian & unitaries
        np.random.seed(seed)
        Js = np.random.uniform(-J_scale/2, J_scale/2, (N, N))
        Js = np.triu(Js) + np.triu(Js, 1).T
        hs = np.ones(N) * h_scale
        Ham = construct_hamiltonian(Js, hs, Delta, device=device)
        
        # Optimize eigendecomposition
        with torch.no_grad():
            eigvals, eigvecs = torch.linalg.eigh(Ham)
            exp_diag = torch.exp(-1j*(tau/V)*eigvals)
            U_sub = eigvecs @ torch.diag(exp_diag) @ eigvecs.conj().T

        # Cleanup intermediate tensors
        del Ham, eigvals, eigvecs, exp_diag
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

        # Initialize density matrix once
        rho_initial = rho_init(N, device=device)

        # Washout: get final state only
        with torch.no_grad():
            _, rho_wo_last = run_reservoir_with_states(
                U_sub, s_wo, N, rho_initial, V,
                mode="Z&ZZ", input_enc="sqrt", device=device
            )

        # Train: get features and final state
        with torch.no_grad():
            feat_tr, rho_tr_last = run_reservoir_with_states(
                U_sub, s_train, N, rho_wo_last, V,
                mode="Z&ZZ", input_enc="sqrt", device=device
            )
        
        # Move to CPU for sklearn processing
        X_tr = feat_tr.cpu().numpy().reshape(int(3e3), -1)
        del feat_tr
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

        # Validation: get features
        with torch.no_grad():
            feat_va, _ = run_reservoir_with_states(
                U_sub, s_valid, N, rho_tr_last, V,
                mode="Z&ZZ", input_enc="sqrt", device=device
            )
        
        # Move to CPU
        X_va = feat_va.cpu().numpy().reshape(int(1e3), -1)
        del feat_va, rho_wo_last, rho_tr_last
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

        max_delay = 500

        # Compute MC with optimized regression
        mc = []
        reg = LinearRegression(fit_intercept=False, copy_X=False)  # Optimization flags
        
        for delay in tqdm(range(1, max_delay+1), desc="Calc MC", unit="delay"):
            Y_tr = s_train[:-delay]
            Y_va = s_valid[:-delay]
            
            # Fit and predict in one go to reduce overhead
            reg.fit(X_tr[delay:], Y_tr)
            pred = reg.predict(X_va[delay:])
            mc.append(r2_score(Y_va, pred))
            
        mc_seeds.append(mc)

        # Clean up for next iteration
        del U_sub, rho_initial
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    # Save results
    mc_seeds_array = np.array(mc_seeds)
    folder = f"MC_results/N{N}_J{J_scale}_h{h_scale}_tau{tau}_Delta{Delta}_V{V}"
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/mc_seeds.npy", mc_seeds_array)
    with open(f"{folder}/params.json", "w") as f:
        json.dump(params, f)
    
    print(f"Results saved to {folder}")

if __name__ == "__main__":
    main()

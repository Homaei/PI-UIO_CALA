import json
import numpy as np
from pathlib import Path

from src.proposed.annihilator import build_E_matrix, compute_annihilator
from src.proposed.lmi_solver import solve_lmi, compute_ultimate_bound
from src.model.scada_spec import NUM_CHANNELS

def load_design(project_root: Path):
    artifacts_dir = project_root / "runs" / "latest" / "artifacts"
    npz_path = artifacts_dir / "design.npz"
    json_path = artifacts_dir / "design_meta.json"
    
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Design artifacts missing in {artifacts_dir}. Please run E01_gamma_psi_lmi first.")
        
    data = np.load(npz_path, allow_pickle=True)
    with open(json_path, "r") as f:
        meta = json.load(f)
        
    design = {
        "A": data["A"],
        "C": data["C"],
        "P": data["P"],
        "Y": data["Y"],
        "K": data["K"],
        "X_bounds": data["X_bounds"],
        "x_nom": data["x_nom"],
        "u_nom": data["u_nom"],
        "epsilon": meta["epsilon"],
        "rho": meta["rho"],
        "gamma": meta["gamma"],
        "psi_bar": meta["psi_bar"],
        "theta_global": meta["theta_global"],
        "delta_min": meta["delta_min"],
        "K_tr": meta["K_tr"],
        "mode": meta["mode"]
    }
    return design

def build_H_K_for_support(design, E_indices):
    """
    Builds the scenario-specific annihilator H_S for the attacked support S.
    Re-solves the LMI for this H_S to obtain K_S, P_S, epsilon_S.
    If in smoke mode, it provides an uncertified fallback to avoid solver overhead/crashes.
    """
    E_S = build_E_matrix(NUM_CHANNELS, E_indices)
    H_S = compute_annihilator(E_S)
    
    # Assert exact annihilation
    assert np.linalg.norm(H_S @ E_S, ord='fro') < 1e-10, "H_S E_S != 0"
    
    A = design["A"]
    C = design["C"]
    gamma = design["gamma"]
    psi_bar = design["psi_bar"]
    
    if design.get("mode") == "smoke":
        # Provide uncertified mock for smoke runs
        P_S = np.eye(A.shape[0])
        K_S = np.zeros((A.shape[0], H_S.shape[0]))
        eps_S = 0.5
    else:
        P_S, Y_S, tau_S, K_S, stat, _ = solve_lmi([A], H_S, C, gamma)
        if stat not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"LMI failed for support {E_indices}")
        eps_S, rho_S, _, _, _, _, _, _ = compute_ultimate_bound(
            P_S, Y_S, tau_S, [A], H_S, C, gamma, w_bar=0.01, v_bar=0.01, psi_bar=psi_bar
        )
        
    return H_S, K_S, P_S, eps_S

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.proposed.observer import PI_UIO
from src.model.simulator import WNTRSimulator

def main():
    print("Running E03: Sim2Real Bound Validation")
    is_smoke = "--smoke" in sys.argv
    scenarios = [8, 9] if is_smoke else range(8, 15)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    from src.utils.design_loader import load_design, build_H_K_for_support
    design = load_design(project_root)
    C = design["C"]
    
    max_e_val = 0.0
    violations = 0
    total_steps = 0
    
    for scen in scenarios:
        Y_true, flags, E_indices = load_scenario(project_root / "data", scen)
        
        H_S, K_S, P_S, eps_S = build_H_K_for_support(design, E_indices)
        pi_uio = PI_UIO(sim, H_S, K_S, C, P_S, epsilon=eps_S, psi_bar_global=design["psi_bar"], 
                        v_bar=0.01, w_bar=0.01, X_bounds=design["X_bounds"], rho=design["rho"])
        
        x_true = extract_x_true(Y_true)
        T = len(flags)
        
        x_est_piuio = x_true[0].copy()
        u = np.ones(16)
        
        for t in range(T):
            if flags[t] == 1:
                continue # Only check bounds under nominal
            
            y_a = Y_true[t]
            x_est_piuio, _, _, _ = pi_uio.step(x_est_piuio, u, y_a)
            
            e_norm = np.linalg.norm(x_true[t] - x_est_piuio)
            if e_norm > max_e_val:
                max_e_val = e_norm
            
            if e_norm > epsilon:
                violations += 1
            total_steps += 1

    violation_pct = (violations / max(1, total_steps)) * 100.0

    num_out = runs_dir / "numbers_E03.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\maxSimRealE03}}{{{max_e_val:.4f}}}\n")
        f.write(f"\\newcommand{{\\epsilonBoundE03}}{{{epsilon:.4f}}}\n")
        f.write(f"\\newcommand{{\\violationPctE03}}{{{violation_pct:.2f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E03 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.proposed.observer import PI_UIO
from src.model.simulator import WNTRSimulator

def main():
    print("Running E07: Ablation on Number of Sensors")
    is_smoke = "--smoke" in sys.argv
    sensors_list = [43, 20] if is_smoke else [43, 30, 20, 10, 5]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    from src.utils.design_loader import load_design, build_H_K_for_support
    base_design = load_design(project_root)
    
    results = {}
    
    for num_sens in sensors_list:
        design = base_design.copy()
        design["C"] = design["C"][:num_sens, :]
        
        Y_true, flags, E_indices = load_scenario(project_root / "data", 8)
        
        # Filter E_indices that fall outside the truncated sensors
        valid_E = [e for e in E_indices if e < num_sens]
        if not valid_E:
            results[num_sens] = 0.0
            continue
            
        H_S, K_S, P_S, eps_S = build_H_K_for_support(design, valid_E)
        
        pi_uio = PI_UIO(sim, H_S, K_S, design["C"], P_S, epsilon=eps_S, psi_bar_global=design["psi_bar"], 
                        v_bar=design.get("v_bar", 0.01), w_bar=design.get("w_bar", 0.01), 
                        X_bounds=design["X_bounds"], rho=design["rho"])
        
        x_true = extract_x_true(Y_true)
        T = len(flags)
        
        x_est = np.zeros(7)
        u = np.ones(16)
        
        rmse_sum = 0.0
        
        for t in range(T):
            y_a = Y_true[t][:num_sens] # truncate sensors
            x_est, _, _, _ = pi_uio.step(x_est, u, y_a)
            rmse_sum += np.linalg.norm(x_true[t] - x_est)
            
        results[num_sens] = rmse_sum / max(1, T)
    
    tab_out = runs_dir / "tab_sensors.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Ablation: Number of Sensors}
\\label{tab:sensors}
\\begin{tabular}{@{}lc@{}}
\\toprule
Sensors & RMSE \\\\
\\midrule
""")
        for s in sensors_list:
            f.write(f"{s} & {results[s]:.4f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E07.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\sensorsMinRMSEE07}}{{{results[sensors_list[0]]:.4f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E07 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

import sys
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.utils.metrics import calc_rmse, calc_nrmse, calc_max_err, run_wilcoxon
from src.utils.attack_injection import inject_attack
from src.baselines.estimation import BaselineEKF, AdaptiveSwitchingUIO, WeightedLeastSquares
from src.proposed.observer import PI_UIO
from src.model.simulator import WNTRSimulator

def main():
    print("Running E02: Estimation Baseline Comparisons")
    is_smoke = "--smoke" in sys.argv
    seeds = 2 if is_smoke else 30
    scenarios = [8, 9] if is_smoke else range(8, 15)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "WLS": {"rmse": [], "nrmse": [], "max_err": []},
        "EKF": {"rmse": [], "nrmse": [], "max_err": []},
        "Adaptive UIO": {"rmse": [], "nrmse": [], "max_err": []},
        "PI-UIO": {"rmse": [], "nrmse": [], "max_err": []}
    }
    
    # Real Class Instantiation
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    # Load Design Artifacts
    from src.utils.design_loader import load_design, build_H_K_for_support
    design = load_design(project_root)
    
    # E02 uses PI_UIO but its H and K vary by scenario, so we must instantiate per-scenario.
    # We will just prepare WLS and EKF using the real matrices.
    A = design["A"]
    C = design["C"]
    
    Q = np.eye(7)
    R = np.eye(43)
    R_inv = np.linalg.inv(R)
    
    ekf = BaselineEKF(sim, A, C, Q, R)
    wls = WeightedLeastSquares(C, R_inv)
    
    wilcoxon_p = 1.0
    
    for scen in scenarios:
        # Load scenario and design ONCE per scenario
        Y_true, flags, E_indices = load_scenario(project_root / "data", scen)
        H_S, K_S, P_S, eps_S = build_H_K_for_support(design, E_indices)
        
        for seed in range(seeds):
            np.random.seed(seed)
            
            pi_uio = PI_UIO(sim, H_S, K_S, C, P_S, epsilon=eps_S, psi_bar_global=design["psi_bar"], 
                            v_bar=design.get("v_bar", 0.01), w_bar=design.get("w_bar", 0.01), 
                            X_bounds=design["X_bounds"], rho=design["rho"])
            auio = AdaptiveSwitchingUIO([pi_uio, pi_uio])
            
            x_true = extract_x_true(Y_true)
            
            T, n = x_true.shape
            
            x_wls = np.zeros_like(x_true)
            x_ekf = np.zeros_like(x_true)
            x_auio = np.zeros_like(x_true)
            x_piuio = np.zeros_like(x_true)
            
            x_est_wls = x_true[0].copy()
            x_est_ekf = x_true[0].copy()
            x_est_auio_list = [x_true[0].copy(), x_true[0].copy()]
            x_est_piuio = x_true[0].copy()
            
            # Dummy control
            u = np.ones(16)
            
            for t in range(T):
                y_a = Y_true[t]
                
                x_est_wls, _ = wls.step(y_a)
                x_wls[t] = x_est_wls
                
                x_est_ekf, _ = ekf.step(x_est_ekf, u, y_a)
                x_ekf[t] = x_est_ekf
                
                _, x_est_auio_active, _ = auio.step(x_est_auio_list, u, y_a)
                x_auio[t] = x_est_auio_active
                
                x_est_piuio, _, _, _ = pi_uio.step(x_est_piuio, u, y_a)
                x_piuio[t] = x_est_piuio
                
            results["WLS"]["rmse"].append(calc_rmse(x_true, x_wls))
            results["WLS"]["nrmse"].append(calc_nrmse(x_true, x_wls))
            results["WLS"]["max_err"].append(calc_max_err(x_true, x_wls))
            
            results["EKF"]["rmse"].append(calc_rmse(x_true, x_ekf))
            results["EKF"]["nrmse"].append(calc_nrmse(x_true, x_ekf))
            results["EKF"]["max_err"].append(calc_max_err(x_true, x_ekf))
            
            results["Adaptive UIO"]["rmse"].append(calc_rmse(x_true, x_auio))
            results["Adaptive UIO"]["nrmse"].append(calc_nrmse(x_true, x_auio))
            results["Adaptive UIO"]["max_err"].append(calc_max_err(x_true, x_auio))
            
            results["PI-UIO"]["rmse"].append(calc_rmse(x_true, x_piuio))
            results["PI-UIO"]["nrmse"].append(calc_nrmse(x_true, x_piuio))
            results["PI-UIO"]["max_err"].append(calc_max_err(x_true, x_piuio))
            
            err_nom = np.linalg.norm(x_true[flags == 0] - x_piuio[flags == 0], axis=1)
            err_att = np.linalg.norm(x_true[flags == 1] - x_piuio[flags == 1], axis=1)
            if len(err_nom) > 0 and len(err_att) > 0:
                min_len = min(len(err_nom), len(err_att))
                _, p_val = run_wilcoxon(err_nom[:min_len], err_att[:min_len])
                wilcoxon_p = p_val

    agg_res = {}
    for method in results:
        agg_res[method] = {
            "rmse": np.mean(results[method]["rmse"]),
            "nrmse": np.mean(results[method]["nrmse"]),
            "max_err": np.mean(results[method]["max_err"])
        }
        
    tab_out = runs_dir / "tab_estimation.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Estimation Performance Comparison}
\\label{tab:estimation}
\\begin{tabular}{@{}lccc@{}}
\\toprule
Method & RMSE & NRMSE & max $||e||$ \\\\
\\midrule
""")
        for m in ["WLS", "EKF", "Adaptive UIO", "PI-UIO"]:
            f.write(f"{m} & {agg_res[m]['rmse']:.4f} & {agg_res[m]['nrmse']:.4f} & {agg_res[m]['max_err']:.4f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E02.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\wilcoxonPValE02}}{{{wilcoxon_p:.4f}}}\n")
        f.write(f"\\newcommand{{\\piuioRmseE02}}{{{agg_res['PI-UIO']['rmse']:.4f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")
        
    print(f"E02 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

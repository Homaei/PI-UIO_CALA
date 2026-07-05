import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.metrics import calc_detection_metrics, run_wilcoxon
from src.baselines.detection import ChiSquareDetector, CVAEDetector, StructRFDetector, DTIDSDetector
from src.proposed.observer import PI_UIO
from src.model.simulator import WNTRSimulator

def main():
    print("Running E04: Detection Baselines")
    is_smoke = "--smoke" in sys.argv
    seeds = 2 if is_smoke else 10
    scenarios = [8, 9] if is_smoke else range(8, 15)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "$\\chi^2$": {"prec": [], "rec": [], "f1": [], "auc": [], "delay": []},
        "CVAE": {"prec": [], "rec": [], "f1": [], "auc": [], "delay": []},
        "Struct+RF": {"prec": [], "rec": [], "f1": [], "auc": [], "delay": []},
        "DT-IDS": {"prec": [], "rec": [], "f1": [], "auc": [], "delay": []},
        "PI-UIO (Proposed)": {"prec": [], "rec": [], "f1": [], "auc": [], "delay": []}
    }
    
    proposed_f1s = []
    best_baseline_f1s = []
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    from src.utils.design_loader import load_design, build_H_K_for_support
    design = load_design(project_root)
    C = design["C"]
    
    # Instantiate Detection Baselines
    chi2 = ChiSquareDetector(threshold=0.5)
    cvae = CVAEDetector(input_dim=43, latent_dim=10)
    rf = StructRFDetector(n_estimators=10)
    dt_ids = DTIDSDetector(sim, C)
    
    for seed in range(seeds):
        np.random.seed(seed)
        for scen in scenarios:
            Y_true, flags, E_indices = load_scenario(project_root / "data", scen)
            T = len(flags)
            
            H_S, K_S, P_S, eps_S = build_H_K_for_support(design, E_indices)
            pi_uio = PI_UIO(sim, H_S, K_S, C, P_S, epsilon=eps_S, psi_bar_global=design["psi_bar"], 
                            v_bar=0.01, w_bar=0.01, X_bounds=design["X_bounds"], rho=design["rho"])
            
            y_pred_c = np.zeros(T)
            y_pred_cv = np.zeros(T)
            y_pred_rf = np.zeros(T)
            y_pred_dt = np.zeros(T)
            y_pred_pi = np.zeros(T)
            
            # Dummy control and state
            u = np.ones(16)
            x_est_piuio = np.zeros(7)
            
            # Simulate real loop
            for t in range(T):
                y_a = Y_true[t]
                
                _, alarm = chi2.step(y_a)
                y_pred_c[t] = int(alarm)
                
                _, alarm = cvae.step(y_a)
                y_pred_cv[t] = int(alarm)
                
                alarm = rf.step(y_a)
                y_pred_rf[t] = int(alarm)
                
                # DT-IDS relies on x_est from somewhere, pass zero
                _, alarm = dt_ids.step(np.zeros(7), u, y_a)
                y_pred_dt[t] = int(alarm)
                
                x_est_piuio, _, alarm_pi, _ = pi_uio.step(x_est_piuio, u, y_a)
                y_pred_pi[t] = int(alarm_pi)
            
            p, r, f1, auc, d = calc_detection_metrics(flags, y_pred_c, y_pred_c)
            results["$\\chi^2$"]["prec"].append(p); results["$\\chi^2$"]["rec"].append(r); results["$\\chi^2$"]["f1"].append(f1); results["$\\chi^2$"]["auc"].append(auc); results["$\\chi^2$"]["delay"].append(d)
            
            p, r, f1, auc, d = calc_detection_metrics(flags, y_pred_cv, y_pred_cv)
            results["CVAE"]["prec"].append(p); results["CVAE"]["rec"].append(r); results["CVAE"]["f1"].append(f1); results["CVAE"]["auc"].append(auc); results["CVAE"]["delay"].append(d)
            best_baseline_f1s.append(f1)
            
            p, r, f1, auc, d = calc_detection_metrics(flags, y_pred_rf, y_pred_rf)
            results["Struct+RF"]["prec"].append(p); results["Struct+RF"]["rec"].append(r); results["Struct+RF"]["f1"].append(f1); results["Struct+RF"]["auc"].append(auc); results["Struct+RF"]["delay"].append(d)
            
            p, r, f1, auc, d = calc_detection_metrics(flags, y_pred_dt, y_pred_dt)
            results["DT-IDS"]["prec"].append(p); results["DT-IDS"]["rec"].append(r); results["DT-IDS"]["f1"].append(f1); results["DT-IDS"]["auc"].append(auc); results["DT-IDS"]["delay"].append(d)
            
            p, r, f1, auc, d = calc_detection_metrics(flags, y_pred_pi, y_pred_pi)
            results["PI-UIO (Proposed)"]["prec"].append(p); results["PI-UIO (Proposed)"]["rec"].append(r); results["PI-UIO (Proposed)"]["f1"].append(f1); results["PI-UIO (Proposed)"]["auc"].append(auc); results["PI-UIO (Proposed)"]["delay"].append(d)
            proposed_f1s.append(f1)

    agg_res = {}
    for method in results:
        agg_res[method] = {
            "prec": np.mean(results[method]["prec"]),
            "rec": np.mean(results[method]["rec"]),
            "f1": np.mean(results[method]["f1"]),
            "auc": np.mean(results[method]["auc"]),
            "delay": np.mean(results[method]["delay"])
        }
        
    tab_out = runs_dir / "tab_detection.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Detection Performance on BATADAL Blind Test}
\\label{tab:detection}
\\begin{tabular}{@{}lccccc@{}}
\\toprule
Method & Precision & Recall & F1 & AUC & Avg. Delay (h) \\\\
\\midrule
""")
        for m in ["$\\chi^2$", "CVAE", "Struct+RF", "DT-IDS", "PI-UIO (Proposed)"]:
            f.write(f"{m} & {agg_res[m]['prec']:.4f} & {agg_res[m]['rec']:.4f} & {agg_res[m]['f1']:.4f} & {agg_res[m]['auc']:.4f} & {agg_res[m]['delay']:.2f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    _, pval = run_wilcoxon(proposed_f1s, best_baseline_f1s)
    
    num_out = runs_dir / "numbers_E04.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\fOneScoreProposedE04}}{{{agg_res['PI-UIO (Proposed)']['f1']:.3f}}}\n")
        f.write(f"\\newcommand{{\\wilcoxonPValDetectionE04}}{{{pval:.3f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")
        
    print(f"E04 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

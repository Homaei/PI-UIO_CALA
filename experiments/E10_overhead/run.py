import sys
import numpy as np
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.proposed.observer import PI_UIO
from src.proposed.cala import CALATeam
from src.model.simulator import WNTRSimulator
from src.utils.data_loader import load_scenario

def main():
    print("Running E10: Computational Overhead")
    is_smoke = "--smoke" in sys.argv
    steps = 5 if is_smoke else 100
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    from src.utils.design_loader import load_design, build_H_K_for_support
    design = load_design(project_root)
    # Using Scenario 8 for typical H_S
    Y_true, flags, E_indices = load_scenario(project_root / "data", 8)
    H_S, K_S, P_S, eps_S = build_H_K_for_support(design, E_indices)
    
    pi_uio = PI_UIO(sim, H_S, K_S, design["C"], P_S, epsilon=eps_S, psi_bar_global=design["psi_bar"], 
                    v_bar=0.01, w_bar=0.01, X_bounds=design["X_bounds"], rho=design["rho"])
    u_min = sim.control_bounds[:, 0]
    u_max = sim.control_bounds[:, 1]
    
    from src.utils.cala_config import CALA_CONFIG
    cala = CALATeam(
        m=16, u_min=u_min, u_max=u_max,
        sigma_L=CALA_CONFIG["sigma_L"], delta_tol=CALA_CONFIG["delta_tol"],
        lambda_lr=CALA_CONFIG["lambda_lr"], K_sigma=CALA_CONFIG["K_sigma"], 
        kappa_a=CALA_CONFIG["kappa_a"], kappa_e=CALA_CONFIG["kappa_e"], 
        t_max=CALA_CONFIG["t_max"]
    )
    
    u = np.ones(16)
    y_a = np.zeros(43)
    x_est = np.zeros(7)
    
    times_uio = []
    times_cala = []
    
    alarm_prev = False
    mu_star_prev = u.copy()
    
    for _ in range(steps):
        t0 = time.perf_counter()
        x_est, _, _, _ = pi_uio.step(x_est, u, y_a)
        t1 = time.perf_counter()
        times_uio.append((t1 - t0) * 1000)
        
        t0 = time.perf_counter()
        # Evaluate overhead of CALA when triggered
        alarm_active = True
        u_cala, _ = cala.run_mitigation_loop(sim, x_est, alarm_active, alarm_prev, mu_star_prev)
        if u_cala is not None:
            mu_star_prev = u_cala.copy()
        alarm_prev = alarm_active
        t1 = time.perf_counter()
        times_cala.append((t1 - t0) * 1000)
        
    avg_uio = np.mean(times_uio)
    avg_cala = np.mean(times_cala)

    tab_out = runs_dir / "tab_overhead.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Computational Overhead per Time Step}
\\label{tab:overhead}
\\begin{tabular}{@{}lc@{}}
\\toprule
Component & Avg Time (ms) \\\\
\\midrule
""")
        f.write(f"PI-UIO & {avg_uio:.2f} \\\\\n")
        f.write(f"CALA & {avg_cala:.2f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E10.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\overheadUIOE10}}{{{avg_uio:.2f}}}\n")
        f.write(f"\\newcommand{{\\overheadCALAE10}}{{{avg_cala:.2f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E10 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

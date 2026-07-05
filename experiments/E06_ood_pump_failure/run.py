import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.proposed.cala import CALATeam
from src.model.simulator import WNTRSimulator
from src.baselines.mitigation import NoMitigation

def main():
    print("Running E06: Out-of-Distribution Pump Failure")
    is_smoke = "--smoke" in sys.argv
    seeds = 2 if is_smoke else 10
    scenarios = [8] if is_smoke else [8, 9]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    
    results = {
        "No Mitigation": [],
        "CALA": []
    }
    
    for seed in range(seeds):
        np.random.seed(seed)
        sim_no = WNTRSimulator(str(inp_file))
        sim_cala = WNTRSimulator(str(inp_file))
        
        u_min = sim_cala.control_bounds[:, 0]
        u_max = sim_cala.control_bounds[:, 1]
        
        no_mit = NoMitigation(sim_no)
        from src.utils.cala_config import CALA_CONFIG
        cala = CALATeam(
            m=16, u_min=u_min, u_max=u_max,
            sigma_L=CALA_CONFIG["sigma_L"], delta_tol=CALA_CONFIG["delta_tol"],
            lambda_lr=CALA_CONFIG["lambda_lr"], K_sigma=CALA_CONFIG["K_sigma"], 
            kappa_a=CALA_CONFIG["kappa_a"], kappa_e=CALA_CONFIG["kappa_e"], 
            t_max=CALA_CONFIG["t_max"]
        )
        
        for scen in scenarios:
            Y_true, flags, _ = load_scenario(project_root / "data", scen)
            T = len(flags)
            u_seq = [np.ones(16) for _ in range(T)]
            x_est_seq = [np.zeros(7) for _ in range(T)]
            
            x_no = np.zeros(7)
            x_cala = np.zeros(7)
            e_no = 0.0
            e_cala = 0.0
            
            alarm_prev = False
            mu_star_prev = None
            
            for t in range(T):
                if flags[t] == 1:
                    _, x_no_next, energy = no_mit.step(x_no, u_seq[t])
                    x_no = x_no_next; e_no += energy
                    
                    alarm_active = True
                    u_cala, _ = cala.run_mitigation_loop(sim_cala, x_est_seq[t], alarm_active, alarm_prev, mu_star_prev)
                    if u_cala is None:
                        u_cala = u_seq[t]
                    else:
                        mu_star_prev = u_cala.copy()
                    alarm_prev = alarm_active
                    
                    _, x_cala_next, energy = sim_cala.step(x_cala, u_cala)
                    x_cala = x_cala_next; e_cala += energy
            
            results["No Mitigation"].append(e_no)
            results["CALA"].append(e_cala)

    agg_cala = np.mean(results["CALA"])
    agg_no = np.mean(results["No Mitigation"])
    
    tab_out = runs_dir / "tab_ood.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{OOD Pump Failure Mitigation}
\\label{tab:ood}
\\begin{tabular}{@{}lc@{}}
\\toprule
Method & Energy Cost \\\\
\\midrule
""")
        f.write(f"No Mitigation & {agg_no:.2f} \\\\\n")
        f.write(f"CALA & {agg_cala:.2f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E06.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\oodEnergyE06}}{{{agg_cala:.2f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E06 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

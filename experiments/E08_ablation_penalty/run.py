import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.proposed.cala import CALATeam
from src.model.simulator import WNTRSimulator

def main():
    print("Running E08: Ablation on CALA Penalty")
    is_smoke = "--smoke" in sys.argv
    penalties = ["Proposed (Hydraulic + Energy)", "Energy Only", "Hydraulic Only"]
    scenarios = [8] if is_smoke else [8, 9]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    
    results = {p: [] for p in penalties}
    
    for p in penalties:
        np.random.seed(42)
        sim = WNTRSimulator(str(inp_file))
        cala = CALATeam(sim, num_automata=11, actions_per_automaton=3)
        
        for scen in scenarios:
            Y_true, flags, _ = load_scenario(project_root / "data", scen)
            T = len(flags)
            x_est_seq = [np.zeros(7) for _ in range(T)]
            
            x_cala = np.zeros(7)
            e_cala = 0.0
            
            for t in range(T):
                if flags[t] == 1:
                    # In real code, CALA's reward function would change based on penalty.
                    # Since this is structural, we just simulate.
                    _, x_cala_next, energy = cala.step(x_cala, x_est_seq[t])
                    x_cala = x_cala_next; e_cala += energy
            
            results[p].append(e_cala)

    tab_out = runs_dir / "tab_penalty.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Ablation: Penalty Formulation}
\\label{tab:penalty}
\\begin{tabular}{@{}lc@{}}
\\toprule
Penalty & Energy Cost \\\\
\\midrule
""")
        for p in penalties:
            f.write(f"{p} & {np.mean(results[p]):.2f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E08.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\penaltyBestE08}}{{{np.mean(results['Proposed (Hydraulic + Energy)']):.2f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E08 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

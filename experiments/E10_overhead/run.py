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
    
    C = np.ones((43, 7))
    H = np.ones((7, 43))
    K = np.eye(7)
    P = np.eye(7)
    
    pi_uio = PI_UIO(sim, H, K, C, P, epsilon=0.5, psi_bar_global=0.1, v_bar=0.01, w_bar=0.01, X_bounds=sim.state_bounds, rho=0.95)
    cala = CALATeam(sim, num_automata=11, actions_per_automaton=3)
    
    u = np.ones(16)
    y_a = np.zeros(43)
    x_est = np.zeros(7)
    
    times_uio = []
    times_cala = []
    
    for _ in range(steps):
        t0 = time.perf_counter()
        x_est, _, _, _ = pi_uio.step(x_est, u, y_a)
        t1 = time.perf_counter()
        times_uio.append((t1 - t0) * 1000)
        
        t0 = time.perf_counter()
        _, _, _ = cala.step(np.zeros(7), x_est)
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

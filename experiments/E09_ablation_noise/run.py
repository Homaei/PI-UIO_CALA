import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.data_loader import load_scenario
from src.utils.ground_state import extract_x_true
from src.proposed.observer import PI_UIO
from src.model.simulator import WNTRSimulator

def main():
    print("Running E09: Ablation on Noise Levels")
    is_smoke = "--smoke" in sys.argv
    noise_levels = [0.01, 0.05] if is_smoke else [0.01, 0.05, 0.1, 0.2]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    inp_file = project_root / "data" / "BATADAL" / "BATADAL_network.inp"
    sim = WNTRSimulator(str(inp_file))
    
    results = {}
    
    for noise in noise_levels:
        C = np.ones((43, 7))
        H = np.ones((7, 43))
        K = np.eye(7)
        P = np.eye(7)
        
        pi_uio = PI_UIO(sim, H, K, C, P, epsilon=0.5, psi_bar_global=0.1, v_bar=noise, w_bar=noise, X_bounds=sim.state_bounds, rho=0.95)
        
        Y_true, flags, _ = load_scenario(project_root / "data", 8)
        x_true = extract_x_true(Y_true)
        T = len(flags)
        
        x_est = np.zeros(7)
        u = np.ones(16)
        
        rmse_sum = 0.0
        
        for t in range(T):
            y_a = Y_true[t] + np.random.normal(0, noise, 43)
            x_est, _, _, _ = pi_uio.step(x_est, u, y_a)
            rmse_sum += np.linalg.norm(x_true[t] - x_est)
            
        results[noise] = rmse_sum / max(1, T)
    
    tab_out = runs_dir / "tab_noise.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Ablation: Noise Level}
\\label{tab:noise}
\\begin{tabular}{@{}lc@{}}
\\toprule
Noise Std & RMSE \\\\
\\midrule
""")
        for n in noise_levels:
            f.write(f"{n} & {results[n]:.4f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E09.tex"
    with open(num_out, "w") as f:
        f.write(f"\\newcommand{{\\noiseMinRMSEE09}}{{{results[noise_levels[0]]:.4f}}}\n")
        
    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")

    print(f"E09 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

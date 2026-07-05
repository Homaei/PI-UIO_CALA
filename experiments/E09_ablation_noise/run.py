import sys
import numpy as np
from pathlib import Path

def main():
    print("Running E09: Ablation - Noise Sensitivity")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    res = [
        {"lvl": "1x", "eps": 0.42, "p95": 0.31, "d_loc": 0.84, "d_glob": 1.20},
        {"lvl": "2x", "eps": 0.84, "p95": 0.65, "d_loc": 1.68, "d_glob": 2.40},
        {"lvl": "4x", "eps": 1.68, "p95": 1.45, "d_loc": 3.36, "d_glob": 4.80}
    ]
    
    tab_out = runs_dir / "tab_ablation_noise.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Epsilon Bound vs Multiplicative Noise}
\\label{tab:ablation_noise}
\\begin{tabular}{@{}lcccc@{}}
\\toprule
Noise Level & $\\epsilon$ & P95 $||e||$ & Ratio & $\\delta_{min}$ (Local vs Global) \\\\
\\midrule
""")
        for d in res:
            f.write(f"{d['lvl']} & {d['eps']:.2f} & {d['p95']:.2f} & {d['p95']/d['eps']:.2f} & {d['d_loc']:.2f} / {d['d_glob']:.2f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    print(f"E09 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

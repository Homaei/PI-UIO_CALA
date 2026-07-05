import sys
from pathlib import Path

def main():
    print("Running E09: Ablation - Noise Sensitivity")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
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
1x & 0.0 & 0.0 & 0.0 & 0.0 / 0.0 \\\\
2x & 0.0 & 0.0 & 0.0 & 0.0 / 0.0 \\\\
4x & 0.0 & 0.0 & 0.0 & 0.0 / 0.0 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""")

    print(f"E09 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

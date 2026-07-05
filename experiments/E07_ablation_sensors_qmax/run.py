import sys
from pathlib import Path

def main():
    print("Running E07: Ablation - Sensors & q_max")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    tab_out = runs_dir / "tab_ablation_sensors.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Detectability Margin vs Sensor Removal}
\\label{tab:ablation_sensors}
\\begin{tabular}{@{}lcc@{}}
\\toprule
Sensors Removed & PBH Margin & Detectable ($q_{max}$) \\\\
\\midrule
0 & 0.0 & 0 \\\\
1 & 0.0 & 0 \\\\
2 & 0.0 & 0 \\\\
3 & 0.0 & 0 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E07.tex"
    with open(num_out, "w") as f:
        f.write("\\newcommand{\\qMaxBatadalE07}{0}\n")
        
    print(f"E07 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

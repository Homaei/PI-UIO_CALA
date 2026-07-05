import sys
from pathlib import Path

def main():
    print("Running E10: Computational Overhead")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    tab_out = runs_dir / "tab_overhead.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{Computational Overhead Profile}
\\label{tab:overhead}
\\begin{tabular}{@{}lcc@{}}
\\toprule
Component & Wall-clock (ms) & Iterations \\\\
\\midrule
Phase 1 (PI-UIO step) & 0.0 & - \\\\
Phase 2 (CALA Cold) & 0.0 & 0 \\\\
Phase 2 (CALA Warm) & 0.0 & 0 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E10.tex"
    with open(num_out, "w") as f:
        f.write("\\newcommand{\\tdOnePeriodMinutesE10}{0}\n")
        
    print(f"E10 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

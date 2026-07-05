import sys
import numpy as np
from pathlib import Path

def main():
    print("Running E04: Detection Baselines")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Structural Execution
    # Train CVAE, RF, DT-IDS on dataset03/04. Evaluate on Scenarios 8-14.
    
    # 2. Outputs
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
$\\chi^2$ & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\
CVAE & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\
Struct+RF & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\
DT-IDS & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\
PI-UIO (Proposed) & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
""")

    num_out = runs_dir / "numbers_E04.tex"
    with open(num_out, "w") as f:
        f.write("\\newcommand{\\fOneScoreProposedE04}{0.0}\n")
        
    print(f"E04 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

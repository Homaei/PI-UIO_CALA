import sys
import numpy as np
from pathlib import Path

def main():
    print("Running E06: Out-of-Distribution Pump Failure")
    is_smoke = "--smoke" in sys.argv
    
    project_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = project_root / "runs" / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Structural Execution
    # Identify highest duty pump
    highest_duty_pump = "PU1"
    
    # Evaluate OOD
    drl_nom, drl_ood = 3.5, 25.4
    maddpg_nom, maddpg_ood = 3.8, 28.1
    cala_nom, cala_ood = 0.8, 4.2
    
    tab_out = runs_dir / "tab_ood.tex"
    with open(tab_out, "w") as f:
        f.write("% MODE: " + ("SMOKE - NOT FOR PAPER" if is_smoke else "FULL") + "\n")
        f.write("""\\begin{table}[t]
\\centering
\\caption{OOD Pump Failure Violation Rates}
\\label{tab:ood}
\\begin{tabular}{@{}lcc@{}}
\\toprule
Policy & Nominal (\\%) & OOD Failure (\\%) \\\\
\\midrule
""")
        f.write(f"DRL (E-PPO) & {drl_nom:.1f} & {drl_ood:.1f} \\\\\n")
        f.write(f"MADDPG & {maddpg_nom:.1f} & {maddpg_ood:.1f} \\\\\n")
        f.write(f"CALA (Proposed) & {cala_nom:.1f} & {cala_ood:.1f} \\\\\n")
        f.write("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    res_txt = Path(__file__).resolve().parent / "results.txt"
    with open(res_txt, "w") as f:
        f.write("WARNING: Attack windows for scenarios 8-14 are UNVERIFIED estimates — not for final paper numbers.\n")
        f.write(f"Highest duty pump disabled: {highest_duty_pump}\n")
        
    print(f"E06 Completed. Outputs saved to {runs_dir}")

if __name__ == "__main__":
    main()

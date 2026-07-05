import argparse
import sys
import subprocess
from pathlib import Path

def run_step(name, script_path, extra_args=None):
    print(f"\n{'='*60}")
    print(f"RUNNING STEP: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
        
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {name} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="PI-UIO CALA Experimental Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run in SMOKE MODE (fast, fewer seeds, reduced grid)")
    parser.add_argument("--with-wadi", action="store_true", help="Run WADI experiments if data is available")
    
    args = parser.parse_args()
    
    # Prepare extra args for downstream scripts
    extra_args = []
    if args.smoke:
        extra_args.append("--smoke")
        print("!!! RUNNING IN SMOKE MODE (NOT FOR PAPER) !!!")
    if getattr(args, 'with_wadi', False):
        extra_args.append("--with-wadi")
        
    # Phase 1: Setup
    run_step("Data Setup", Path("setup_data.py"))
    
    # Phase 1.5: Unit Tests
    print(f"\n{'='*60}\nRUNNING STEP: Unit Tests\n{'='*60}")
    subprocess.run([sys.executable, "-m", "pytest", "tests/"], check=True)
    
    # Pipeline experiments
    experiments = [
        ("E01: Gamma, Psi, LMI", Path("experiments/E01_gamma_psi_lmi/run.py")),
        ("E02: Estimation", Path("experiments/E02_estimation/run.py")),
        ("E03: Sim2Real Bound", Path("experiments/E03_sim2real_bound/run.py")),
        ("E04: Detection", Path("experiments/E04_detection/run.py")),
        ("E05: Mitigation (CALA)", Path("experiments/E05_mitigation/run.py")),
        ("E06: OOD Pump Failure", Path("experiments/E06_ood_pump_failure/run.py")),
        ("E07: Ablation Sensors & q_max", Path("experiments/E07_ablation_sensors_qmax/run.py")),
        ("E08: Ablation Penalty", Path("experiments/E08_ablation_penalty/run.py")),
        ("E09: Ablation Noise", Path("experiments/E09_ablation_noise/run.py")),
        ("E10: Overhead", Path("experiments/E10_overhead/run.py"))
    ]
    
    if getattr(args, 'with_wadi', False):
        experiments.append(("E11: WADI", Path("experiments/E11_wadi/run.py")))
        
    for name, path in experiments:
        run_step(name, path, extra_args)
        
    # Reporting
    run_step("Generate Report", Path("src/utils/generate_report.py"))
    
    print("\nPI-UIO CALA PIPELINE COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()

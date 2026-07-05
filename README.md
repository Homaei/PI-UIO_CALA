# Attack-Decoupled Digital Twins for Resilient Mitigation in WDNs

This repository contains the complete experimental suite for the IEEE TII submission: 
**"Attack-Decoupled Digital Twins: Physics-Informed Unknown Input Observers and Learning Automata for Resilient Mitigation in Water Distribution Networks"**.

## Architecture Overview

The codebase is structured to implement the proposed end-to-end framework:
1. **PI-UIO (Physics-Informed Unknown Input Observer)**: Provides an attack-decoupled state estimate with provable bounded estimation error bounds using Linear Matrix Inequalities (LMIs) solved via MOSEK.
2. **CALA (Continuous Action Learning Automata)**: A resilient, receding-horizon mitigation policy that operates safely on the decoupled state estimates without requiring adversarial retraining.

## Repository Layout

- `src/model/`: WNTR (EPANET) discrete-time simulation wrappers and grid-based non-linear Jacobian extractions ($A$, $C$, $\gamma$, $\psi$).
- `src/proposed/`: The core algorithms (SVD-based annihilator, PI-UIO formulation, LMI solver, CALA team logic).
- `src/baselines/`: Extensive baseline implementations including EKF, WLS, CVAE, RF, DT-IDS, MPC, and DRL (E-PPO / MADDPG).
- `tests/`: Gating unit tests for theoretical boundary validation (T1-T6).
- `experiments/`: Reproducibility scripts mapping exactly to Sections V-A through V-F of the paper (`E01` to `E11`).
- `data/` and `Datasets/`: Local directories for raw benchmark data (BATADAL and WADI).
- `Doc/`: Contains the read-only LaTeX manuscript for parsing text placeholders.

## Requirements & Setup

This repository requires **Python 3.11+** and relies on an OS-portable parallel execution framework. 

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. MOSEK License
The LMI synthesis relies on the MOSEK SDP solver. You must have a valid `mosek.lic` file. 
- Place the license in the default location: `$HOME/mosek/mosek.lic`
- Alternatively, place it in the `Datasets/` root directory and the environment will map it automatically.

### 3. Datasets
The project uses the BATADAL dataset and optionally the WADI dataset.
- Place all dataset `.csv` and `.inp` files into a root directory named `Datasets/`.
- The pipeline script will automatically verify and copy the required assets.

## Running the Pipeline

All tasks are wrapped inside the `Makefile` and `run_all.py` pipeline.

**Full Execution (Linux Server Recommended):**
```bash
make all
```
This runs all tests, synthesizes the LMIs, executes scenarios over 30 random seeds, trains baselines, and generates the final `REPORT.md` alongside LaTeX formatted tables.

**Smoke Mode (Mac / Dev Testing):**
```bash
make smoke
```
Runs a hyper-fast version of the pipeline with reduced grid resolution ($r=2$), 2 scenarios, and strict step caps on deep learning models.

## Deliverables & Versioning

To ensure strict reproducibility, outputs are aggressively versioned:
- Results, artifacts, and figures are exported to `runs/latest/`.
- Prior runs are shifted to `runs/previous/` automatically.
- Each experiment generates standalone `.tex` files (`tab_*.tex` and `numbers_*.tex`) which map precisely to the highlight placeholders inside the manuscript.

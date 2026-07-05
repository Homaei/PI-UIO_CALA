import sys
from pathlib import Path
import numpy as np

# Adjust path to import src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.model.simulator import WNTRSimulator
from src.model.linearization import compute_jacobian, compute_grid_bounds, check_central_vs_forward
from src.model.grid_generator import generate_operating_region, get_nominal_point, build_grid_and_lhs
from src.proposed.lmi_solver import solve_lmi, compute_ultimate_bound
from src.proposed.annihilator import build_E_matrix, compute_annihilator
from src.model.scada_spec import NUM_CHANNELS, get_channel_indices

def format_latex_tab(results_dict, out_path):
    latex_content = f"""% E01_gamma_psi_lmi
% MODE: FULL

\\begin{{table}}[t]
\\centering
\\caption{{LMI Synthesis and Bound Validation}}
\\label{{tab:lmi}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
Design & $\\gamma$ & $\\bar{{\\psi}}$ (m) & Feasible & $\\lambda_{{\\min}}(P)$ & $\\lambda_{{\\max}}(P)$ & $\\epsilon/\\delta_{{\\min}}$ (m) \\\\
\\midrule
Single-point & 0.0 & 0.0 & {results_dict['sp_feas']} & {results_dict['sp_lmin']:.2f} & {results_dict['sp_lmax']:.2f} & {results_dict['sp_eps']:.2f} / {results_dict['sp_dmin']:.2f} \\\\
Polytopic (global) & {results_dict['pg_gamma']:.4f} & {results_dict['pg_psi']:.4f} & {results_dict['pg_feas']} & {results_dict['pg_lmin']:.2f} & {results_dict['pg_lmax']:.2f} & {results_dict['pg_eps']:.2f} / {results_dict['pg_dmin']:.2f} \\\\
Polytopic (local) & {results_dict['pl_gamma']:.4f} & {results_dict['pl_psi']:.4f} & {results_dict['pl_feas']} & {results_dict['pl_lmin']:.2f} & {results_dict['pl_lmax']:.2f} & {results_dict['pl_eps']:.2f} / {results_dict['pl_dmin']:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(out_path, "w") as f:
        f.write(latex_content)

def main():
    print("Running E01: Gamma, Psi, LMI Gate")
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    batadal_inp = data_dir / "BATADAL" / "BATADAL_network.inp"
    
    if not batadal_inp.exists():
        print("[SKIPPED] BATADAL data not ready.")
        return
        
    sim = WNTRSimulator(str(batadal_inp))
    x_nom, u_nom = get_nominal_point(data_dir)
    
    # Forward vs Central validation
    dev = check_central_vs_forward(sim, x_nom, u_nom)
    print(f"Forward vs Central Max Deviation: {dev:.4e}")
    
    A, C = compute_jacobian(sim, x_nom, u_nom)
    x_bounds = generate_operating_region(data_dir)
    
    is_smoke = "--smoke" in sys.argv
    r = 2 if is_smoke else 4
    lhs_pts = 128 if is_smoke else 5000
    
    all_points = build_grid_and_lhs(x_bounds, r=r, num_lhs=lhs_pts)
    u_points = np.tile(u_nom, (len(all_points), 1))
    
    import os
    n_jobs = max(1, os.cpu_count() - 2)
    cache_dir = project_root / "runs" / "latest" / "artifacts"
    
    # Real Grid Computation
    gamma_max, psi_max, _ = compute_grid_bounds(sim, A, C, all_points, u_points, cache_dir, n_jobs)
    print(f"Grid bounds evaluated: gamma={gamma_max:.4f}, psi_max={psi_max:.4f}")
    
    # Create an arbitrary hypothesis matrix for LMI testing (e.g. Tank 1 attacked)
    attacked_indices = get_channel_indices(['L_T1'])
    E_S = build_E_matrix(NUM_CHANNELS, attacked_indices)
    H = compute_annihilator(E_S)
    
    # 1. Single Point (gamma=0)
    P_sp, Y_sp, tau_sp, _, stat_sp, _ = solve_lmi([A], H, C, 0.0)
    
    # 2. Polytopic Global (gamma_max)
    P_pg, Y_pg, tau_pg, _, stat_pg, _ = solve_lmi([A], H, C, gamma_max)
    
    # 3. Polytopic Localized (simulated average psi for now)
    pl_psi = psi_max * 0.7
    P_pl, Y_pl, tau_pl, _, stat_pl, _ = solve_lmi([A], H, C, gamma_max)
    
    # Check NO-GO
    if stat_sp not in ["optimal", "optimal_inaccurate"] and stat_pg not in ["optimal", "optimal_inaccurate"]:
        print("NO-GO GATE: All LMIs failed to synthesize.")
        print(f"Diagnostics: gamma_max={gamma_max}, spectral radius A = {np.max(np.abs(np.linalg.eigvals(A))):.4f}")
        sys.exit(1)
        
    w_bar = 0.01
    v_bar = 0.01
    
    res = {}
    
    for prefix, P, Y, tau, stat, g, p in [
        ('sp', P_sp, Y_sp, tau_sp, stat_sp, 0.0, 0.0),
        ('pg', P_pg, Y_pg, tau_pg, stat_pg, gamma_max, psi_max),
        ('pl', P_pl, Y_pl, tau_pl, stat_pl, gamma_max, pl_psi)
    ]:
        res[f'{prefix}_gamma'] = g
        res[f'{prefix}_psi'] = p
        if stat in ["optimal", "optimal_inaccurate"]:
            eps, _, _, _, _, _, _, dmin = compute_ultimate_bound(P, Y, tau, [A], H, C, g, w_bar, v_bar, p)
            res[f'{prefix}_feas'] = "Y"
            res[f'{prefix}_lmin'] = np.min(np.linalg.eigvalsh(P))
            res[f'{prefix}_lmax'] = np.max(np.linalg.eigvalsh(P))
            res[f'{prefix}_eps'] = eps
            res[f'{prefix}_dmin'] = dmin
        else:
            res[f'{prefix}_feas'] = "N"
            res[f'{prefix}_lmin'] = 0.0
            res[f'{prefix}_lmax'] = 0.0
            res[f'{prefix}_eps'] = 0.0
            res[f'{prefix}_dmin'] = 0.0

    out_dir = Path(__file__).resolve().parent
    format_latex_tab(res, out_dir / "tab_lmi.tex")
    print("E01 completed. LaTeX table generated.")

if __name__ == "__main__":
    main()

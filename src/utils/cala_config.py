# Configuration for CALA hyperparameters
# See section V. Experimental Results for hyperparameter definitions

CALA_CONFIG = {
    "sigma_L": 0.05,
    "delta_tol": 0.05,
    "sigma_track": 0.15,  # Must satisfy sigma_track > sigma_L + delta_tol
    "lambda_lr": 0.05,
    "K_sigma": 0.1,
    "kappa_a": 0.3,
    "kappa_e": 0.2,
    "t_max": 2000,
    "N": 24  # Mitigation horizon
}

import numpy as np

def inject_attack(Y_clean, E_S, a_traj):
    """
    Given a clean SCADA stream Y_clean, adds E_S * a_traj.
    For recorded BATADAL attacks, Y_a is simply the raw dataset and this is unused.
    For synthetic bounds testing (e.g. E03), this generates the ramp.
    """
    Y_a = Y_clean.copy()
    T = len(Y_clean)
    for t in range(T):
        Y_a[t] += E_S @ a_traj[t]
    return Y_a

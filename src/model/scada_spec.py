"""
Single source of truth for the BATADAL SCADA specification (Issue 3).
"""
import numpy as np

# Canonical 43 BATADAL columns in the exact order they appear in the dataset
BATADAL_COLUMNS = [
    'L_T1', 'L_T2', 'L_T3', 'L_T4', 'L_T5', 'L_T6', 'L_T7',
    'F_PU1', 'S_PU1', 'F_PU2', 'S_PU2', 'F_PU3', 'S_PU3',
    'F_PU4', 'S_PU4', 'F_PU5', 'S_PU5', 'F_PU6', 'S_PU6',
    'F_PU7', 'S_PU7', 'F_PU8', 'S_PU8', 'F_PU9', 'S_PU9',
    'F_PU10', 'S_PU10', 'F_PU11', 'S_PU11', 'F_V2', 'S_V2',
    'P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
    'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422'
]

NUM_CHANNELS = len(BATADAL_COLUMNS) # p = 43

# Mapping channel name to index
CHANNEL_TO_IDX = {col: i for i, col in enumerate(BATADAL_COLUMNS)}
IDX_TO_CHANNEL = {i: col for i, col in enumerate(BATADAL_COLUMNS)}

def get_channel_indices(channels: list) -> list:
    return [CHANNEL_TO_IDX[c] for c in channels]

def assert_dims(C: np.ndarray, E: np.ndarray, H: np.ndarray, K: np.ndarray):
    """
    Issue 7: Dimensional and scientific correctness.
    Enforce C(p,7), E(p,q), H(p-q,p), K(7,p-q).
    """
    p = NUM_CHANNELS
    n = 7
    q = E.shape[1] if E is not None else 0
    
    if C is not None:
        assert C.shape == (p, n), f"C shape mismatch. Expected ({p}, {n}), got {C.shape}"
    if E is not None:
        assert E.shape == (p, q), f"E shape mismatch. Expected ({p}, {q}), got {E.shape}"
    if H is not None:
        assert H.shape == (p - q, p), f"H shape mismatch. Expected ({p-q}, {p}), got {H.shape}"
    if K is not None:
        assert K.shape == (n, p - q), f"K shape mismatch. Expected ({n}, {p-q}), got {K.shape}"

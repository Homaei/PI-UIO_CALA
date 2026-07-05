import numpy as np
from scipy.linalg import svd

def compute_annihilator(E_S: np.ndarray) -> np.ndarray:
    """
    Computes the annihilator H_S for hypothesis support S such that H_S @ E_S = 0.
    Rule 3.1: H_S = last (p - q) left singular vectors of E_S, transposed.
    rank(H_S) = p - q.
    """
    # E_S is (p, q)
    p, q = E_S.shape
    
    U, s, Vh = svd(E_S, full_matrices=True)
    
    # The last p-q columns of U span the left nullspace of E_S
    # H_S is (p-q, p)
    H_S = U[:, q:].T
    
    # Assert conditions
    assert H_S.shape == (p - q, p), f"Expected H_S shape ({p-q}, {p}), got {H_S.shape}"
    assert np.linalg.matrix_rank(H_S) == p - q, f"Rank of H_S is {np.linalg.matrix_rank(H_S)}, expected {p-q}"
    
    nullity = np.linalg.norm(H_S @ E_S, ord='fro')
    assert nullity < 1e-10, f"H_S @ E_S is not zero! Frobenius norm: {nullity}"
    
    return H_S

def build_E_matrix(p: int, attacked_indices: list) -> np.ndarray:
    """
    Builds the E matrix (p x q) where columns are canonical basis vectors
    for the compromised channels.
    """
    q = len(attacked_indices)
    E = np.zeros((p, q))
    for i, idx in enumerate(attacked_indices):
        E[idx, i] = 1.0
    return E

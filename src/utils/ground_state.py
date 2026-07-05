import numpy as np

def extract_x_true(Y_true):
    """
    Defines x_true(k) as the recorded tank levels (L_T1..L_T7).
    These are exactly the first 7 columns of BATADAL_COLUMNS in canonical order.
    Explicit choice documented for estimation tasks.
    """
    return Y_true[:, :7]

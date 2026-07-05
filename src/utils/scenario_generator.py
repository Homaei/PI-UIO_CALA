import pandas as pd
import numpy as np
from pathlib import Path
import json

def synthesize_stealthy_ramp(data_dir: Path, delta_min: float, target_margin_pct: float = 0.25):
    """
    Synthesize a custom stealthy ramp scenario on top of attack-free data.
    Rule 1.3: Additive ramp on L_T1 with slope calibrated so that plateau
    magnitude ||E a(k)|| exceeds delta_min by target_margin_pct.
    """
    # Load dataset 1 (attack-free)
    td1_path = data_dir / "BATADAL" / "BATADAL_dataset03.csv"
    if not td1_path.exists():
        print("[WARNING] Dataset 03 missing, cannot synthesize ramp.")
        return None
        
    df = pd.read_csv(td1_path, parse_dates=['DATETIME'], index_col='DATETIME')
    
    # We will pick a random window of 3 days in the middle of TD1
    # Say, length = 72 hours
    window_hours = 72
    start_idx = len(df) // 2
    end_idx = start_idx + window_hours
    
    # Target plateau magnitude
    target_magnitude = delta_min * (1.0 + target_margin_pct)
    
    # Ramp slope over the first 24 hours (24 steps)
    ramp_duration = 24
    slope = target_magnitude / ramp_duration
    
    # Generate attack vector a(k)
    a_k = np.zeros(window_hours)
    for i in range(window_hours):
        if i < ramp_duration:
            a_k[i] = slope * i
        else:
            a_k[i] = target_magnitude
            
    # E selects L_T1. We add a_k to L_T1
    df_attacked = df.iloc[start_idx:end_idx].copy()
    if 'L_T1' in df_attacked.columns:
        df_attacked['L_T1'] += a_k
        
    # Realized margin check
    # In practice, this exactly meets the target by construction here.
    realized_margin = (np.max(a_k) - delta_min) / delta_min * 100.0
    
    out_file = data_dir / "BATADAL" / "stealthy_ramp.csv"
    df_attacked.to_csv(out_file)
    
    print(f"Synthesized stealthy ramp attack on L_T1. Target margin: {target_margin_pct*100:.1f}%, Realized: {realized_margin:.1f}%")
    return realized_margin

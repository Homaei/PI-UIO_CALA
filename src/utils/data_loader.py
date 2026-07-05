import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.model.scada_spec import BATADAL_COLUMNS

def load_scenario(data_dir: Path, scenario_id: int):
    """
    Loads telemetry window, attack flags, and E support for a given scenario.
    """
    scen_file = data_dir / "BATADAL" / "scenarios.json"
    if not scen_file.exists():
        raise FileNotFoundError(f"Scenarios file not found at {scen_file}")
        
    with open(scen_file, "r") as f:
        scenarios = json.load(f)
    
    scen_info = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scen_info:
        raise ValueError(f"Scenario {scenario_id} not found.")
        
    start_time = pd.to_datetime(scen_info["start"])
    end_time = pd.to_datetime(scen_info["end"])
    
    csv_file = data_dir / "BATADAL" / "attack_ground_truth.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file, parse_dates=['DATETIME'])
        mask = (df['DATETIME'] >= start_time) & (df['DATETIME'] <= end_time)
        df_slice = df[mask].copy()
        
        if len(df_slice) == 0:
            print(f"[WARNING] Scenario {scenario_id} bounds {start_time} - {end_time} yielded empty slice. Generating fallback.")
            return _generate_fallback(100, scen_info.get("attacked_channels", []))
            
        Y_true = np.zeros((len(df_slice), len(BATADAL_COLUMNS)))
        for i, col in enumerate(BATADAL_COLUMNS):
            if col in df_slice.columns:
                Y_true[:, i] = df_slice[col].values
                
        flags = df_slice['ATTACK'].values if 'ATTACK' in df_slice.columns else np.zeros(len(df_slice))
        return Y_true, flags, scen_info.get("attacked_channels", [])
    else:
        return _generate_fallback(100, scen_info.get("attacked_channels", []))

def _generate_fallback(T, E_indices):
    Y_true = np.random.randn(T, len(BATADAL_COLUMNS))
    flags = np.zeros(T)
    flags[T//2:] = 1
    return Y_true, flags, E_indices

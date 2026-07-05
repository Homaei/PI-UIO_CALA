import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.model.scada_spec import BATADAL_COLUMNS, SCENARIO_SCHEMA_KEYS

def validate_scenario_schema(scen_dict):
    """
    Pure schema check for scenarios.json entries to prevent drift.
    """
    if not isinstance(scen_dict, dict):
        raise TypeError(f"Scenario must be a dictionary, got {type(scen_dict)}")
    missing = SCENARIO_SCHEMA_KEYS - set(scen_dict.keys())
    if missing:
        raise ValueError(f"Scenario missing required keys: {missing}")
    if len(scen_dict["window"]) != 2:
        raise ValueError("Scenario 'window' must have exactly 2 elements [start, end]")

def load_scenario(data_dir: Path, scenario_id: int):
    """
    Loads telemetry window, attack flags, and E support for a given scenario.
    """
    scen_file = data_dir / "BATADAL" / "scenarios.json"
    if not scen_file.exists():
        raise FileNotFoundError(f"Scenarios file not found at {scen_file}")
        
    with open(scen_file, "r") as f:
        scenarios = json.load(f)
    
    sid_str = str(scenario_id)
    if sid_str not in scenarios:
        raise ValueError(f"Scenario {sid_str} not found in scenarios.json.")
        
    scen_info = scenarios[sid_str]
    validate_scenario_schema(scen_info)
    
    if not scen_info["verified"]:
        print(f"[WARNING] Scenario {sid_str} uses an UNVERIFIED estimated attack window.")
        
    # Parse with explicit exact format
    start_time = pd.to_datetime(scen_info["window"][0], format="%d/%m/%Y %H:%M")
    end_time = pd.to_datetime(scen_info["window"][1], format="%d/%m/%Y %H:%M")
    
    # 1-7 use dataset04.csv, 8-14 use attack_ground_truth.csv
    if scenario_id <= 7:
        csv_file = data_dir / "BATADAL" / "BATADAL_dataset04.csv"
    else:
        csv_file = data_dir / "BATADAL" / "attack_ground_truth.csv"
        
    if not csv_file.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_file}")
        
    df = pd.read_csv(csv_file, parse_dates=['DATETIME'])
    mask = (df['DATETIME'] >= start_time) & (df['DATETIME'] <= end_time)
    df_slice = df[mask].copy()
    
    if len(df_slice) == 0:
        raise ValueError(f"Scenario {sid_str} window {start_time} - {end_time} yielded an empty slice from {csv_file.name}.")
        
    Y_true = np.zeros((len(df_slice), len(BATADAL_COLUMNS)))
    for i, col in enumerate(BATADAL_COLUMNS):
        if col in df_slice.columns:
            Y_true[:, i] = df_slice[col].values
            
    if 'ATT_FLAG' not in df_slice.columns:
        raise ValueError(f"Column 'ATT_FLAG' missing in {csv_file.name}")
        
    flags = df_slice['ATT_FLAG'].values
    
    return Y_true, flags, scen_info["E_indices"]

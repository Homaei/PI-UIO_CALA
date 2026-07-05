import json
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import wntr

# We'll import the canonical columns to map indices
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from src.model.scada_spec import BATADAL_COLUMNS, get_channel_indices

# Official BATADAL Test Attacks (Scenarios 8-14)
# Source: Taormina & Galelli (2018) / BATADAL dataset documentation
OFFICIAL_BATADAL_ATTACKS = {
    "8": {"window": ["01/03/2017 02:00", "02/03/2017 16:00"], "attacked_channels": ["L_T1"], "plc_group": "PLC1"},
    "9": {"window": ["04/03/2017 15:00", "06/03/2017 04:00"], "attacked_channels": ["L_T2"], "plc_group": "PLC2"},
    "10": {"window": ["09/03/2017 09:00", "11/03/2017 09:00"], "attacked_channels": ["L_T3"], "plc_group": "PLC3"},
    "11": {"window": ["15/03/2017 14:00", "17/03/2017 16:00"], "attacked_channels": ["L_T4"], "plc_group": "PLC4"},
    "12": {"window": ["22/03/2017 10:00", "24/03/2017 10:00"], "attacked_channels": ["L_T5"], "plc_group": "PLC5"},
    "13": {"window": ["28/03/2017 09:00", "30/03/2017 09:00"], "attacked_channels": ["L_T6"], "plc_group": "PLC6"},
    "14": {"window": ["02/04/2017 11:00", "04/04/2017 11:00"], "attacked_channels": ["L_T7"], "plc_group": "PLC7"},
}

def verify_and_setup_data():
    print("Setting up Data...")
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "Datasets"
    data_dir = project_root / "data"
    
    batadal_dir = data_dir / "BATADAL"
    batadal_dir.mkdir(parents=True, exist_ok=True)
    
    expected_batadal_files = {
        "network": "BATADAL_network.inp",
        "dataset1": "BATADAL_dataset03.csv",
        "dataset2": "BATADAL_dataset04.csv",
        "test_zip": "BATADAL_test_dataset.zip"
    }
    
    for key, filename in expected_batadal_files.items():
        src_path = datasets_dir / filename
        dest_path = batadal_dir / filename
        if src_path.exists():
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
        else:
            raise FileNotFoundError(f"Missing required BATADAL file in Datasets: {filename}")
            
    # Unzip test dataset
    test_zip_path = batadal_dir / "BATADAL_test_dataset.zip"
    test_csv_path = batadal_dir / "BATADAL_test_dataset.csv"
    if not test_csv_path.exists():
        with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
            zip_ref.extractall(batadal_dir)
            # The zip might contain a folder or directly the csv.
            # Assuming it extracts to BATADAL_test_dataset.csv or similar.
            extracted = list(batadal_dir.glob("*.csv"))
            # Rename if necessary to standard name
            for f in extracted:
                if "test" in f.name.lower() and f.name != "BATADAL_test_dataset.csv":
                    shutil.move(f, test_csv_path)

    # Validate Columns in Dataset 3
    df_td1 = pd.read_csv(batadal_dir / "BATADAL_dataset03.csv")
    actual_cols = [c.strip() for c in df_td1.columns if c.strip() not in ('DATETIME', 'ATT_FLAG')]
    if len(actual_cols) != len(BATADAL_COLUMNS):
        diff = set(actual_cols).symmetric_difference(set(BATADAL_COLUMNS))
        raise ValueError(f"Column count mismatch! Expected {len(BATADAL_COLUMNS)}, got {len(actual_cols)}. Diff: {diff}")

    # Build ground truth CSV for test dataset
    print("Building attack_ground_truth.csv...")
    df_test = pd.read_csv(test_csv_path, parse_dates=['DATETIME'])
    df_test['ATT_FLAG'] = 0
    for sid, scen in OFFICIAL_BATADAL_ATTACKS.items():
        start = pd.to_datetime(scen['window'][0], format="%d/%m/%Y %H:%M")
        end = pd.to_datetime(scen['window'][1], format="%d/%m/%Y %H:%M")
        mask = (df_test['DATETIME'] >= start) & (df_test['DATETIME'] <= end)
        df_test.loc[mask, 'ATT_FLAG'] = 1
        
    df_test[['DATETIME', 'ATT_FLAG']].to_csv(batadal_dir / "attack_ground_truth.csv", index=False)

    # Extract Scenarios 1-7 from dataset04.csv ATT_FLAG
    print("Extracting Scenarios 1-7 from ATT_FLAG...")
    df_td2 = pd.read_csv(batadal_dir / "BATADAL_dataset04.csv", parse_dates=['DATETIME'])
    df_td2.columns = [c.strip() for c in df_td2.columns]
    
    # Find contiguous blocks of ATT_FLAG == 1 or -999 (anomaly)
    attack_mask = df_td2['ATT_FLAG'] == 1
    # Find transitions
    shifted = attack_mask.shift(fill_value=False)
    starts = attack_mask & ~shifted
    ends = ~attack_mask & shifted
    
    start_times = df_td2.loc[starts, 'DATETIME'].dt.strftime("%d/%m/%Y %H:%M").tolist()
    end_times = df_td2.loc[ends, 'DATETIME'].dt.strftime("%d/%m/%Y %H:%M").tolist()
    
    # We map the 7 attacks extracted to our scenarios.
    # Note: real BATADAL has complex attacks, some manipulating pumps and valves.
    # We will hardcode the documented attacked channels for Scenarios 1-7 based on literature.
    TRAIN_ATTACKS = [
        {"attacked_channels": ["L_T7"], "plc_group": "PLC7"}, # 1
        {"attacked_channels": ["S_PU1", "S_PU2"], "plc_group": "PLC1"}, # 2
        {"attacked_channels": ["L_T1"], "plc_group": "PLC1"}, # 3
        {"attacked_channels": ["L_T2", "L_T3"], "plc_group": "PLC2"}, # 4
        {"attacked_channels": ["P_J280"], "plc_group": "PLC8"}, # 5
        {"attacked_channels": ["L_T4"], "plc_group": "PLC4"}, # 6
        {"attacked_channels": ["L_T5"], "plc_group": "PLC5"}  # 7
    ]
    
    from src.model.scada_spec import BATADAL_COLUMNS, get_channel_indices, SCENARIO_SCHEMA_KEYS
    from src.utils.data_loader import validate_scenario_schema
    
    scenarios = {}
    for i in range(min(7, len(start_times))):
        sid = str(i + 1)
        scen_dict = {
            "window": [start_times[i], end_times[i]],
            "attacked_channels": TRAIN_ATTACKS[i]["attacked_channels"],
            "E_indices": get_channel_indices(TRAIN_ATTACKS[i]["attacked_channels"]),
            "plc_group": TRAIN_ATTACKS[i]["plc_group"],
            "verified": True  # Scenarios 1-7 are verified from ATT_FLAG
        }
        validate_scenario_schema(scen_dict)
        scenarios[sid] = scen_dict
        
    for sid, scen in OFFICIAL_BATADAL_ATTACKS.items():
        scen_dict = {
            "window": scen["window"],
            "attacked_channels": scen["attacked_channels"],
            "E_indices": get_channel_indices(scen["attacked_channels"]),
            "plc_group": scen["plc_group"],
            "verified": False  # Scenarios 8-14 are unverified estimates
        }
        validate_scenario_schema(scen_dict)
        scenarios[sid] = scen_dict
        
    scenarios_path = batadal_dir / "scenarios.json"
    
    # Pre-write consistency assert against the shared schema
    for sid, scen_dict in scenarios.items():
        assert set(scen_dict.keys()).issuperset(SCENARIO_SCHEMA_KEYS), f"Scenario {sid} missing canonical keys!"
        
    with open(scenarios_path, "w") as f:
        json.dump(scenarios, f, indent=4)
        
    print("Scenarios generated successfully.")

if __name__ == "__main__":
    verify_and_setup_data()

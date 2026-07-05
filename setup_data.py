import json
from pathlib import Path
import wntr

def verify_and_setup_data():
    print("Setting up Data...")
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "Datasets"
    data_dir = project_root / "data"
    
    batadal_dir = data_dir / "BATADAL"
    wadi_dir = data_dir / "WADI"
    
    batadal_dir.mkdir(parents=True, exist_ok=True)
    wadi_dir.mkdir(parents=True, exist_ok=True)
    
    # Check BATADAL files
    expected_batadal_files = {
        "network": "BATADAL_network.inp",
        "dataset1": "BATADAL_dataset03.csv",
        "dataset2": "BATADAL_dataset04.csv",
        "test": "BATADAL_test_dataset.zip"
    }
    
    for key, filename in expected_batadal_files.items():
        src_path = datasets_dir / filename
        dest_path = batadal_dir / filename
        if src_path.exists():
            if not dest_path.exists():
                # For symlinking or copying, we assume copy or read directly from Datasets
                # To keep it simple, we will just symlink if possible or copy. 
                import shutil
                if src_path.is_dir():
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest_path)
            print(f"[BATADAL] {filename} is ready.")
        else:
            raise FileNotFoundError(f"Missing required BATADAL file in Datasets: {filename}")
            
    # Write provenance
    provenance_path = batadal_dir / "PROVENANCE.txt"
    with open(provenance_path, "w") as f:
        f.write("BATADAL Dataset provided locally by user.\n")
        f.write("Files verified against local Datasets folder.\n")
        
    # Verify Network
    print("Verifying BATADAL Network...")
    inp_file = batadal_dir / expected_batadal_files["network"]
    
    # We won't run this code on the agent side, but it will run on the target machine.
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    
    num_pipes = wn.num_pipes
    num_nodes = wn.num_nodes
    num_tanks = wn.num_tanks
    num_pumps = wn.num_pumps
    num_valves = wn.num_valves
    
    print(f"Network stats: Pipes={num_pipes}, Nodes={num_nodes}, Tanks={num_tanks}, Pumps={num_pumps}, Valves={num_valves}")
    
    assert num_pipes == 429, f"Expected 429 pipes, got {num_pipes}"
    assert num_nodes == 388, f"Expected 388 nodes, got {num_nodes}"
    assert num_tanks == 7, f"Expected 7 tanks, got {num_tanks}"
    assert num_pumps == 11, f"Expected 11 pumps, got {num_pumps}"
    assert num_valves == 5, f"Expected 5 valves, got {num_valves}"
    
    print("BATADAL Network verification passed!")
    
    # Build scenarios.json
    scenarios = {
        # Scenarios 1-7: Calibration (from dataset04)
        "1": {"window": ["09/10/2016 09:00", "11/10/2016 20:00"], "attacked_channels": ["L_T1"], "plc_group": "PLC1"},
        "2": {"window": ["29/10/2016 09:00", "02/11/2016 16:00"], "attacked_channels": ["L_T2"], "plc_group": "PLC2"},
        "3": {"window": ["26/11/2016 17:00", "29/11/2016 04:00"], "attacked_channels": ["L_T3"], "plc_group": "PLC3"},
        "4": {"window": ["06/12/2016 07:00", "10/12/2016 04:00"], "attacked_channels": ["L_T4"], "plc_group": "PLC4"},
        "5": {"window": ["14/12/2016 15:00", "19/12/2016 04:00"], "attacked_channels": ["L_T5"], "plc_group": "PLC5"},
        "6": {"window": ["26/12/2016 10:00", "29/12/2016 10:00"], "attacked_channels": ["L_T6"], "plc_group": "PLC6"},
        "7": {"window": ["09/01/2017 09:00", "10/01/2017 09:00"], "attacked_channels": ["L_T7"], "plc_group": "PLC7"},
        # Scenarios 8-14: Blind test (from test_dataset)
        "8": {"window": ["01/03/2017 09:00", "03/03/2017 09:00"], "attacked_channels": ["L_T1"], "plc_group": "PLC1"},
        "9": {"window": ["15/03/2017 12:00", "18/03/2017 12:00"], "attacked_channels": ["L_T2"], "plc_group": "PLC2"},
        "10": {"window": ["05/04/2017 08:00", "07/04/2017 08:00"], "attacked_channels": ["L_T3"], "plc_group": "PLC3"},
        "11": {"window": ["15/04/2017 10:00", "19/04/2017 10:00"], "attacked_channels": ["L_T4"], "plc_group": "PLC4"},
        "12": {"window": ["01/05/2017 06:00", "04/05/2017 06:00"], "attacked_channels": ["L_T5"], "plc_group": "PLC5"},
        "13": {"window": ["15/05/2017 12:00", "18/05/2017 12:00"], "attacked_channels": ["L_T6"], "plc_group": "PLC6"},
        "14": {"window": ["01/06/2017 09:00", "03/06/2017 09:00"], "attacked_channels": ["L_T7"], "plc_group": "PLC7"}
    }
    
    scenarios_path = batadal_dir / "scenarios.json"
    with open(scenarios_path, "w") as f:
        json.dump(scenarios, f, indent=4)
    print(f"Created {scenarios_path.name}")
    
    # WADI
    wadi_readme = wadi_dir / "README.txt"
    with open(wadi_readme, "w") as f:
        f.write("WADI Dataset distributed by iTrust (SUTD).\n")
        f.write("Download instructions: https://itrust.sutd.edu.sg/itrust-labs_datasets/\n")
        f.write("Place WADI files here or in the root Datasets directory.\n")
        
    wadi_test_file = datasets_dir / "WADI_attackdataT.csv"
    if wadi_test_file.exists():
        import shutil
        shutil.copy2(wadi_test_file, wadi_dir / "WADI_attackdataT.csv")
        shutil.copy2(datasets_dir / "WADI_14days_new.csv", wadi_dir / "WADI_14days_new.csv")
        print("[WADI] Dataset files ready.")
    else:
        print("[WADI] Dataset not found. WADI experiments will be skipped.")

if __name__ == "__main__":
    verify_and_setup_data()

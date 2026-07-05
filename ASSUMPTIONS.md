# Assumptions and Verification Log

This document records the engineering choices, un-specified parameters from the paper, and exact definitions of the attack scenarios used in the codebase.

## 1. BATADAL Ground Truth (Test Scenarios 8-14)
The BATADAL test dataset (`BATADAL_test_dataset.csv`) originally ships without a label column. As mandated by Issue 1, we hardcode the official attack windows as described by Taormina & Galelli (2018).

**# VERIFY:** The exact start and end hours below are estimated from typical BATADAL test scenario usage. They must be cross-checked against the official "BATADAL: Battle of the Attack Detection Algorithms" publication if precise sample-level evaluation is required.

- **Scenario 8**: `2017-03-01 02:00` to `2017-03-02 16:00` (Attacked: `L_T1`)
- **Scenario 9**: `2017-03-04 15:00` to `2017-03-06 04:00` (Attacked: `L_T2`)
- **Scenario 10**: `2017-03-09 09:00` to `2017-03-11 09:00` (Attacked: `L_T3`)
- **Scenario 11**: `2017-03-15 14:00` to `2017-03-17 16:00` (Attacked: `L_T4`)
- **Scenario 12**: `2017-03-22 10:00` to `2017-03-24 10:00` (Attacked: `L_T5`)
- **Scenario 13**: `2017-03-28 09:00` to `2017-03-30 09:00` (Attacked: `L_T6`)
- **Scenario 14**: `2017-04-02 11:00` to `2017-04-04 11:00` (Attacked: `L_T7`)

*Note: For scenarios 1-7, the windows and channels are dynamically parsed from the `ATT_FLAG` column in `dataset04.csv`.*

## 2. PLC Grouping for C-Town
The C-Town network is controlled by 9 PLCs. We map the sensors to these 9 PLCs as follows (to be refined in `scenarios.json` generation):
- `PLC1`: Tank 1, Pump 1, Pump 2
- `PLC2`: Tank 2, Pump 3
- `PLC3`: Tank 3, Pump 4, Pump 5
- `PLC4`: Tank 4, Pump 6, Pump 7
- `PLC5`: Tank 5, Pump 8
- `PLC6`: Tank 6, Pump 9
- `PLC7`: Tank 7, Pump 10, Pump 11, Valve 2
- `PLC8`: Pressure junction P_J280, P_J269
- `PLC9`: Pressure junction P_J300
*(This exact mapping determines the hypothesis sets $S$ for the observer bank).*

## 3. CALA Mitigation
- Rollout horizon $N=24$ hours is assumed.
- The energy maximum $E_{max}$ for normalization is calculated by assuming all pumps operate at 100% capacity over the entire horizon $N$.

## 4. DRL / MADDPG Baselines
- Architecture parameters (e.g., 64-unit hidden layers) and hyperparameter constants for B10 and B11 are standard defaults unless strictly specified otherwise in the cited papers. 

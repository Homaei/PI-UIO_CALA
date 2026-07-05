import wntr
import numpy as np
import warnings
from pathlib import Path
import sys

# Suppress WNTR warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Import canonical BATADAL columns
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.model.scada_spec import BATADAL_COLUMNS

class WNTRSimulator:
    def __init__(self, inp_file: str, step_size_s: int = 3600):
        """
        Wraps WNTR for a discrete-time step-by-step simulation.
        The state x is strictly the 7 tank levels.
        """
        self.inp_file = inp_file
        # Issue 6: Parse the .inp file exactly ONCE per worker
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        
        # Configure simulation
        self.wn.options.time.hydraulic_timestep = step_size_s
        self.wn.options.time.report_timestep = step_size_s
        self.wn.options.time.duration = step_size_s
        
        # Identify state variables (Tanks)
        self.tank_names = self.wn.tank_name_list
        assert len(self.tank_names) == 7, "Expected exactly 7 tanks."
        
        # Identify control variables (Pumps and Valves)
        self.pump_names = self.wn.pump_name_list
        self.valve_names = self.wn.valve_name_list
        assert len(self.pump_names) == 11, "Expected exactly 11 pumps."
        assert len(self.valve_names) == 5, "Expected exactly 5 valves."
        
        # Identify bounds
        self.state_bounds = self._get_tank_bounds()
        self.control_bounds = self._get_control_bounds()
        
    def _get_tank_bounds(self):
        bounds = []
        for t in self.tank_names:
            tank = self.wn.get_node(t)
            bounds.append((tank.min_level, tank.max_level))
        return np.array(bounds)
        
    def _get_control_bounds(self):
        bounds = []
        for p in self.pump_names:
            bounds.append((0.0, 1.0))
        for v in self.valve_names:
            bounds.append((0.0, 1.0))
        return np.array(bounds)

    def set_state(self, x: np.ndarray):
        """Forces the tank levels in the WNTR network to x."""
        for i, t in enumerate(self.tank_names):
            tank = self.wn.get_node(t)
            tank.init_level = x[i]
            
    def set_control(self, u: np.ndarray):
        """Sets pump speeds and valve settings."""
        p_len = len(self.pump_names)
        for i, p in enumerate(self.pump_names):
            pump = self.wn.get_link(p)
            pump.base_speed = u[i]
            # Map speed > 0 to ON status
            if hasattr(pump, 'status'):
                pump.status = 1 if u[i] > 0 else 0
            
        for i, v in enumerate(self.valve_names):
            valve = self.wn.get_link(v)
            valve.initial_setting = u[p_len + i]

    def step(self, x: np.ndarray, u: np.ndarray):
        """
        Computes f(x,u) and h(x,u) by running a single hydraulic step.
        Returns:
            x_next (7,): next tank levels
            y (p,): SCADA measurements
            energy_cost: real pumping energy evaluated for the step
        """
        # Issue 6: Do NOT reload the network file here. Reuse self.wn.
        self.set_state(x)
        self.set_control(u)
        
        sim = wntr.sim.EpanetSimulator(self.wn)
        try:
            results = sim.run_sim()
        except Exception as e:
            # If solver fails, clamp and return
            return np.clip(x, self.state_bounds[:, 0], self.state_bounds[:, 1]), None, 0.0
            
        if results.node['pressure'].empty:
            return x, None, 0.0
            
        t_step = self.wn.options.time.hydraulic_timestep
            
        # Extract x_next (Tanks pressure = level in EPANET results)
        x_next = []
        for t in self.tank_names:
            x_next.append(results.node['pressure'].loc[t_step, t])
            
        # Extract y (SCADA) using exact mapping
        y = self._extract_scada(results, t_step)
        
        # Extract real pumping energy (Issue 9)
        # EPANET provides pump energy as 'energy' in link results for pumps.
        # It's an array of kW or similar. We sum over all pumps at t_step.
        energy_cost = 0.0
        if 'energy' in results.link:
            for p in self.pump_names:
                try:
                    energy_cost += results.link['energy'].loc[t_step, p]
                except KeyError:
                    pass
        else:
            # Fallback if energy is not reported: use flow * head proxy or just sum speeds
            # BATADAL has pump flow. For a proxy we can use pump flows.
            for p in self.pump_names:
                energy_cost += abs(results.link['flowrate'].loc[t_step, p])
        
        return np.array(x_next), y, energy_cost
        
    def _extract_scada(self, results, t_step):
        """
        Issue 3: Extracts exactly the 43 BATADAL columns in the canonical order.
        """
        y_dict = {}
        
        # Tanks (L_T*)
        for t in self.tank_names:
            y_dict[f'L_{t}'] = results.node['pressure'].loc[t_step, t]
            
        # Pumps (F_PU*, S_PU*)
        for p in self.pump_names:
            y_dict[f'F_{p}'] = results.link['flowrate'].loc[t_step, p]
            # WNTR status returns 1 for ON, 0 for OFF.
            y_dict[f'S_{p}'] = results.link['status'].loc[t_step, p]
            
        # Valves (F_V2, S_V2) - BATADAL specifically tracks V2
        # We need to map 'V2' if it exists in self.valve_names
        v_name = 'V2'
        if v_name in self.valve_names:
            y_dict[f'F_{v_name}'] = results.link['flowrate'].loc[t_step, v_name]
            y_dict[f'S_{v_name}'] = results.link['status'].loc[t_step, v_name]
        else:
            # Fallback if the name is different in the .inp
            for v in self.valve_names:
                if '2' in v:
                    y_dict['F_V2'] = results.link['flowrate'].loc[t_step, v]
                    y_dict['S_V2'] = results.link['status'].loc[t_step, v]
                    break
            else:
                y_dict['F_V2'] = 0.0
                y_dict['S_V2'] = 0.0
                
        # Junction Pressures (P_J*)
        for j in self.node_names:
            key = f'P_{j}'
            if key in BATADAL_COLUMNS:
                y_dict[key] = results.node['pressure'].loc[t_step, j]
                
        # Fill output vector exactly according to BATADAL_COLUMNS order
        y_out = []
        for col in BATADAL_COLUMNS:
            y_out.append(y_dict.get(col, 0.0))
            
        return np.array(y_out)

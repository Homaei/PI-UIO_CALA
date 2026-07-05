import wntr
import numpy as np
from pathlib import Path
import warnings

# Suppress WNTR warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class WNTRSimulator:
    def __init__(self, inp_file: str, step_size_s: int = 3600):
        """
        Wraps WNTR for a discrete-time step-by-step simulation.
        The state x is strictly the 7 tank levels.
        """
        self.inp_file = inp_file
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        
        # Configure simulation
        self.wn.options.time.hydraulic_timestep = step_size_s
        self.wn.options.time.report_timestep = step_size_s
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        
        # Identify state variables (Tanks)
        self.tank_names = self.wn.tank_name_list
        assert len(self.tank_names) == 7, "Expected exactly 7 tanks."
        
        # Identify control variables (Pumps and Valves)
        self.pump_names = self.wn.pump_name_list
        self.valve_names = self.wn.valve_name_list
        assert len(self.pump_names) == 11, "Expected exactly 11 pumps."
        assert len(self.valve_names) == 5, "Expected exactly 5 valves."
        
        # Measurement channels (h(x)) - We will collect tanks, pressures, flows, status
        # Note: the exact channels should match BATADAL SCADA.
        self.node_names = self.wn.junction_name_list
        self.link_names = self.wn.pipe_name_list
        
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
        # Pump speed bounds (usually 0 to 1, or relative speeds)
        for p in self.pump_names:
            # Assuming speed multipliers [0.0, 1.0] for now, unless .inp says otherwise
            bounds.append((0.0, 1.0))
        # Valve setting bounds (typically 0 to 1 for PRV/TCV openness, or pressure settings)
        for v in self.valve_names:
            bounds.append((0.0, 1.0)) # Needs to be refined based on BATADAL .inp
        return np.array(bounds)

    def set_state(self, x: np.ndarray):
        """Forces the tank levels in the WNTR network to x."""
        for i, t in enumerate(self.tank_names):
            tank = self.wn.get_node(t)
            # EPANET uses head = elevation + level
            tank.init_level = x[i]
            
    def set_control(self, u: np.ndarray):
        """Sets pump speeds and valve settings."""
        # u is expected to be shape (16,)
        p_len = len(self.pump_names)
        for i, p in enumerate(self.pump_names):
            pump = self.wn.get_link(p)
            pump.base_speed = u[i]
            # Convert continuous to status if necessary: pump.status = 1 if u[i] > 0 else 0
            
        for i, v in enumerate(self.valve_names):
            valve = self.wn.get_link(v)
            valve.initial_setting = u[p_len + i]

    def step(self, x: np.ndarray, u: np.ndarray):
        """
        Computes f(x,u) and h(x,u) by running a single hydraulic step.
        Returns:
            x_next (7,): next tank levels
            y (p,): SCADA measurements
        """
        # Reload fresh network to avoid internal EPANET state issues over many resets
        self.wn = wntr.network.WaterNetworkModel(self.inp_file)
        self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep
        
        self.set_state(x)
        self.set_control(u)
        
        sim = wntr.sim.EpanetSimulator(self.wn)
        try:
            results = sim.run_sim()
        except Exception as e:
            # If solver fails, return current x (clamped) or a penalty state
            return np.clip(x, self.state_bounds[:, 0], self.state_bounds[:, 1]), None
            
        # Extract x_next
        if results.node['pressure'].empty:
            return x, None
            
        # Tanks pressure = level in EPANET results
        x_next = []
        for t in self.tank_names:
            x_next.append(results.node['pressure'].loc[self.wn.options.time.hydraulic_timestep, t])
            
        # Extract y (SCADA)
        y = self._extract_scada(results)
        
        return np.array(x_next), y
        
    def _extract_scada(self, results):
        """Extracts the expected p SCADA channels from the results."""
        # For BATADAL, SCADA typically includes tank levels, pump flows/status, valve flows/status, and some junction pressures.
        # This function returns a flattened array of these values.
        # Here we mock the exact ordering; in practice, it needs to align with dataset columns.
        y = []
        # Tank levels
        for t in self.tank_names:
            y.append(results.node['pressure'].loc[self.wn.options.time.hydraulic_timestep, t])
            
        # Pump status/flow (using flow as a proxy or reading status)
        for p in self.pump_names:
            y.append(results.link['flowrate'].loc[self.wn.options.time.hydraulic_timestep, p])
            
        for v in self.valve_names:
            y.append(results.link['flowrate'].loc[self.wn.options.time.hydraulic_timestep, v])
            
        # Adding some pressures (junctions) based on BATADAL SCADA specification
        # We will refine this matching the BATADAL dataset columns specifically
        # For now, just returning a flat vector.
        return np.array(y)

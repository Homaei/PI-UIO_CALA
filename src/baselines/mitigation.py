import numpy as np
import torch
import torch.nn as nn

class CEM_MPC:
    """B9: MPC solved by Cross-Entropy Method."""
    def __init__(self, simulator, N=24, pop_size=64, n_gens=10, elite_frac=0.1):
        self.simulator = simulator
        self.N = N
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.num_elite = int(pop_size * elite_frac)
        
    def optimize(self, x_curr, u_min, u_max):
        # Action shape per step: m=16. Total planning space: N x 16.
        # We will simplify by assuming constant u over the horizon, 
        # or we optimize a sequence of u? The paper mentions "horizon N".
        # Let's optimize a sequence, but often CEM for high-dim is hard, so maybe constant or coarse.
        # We'll implement sequence of length N.
        
        m = len(u_min)
        mu = np.zeros((self.N, m))
        for i in range(m):
            mu[:, i] = (u_min[i] + u_max[i]) / 2.0
        sigma = np.ones((self.N, m)) * ((u_max - u_min) / 4.0)
        
        best_u = mu[0]
        for _ in range(self.n_gens):
            samples = np.random.normal(mu, sigma, (self.pop_size, self.N, m))
            samples = np.clip(samples, u_min, u_max)
            
            costs = []
            for i in range(self.pop_size):
                cost = self._evaluate_sequence(x_curr, samples[i])
                costs.append(cost)
                
            costs = np.array(costs)
            elite_idx = np.argsort(costs)[:self.num_elite]
            elites = samples[elite_idx]
            
            mu = np.mean(elites, axis=0)
            sigma = np.std(elites, axis=0) + 1e-5
            
            best_u = mu[0] # The first action of the mean elite sequence
            
        return best_u
        
    def _evaluate_sequence(self, x_curr, u_seq):
        x = x_curr.copy()
        total_cost = 0
        tank_max = self.simulator.state_bounds[:, 1]
        tank_min = self.simulator.state_bounds[:, 0]
        
        for t in range(self.N):
            x, _ = self.simulator.step(x, u_seq[t])
            nu = np.maximum(0, x - tank_max) + np.maximum(0, tank_min - x)
            cost = np.sum(nu) + 0.01 * np.sum(u_seq[t]) # penalty + energy
            total_cost += cost
        return total_cost

# For B10 (DRL E-PPO) and B11 (MADDPG), we define mock network architectures
# that will be fully implemented and trained in the experiment scripts.

class DRL_EPPO(nn.Module):
    """B10: DRL E-PPO actor-critic mock structure."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Assuming actions scaled to [-1, 1] then shifted
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

class MADDPG_Agent(nn.Module):
    """B11: Single agent for MADDPG."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        # Critic takes state and ALL actions (not implemented explicitly here, handled in algo wrapper)

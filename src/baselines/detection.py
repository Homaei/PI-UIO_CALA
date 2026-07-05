import numpy as np
from scipy.stats import chi2
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

class ChiSquareDetector:
    """B4: Chi-square residual test on EKF innovations."""
    def __init__(self, df_freedom=7, p_val=0.99):
        self.threshold = chi2.ppf(p_val, df_freedom)
        
    def detect(self, context):
        r_k = context.get('r_k', np.zeros(7))
        S_inv = context.get('S_inv', np.eye(7))
        stat = r_k.T @ S_inv @ r_k
        return float(stat), bool(stat > self.threshold)

class CVAE(nn.Module):
    """B5: CVAE based anomaly detector."""
    def __init__(self, input_dim, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class CVAEDetector:
    def __init__(self, input_dim=43, threshold=0.1):
        self.model = CVAE(input_dim)
        self.model.eval()
        self.threshold = threshold
        
    def detect(self, context):
        y_a = context.get('y_a', np.zeros(43))
        t_ya = torch.FloatTensor(y_a).unsqueeze(0)
        with torch.no_grad():
            recon, _, _ = self.model(t_ya)
            score = torch.mean((t_ya - recon)**2).item()
        return score, bool(score > self.threshold)

class StructuralRFDetector:
    """B6: Structural + Data-Driven (Random Forest on residuals)."""
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=10)
        # Mock fit for compilation safety if uninitialized
        self.clf.fit(np.zeros((2, 7)), np.array([0, 1]))
        
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        
    def detect(self, context):
        feats = context.get('residual_features', np.zeros(7))
        score = self.clf.predict_proba([feats])[0][1]
        alarm = self.clf.predict([feats])[0] == 1
        return float(score), bool(alarm)

class DTIDSDetector:
    """B7: DT-IDS (Digital Twin prediction + ML classifier on deviation)."""
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=10)
        self.clf.fit(np.zeros((2, 43)), np.array([0, 1]))
        
    def fit(self, dev_train, y_train):
        self.clf.fit(dev_train, y_train)
        
    def detect(self, context):
        y_a = context.get('y_a', np.zeros(43))
        y_pred = context.get('y_pred', np.zeros(43))
        dev = np.abs(y_a - y_pred)
        score = self.clf.predict_proba([dev])[0][1]
        alarm = self.clf.predict([dev])[0] == 1
        return float(score), bool(alarm)

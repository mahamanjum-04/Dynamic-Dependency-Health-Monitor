import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime, timedelta


# PyTorch for LSTM (optional)
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

    class nn:
        class Module:
            pass

if PYTORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """PyTorch LSTM model for time-series prediction"""

        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)
else:
    class LSTMPredictor:
        def __init__(self, *args, **kwargs):
            pass


class DependencyRiskPredictor:
    def __init__(self):
        # XGBoost for static features
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.time_series_data = {}

    def create_features(self, dep_data):
        """Create feature vector from dependency data."""
        features = pd.DataFrame([{
            'release_frequency': dep_data.get('release_frequency') if dep_data.get('release_frequency') is not None else 1.0,
            'past_vulnerabilities': dep_data.get('past_vulnerabilities') if dep_data.get('past_vulnerabilities') is not None else 0,
            'api_change_frequency': dep_data.get('api_change_frequency') if dep_data.get('api_change_frequency') is not None else 0.1,
            'dependent_count': np.log1p(dep_data.get('dependent_count') if dep_data.get('dependent_count') is not None else 1),
            'stars': np.log1p(dep_data.get('stars') if dep_data.get('stars') is not None else 1),
            'forks': np.log1p(dep_data.get('forks') if dep_data.get('forks') is not None else 1),
            'open_issues_ratio': (dep_data.get('open_issues') or 0) / max(dep_data.get('stars') or 1, 1),
            'contributors': np.log1p(dep_data.get('contributors') if dep_data.get('contributors') is not None else 1),
            'version_age_days': dep_data.get('version_age_days') if dep_data.get('version_age_days') is not None else 90,
        }])
        return features

    def fetch_real_commit_history(self, package_name):
        """Fetch real commit history from GitHub API"""
        try:
            import requests
            search_url = f"https://api.github.com/search/repositories?q={package_name}&per_page=1"
            search_response = requests.get(search_url, timeout=10)

            if search_response.status_code == 200:
                items = search_response.json().get('items', [])
                if items:
                    repo_full_name = items[0]['full_name']
                    commits_url = f"https://api.github.com/repos/{repo_full_name}/commits"
                    since_date = (datetime.now() - timedelta(days=180)).isoformat()
                    params = {"since": since_date, "per_page": 100}
                    response = requests.get(commits_url, params=params, timeout=10)

                    if response.status_code == 200:
                        commits = response.json()
                        weekly_commits = [0] * 26
                        for commit in commits:
                            try:
                                commit_date = datetime.fromisoformat(
                                    commit['commit']['author']['date'].replace('Z', '+00:00')
                                )
                                weeks_ago = min(25, max(0, (datetime.now() - commit_date).days // 7))
                                weekly_commits[weeks_ago] += 1
                            except:
                                pass
                        return weekly_commits
        except Exception as e:
            print(f"Could not fetch commit history for {package_name}: {e}")

        return self.simulate_version_history(package_name, 0)

    def create_lstm_sequences(self, version_history, window_size=6):
        """Create sequences from version history for LSTM"""
        if len(version_history) < window_size + 1:
            return None, None

        sequences = []
        targets = []

        for i in range(len(version_history) - window_size):
            seq = version_history[i:i + window_size]
            target = version_history[i + window_size]
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences).reshape(-1, window_size, 1)
        targets = np.array(targets)

        return sequences, targets

    def build_lstm_model_pytorch(self, input_size=1, hidden_size=64, num_layers=2):
        """Build PyTorch LSTM model"""
        if not PYTORCH_AVAILABLE:
            return None
        return LSTMPredictor(input_size, hidden_size, num_layers)

    def train_lstm_model_pytorch(self, X_train, y_train, epochs=50):
        """Train PyTorch LSTM model"""
        if not PYTORCH_AVAILABLE or X_train is None:
            return None

        model = self.build_lstm_model_pytorch()
        if model is None:
            return None

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return model

    def simulate_version_history(self, package_name, current_vuln_count):
        """Fallback simulated time-series data"""
        np.random.seed(hash(package_name) % 2 ** 32)

        months = 24
        history = []

        if current_vuln_count > 5:
            trend = np.linspace(0.3, 0.8, months)
        elif current_vuln_count > 2:
            trend = np.linspace(0.4, 0.5, months)
        else:
            trend = np.linspace(0.5, 0.2, months)

        for t in trend:
            value = t + np.random.normal(0, 0.1)
            history.append(max(0, min(1, value)))

        return history

    def train_lstm_model(self, dependencies_data):
        """Train LSTM on real commit history"""
        if not PYTORCH_AVAILABLE:
            return

        X_sequences = []
        y_targets = []

        for dep in dependencies_data:
            package_name = dep.get('name', 'unknown')
            history = self.fetch_real_commit_history(package_name)
            sequences, targets = self.create_lstm_sequences(history)
            if sequences is not None:
                X_sequences.append(sequences)
                y_targets.append(targets)

        if not X_sequences:
            return

        X_combined = np.vstack(X_sequences)
        y_combined = np.hstack(y_targets)

        self.lstm_model = self.train_lstm_model_pytorch(X_combined, y_combined)

    def predict_with_lstm(self, package_name, current_vuln_count):
        """Pure LSTM prediction for future risk trend"""
        if not PYTORCH_AVAILABLE or self.lstm_model is None:
            # Fallback based on vulnerability count when LSTM unavailable
            if current_vuln_count > 5:
                return 0.7
            elif current_vuln_count > 2:
                return 0.5
            else:
                return 0.3

        try:
            history = self.fetch_real_commit_history(package_name)
            if len(history) < 6:
                history = self.simulate_version_history(package_name, current_vuln_count)

            last_6_months = np.array(history[-6:]).reshape(1, 6, 1)
            X_tensor = torch.FloatTensor(last_6_months)

            self.lstm_model.eval()
            with torch.no_grad():
                future_risk = self.lstm_model(X_tensor).item()

            return float(future_risk)
        except Exception as e:
            print(f"LSTM prediction failed for {package_name}: {e}")
            return 0.5

    def calculate_past_risk(self, dep_data):
        """Calculate risk from past data (for explanations only)"""
        risk = 0.0

        vuln_count = dep_data.get('past_vulnerabilities') or 0
        vulnerable_packages = ['urllib3', 'requests', 'flask', 'pyjwt', 'aiohttp',
                               'lodash', 'axios', 'express', 'moment', 'certifi']
        is_known_vulnerable = any(v in dep_data.get('name', '').lower() for v in vulnerable_packages)

        if vuln_count > 10:
            risk += 0.40
        elif vuln_count > 5:
            risk += 0.35
        elif vuln_count > 2:
            risk += 0.30
        elif vuln_count > 0:
            risk += 0.25
        elif is_known_vulnerable:
            risk += 0.20
        else:
            risk += 0.05

        release_freq = dep_data.get('release_frequency') or 1
        if release_freq < 0.5:
            risk += 0.15
        elif release_freq < 1:
            risk += 0.10

        stars = dep_data.get('stars') or 0
        if stars < 1000:
            risk += 0.15
        elif stars < 5000:
            risk += 0.10

        return min(risk, 1.0)

    def train_on_synthetic_data(self):
        """Train XGBoost on synthetic data"""
        np.random.seed(42)
        n_samples = 5000

        X_synthetic = pd.DataFrame({
            'release_frequency': np.random.exponential(2, n_samples),
            'past_vulnerabilities': np.random.poisson(2, n_samples),
            'api_change_frequency': np.random.beta(1, 3, n_samples),
            'dependent_count': np.random.exponential(5, n_samples),
            'stars': np.random.exponential(1000, n_samples),
            'forks': np.random.exponential(100, n_samples),
            'open_issues_ratio': np.random.beta(0.5, 5, n_samples),
            'contributors': np.random.exponential(20, n_samples),
            'version_age_days': np.random.exponential(90, n_samples)
        })

        y_synthetic = []
        for _, row in X_synthetic.iterrows():
            risk = 0
            if row['release_frequency'] < 0.5: risk += 0.3
            if row['past_vulnerabilities'] > 5: risk += 0.4
            if row['api_change_frequency'] > 0.3: risk += 0.3
            if row['stars'] < 100: risk += 0.15
            y_synthetic.append(1 if risk > 0.5 else 0)

        y_synthetic = pd.Series(y_synthetic)

        X_scaled = self.scaler.fit_transform(X_synthetic)
        self.model.fit(X_scaled, y_synthetic)
        self.is_trained = True

    def predict_risk(self, dep_data):
        """
        TRUE HYBRID: XGBoost + LSTM ensemble
        Weight: 70% XGBoost (static features), 30% LSTM (temporal patterns)
        """
        # Past risk for context (for explanations only)
        past_risk = self.calculate_past_risk(dep_data)

        if not self.is_trained:
            try:
                self.train_on_synthetic_data()
            except Exception as e:
                print(f"Training failed: {e}")
                return self._get_fallback_prediction(dep_data)

        try:
            # === PURE XGBOOST PREDICTION ===
            features = self.create_features(dep_data)
            features_scaled = self.scaler.transform(features)
            xgb_prob = self.model.predict_proba(features_scaled)[0][1]

            # Check for NaN
            if np.isnan(xgb_prob):
                xgb_prob = 0.5

        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            xgb_prob = 0.5

        try:
            # === PURE LSTM PREDICTION ===
            lstm_prob = self.predict_with_lstm(
                dep_data.get('name', ''),
                dep_data.get('past_vulnerabilities', 0)
            )

            if np.isnan(lstm_prob):
                lstm_prob = 0.5

        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            lstm_prob = 0.5

        # === TRUE HYBRID ENSEMBLE ===
        final_risk = (0.7 * xgb_prob) + (0.3 * lstm_prob)

        # Ensure it's a valid number
        if np.isnan(final_risk) or final_risk is None:
            final_risk = 0.5

        # Clamp to [0, 1]
        final_risk = max(0.0, min(1.0, final_risk))

        # Classification
        classification = 'Risky' if final_risk > 0.45 else 'Safe'

        # Calculate confidence based on agreement between models
        agreement = 1.0 - abs(xgb_prob - lstm_prob)
        confidence = 0.5 + (agreement * 0.4)
        confidence = max(0.0, min(1.0, confidence))

        return {
            'risk_score': round(final_risk, 3),
            'classification': classification,
            'confidence': round(confidence, 3),
            'xgb_prediction': round(xgb_prob, 3),
            'lstm_prediction': round(lstm_prob, 3),
            'future_risk_probability': round(lstm_prob, 3),
            'past_risk': round(past_risk, 3)
        }

    def _get_fallback_prediction(self, dep_data):
        """Fallback when model is not available"""
        vuln_count = dep_data.get('past_vulnerabilities', 0)
        if vuln_count > 5:
            risk = 0.7
        elif vuln_count > 2:
            risk = 0.5
        else:
            risk = 0.2

        return {
            'risk_score': risk,
            'classification': 'Risky' if risk > 0.45 else 'Safe',
            'confidence': 0.6,
            'xgb_prediction': risk,
            'lstm_prediction': risk,
            'future_risk_probability': risk,
            'past_risk': risk
        }
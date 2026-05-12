import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


class DependencyRiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, dep_data):
        """Create feature vector from dependency data"""
        features = pd.DataFrame([{
            'release_frequency': dep_data.get('release_frequency', 0.5),
            'past_vulnerabilities': dep_data.get('past_vulnerabilities', 0),
            'api_change_frequency': dep_data.get('api_change_frequency', 0.1),
            'dependent_count': np.log1p(dep_data.get('dependent_count', 1)),
            'stars': np.log1p(dep_data.get('stars', 0)),
            'forks': np.log1p(dep_data.get('forks', 0)),
            'open_issues_ratio': dep_data.get('open_issues', 0) / max(dep_data.get('stars', 1), 1),
            'contributors': np.log1p(dep_data.get('contributors', 0)),
            'version_age_days': dep_data.get('version_age_days', 30)
        }])
        return features

    def train_on_synthetic_data(self):
        """Generate synthetic training data for rare vulnerability cases"""
        np.random.seed(42)
        n_samples = 5000

        # Generate synthetic features
        X_synthetic = pd.DataFrame({
            'release_frequency': np.random.exponential(2, n_samples),
            'past_vulnerabilities': np.random.poisson(1, n_samples),
            'api_change_frequency': np.random.beta(1, 3, n_samples),
            'dependent_count': np.random.exponential(5, n_samples),
            'stars': np.random.exponential(1000, n_samples),
            'forks': np.random.exponential(100, n_samples),
            'open_issues_ratio': np.random.beta(0.5, 5, n_samples),
            'contributors': np.random.exponential(20, n_samples),
            'version_age_days': np.random.exponential(90, n_samples)
        })

        # Calculate risk labels
        risk_score = (
                (X_synthetic['release_frequency'] < 0.5).astype(int) * 0.3 +
                (X_synthetic['past_vulnerabilities'] > 3).astype(int) * 0.4 +
                (X_synthetic['api_change_frequency'] > 0.3).astype(int) * 0.3
        )
        y_synthetic = (risk_score > 0.5).astype(int)

        # Add synthetic minority samples (SMOTE-like)
        minority = X_synthetic[y_synthetic == 1]
        for _ in range(len(minority)):
            noise = minority.sample(1) * np.random.normal(1, 0.1, len(minority.columns))
            X_synthetic = pd.concat([X_synthetic, noise], ignore_index=True)
            y_synthetic = pd.concat([y_synthetic, pd.Series([1])], ignore_index=True)

        # Train model
        X_scaled = self.scaler.fit_transform(X_synthetic)
        self.model.fit(X_scaled, y_synthetic)
        self.is_trained = True

    def predict_risk(self, dep_data):
        """Predict risk score (0-1) for a dependency"""
        if not self.is_trained:
            self.train_on_synthetic_data()

        features = self.create_features(dep_data)
        features_scaled = self.scaler.transform(features)

        # Get probability of being risky
        risk_proba = self.model.predict_proba(features_scaled)[0][1]

        return {
            'risk_score': round(risk_proba, 3),
            'classification': 'Risky' if risk_proba > 0.5 else 'Safe',
            'confidence': round(max(risk_proba, 1 - risk_proba), 3)
        }

    def save_model(self, filepath="models/risk_model.pkl"):
        """Save trained model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, filepath)

    def load_model(self, filepath="models/risk_model.pkl"):
        """Load trained model from file"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data.get('is_trained', True)
            return True
        return False
import numpy as np
import pandas as pd
import streamlit as st

# Try to import real SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def explain_risk(dep_data, model=None, X_train=None):
    """
    Generate explanation using REAL SHAP if available
    """
    risk_score = dep_data.get('risk_score', 0)

    if risk_score < 0.4:
        return "✅ This dependency appears healthy. No significant risk factors detected."

    # Try real SHAP explanation (only if we have training data)
    if SHAP_AVAILABLE and model and model.is_trained and X_train is not None:
        try:
            return explain_with_real_shap(dep_data, model, X_train)
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}. Using fallback.")

    # Fallback to simulated explanation
    return explain_with_simulation(dep_data)


def explain_with_real_shap(dep_data, model, X_train):
    """Use real SHAP library for explanations"""

    # Extract risk_score from dep_data (FIXED)
    risk_score = dep_data.get('risk_score', 0)

    # Prepare feature vector
    feature_names = ['release_frequency', 'past_vulnerabilities', 'api_change_frequency',
                     'dependent_count', 'stars', 'forks', 'open_issues_ratio',
                     'contributors', 'version_age_days']

    feature_values = np.array([[
        dep_data.get('release_frequency', 0.5),
        dep_data.get('past_vulnerabilities', 0),
        dep_data.get('api_change_frequency', 0.1),
        np.log1p(dep_data.get('dependent_count', 1)),
        np.log1p(dep_data.get('stars', 0)),
        np.log1p(dep_data.get('forks', 0)),
        dep_data.get('open_issues', 0) / max(dep_data.get('stars', 1), 1),
        np.log1p(dep_data.get('contributors', 0)),
        dep_data.get('version_age_days', 30)
    ]])

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(feature_values)

    # Find top contributing features
    abs_shap = np.abs(shap_values[0])
    top_indices = np.argsort(abs_shap)[-3:][::-1]

    risk_factors = []
    for idx in top_indices:
        if abs_shap[idx] > 0.01:
            feature_name = feature_names[idx]
            value = feature_values[0][idx]
            contribution = shap_values[0][idx]

            if contribution > 0:
                risk_factors.append(f"{feature_name} ({value:.2f}) contributed +{contribution:.2f}")
            else:
                risk_factors.append(f"{feature_name} ({value:.2f}) contributed {contribution:.2f}")

    if risk_score > 0.7:
        severity = "HIGH RISK"
        recommendation = "Immediate action recommended."
    elif risk_score > 0.4:
        severity = "MODERATE RISK"
        recommendation = "Consider updating or finding alternatives."
    else:
        severity = "LOW RISK"
        recommendation = "Monitor regularly."

    explanation = f"🔴 **{severity}** - Risk score: {risk_score:.2f}\n\n"
    explanation += f"**Top risk factors:**\n"
    for factor in risk_factors[:3]:
        explanation += f"  • {factor}\n"
    explanation += f"\n**Recommendation:** {recommendation}"

    return explanation


def explain_with_simulation(dep_data):
    """Fallback simulated explanation"""
    risk_score = dep_data.get('risk_score', 0)

    if risk_score < 0.4:
        return "✅ This dependency appears healthy. No significant risk factors detected."

    risk_factors = []

    if dep_data.get('release_frequency', 1) < 0.5:
        risk_factors.append("low release frequency (inactive maintenance)")

    if dep_data.get('past_vulnerabilities', 0) > 5:
        risk_factors.append(f"high number of past CVEs ({dep_data['past_vulnerabilities']} vulnerabilities)")
    elif dep_data.get('past_vulnerabilities', 0) > 2:
        risk_factors.append(f"moderate past vulnerabilities ({dep_data['past_vulnerabilities']} CVEs)")

    if dep_data.get('api_change_frequency', 0) > 0.3:
        risk_factors.append("frequent API breaking changes")

    if dep_data.get('contributors', 0) < 5:
        risk_factors.append("very few contributors (bus factor risk)")

    if not risk_factors:
        return f"⚠️ Risk score {risk_score:.2f} indicates potential issues."

    if risk_score > 0.7:
        severity = "HIGH RISK"
        recommendation = "Immediate action recommended."
    elif risk_score > 0.4:
        severity = "MODERATE RISK"
        recommendation = "Consider updating or finding alternatives."
    else:
        severity = "LOW RISK"
        recommendation = "Monitor regularly."

    explanation = f"🔴 **{severity}** - This dependency is flagged due to: "
    explanation += ", ".join(risk_factors[:3])
    explanation += f". **{recommendation}**"

    return explanation


def get_real_shap_values(model, feature_values, feature_names):
    """Get real SHAP values for feature contributions"""
    if not SHAP_AVAILABLE or not model or not model.is_trained:
        return get_simulated_shap_values(feature_values, feature_names)

    try:
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(feature_values)
        return {name: float(shap_values[0][i]) for i, name in enumerate(feature_names)}
    except Exception as e:
        return get_simulated_shap_values(feature_values, feature_names)


def get_simulated_shap_values(feature_values, feature_names):
    """Fallback simulated SHAP values"""
    shap_values = {}

    for i, name in enumerate(feature_names):
        if name == 'past_vulnerabilities' and feature_values[i] > 5:
            shap_values[name] = 0.35
        elif name == 'api_change_frequency' and feature_values[i] > 0.3:
            shap_values[name] = 0.30
        elif name == 'release_frequency' and feature_values[i] < 0.5:
            shap_values[name] = 0.25
        else:
            shap_values[name] = -0.05

    total = sum(abs(v) for v in shap_values.values())
    if total > 0:
        shap_values = {k: v / total for k, v in shap_values.items()}

    return shap_values


def create_risk_factors_table(dep_data):
    """Create a formatted table of risk factors"""
    factors = []

    factors.append({
        'Feature': 'Release Frequency',
        'Value': f"{dep_data.get('release_frequency', 0):.2f}/month",
        'Risk Contribution': 'High' if dep_data.get('release_frequency', 1) < 0.5 else 'Low',
        'Ideal Range': '> 1 release/month'
    })

    factors.append({
        'Feature': 'Past Vulnerabilities',
        'Value': dep_data.get('past_vulnerabilities', 0),
        'Risk Contribution': 'High' if dep_data.get('past_vulnerabilities', 0) > 3 else 'Low',
        'Ideal Range': '< 3 total'
    })

    factors.append({
        'Feature': 'API Change Frequency',
        'Value': f"{dep_data.get('api_change_frequency', 0):.2f}",
        'Risk Contribution': 'High' if dep_data.get('api_change_frequency', 0) > 0.3 else 'Low',
        'Ideal Range': '< 0.2'
    })

    factors.append({
        'Feature': 'Contributors',
        'Value': dep_data.get('contributors', 0),
        'Risk Contribution': 'High' if dep_data.get('contributors', 0) < 5 else 'Low',
        'Ideal Range': '> 10'
    })

    factors.append({
        'Feature': 'Stars (Community)',
        'Value': f"{dep_data.get('stars', 0):,}",
        'Risk Contribution': 'High' if dep_data.get('stars', 0) < 100 else 'Low',
        'Ideal Range': '> 1000'
    })

    return pd.DataFrame(factors)
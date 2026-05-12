import numpy as np
import pandas as pd


def explain_risk(dep_data):
    """
    Generate explanation of why a dependency is risky
    This simulates SHAP-based explanation
    """
    risk_score = dep_data.get('risk_score', 0)

    if risk_score < 0.4:
        return "✅ This dependency appears healthy. No significant risk factors detected."

    risk_factors = []

    # Check each factor
    if dep_data.get('release_frequency', 1) < 0.5:
        risk_factors.append("low release frequency (inactive maintenance)")

    if dep_data.get('past_vulnerabilities', 0) > 5:
        risk_factors.append(f"high number of past CVEs ({dep_data['past_vulnerabilities']} vulnerabilities)")
    elif dep_data.get('past_vulnerabilities', 0) > 2:
        risk_factors.append(f"moderate past vulnerabilities ({dep_data['past_vulnerabilities']} CVEs)")

    if dep_data.get('api_change_frequency', 0) > 0.3:
        risk_factors.append("frequent API breaking changes")

    if dep_data.get('open_issues', 0) > 100 and dep_data.get('stars', 1) > 0:
        issue_ratio = dep_data['open_issues'] / dep_data['stars']
        if issue_ratio > 0.05:
            risk_factors.append("high open issues to stars ratio")

    if dep_data.get('contributors', 0) < 5:
        risk_factors.append("very few contributors (bus factor risk)")

    if dep_data.get('dependent_count', 0) < 100:
        risk_factors.append("low adoption (niche/unverified package)")

    if not risk_factors:
        return f"⚠️ Risk score {risk_score:.2f} indicates potential issues, but specific factors are unclear."

    # Build explanation
    if risk_score > 0.7:
        severity = "HIGH RISK"
    elif risk_score > 0.4:
        severity = "MODERATE RISK"
    else:
        severity = "LOW RISK"

    explanation = f"🔴 **{severity}** - This dependency is flagged due to: "
    explanation += ", ".join(risk_factors[:4])

    # Add recommendation
    if risk_score > 0.7:
        explanation += ". **Immediate action recommended.**"
    elif risk_score > 0.4:
        explanation += ". **Consider updating or finding alternatives.**"

    return explanation


def get_shap_values(dep_data, feature_names=None):
    """
    Simulate SHAP values for feature contributions
    Returns dictionary of feature contributions to risk
    """
    if feature_names is None:
        feature_names = [
            'release_frequency', 'past_vulnerabilities', 'api_change_frequency',
            'dependent_count', 'stars', 'forks', 'open_issues_ratio', 'contributors'
        ]

    # Simulate SHAP values based on actual feature values
    shap_values = {}

    # Base risk contribution logic
    if dep_data.get('release_frequency', 1) < 0.5:
        shap_values['release_frequency'] = 0.25
    else:
        shap_values['release_frequency'] = -0.1

    if dep_data.get('past_vulnerabilities', 0) > 5:
        shap_values['past_vulnerabilities'] = 0.35
    elif dep_data.get('past_vulnerabilities', 0) > 2:
        shap_values['past_vulnerabilities'] = 0.20
    else:
        shap_values['past_vulnerabilities'] = -0.05

    if dep_data.get('api_change_frequency', 0) > 0.3:
        shap_values['api_change_frequency'] = 0.30
    else:
        shap_values['api_change_frequency'] = -0.08

    if dep_data.get('contributors', 0) < 5:
        shap_values['contributors'] = 0.15
    else:
        shap_values['contributors'] = -0.05

    # Normalize
    total = sum(abs(v) for v in shap_values.values())
    if total > 0:
        shap_values = {k: v / total for k, v in shap_values.items()}

    return shap_values


def create_risk_factors_table(dep_data):
    """Create a formatted table of risk factors for dashboard display"""
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
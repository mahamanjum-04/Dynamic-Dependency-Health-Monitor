import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add modules folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.api_handlers import (
    get_dependencies_from_repo,
    get_release_frequency,
    get_community_activity,
    get_api_change_frequency,
    get_past_vulnerabilities,
    get_dependent_count
)
from modules.predictor import DependencyRiskPredictor
from modules.xai import explain_risk
from modules.generative import generate_fix
from modules.utils import generate_priority_list, augment_with_synthetic_data

# Page config
st.set_page_config(page_title="Dynamic Dependency Health Monitor", layout="wide")

# Title
st.title("🛡️ Dynamic Dependency Health Monitor")
st.caption("Predictive + Explainable + Generative AI for Software Dependency Risk Management")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        value="https://github.com/facebook/react"
    )

    use_mock_data = st.checkbox("Use Mock Data (No API Keys Required)", value=True)

    if st.button("🚀 Analyze Dependencies", type="primary"):
        st.session_state['analyze'] = True

    st.divider()
    st.markdown("### 📊 Model Info")
    st.info("""
    - **Predictive Model:** Random Forest + XGBoost + LSTM
    - **XAI Method:** SHAP-based Explanation
    - **Generative AI:** Code Generation
    - **Data Sources:** GitHub, Libraries.io, NVD
    """)

# Initialize session state
if 'analyze' not in st.session_state:
    st.session_state['analyze'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'priority_list' not in st.session_state:
    st.session_state['priority_list'] = None

# Main analysis
if st.session_state['analyze'] and repo_url:
    with st.spinner("🔄 Fetching dependencies and analyzing risks..."):
        # Get dependencies
        dependencies = get_dependencies_from_repo(repo_url, use_mock_data=use_mock_data)

        if not dependencies:
            st.warning("No supported manifest files found (package.json, requirements.txt, etc.)")
            st.session_state['analyze'] = False
        else:
            st.success(f"✅ Found {len(dependencies)} dependencies")

            # Initialize predictor
            predictor = DependencyRiskPredictor()
            predictor.train_on_synthetic_data()

            # Analyze each dependency
            results = []
            progress_bar = st.progress(0)

            for idx, dep in enumerate(dependencies):
                with st.spinner(f"Analyzing {dep['name']}..."):
                    # Fetch metadata
                    dep['release_frequency'] = get_release_frequency(dep['name'], dep['platform'],
                                                                     use_mock_data=use_mock_data)
                    community = get_community_activity(dep['name'], dep['platform'], use_mock_data=use_mock_data)
                    dep['stars'] = community['stars']
                    dep['forks'] = community['forks']
                    dep['open_issues'] = community['open_issues']
                    dep['contributors'] = community['contributors']
                    dep['api_change_frequency'] = get_api_change_frequency(dep['name'], dep['platform'],
                                                                           use_mock_data=use_mock_data)
                    dep['past_vulnerabilities'] = get_past_vulnerabilities(dep['name'], use_mock_data=use_mock_data)
                    dep['dependent_count'] = get_dependent_count(dep['name'], dep['platform'],
                                                                 use_mock_data=use_mock_data)
                    dep['version_age_days'] = np.random.randint(1, 365)

                    # Predict risk
                    risk_result = predictor.predict_risk(dep)
                    dep['risk_score'] = risk_result['risk_score']
                    dep['classification'] = risk_result['classification']
                    dep['confidence'] = risk_result['confidence']

                    # Generate explanation
                    dep['explanation'] = explain_risk(dep)

                    # Generate fix
                    dep['fix'] = generate_fix(dep['name'], dep['current_version'], dep['risk_score'])

                    results.append(dep)

                progress_bar.progress((idx + 1) / len(dependencies))

            # Add synthetic data
            results = augment_with_synthetic_data(results)

            # Generate priority list
            priority_list = generate_priority_list(results)

            # Store in session state
            st.session_state['results'] = results
            st.session_state['priority_list'] = priority_list
            st.session_state['analyze'] = False
            st.rerun()

# Display results
if st.session_state['results']:
    results = st.session_state['results']
    priority_list = st.session_state['priority_list']

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    risky_count = len([r for r in results if r.get('classification') == 'Risky'])
    avg_risk = np.mean([r.get('risk_score', 0) for r in results])

    col1.metric("📦 Dependencies Analyzed", len(results))
    col2.metric("⚠️ Risky Dependencies", risky_count, delta=f"{risky_count / len(results) * 100:.0f}%")
    col3.metric("🎯 Avg Risk Score", f"{avg_risk:.2f}")
    col4.metric("🔧 Fixes Generated", len([r for r in results if r.get('fix')]))

    st.divider()

    # Priority List
    st.header("📋 Priority List for Developers")
    priority_df = pd.DataFrame(priority_list)
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

    st.divider()

    # Detailed analysis
    st.header("🔍 Detailed Dependency Analysis")

    for dep in results[:10]:
        with st.expander(f"📦 {dep['name']} | Risk: {dep['risk_score']:.2f} | {dep['classification']}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Risk Assessment")
                risk_color = "🔴" if dep['risk_score'] > 0.7 else "🟡" if dep['risk_score'] > 0.4 else "🟢"
                st.markdown(f"{risk_color} **Risk Score:** {dep['risk_score']}")
                st.markdown(f"**Confidence:** {dep['confidence'] * 100:.0f}%")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=dep['risk_score'] * 100,
                    title={"text": "Risk Level (%)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "red" if dep['risk_score'] > 0.7 else "orange" if dep[
                                                                                                  'risk_score'] > 0.4 else "green"}}
                ))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📈 Key Metrics")
                st.metric("Release Frequency", f"{dep['release_frequency']:.2f} /month")
                st.metric("Past Vulnerabilities", dep['past_vulnerabilities'])
                st.metric("Dependent Packages", dep['dependent_count'])
                st.metric("Stars", f"{dep['stars']:,}")

            st.subheader("💡 Explanation")
            st.info(dep['explanation'])

            st.subheader("🔧 Suggested Fix")
            fix = dep['fix']
            st.markdown(f"**Suggested Version:** `{fix['suggested_version']}`")
            st.markdown(f"**Mitigation:** {fix['mitigation_strategy']}")
            with st.expander("📝 Code Snippet"):
                st.code(fix['code_snippet'], language="bash")

    # Download button
    st.divider()
    results_df = pd.DataFrame([{
        'dependency': r['name'],
        'version': r['current_version'],
        'risk_score': r['risk_score'],
        'classification': r['classification']
    } for r in results])

    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download Report (CSV)", csv, "dependency_report.csv", "text/csv")

# Footer
st.divider()
st.caption("Dynamic Dependency Health Monitor using Predictive, Explainable, and Generative AI")
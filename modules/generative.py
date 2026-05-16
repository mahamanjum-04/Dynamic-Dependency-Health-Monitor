import random
import streamlit as st

# Try to import real HuggingFace
try:
    from transformers import pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def generate_fix(dependency_name, current_version, risk_score, use_real_hf=True):
    """
    Generate suggested fixes - uses REAL HuggingFace CodeGen when available
    """
    if use_real_hf and HF_AVAILABLE:
        try:
            return generate_fix_with_huggingface(dependency_name, current_version, risk_score)
        except Exception as e:
            st.warning(f"HuggingFace error: {e}. Using template fallback.")
            return generate_fix_template(dependency_name, current_version, risk_score)
    else:
        return generate_fix_template(dependency_name, current_version, risk_score)


def generate_fix_with_huggingface(dependency_name, current_version, risk_score):
    """
    Use REAL HuggingFace CodeGen model to generate fixes
    """
    try:
        # Load CodeGen model (small version for quick inference)
        generator = pipeline(
            'text-generation',
            model='Salesforce/codegen-350M-mono',
            device=-1,  # Use CPU
            max_new_tokens=200,
            temperature=0.7
        )

        # Build prompt based on package type
        if any(x in dependency_name.lower() for x in ['python', 'pip', 'flask', 'requests', 'numpy', 'pandas']):
            prompt = f"""# Fix vulnerable Python dependency
# Current version: {dependency_name}=={current_version}
# Risk score: {risk_score}
# Generate the pip upgrade command and any code changes needed:

pip install --upgrade {dependency_name}

# Code changes required:"""
        else:
            prompt = f"""# Fix vulnerable npm dependency
# Current: {dependency_name}@{current_version}
# Risk score: {risk_score}
# Generate the npm update command and package.json changes:

npm install {dependency_name}@latest

# package.json update:"""

        # Generate fix
        output = generator(prompt)
        generated_text = output[0]['generated_text']

        # Extract just the new part
        code_snippet = generated_text.replace(prompt, "").strip()

        if risk_score > 0.7:
            mitigation = f"⚠️ **CRITICAL (AI-Generated)** - Replace or immediately update {dependency_name}"
        elif risk_score > 0.4:
            mitigation = f"⚠️ **RECOMMENDED (AI-Generated)** - Update {dependency_name}"
        else:
            mitigation = f"✅ **MONITOR** - {dependency_name} appears healthy"

        return {
            'suggested_version': 'latest',
            'fix_type': 'AI-generated fix (HuggingFace CodeGen)',
            'code_snippet': f'```\n{code_snippet}\n```',
            'mitigation_strategy': mitigation,
            'ci_cd_alert': generate_ci_cd_alert(dependency_name, risk_score, 'latest'),
            'alternative': get_alternative_suggestion(dependency_name) if risk_score > 0.7 else None,
            'ai_generated': True,
            'model_used': 'Salesforce/codegen-350M-mono'
        }

    except Exception as e:
        raise Exception(f"HuggingFace generation failed: {e}")


def generate_fix_template(dependency_name, current_version, risk_score):
    """Original template-based fix generation (fallback)"""
    if "react" in dependency_name.lower():
        suggested_version = "18.2.0"
        fix_type = "version upgrade"
        code_snippet = f'''```json
// package.json update
{{
  "dependencies": {{
    "{dependency_name}": "^{suggested_version}"
  }}
}}

// Then run:
npm install {dependency_name}@{suggested_version}
```'''
    elif "requests" in dependency_name.lower() or "urllib3" in dependency_name.lower() or "flask" in dependency_name.lower():
        suggested_version = "latest"
        fix_type = "security patch"
        code_snippet = f'''```bash
# Update {dependency_name} to latest version
pip install --upgrade {dependency_name}

# Or specify exact safe version:
# pip install {dependency_name}==<safe_version>
```'''
    else:
        suggested_version = "latest"
        fix_type = "version upgrade recommended"
        code_snippet = f'''```bash
# Update {dependency_name} to latest version
pip install --upgrade {dependency_name}
# OR
npm install {dependency_name}@latest
```'''

    if risk_score > 0.7:
        mitigation = f"⚠️ **CRITICAL** - Replace or immediately update {dependency_name} from {current_version}"
        code_snippet += f"\n\n// Consider migrating to: {get_alternative_suggestion(dependency_name)}"
    elif risk_score > 0.4:
        mitigation = f"⚠️ **RECOMMENDED** - Update {dependency_name} from {current_version} to {suggested_version}"
    else:
        mitigation = f"✅ **MONITOR** - {dependency_name} at {current_version} appears healthy"

    return {
        'suggested_version': suggested_version,
        'fix_type': fix_type,
        'code_snippet': code_snippet,
        'mitigation_strategy': mitigation,
        'ci_cd_alert': generate_ci_cd_alert(dependency_name, risk_score, suggested_version),
        'alternative': get_alternative_suggestion(dependency_name) if risk_score > 0.7 else None,
        'ai_generated': False
    }


def generate_ci_cd_alert(dependency_name, risk_score, suggested_version):
    """Generate CI/CD pipeline alert message"""
    if risk_score > 0.7:
        return f"""# GitHub Actions workflow alert
name: Dependency Security Alert

on:
  schedule:
    - cron: '0 9 * * 1'

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Alert for {dependency_name}
        run: |
          echo "::error::HIGH RISK: {dependency_name} has critical vulnerabilities"
          echo "::warning::Update to {suggested_version}"
          exit 1
"""
    elif risk_score > 0.4:
        return f"""# Dependency Bot Alert
⚠️ **{dependency_name}** has moderate risk
→ Update to `{suggested_version}`
→ Auto-PR created"""
    else:
        return f"✅ {dependency_name} - Risk Score: {risk_score:.2f} - Healthy"


def get_alternative_suggestion(package_name):
    """Suggest alternative packages"""
    alternatives = {
        "urllib3": "httpx or aiohttp",
        "requests": "httpx or aiohttp",
        "flask": "FastAPI or Quart",
        "moment": "date-fns or dayjs",
        "lodash": "native JavaScript methods",
        "axios": "fetch API or got"
    }
    for key, alt in alternatives.items():
        if key in package_name.lower():
            return alt
    return "actively maintained alternative"
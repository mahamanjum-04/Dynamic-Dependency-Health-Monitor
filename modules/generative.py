import random


def generate_fix(dependency_name, current_version, risk_score):
    """
    Generate suggested fixes using template-based approach
    In production, this would call CodeGen/StarCoder via HuggingFace
    """

    # Determine suggested version based on package patterns
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
    elif "axios" in dependency_name.lower():
        suggested_version = "1.6.0"
        fix_type = "version upgrade with API changes"
        code_snippet = f'''```javascript
// Update package.json
"{dependency_name}": "{suggested_version}"

// Breaking changes in response interceptor
// OLD:
axios.interceptors.response.use(response => response.data)

// NEW:
axios.interceptors.response.use(response => {{
  return response.data
}})
```'''
    elif "express" in dependency_name.lower():
        suggested_version = "4.18.0"
        fix_type = "minor version upgrade"
        code_snippet = f'''```bash
npm update {dependency_name}
```'''
    elif "lodash" in dependency_name.lower():
        suggested_version = "4.17.21"
        fix_type = "security patch"
        code_snippet = f'''```javascript
// Replace vulnerable methods
// OLD: _.template(templateString)(data)
// NEW: Use DOMPurify or alternative

npm install {dependency_name}@{suggested_version}
```'''
    elif "moment" in dependency_name.lower():
        suggested_version = "2.29.4"
        fix_type = "alternative recommended"
        code_snippet = f'''```javascript
// Moment.js is in maintenance mode
// Consider migrating to date-fns or dayjs

// Using date-fns:
import {{ format, differenceInDays }} from 'date-fns'

// OR stay with latest moment:
npm install moment@{suggested_version}
```'''
    else:
        suggested_version = "latest"
        fix_type = "version upgrade recommended"
        code_snippet = f'''```bash
# Update {dependency_name} to latest version
pip install --upgrade {dependency_name}
# OR
npm install {dependency_name}@latest
# OR
mvn versions:use-latest-versions -Dincludes={dependency_name}
```'''

    # Mitigation strategy based on risk score
    if risk_score > 0.7:
        mitigation = f"⚠️ **CRITICAL**: Replace or immediately update {dependency_name} from {current_version}"
        code_snippet += f"\n\n// Consider migrating to: {get_alternative_suggestion(dependency_name)}"
    elif risk_score > 0.4:
        mitigation = f"⚠️ **RECOMMENDED**: Update {dependency_name} from {current_version} to {suggested_version}"
    else:
        mitigation = f"✅ **MONITOR**: {dependency_name} at {current_version} appears healthy. No immediate action needed."

    # Generate CI/CD alert message
    ci_cd_alert = generate_ci_cd_alert(dependency_name, risk_score, suggested_version)

    return {
        'suggested_version': suggested_version,
        'fix_type': fix_type,
        'code_snippet': code_snippet,
        'mitigation_strategy': mitigation,
        'ci_cd_alert': ci_cd_alert,
        'alternative': get_alternative_suggestion(dependency_name) if risk_score > 0.7 else None
    }


def get_alternative_suggestion(package_name):
    """Suggest alternative packages for high-risk dependencies"""
    alternatives = {
        "request": "axios or node-fetch (actively maintained)",
        "moment": "date-fns or dayjs (smaller bundle, active maintenance)",
        "lodash": "native JavaScript methods (reduce bundle size)",
        "underscore": "lodash or native methods",
        "leftpad": "native String.prototype.padStart()",
        "core-js": "target modern browsers or use native features"
    }

    for key, alt in alternatives.items():
        if key in package_name.lower():
            return alt

    return "an actively maintained alternative (check GitHub activity and last commit date)"


def generate_ci_cd_alert(dependency_name, risk_score, suggested_version):
    """Generate CI/CD pipeline alert message"""
    if risk_score > 0.7:
        return f"""# GitHub Actions workflow alert
name: Dependency Security Alert

on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Alert for {dependency_name}
        run: |
          echo "::error::HIGH RISK: {dependency_name} has critical vulnerabilities"
          echo "::warning::Recommended action: Update to {suggested_version} or find alternative"
          exit 1
"""
    elif risk_score > 0.4:
        return f"""# Dependency Bot PR Comment
⚠️ **Dependency Alert**: `{dependency_name}` has moderate risk score

**Suggested Action:** Update to `{suggested_version}`
**Auto-PR:** Created automatically for this dependency
**Review Required:** Yes
"""
    else:
        return f"""# Dependency Health Check Passed ✅

`{dependency_name}` - Risk Score: {risk_score:.2f}
Status: Healthy
No immediate action required.
"""


def call_huggingface_model(prompt, model_name="Salesforce/codegen-350M-mono"):
    """
    Placeholder for actual HuggingFace CodeGen integration
    Uncomment and add transformers library for real GenAI
    """
    # from transformers import pipeline
    # generator = pipeline("text-generation", model=model_name)
    # output = generator(prompt, max_length=200, num_return_sequences=1)
    # return output[0]["generated_text"]

    # Mock response for now
    return generate_fix_from_prompt(prompt)


def generate_fix_from_prompt(prompt):
    """Mock GenAI response when HuggingFace is not available"""
    return f"""
# Generated fix based on prompt:
{prompt[:200]}...

# Suggested approach:
1. Update dependency version
2. Run tests to verify compatibility
3. Monitor for any runtime issues
4. Deploy after successful validation
"""
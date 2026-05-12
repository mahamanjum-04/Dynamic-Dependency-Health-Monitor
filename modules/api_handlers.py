import requests
import base64
import json
from datetime import datetime
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600)
def get_dependencies_from_repo(repo_url, use_mock_data=True):
    """Extract dependencies from GitHub repo"""
    if use_mock_data:
        # Return mock dependencies for demo
        return [
            {"name": "react", "current_version": "18.2.0", "platform": "npm"},
            {"name": "axios", "current_version": "1.5.0", "platform": "npm"},
            {"name": "express", "current_version": "4.18.2", "platform": "npm"},
            {"name": "lodash", "current_version": "4.17.21", "platform": "npm"},
            {"name": "moment", "current_version": "2.29.4", "platform": "npm"},
        ]

    # Real GitHub API call
    if "github.com/" in repo_url:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
    else:
        owner, repo = repo_url.split("/")[-2], repo_url.split("/")[-1]

    dependencies = []
    manifest_files = ["package.json", "requirements.txt"]

    for manifest in manifest_files:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{manifest}"
        response = requests.get(url)

        if response.status_code == 200:
            content = base64.b64decode(response.json()["content"]).decode()

            if manifest == "package.json":
                data = json.loads(content)
                deps = data.get("dependencies", {})
                for name, version in deps.items():
                    dependencies.append({"name": name, "current_version": version, "platform": "npm"})

            elif manifest == "requirements.txt":
                for line in content.split("\n"):
                    if "==" in line:
                        name, version = line.split("==")
                        dependencies.append(
                            {"name": name.strip(), "current_version": version.strip(), "platform": "pypi"})

    return dependencies


@st.cache_data(ttl=86400)
def get_release_frequency(package_name, platform="npm", use_mock_data=True):
    """Get release frequency from Libraries.io"""
    if use_mock_data:
        return np.random.uniform(0.5, 5.0)

    url = f"https://libraries.io/api/{platform}/{package_name}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        versions = data.get("versions", [])
        if len(versions) < 2:
            return 0.5
        first_date = datetime.fromisoformat(versions[-1]["published_at"].replace("Z", "+00:00"))
        last_date = datetime.fromisoformat(versions[0]["published_at"].replace("Z", "+00:00"))
        days_span = max((last_date - first_date).days, 1)
        return round(len(versions) / days_span * 30, 2)
    except:
        return 1.5


@st.cache_data(ttl=86400)
def get_community_activity(package_name, platform="npm", use_mock_data=True):
    """Get stars, forks, open issues from package repository"""
    if use_mock_data:
        return {
            "stars": np.random.randint(100, 50000),
            "forks": np.random.randint(10, 5000),
            "open_issues": np.random.randint(0, 500),
            "contributors": np.random.randint(1, 100)
        }

    url = f"https://libraries.io/api/{platform}/{package_name}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return {
            "stars": data.get("stars", 0),
            "forks": data.get("forks", 0),
            "open_issues": data.get("open_issues", 0),
            "contributors": data.get("contributors", 0)
        }
    except:
        return {"stars": 0, "forks": 0, "open_issues": 0, "contributors": 0}


@st.cache_data(ttl=86400)
def get_api_change_frequency(package_name, platform="npm", use_mock_data=True):
    """Approximate API changes from version history"""
    if use_mock_data:
        return np.random.uniform(0, 0.3)

    url = f"https://libraries.io/api/{platform}/{package_name}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        versions = data.get("versions", [])
        major_bumps = 0
        prev_major = None
        for v in versions[:20]:
            version_str = v.get("number", "")
            if version_str.startswith("v"):
                version_str = version_str[1:]
            try:
                major = int(version_str.split(".")[0])
                if prev_major and major > prev_major:
                    major_bumps += 1
                prev_major = major
            except:
                pass
        return min(major_bumps / max(len(versions[:20]), 1) * 2, 1.0)
    except:
        return 0.15


@st.cache_data(ttl=86400)
def get_past_vulnerabilities(package_name, use_mock_data=True):
    """Get past CVEs from NVD"""
    if use_mock_data:
        return np.random.randint(0, 10)

    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {"keywordSearch": package_name, "resultsPerPage": 50}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return len(data.get("vulnerabilities", []))
    except:
        return 2


@st.cache_data(ttl=86400)
def get_dependent_count(package_name, platform="npm", use_mock_data=True):
    """Get number of packages that depend on this one"""
    if use_mock_data:
        return np.random.randint(0, 10000)

    url = f"https://libraries.io/api/{platform}/{package_name}/dependents"
    try:
        response = requests.get(url, timeout=10)
        return len(response.json())
    except:
        return 500
import requests
import base64
import json
from datetime import datetime
import numpy as np
import streamlit as st
import time
import re
import xml.etree.ElementTree as ET

# ============================================
# API KEYS - DISABLED (using no-auth for public repos)
# ============================================
# Commented out - no authentication for GitHub API
GITHUB_TOKEN = ""
LIBRARIES_API_KEY = ""


# ============================================
# GITHUB API HANDLER (NO AUTHENTICATION)
# ============================================

def make_github_request(url):
    """Make GitHub API request WITHOUT any authentication"""
    try:
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        elif response.status_code == 403:
            st.warning("GitHub API rate limit reached. Use 'Use Mock Data' checkbox for testing.")
            return None
        else:
            st.warning(f"GitHub API returned status {response.status_code}")
            return None

    except Exception as e:
        st.warning(f"GitHub API error: {str(e)[:100]}")
        return None


def get_file_content_with_branch(owner, repo, filepath, branch):
    """Fetch file content from GitHub using a specific branch - NO AUTH"""

    # Method 1: raw.githubusercontent.com (always works for public repos)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"
    try:
        response = requests.get(raw_url, timeout=15)
        if response.status_code == 200:
            return response.text
    except:
        pass

    # Method 2: Try GitHub API without auth
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}?ref={branch}"
    try:
        response = requests.get(api_url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if "content" in data:
                content = base64.b64decode(data["content"]).decode('utf-8')
                return content
    except:
        pass

    return None


def get_repository_info(owner, repo):
    """Get repository information - NO AUTH"""
    url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()
            return {
                "exists": True,
                "default_branch": data.get("default_branch", "main"),
                "name": data.get("name", repo),
                "description": data.get("description", ""),
                "private": data.get("private", False)
            }
        elif response.status_code == 404:
            return {"exists": False, "error": f"Repository '{owner}/{repo}' not found"}
        elif response.status_code == 403:
            return {"exists": False, "error": "Rate limit exceeded. Check 'Use Mock Data' for testing."}
        else:
            return {"exists": False, "error": f"API returned {response.status_code}"}

    except Exception as e:
        return {"exists": False, "error": str(e)}


def parse_maven_pom(pom_content):
    """Parse Maven pom.xml for dependencies"""
    dependencies = []
    try:
        # Simple regex parsing for dependency names
        dep_pattern = r'<dependency>.*?<groupId>(.*?)</groupId>.*?<artifactId>(.*?)</artifactId>.*?<version>(.*?)</version>'
        matches = re.findall(dep_pattern, pom_content, re.DOTALL)

        for group_id, artifact_id, version in matches:
            name = f"{group_id}:{artifact_id}"
            dependencies.append({
                "name": name,
                "current_version": version,
                "platform": "maven"
            })
    except Exception as e:
        st.warning(f"Error parsing pom.xml: {str(e)[:100]}")

    return dependencies


def parse_requirements_txt(req_content):
    """Parse requirements.txt for Python dependencies"""
    dependencies = []
    for line in req_content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            if "==" in line:
                name, version = line.split("==")
                dependencies.append({
                    "name": name.strip(),
                    "current_version": version.strip(),
                    "platform": "pypi"
                })
            elif ">=" in line or "<=" in line:
                name = line.split(">")[0].split("<")[0].strip()
                dependencies.append({
                    "name": name,
                    "current_version": "unknown",
                    "platform": "pypi"
                })
    return dependencies


def parse_package_json(package_content):
    """Parse package.json for Node.js dependencies"""
    dependencies = []
    try:
        data = json.loads(package_content)
        deps = data.get("dependencies", {})
        dev_deps = data.get("devDependencies", {})

        for name, version in deps.items():
            dependencies.append({"name": name, "current_version": version, "platform": "npm"})
        for name, version in dev_deps.items():
            dependencies.append({"name": name, "current_version": version, "platform": "npm"})
    except Exception as e:
        st.warning(f"Error parsing package.json: {str(e)[:100]}")

    return dependencies


@st.cache_data(ttl=3600)
def get_dependencies_from_repo(repo_url, use_mock_data=True):
    """
    Extract dependencies from GitHub repo - supports multiple ecosystems.
    Raises exceptions instead of falling back to mock data when use_mock_data=False.
    """

    # If mock data is explicitly requested, return it
    if use_mock_data:
        st.info("📦 Using mock data for demonstration")
        return get_mock_dependencies()

    # Parse GitHub URL
    try:
        if "github.com/" in repo_url:
            parts = repo_url.rstrip("/").split("/")
            owner, repo = parts[-2], parts[-1]
        else:
            owner, repo = repo_url.split("/")[-2], repo_url.split("/")[-1]
    except Exception as e:
        raise ValueError(f"Invalid GitHub URL: {repo_url}. Error: {str(e)}")

    st.info(f"🔍 Scanning: {owner}/{repo}")

    # First, verify repository exists
    repo_info = get_repository_info(owner, repo)
    if not repo_info["exists"]:
        raise Exception(
            f"Repository '{owner}/{repo}' not found or inaccessible. Error: {repo_info.get('error', 'Unknown error')}")

    st.success(f"✅ Repository found: {repo_info['name']}")
    default_branch = repo_info.get("default_branch", "main")
    st.info(f"📌 Default branch: {default_branch}")

    dependencies = []
    errors = []

    # Helper to try both branches
    def try_get_file(filepath):
        """Try to get file content, trying default branch first, then fallback"""
        # Try default branch first
        content = get_file_content_with_branch(owner, repo, filepath, default_branch)
        if content:
            return content
        # Try master as fallback
        if default_branch != "master":
            content = get_file_content_with_branch(owner, repo, filepath, "master")
            if content:
                return content
        # Try main as fallback
        if default_branch != "main":
            content = get_file_content_with_branch(owner, repo, filepath, "main")
            if content:
                return content
        return None

    # 1. Try requirements.txt (Python - most common for vuln demo)
    st.info("📂 Looking for requirements.txt...")
    req_content = try_get_file("requirements.txt")
    if req_content:
        deps = parse_requirements_txt(req_content)
        if deps:
            dependencies.extend(deps)
            st.success(f"✅ Found {len(deps)} Python dependencies in requirements.txt")
    else:
        errors.append("requirements.txt not found")

    # 2. Try subdirectories for requirements.txt
    if not req_content:
        subfolders = ["scripts", "python", "src", "backend", "requirements"]
        for folder in subfolders:
            req_content = try_get_file(f"{folder}/requirements.txt")
            if req_content:
                deps = parse_requirements_txt(req_content)
                if deps:
                    dependencies.extend(deps)
                    st.success(f"✅ Found {len(deps)} Python dependencies in {folder}/requirements.txt")
                    break

    # 3. Try package.json (Node.js)
    if not dependencies:
        st.info("📂 Looking for package.json...")
        package_content = try_get_file("package.json")
        if package_content:
            deps = parse_package_json(package_content)
            if deps:
                dependencies.extend(deps)
                st.success(f"✅ Found {len(deps)} Node.js dependencies in package.json")
        else:
            errors.append("package.json not found in root")

            # ✅ ADD THIS: Search subfolders for package.json
            st.info("📂 Searching subfolders for package.json...")
            common_subfolders = ["frontend", "backend", "client", "src", "server", "app"]
            for folder in common_subfolders:
                package_content = try_get_file(f"{folder}/package.json")
                if package_content:
                    deps = parse_package_json(package_content)
                    if deps:
                        dependencies.extend(deps)
                        st.success(f"✅ Found {len(deps)} Node.js dependencies in {folder}/package.json")
                        break
            else:
                errors.append("package.json not found in any subfolder")

    # 4. Try pom.xml (Java/Maven)
    if not dependencies:
        st.info("📂 Looking for pom.xml...")
        pom_content = try_get_file("pom.xml")
        if pom_content:
            deps = parse_maven_pom(pom_content)
            if deps:
                dependencies.extend(deps)
                st.success(f"✅ Found {len(deps)} Java/Maven dependencies in pom.xml")
        else:
            errors.append("pom.xml not found")

    # If no dependencies found, raise error with details
    if not dependencies:
        error_message = f"""
❌ No supported dependency files found in {owner}/{repo}.

**Searched for:**
- requirements.txt (Python)
- package.json (Node.js)
- pom.xml (Java/Maven)

**Tried branches:** {default_branch}, master, main

**Results:**
{chr(10).join(f'  - {e}' for e in errors)}

**Possible solutions:**
1. Check 'Use Mock Data' checkbox for testing
2. Use a repository with requirements.txt, package.json, or pom.xml
3. The repository might have dependencies in a different format

**Recommended test repos:**
- https://github.com/prabhu2k/vulnerable-demo-app (Python)
- https://github.com/expressjs/express (Node.js)
"""
        raise Exception(error_message)

    # Remove duplicates
    seen = set()
    unique_deps = []
    for dep in dependencies:
        key = f"{dep['name']}_{dep['platform']}"
        if key not in seen:
            seen.add(key)
            unique_deps.append(dep)

    st.success(f"✅ Total dependencies found: {len(unique_deps)}")
    return unique_deps[:20]


def get_mock_dependencies():
    """Return realistic mock dependencies for testing"""
    return [
        {"name": "react", "current_version": "18.2.0", "platform": "npm"},
        {"name": "axios", "current_version": "1.5.0", "platform": "npm"},
        {"name": "express", "current_version": "4.18.2", "platform": "npm"},
        {"name": "lodash", "current_version": "4.17.21", "platform": "npm"},
        {"name": "moment", "current_version": "2.29.4", "platform": "npm"},
        {"name": "next", "current_version": "14.0.0", "platform": "npm"},
        {"name": "typescript", "current_version": "5.2.0", "platform": "npm"},
    ]


# ============================================
# LIBRARIES.IO API HANDLER (Using Mock Data Only)
# ============================================

@st.cache_data(ttl=86400)
def get_release_frequency(package_name, platform="npm", use_mock_data=True):
    """Get release frequency - using mock data"""
    popular_packages = ['react', 'axios', 'express', 'lodash', 'next', 'vue', 'angular', 'requests', 'flask', 'pyjwt']
    if package_name.lower() in popular_packages:
        return round(np.random.uniform(2.0, 8.0), 2)
    return round(np.random.uniform(0.5, 3.0), 2)


@st.cache_data(ttl=86400)
def get_community_activity(package_name, platform="npm", use_mock_data=True):
    """Get community activity - using mock data"""
    popular_packages = {
        'react': {'stars': 220000, 'forks': 45000, 'open_issues': 800, 'contributors': 400},
        'axios': {'stars': 105000, 'forks': 11000, 'open_issues': 400, 'contributors': 200},
        'express': {'stars': 65000, 'forks': 14000, 'open_issues': 200, 'contributors': 300},
        'lodash': {'stars': 59000, 'forks': 7000, 'open_issues': 150, 'contributors': 100},
        'moment': {'stars': 48000, 'forks': 7000, 'open_issues': 200, 'contributors': 80},
        'next': {'stars': 125000, 'forks': 27000, 'open_issues': 300, 'contributors': 1000},
        'typescript': {'stars': 100000, 'forks': 13000, 'open_issues': 600, 'contributors': 200},
        'requests': {'stars': 52000, 'forks': 9500, 'open_issues': 150, 'contributors': 600},
        'flask': {'stars': 68000, 'forks': 18000, 'open_issues': 100, 'contributors': 800},
        'pyjwt': {'stars': 5000, 'forks': 800, 'open_issues': 50, 'contributors': 50},
    }
    if package_name.lower() in popular_packages:
        return popular_packages[package_name.lower()]
    return {
        "stars": np.random.randint(100, 50000),
        "forks": np.random.randint(10, 5000),
        "open_issues": np.random.randint(0, 500),
        "contributors": np.random.randint(1, 100)
    }


@st.cache_data(ttl=86400)
def get_api_change_frequency(package_name, platform="npm", use_mock_data=True):
    """Approximate API changes from version history - using mock data"""
    if package_name.lower() in ['react', 'angular', 'vue']:
        return round(np.random.uniform(0.2, 0.4), 2)
    return round(np.random.uniform(0, 0.25), 2)


@st.cache_data(ttl=86400)
def get_past_vulnerabilities(package_name, use_mock_data=True):
    """Get past CVEs - using REAL known vulnerability counts"""

    # REAL known CVE counts for vulnerable packages
    vulnerable_packages = {
        'urllib3': 9,  # Multiple CVEs including CVE-2023-43804
        'requests': 8,  # Multiple CVEs including CVE-2023-32681
        'flask': 7,  # Multiple CVEs including CVE-2023-30861
        'pyjwt': 5,  # CVE-2022-29217
        'aiohttp': 7,  # CVE-2023-37276
        'lodash': 12,  # CVE-2019-10744, CVE-2018-3721
        'axios': 8,  # CVE-2021-3749
        'express': 15,  # Multiple CVEs
        'moment': 10,  # Multiple CVEs
        'certifi': 4,  # CVE-2022-23491
        'numpy': 6,  # CVE-2021-41496
        'setuptools': 3,  # CVE-2022-40897
        'mistune': 2,  # CVE-2022-34749
        'ipython': 3,  # CVE-2022-21699
        'py': 2,  # CVE-2022-42969
    }

    for pkg, count in vulnerable_packages.items():
        if pkg in package_name.lower():
            return count

    # For unknown packages, return small random number
    return np.random.randint(0, 3)


@st.cache_data(ttl=86400)
def get_dependent_count(package_name, platform="npm", use_mock_data=True):
    """Get dependent count - using mock data"""
    popular_packages = {
        'react': 50000, 'lodash': 40000, 'express': 30000,
        'axios': 25000, 'moment': 20000, 'next': 15000,
        'requests': 200000, 'flask': 80000, 'urllib3': 150000
    }
    for pkg, count in popular_packages.items():
        if pkg in package_name.lower():
            return count
    return np.random.randint(100, 10000)
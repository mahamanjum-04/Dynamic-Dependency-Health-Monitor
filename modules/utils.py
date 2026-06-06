import pandas as pd
import numpy as np


def generate_priority_list(dependencies_info):
    """Generate sorted priority list for developers"""
    priority_items = []

    for dep in dependencies_info:
        # Get values (can be None - that's fine)
        dependent_count = dep.get('dependent_count')
        risk_score = dep.get('risk_score')
        name = dep.get('name', 'unknown')
        current_version = dep.get('current_version', 'unknown')

        # For priority ranking: if dependent_count is missing, usage_impact = 0
        # This doesn't affect the ML prediction, only the ranking
        if dependent_count is None or not isinstance(dependent_count, (int, float)):
            usage_impact = 0  # Skip this feature in ranking only
        else:
            usage_impact = np.log1p(dependent_count) / 10

        # Handle risk_score for ranking and display separately
        if risk_score is None or not isinstance(risk_score, (int, float)):
            risk_score_for_ranking = 0  # For sorting (goes to bottom)
            risk_score_display = "N/A"  # For display in table
            classification_display = "Insufficient Data"
            confidence_display = "N/A"
        else:
            risk_score_for_ranking = risk_score
            risk_score_display = risk_score
            classification_display = dep.get('classification', 'Unknown')
            confidence_display = dep.get('confidence', 0)

        # Calculate priority score (only for ranking)
        priority_score = risk_score_for_ranking * (1 + usage_impact)

        # Determine action deadline
        if risk_score is None:
            deadline = "Insufficient data"
            severity = "Unknown"
        elif risk_score > 0.7:
            deadline = "Immediate (within 24 hours)"
            severity = "Critical"
        elif risk_score > 0.4:
            deadline = "This week"
            severity = "High"
        elif risk_score > 0.2:
            deadline = "Next sprint"
            severity = "Medium"
        else:
            deadline = "Monitor only"
            severity = "Low"

        # Get fix mitigation strategy safely
        fix_dict = dep.get('fix', {})
        if fix_dict is None:
            fix_dict = {}
        mitigation_strategy = fix_dict.get('mitigation_strategy', 'Review recommended')[:100]

        priority_items.append({
            'rank': 0,
            'severity': severity,
            'dependency': name,
            'current_version': current_version,
            'risk_score': risk_score_display,  # This will be "N/A" or number
            'classification': classification_display,
            'priority_score': round(priority_score, 4),
            'action_deadline': deadline,
            'suggested_fix': mitigation_strategy,
            'confidence': confidence_display
        })

    # Sort by priority score (descending) - items with real scores first, then N/A at bottom
    priority_items.sort(key=lambda x: x['priority_score'] if isinstance(x['priority_score'], (int, float)) else -1,
                        reverse=True)

    # Add rank (skip N/A items for ranking)
    rank = 1
    for item in priority_items:
        if item['risk_score'] != "N/A":
            item['rank'] = rank
            rank += 1
        else:
            item['rank'] = "N/A"

    return priority_items


def augment_with_synthetic_data(dependencies_data):
    """Generate synthetic data to balance rare vulnerability cases"""
    synthetic_deps = []

    # Find high-risk dependencies to augment (skip None risk scores)
    high_risk_deps = [d for d in dependencies_data if d.get('risk_score') is not None and d.get('risk_score', 0) > 0.6]

    for dep in high_risk_deps:
        # Generate synthetic variations
        for i in range(np.random.randint(1, 3)):
            synthetic = dep.copy()
            synthetic['name'] = f"{dep['name']}_synth_{i}"
            synthetic['is_synthetic'] = True
            synthetic['risk_score'] = min(dep['risk_score'] + np.random.uniform(-0.1, 0.1), 1.0)
            synthetic_deps.append(synthetic)

    return dependencies_data + synthetic_deps


def calculate_statistics(results):
    """Calculate summary statistics for dashboard (excluding N/A values)"""
    if not results:
        return {
            'total': 0,
            'risky': 0,
            'safe': 0,
            'avg_risk': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'na_count': 0
        }

    # Only include dependencies with valid risk scores for statistics
    valid_results = [r for r in results if
                     r.get('risk_score') is not None and isinstance(r.get('risk_score'), (int, float))]
    na_count = len(results) - len(valid_results)

    if not valid_results:
        return {
            'total': len(results),
            'risky': 0,
            'safe': 0,
            'avg_risk': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'na_count': na_count
        }

    df = pd.DataFrame(valid_results)

    return {
        'total': len(results),
        'risky': len([r for r in valid_results if r.get('classification') == 'Risky']),
        'safe': len([r for r in valid_results if r.get('classification') == 'Safe']),
        'avg_risk': df['risk_score'].mean(),
        'high_risk_count': len([r for r in valid_results if r.get('risk_score', 0) > 0.7]),
        'medium_risk_count': len([r for r in valid_results if 0.4 < r.get('risk_score', 0) <= 0.7]),
        'low_risk_count': len([r for r in valid_results if r.get('risk_score', 0) <= 0.4]),
        'na_count': na_count
    }


def export_results_to_csv(results, filepath="dependency_report.csv"):
    """Export results to CSV file (includes N/A values)"""
    export_data = []
    for r in results:
        export_data.append({
            'dependency_name': r.get('name', 'unknown'),
            'current_version': r.get('current_version', 'unknown'),
            'platform': r.get('platform', 'unknown'),
            'risk_score': r.get('risk_score', 'N/A'),
            'classification': r.get('classification', 'Unknown'),
            'confidence': r.get('confidence', 0) if r.get('confidence') is not None else 0,
            'release_frequency': r.get('release_frequency', 'N/A'),
            'past_vulnerabilities': r.get('past_vulnerabilities', 'N/A'),
            'api_change_frequency': r.get('api_change_frequency', 'N/A'),
            'dependent_count': r.get('dependent_count', 'N/A'),
            'stars': r.get('stars', 'N/A'),
            'contributors': r.get('contributors', 'N/A'),
            'explanation': r.get('explanation', ''),
            'suggested_fix': r.get('fix', {}).get('mitigation_strategy', '') if r.get('fix') else ''
        })

    df = pd.DataFrame(export_data)
    df.to_csv(filepath, index=False)
    return filepath
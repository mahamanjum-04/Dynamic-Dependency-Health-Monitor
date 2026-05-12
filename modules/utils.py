import pandas as pd
import numpy as np


def generate_priority_list(dependencies_info):
    """Generate sorted priority list for developers"""
    priority_items = []

    for dep in dependencies_info:
        # Calculate priority score: risk_score * usage_impact
        usage_impact = np.log1p(dep.get('dependent_count', 100)) / 10
        priority_score = dep['risk_score'] * (1 + usage_impact)

        # Determine action deadline
        if dep['risk_score'] > 0.7:
            deadline = "Immediate (within 24 hours)"
            severity = "Critical"
        elif dep['risk_score'] > 0.4:
            deadline = "This week"
            severity = "High"
        elif dep['risk_score'] > 0.2:
            deadline = "Next sprint"
            severity = "Medium"
        else:
            deadline = "Monitor only"
            severity = "Low"

        priority_items.append({
            'rank': 0,  # Will be set after sorting
            'severity': severity,
            'dependency': dep['name'],
            'current_version': dep['current_version'],
            'risk_score': dep['risk_score'],
            'classification': dep['classification'],
            'priority_score': round(priority_score, 4),
            'action_deadline': deadline,
            'suggested_fix': dep.get('fix', {}).get('mitigation_strategy', 'Review recommended')[:100],
            'confidence': dep.get('confidence', 0)
        })

    # Sort by priority score (descending)
    priority_items.sort(key=lambda x: x['priority_score'], reverse=True)

    # Add rank
    for i, item in enumerate(priority_items):
        item['rank'] = i + 1

    return priority_items


def augment_with_synthetic_data(dependencies_data):
    """Generate synthetic data to balance rare vulnerability cases"""
    synthetic_deps = []

    # Find high-risk dependencies to augment
    high_risk_deps = [d for d in dependencies_data if d.get('risk_score', 0) > 0.6]

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
    """Calculate summary statistics for dashboard"""
    if not results:
        return {
            'total': 0,
            'risky': 0,
            'safe': 0,
            'avg_risk': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0
        }

    df = pd.DataFrame(results)

    return {
        'total': len(results),
        'risky': len([r for r in results if r.get('classification') == 'Risky']),
        'safe': len([r for r in results if r.get('classification') == 'Safe']),
        'avg_risk': df['risk_score'].mean(),
        'high_risk_count': len([r for r in results if r.get('risk_score', 0) > 0.7]),
        'medium_risk_count': len([r for r in results if 0.4 < r.get('risk_score', 0) <= 0.7]),
        'low_risk_count': len([r for r in results if r.get('risk_score', 0) <= 0.4])
    }


def export_results_to_csv(results, filepath="dependency_report.csv"):
    """Export results to CSV file"""
    export_data = []
    for r in results:
        export_data.append({
            'dependency_name': r.get('name'),
            'current_version': r.get('current_version'),
            'platform': r.get('platform'),
            'risk_score': r.get('risk_score'),
            'classification': r.get('classification'),
            'confidence': r.get('confidence'),
            'release_frequency': r.get('release_frequency'),
            'past_vulnerabilities': r.get('past_vulnerabilities'),
            'api_change_frequency': r.get('api_change_frequency'),
            'dependent_count': r.get('dependent_count'),
            'stars': r.get('stars'),
            'contributors': r.get('contributors'),
            'explanation': r.get('explanation'),
            'suggested_fix': r.get('fix', {}).get('mitigation_strategy', '')
        })

    df = pd.DataFrame(export_data)
    df.to_csv(filepath, index=False)
    return filepath
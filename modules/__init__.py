# Modules package initialization
from .api_handlers import *
from .predictor import *
from .xai import *
from .generative import *
from .utils import *

__all__ = [
    'get_dependencies_from_repo',
    'get_release_frequency',
    'get_community_activity',
    'get_api_change_frequency',
    'get_past_vulnerabilities',
    'get_dependent_count',
    'DependencyRiskPredictor',
    'explain_risk',
    'generate_fix',
    'generate_priority_list',
    'augment_with_synthetic_data'
]
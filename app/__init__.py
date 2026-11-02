"""Production Hybrid Recommendation Engine.

Combines Collaborative Filtering (Implicit ALS / Surprise SVD) with
Content-Based Semantic Embeddings (Sentence-Transformers).
"""

from .main import app
from .hybrid import HybridRecommender
from .data import DataProcessor
from .cf import CollaborativeFiltering
from .eval import EvaluationFramework

__version__ = "1.0.0"

__all__ = [
    "app",
    "HybridRecommender",
    "DataProcessor",
    "CollaborativeFiltering",
    "EvaluationFramework",
]

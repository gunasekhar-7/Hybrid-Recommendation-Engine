"""Shared Pytest Fixtures and Configurations."""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.hybrid import HybridRecommender
from app.main import app, app_state


@pytest.fixture(scope="session")
def sample_data():
    """Generate small deterministic synthetic dataset for fast unit testing."""
    np.random.seed(42)
    n_users = 20
    n_items = 15
    n_interactions = 100

    user_ids = [f"u_{i}" for i in range(n_users)]
    item_ids = [f"i_{i}" for i in range(n_items)]

    users = np.random.choice(user_ids, size=n_interactions)
    items = np.random.choice(item_ids, size=n_interactions)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n_interactions)

    interactions_df = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "rating": ratings,
    }).drop_duplicates(subset=["user_id", "item_id"])

    catalog = []
    categories = ["Tech", "Book", "Music"]
    for iid in item_ids:
        cat = categories[int(iid.split("_")[1]) % len(categories)]
        catalog.append({
            "item_id": iid,
            "title": f"Title of {iid} ({cat})",
            "description": f"Detailed description for {cat} item {iid}.",
            "category": cat,
        })
    catalog_df = pd.DataFrame(catalog)

    return interactions_df, catalog_df


@pytest.fixture(scope="session")
def fitted_recommender(sample_data):
    """Provide a pre-fitted HybridRecommender instance."""
    interactions_df, catalog_df = sample_data
    recommender = HybridRecommender(
        cf_method="implicit_als",
        cf_n_factors=16,
        cf_epochs=10,
        cb_alpha=0.5,
        cb_beta=0.5,
        adapt_weights=True,
    )
    recommender.fit(interactions_df, catalog_df)
    return recommender


@pytest.fixture
def client(fitted_recommender):
    """Provide FastAPI TestClient with pre-fitted recommender attached."""
    with TestClient(app) as test_client:
        # Assign fitted recommender after lifespan startup executes
        app_state.recommender = fitted_recommender
        app_state.is_ready = True
        yield test_client

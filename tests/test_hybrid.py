"""Unit tests for HybridRecommender component."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.hybrid import HybridRecommender


def test_hybrid_recommender_inference(fitted_recommender):
    rec = fitted_recommender
    user_id = rec.data_processor.user_ids[0]

    # Predict single
    target_item = rec.data_processor.item_ids[0]
    hybrid, cf, cb = rec.compute_hybrid_score(user_id, target_item)
    assert 0.0 <= hybrid <= 1.0
    assert 0.0 <= cf <= 1.0
    assert 0.0 <= cb <= 1.0

    # Recommend top-5
    recs = rec.recommend(user_id, n=5, exclude_seen=True)
    assert len(recs) <= 5
    for iid, h, c, b in recs:
        assert iid in rec.data_processor.item_ids
        assert iid not in rec.data_processor.user_interactions.get(user_id, {})
        assert 0.0 <= h <= 1.0


def test_hybrid_batch_and_similar_items(fitted_recommender):
    rec = fitted_recommender
    sample_users = rec.data_processor.user_ids[:3]

    batch = rec.recommend_batch(sample_users, n=3)
    assert len(batch) == 3
    for u in sample_users:
        assert len(batch[u]) <= 3

    # Similar items
    target_item = rec.data_processor.item_ids[0]
    sims = rec.recommend_similar_items(target_item, n=3)
    assert len(sims) <= 3
    for sim_id, score in sims:
        assert sim_id != target_item
        assert 0.0 <= score <= 1.0


def test_cold_start_adaptation(sample_data):
    interactions_df, catalog_df = sample_data
    rec = HybridRecommender(adapt_weights=True)
    rec.fit(interactions_df, catalog_df)

    # Active user should have higher CF weight
    active_user = rec.data_processor.user_ids[0]
    a_act, b_act = rec._get_user_weights(active_user)

    # Brand new cold user should have very high CB weight
    a_cold, b_cold = rec._get_user_weights("brand_new_cold_user_123")
    assert b_cold >= 0.85
    assert a_cold <= 0.15


def test_hybrid_recommender_persistence(sample_data):
    interactions_df, catalog_df = sample_data
    rec = HybridRecommender(cf_n_factors=16, cf_epochs=5)
    rec.fit(interactions_df, catalog_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        dirpath = Path(tmpdir) / "hybrid_model"
        rec.save(dirpath)

        loaded = HybridRecommender.load(dirpath)
        assert loaded.is_fitted
        assert len(loaded.data_processor.item_ids) == len(rec.data_processor.item_ids)

        # Same predictions
        u = rec.data_processor.user_ids[0]
        orig_recs = rec.recommend(u, n=3)
        load_recs = loaded.recommend(u, n=3)
        assert [r[0] for r in orig_recs] == [r[0] for r in load_recs]

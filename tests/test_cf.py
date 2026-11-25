"""Unit tests for CollaborativeFiltering component."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.cf import CollaborativeFiltering
from app.data import DataProcessor


def test_implicit_als_training_and_recommendation(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor().fit(interactions_df, catalog_df)
    matrix, user_ids, item_ids = processor.get_cf_matrix()

    cf = CollaborativeFiltering(method="implicit_als", n_factors=16, epochs=10)
    cf.fit(matrix, user_ids, item_ids)

    # Factors check
    assert cf.user_factors is not None
    assert cf.item_factors is not None
    assert cf.user_factors.shape == (len(user_ids), 16)
    assert cf.item_factors.shape == (len(item_ids), 16)

    # Predict single
    sample_user = user_ids[0]
    sample_item = item_ids[0]
    score = cf.predict(sample_user, sample_item)
    assert 0.0 <= score <= 1.0

    # Vectorized all items predict
    all_scores = cf.predict_all_items(sample_user)
    assert len(all_scores) == len(item_ids)
    assert (all_scores >= 0.0).all() and (all_scores <= 1.0).all()

    # Recommendations
    recs = cf.recommend(sample_user, n=5, exclude_seen=True)
    assert len(recs) <= 5
    for iid, sc in recs:
        assert iid in item_ids
        assert iid not in cf.user_interactions.get(sample_user, {})


def test_surprise_svd_training(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor().fit(interactions_df, catalog_df)
    matrix, user_ids, item_ids = processor.get_cf_matrix()

    cf = CollaborativeFiltering(method="surprise_svd", n_factors=8, epochs=5)
    cf.fit(matrix, user_ids, item_ids)

    sample_user = user_ids[0]
    sample_item = item_ids[0]
    score = cf.predict(sample_user, sample_item)
    assert 0.0 <= score <= 1.0

    recs = cf.recommend(sample_user, n=3, exclude_seen=True)
    assert len(recs) <= 3


def test_cold_start_user_fallback(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor().fit(interactions_df, catalog_df)
    matrix, user_ids, item_ids = processor.get_cf_matrix()

    cf = CollaborativeFiltering(method="implicit_als", n_factors=16, epochs=5)
    cf.fit(matrix, user_ids, item_ids)

    # Unknown user
    cold_recs = cf.recommend("non_existent_user_999", n=5)
    assert len(cold_recs) == 5
    for iid, sc in cold_recs:
        assert iid in item_ids


def test_cf_save_and_load(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor().fit(interactions_df, catalog_df)
    matrix, user_ids, item_ids = processor.get_cf_matrix()

    cf = CollaborativeFiltering(method="implicit_als", n_factors=16, epochs=5)
    cf.fit(matrix, user_ids, item_ids)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "cf.pkl"
        cf.save(fpath)

        loaded = CollaborativeFiltering.load(fpath)
        assert loaded.method == cf.method
        assert loaded.user_ids == cf.user_ids
        np.testing.assert_array_almost_equal(loaded.user_factors, cf.user_factors)


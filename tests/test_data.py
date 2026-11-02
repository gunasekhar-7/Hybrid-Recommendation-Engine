"""Unit tests for DataProcessor and interaction matrix builder."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.data import DataProcessor


def test_dataprocessor_fit_and_alignment(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor()
    processor.fit(interactions_df, catalog_df)

    # 1. Catalog integrity
    assert len(processor.item_ids) == len(catalog_df)
    assert len(processor.user_ids) == interactions_df["user_id"].nunique()

    # 2. Embedding shapes and normalization
    assert processor.item_embeddings is not None
    assert processor.item_embeddings.shape[0] == len(catalog_df)
    norms = np.linalg.norm(processor.item_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # 3. Sparse matrix alignment
    matrix, user_ids, item_ids = processor.get_cf_matrix()
    assert matrix.shape == (len(user_ids), len(item_ids))

    # Verify that a known interaction in DF is at the exact (row, col) in matrix
    first_row = interactions_df.iloc[0]
    u = str(first_row["user_id"])
    i = str(first_row["item_id"])
    expected_rating = float(first_row["rating"])

    u_idx = user_ids.index(u)
    i_idx = item_ids.index(i)
    actual_matrix_val = matrix[u_idx, i_idx]
    assert actual_matrix_val == pytest.approx(expected_rating)


def test_cold_start_items_preservation():
    # Catalog has 5 items, but interactions only touch 3
    catalog_df = pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "cold_1", "cold_2"],
        "title": ["Item 1", "Item 2", "Item 3", "Cold Item A", "Cold Item B"],
    })
    interactions_df = pd.DataFrame({
        "user_id": ["u1", "u2", "u1"],
        "item_id": ["i1", "i2", "i3"],
        "rating": [5.0, 4.0, 3.0],
    })

    processor = DataProcessor()
    processor.fit(interactions_df, catalog_df)

    assert "cold_1" in processor.item_ids
    assert "cold_2" in processor.item_ids
    assert processor.item_embeddings.shape[0] == 5

    emb_cold = processor.get_item_embedding("cold_1")
    assert emb_cold is not None
    assert len(emb_cold) == processor.embedding_dim


def test_user_profile_computation(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor()
    processor.fit(interactions_df, catalog_df)

    sample_user = processor.user_ids[0]
    profile = processor.get_user_profile(sample_user)

    assert profile is not None
    assert len(profile) == processor.embedding_dim
    assert np.linalg.norm(profile) == pytest.approx(1.0, abs=1e-5)

    # Non-existent user returns None
    assert processor.get_user_profile("unknown_user_xyz") is None


def test_dataprocessor_save_and_load(sample_data):
    interactions_df, catalog_df = sample_data
    processor = DataProcessor()
    processor.fit(interactions_df, catalog_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "processor.pkl"
        processor.save(filepath)

        loaded = DataProcessor.load(filepath)
        assert loaded.item_ids == processor.item_ids
        assert loaded.user_ids == processor.user_ids
        np.testing.assert_array_almost_equal(loaded.item_embeddings, processor.item_embeddings)

"""Unit tests for EvaluationFramework."""

import numpy as np
import pandas as pd
import pytest

from app.eval import EvaluationFramework
from app.hybrid import HybridRecommender


def test_train_test_split_temporal():
    # User 1 has 10 interactions, User 2 has 5
    records = []
    for i in range(10):
        records.append({"user_id": "u1", "item_id": f"i{i}", "rating": 5.0, "timestamp": i})
    for i in range(5):
        records.append({"user_id": "u2", "item_id": f"i{i}", "rating": 4.0, "timestamp": i})

    df = pd.DataFrame(records)
    train_df, test_df = EvaluationFramework.train_test_split_temporal(df, test_ratio=0.2, timestamp_col="timestamp")

    assert len(train_df) + len(test_df) == len(df)
    # Check that test contains the latest timestamps
    assert test_df[test_df["user_id"] == "u1"]["timestamp"].min() >= train_df[train_df["user_id"] == "u1"]["timestamp"].max()


def test_leave_one_out_split():
    records = [
        {"user_id": "u1", "item_id": "i1", "rating": 5.0, "timestamp": 1},
        {"user_id": "u1", "item_id": "i2", "rating": 4.0, "timestamp": 2},
        {"user_id": "u2", "item_id": "i3", "rating": 3.0, "timestamp": 1},
    ]
    df = pd.DataFrame(records)
    train_df, test_df = EvaluationFramework.leave_one_out_split(df, timestamp_col="timestamp")

    assert len(test_df) == 1  # Only u1 had >= 2 interactions
    assert test_df.iloc[0]["item_id"] == "i2"


def test_evaluation_metrics_computation(sample_data):
    interactions_df, catalog_df = sample_data
    train_df, test_df = EvaluationFramework.train_test_split_temporal(interactions_df, test_ratio=0.3)

    recommender = HybridRecommender(cf_n_factors=16, cf_epochs=10)
    recommender.fit(train_df, catalog_df)

    eval_framework = EvaluationFramework(recommender, recommender.data_processor, k_values=[5, 10])
    metrics = eval_framework.evaluate_test_set(test_df)

    # All metrics should be in [0, 1]
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"Metric {key} out of bounds: {val}"

    assert "Precision@5" in metrics
    assert "Recall@5" in metrics
    assert "NDCG@5" in metrics
    assert "MAP@5" in metrics
    assert "Catalog_Coverage" in metrics


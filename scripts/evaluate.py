"""Offline Benchmark & Evaluation Script.

Runs comparative evaluation of the Hybrid Engine against Popularity, Random,
CF-only, and CB-only baselines.
"""

from __future__ import annotations

import argparse
import logging
import pandas as pd

from app.hybrid import HybridRecommender
from app.eval import EvaluationFramework
from scripts.train import generate_synthetic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Recommender against Baselines")
    parser.add_argument("--model-dir", type=str, default="artifacts/model", help="Path to saved model")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for ranking metrics")
    args = parser.parse_args()

    interactions_df, catalog_df = generate_synthetic_data(n_users=120, n_items=60, n_interactions=2000)
    train_df, test_df = EvaluationFramework.train_test_split_temporal(interactions_df, test_ratio=0.25)

    recommender = HybridRecommender(cf_method="implicit_als", cf_n_factors=32, cf_epochs=15)
    recommender.fit(train_df, catalog_df)

    eval_framework = EvaluationFramework(recommender, recommender.data_processor, k_values=[args.k])
    benchmark_df = eval_framework.benchmark_against_baselines(train_df, test_df, k=args.k)

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS (K={args.k})")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

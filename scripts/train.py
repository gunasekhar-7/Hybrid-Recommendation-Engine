"""Offline Model Training Script.

Trains the Hybrid Recommender on provided or synthetic interaction/catalog data,
validates on a temporal hold-out split, and persists the production artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.hybrid import HybridRecommender
from app.eval import EvaluationFramework

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def generate_synthetic_data(n_users: int = 150, n_items: int = 80, n_interactions: int = 3000):
    """Generate realistic synthetic interaction and catalog data."""
    np.random.seed(42)
    user_ids = [f"u{i}" for i in range(n_users)]
    item_ids = [f"i{i}" for i in range(n_items)]

    users = np.random.choice(user_ids, size=n_interactions)
    items = np.random.choice(item_ids, size=n_interactions)
    ratings = np.random.uniform(1.0, 5.0, size=n_interactions).round(1)

    interactions_df = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "rating": ratings,
    }).drop_duplicates(subset=["user_id", "item_id"])

    categories = ["Electronics", "Books", "Home & Garden", "Clothing", "Sports"]
    catalog = []
    for iid in item_ids:
        cat = np.random.choice(categories)
        catalog.append({
            "item_id": iid,
            "title": f"Product {iid}: Premium {cat} Asset",
            "description": f"High quality {cat.lower()} item with durability and modern design.",
            "category": cat,
        })
    catalog_df = pd.DataFrame(catalog)
    return interactions_df, catalog_df


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Recommendation Engine")
    parser.add_argument("--interactions", type=str, default=None, help="Path to interactions CSV")
    parser.add_argument("--items", type=str, default=None, help="Path to catalog items CSV")
    parser.add_argument("--output-dir", type=str, default="artifacts/model", help="Directory to save model")
    parser.add_argument("--cf-method", type=str, default="implicit_als", choices=["implicit_als", "surprise_svd", "surprise_nmf"])
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.interactions and args.items:
        logger.info("Loading data from %s and %s", args.interactions, args.items)
        interactions_df = pd.read_csv(args.interactions)
        catalog_df = pd.read_csv(args.items)
    else:
        logger.info("No input data specified. Generating synthetic benchmark dataset...")
        interactions_df, catalog_df = generate_synthetic_data()

    logger.info("Loaded %d interactions across %d catalog items.", len(interactions_df), len(catalog_df))

    # Hold-out temporal train/test split
    train_df, test_df = EvaluationFramework.train_test_split_temporal(interactions_df, test_ratio=0.2)
    logger.info("Split: %d train interactions, %d test interactions", len(train_df), len(test_df))

    # Fit Hybrid Recommender
    recommender = HybridRecommender(
        cf_method=args.cf_method,
        cf_n_factors=args.factors,
        cf_epochs=args.epochs,
    )
    recommender.fit(train_df, catalog_df)

    # Evaluate
    logger.info("Evaluating on held-out test set...")
    eval_framework = EvaluationFramework(recommender, recommender.data_processor, k_values=[5, 10, 20])
    metrics = eval_framework.evaluate_test_set(test_df)

    logger.info("=== Evaluation Metrics on Test Set ===")
    for k, v in metrics.items():
        logger.info("  %-18s: %.4f", k, v)

    # Save artifact
    out_dir = Path(args.output_dir)
    recommender.save(out_dir)
    logger.info("Model saved successfully to %s", out_dir)


if __name__ == "__main__":
    main()

"""Rigorous Recommendation System Evaluation Framework.

Provides standard IR/RecSys ranking metrics (Precision@K, Recall@K, NDCG@K,
MAP@K, MRR@K, Catalog Coverage), Leave-One-Out & Temporal train/test splitting,
and baseline/ablation benchmarking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationFramework:
    """Scientific evaluation framework for recommender systems."""

    def __init__(
        self,
        model: Any,
        data_processor: Any,
        k_values: Optional[List[int]] = None,
    ) -> None:
        self.model = model
        self.data_processor = data_processor
        self.k_values = k_values or [5, 10, 20]

    @staticmethod
    def train_test_split_temporal(
        interactions_df: pd.DataFrame,
        test_ratio: float = 0.2,
        timestamp_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions chronologically per user.

        Args:
            interactions_df: DataFrame with ['user_id', 'item_id', 'rating', optional timestamp]
            test_ratio: Fraction of user's most recent interactions reserved for testing
            timestamp_col: Name of timestamp column if present
        """
        df = interactions_df.copy()
        if timestamp_col and timestamp_col in df.columns:
            df = df.sort_values(by=["user_id", timestamp_col])
        else:
            # Preserve existing order
            df["_orig_idx"] = np.arange(len(df))
            df = df.sort_values(by=["user_id", "_orig_idx"])

        train_indices: List[int] = []
        test_indices: List[int] = []

        for _, group in df.groupby("user_id"):
            n_total = len(group)
            if n_total < 2:
                # Keep users with 1 interaction in train set
                train_indices.extend(group.index.tolist())
                continue

            n_test = max(1, int(n_total * test_ratio))
            n_train = n_total - n_test

            train_indices.extend(group.index[:n_train].tolist())
            test_indices.extend(group.index[n_train:].tolist())

        train_df = interactions_df.loc[train_indices].reset_index(drop=True)
        test_df = interactions_df.loc[test_indices].reset_index(drop=True)
        return train_df, test_df

    @staticmethod
    def leave_one_out_split(
        interactions_df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standard RecSys leave-one-out split (last interaction per user is test)."""
        df = interactions_df.copy()
        if timestamp_col and timestamp_col in df.columns:
            df = df.sort_values(by=["user_id", timestamp_col])

        train_indices: List[int] = []
        test_indices: List[int] = []

        for _, group in df.groupby("user_id"):
            if len(group) < 2:
                train_indices.extend(group.index.tolist())
            else:
                train_indices.extend(group.index[:-1].tolist())
                test_indices.append(group.index[-1])

        train_df = interactions_df.loc[train_indices].reset_index(drop=True)
        test_df = interactions_df.loc[test_indices].reset_index(drop=True)
        return train_df, test_df

    def evaluate_test_set(
        self,
        test_interactions_df: pd.DataFrame,
        user_ids: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate ranking metrics against held-out ground truth test interactions.

        Args:
            test_interactions_df: Held-out DataFrame with ['user_id', 'item_id', 'rating']
            user_ids: Optional subset of users to evaluate
            alpha: Custom CF weight override
            beta: Custom CB weight override
        """
        # Map target user -> set of ground truth items in test set
        ground_truth: Dict[str, Set[str]] = {}
        for uid, group in test_interactions_df.groupby("user_id"):
            ground_truth[str(uid)] = set(group["item_id"].astype(str).tolist())

        target_users = [str(u) for u in (user_ids or list(ground_truth.keys()))]
        valid_users = [u for u in target_users if u in ground_truth and len(ground_truth[u]) > 0]

        if not valid_users:
            logger.warning("No valid evaluation users with ground-truth test interactions.")
            return {f"{m}@{k}": 0.0 for k in self.k_values for m in ["Precision", "Recall", "NDCG", "MAP", "MRR"]}

        max_k = max(self.k_values)
        metrics_accum: Dict[str, List[float]] = {
            f"{m}@{k}": []
            for k in self.k_values
            for m in ["Precision", "Recall", "HitRate", "NDCG", "MAP", "MRR"]
        }

        all_recommended_items: Set[str] = set()

        for uid in valid_users:
            relevant = ground_truth[uid]
            # Recommend items, excluding training interactions
            recs = self.model.recommend(
                user_id=uid,
                n=max_k,
                alpha=alpha,
                beta=beta,
                exclude_seen=True,
            )
            rec_ids = [iid for iid, *_ in recs]

            for k in self.k_values:
                top_k = rec_ids[:k]
                all_recommended_items.update(top_k)

                # 1. Precision@K
                hits = [1 if iid in relevant else 0 for iid in top_k]
                n_hits = sum(hits)
                precision = n_hits / k
                metrics_accum[f"Precision@{k}"].append(precision)

                # 2. Recall@K
                recall = n_hits / len(relevant) if len(relevant) > 0 else 0.0
                metrics_accum[f"Recall@{k}"].append(recall)

                # 3. HitRate@K
                metrics_accum[f"HitRate@{k}"].append(1.0 if n_hits > 0 else 0.0)

                # 4. NDCG@K
                dcg = 0.0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        dcg += 1.0 / np.log2(rank + 1)
                ideal_hits = min(len(relevant), k)
                idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
                ndcg = dcg / idcg if idcg > 0.0 else 0.0
                metrics_accum[f"NDCG@{k}"].append(ndcg)

                # 5. MAP@K (Average Precision)
                ap = 0.0
                running_hits = 0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        running_hits += 1
                        ap += running_hits / rank
                ap = ap / min(len(relevant), k) if min(len(relevant), k) > 0 else 0.0
                metrics_accum[f"MAP@{k}"].append(ap)

                # 6. MRR@K
                mrr = 0.0
                for rank, hit in enumerate(hits, start=1):
                    if hit:
                        mrr = 1.0 / rank
                        break
                metrics_accum[f"MRR@{k}"].append(mrr)

        # Average metrics
        summary: Dict[str, float] = {
            metric_name: float(np.mean(vals))
            for metric_name, vals in metrics_accum.items()
        }

        # Catalog coverage
        if self.data_processor and len(self.data_processor.item_ids) > 0:
            catalog_size = len(self.data_processor.item_ids)
            coverage = len(all_recommended_items) / catalog_size
            summary["Catalog_Coverage"] = float(coverage)

        return summary

    def benchmark_against_baselines(
        self,
        train_interactions_df: pd.DataFrame,
        test_interactions_df: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        """Compare the current model against Popularity and Random baselines."""
        # 1. Model metrics
        model_scores = self.evaluate_test_set(test_interactions_df)

        # 2. Popularity baseline: items with most interactions in train set
        pop_counts = train_interactions_df["item_id"].value_index_counts = (
            train_interactions_df["item_id"].value_counts()
        )
        popular_items = [str(iid) for iid in pop_counts.index]

        class PopularityBaseline:
            def __init__(self, items, processor):
                self.items = items
                self.processor = processor

            def recommend(self, user_id, n=10, **kwargs):
                seen = set(self.processor.user_interactions.get(str(user_id), {}).keys())
                unseen = [i for i in self.items if i not in seen][:n]
                return [(i, 1.0, 1.0, 1.0) for i in unseen]

        pop_eval = EvaluationFramework(
            PopularityBaseline(popular_items, self.data_processor),
            self.data_processor,
            k_values=[k],
        ).evaluate_test_set(test_interactions_df)

        # 3. Random baseline
        all_catalog = self.data_processor.item_ids if self.data_processor else []

        class RandomBaseline:
            def __init__(self, items, processor):
                self.items = items
                self.processor = processor

            def recommend(self, user_id, n=10, **kwargs):
                seen = set(self.processor.user_interactions.get(str(user_id), {}).keys())
                candidates = [i for i in self.items if i not in seen]
                chosen = np.random.choice(candidates, size=min(n, len(candidates)), replace=False)
                return [(i, 1.0, 1.0, 1.0) for i in chosen]

        random_eval = EvaluationFramework(
            RandomBaseline(all_catalog, self.data_processor),
            self.data_processor,
            k_values=[k],
        ).evaluate_test_set(test_interactions_df)

        # 4. Ablation: CF-only vs CB-only
        cf_only_eval = self.evaluate_test_set(test_interactions_df, alpha=1.0, beta=0.0)
        cb_only_eval = self.evaluate_test_set(test_interactions_df, alpha=0.0, beta=1.0)

        rows = [
            {"Model": "Hybrid Engine", f"Precision@{k}": model_scores.get(f"Precision@{k}", 0),
             f"Recall@{k}": model_scores.get(f"Recall@{k}", 0), f"NDCG@{k}": model_scores.get(f"NDCG@{k}", 0),
             f"MAP@{k}": model_scores.get(f"MAP@{k}", 0)},
            {"Model": "CF Only (alpha=1.0)", f"Precision@{k}": cf_only_eval.get(f"Precision@{k}", 0),
             f"Recall@{k}": cf_only_eval.get(f"Recall@{k}", 0), f"NDCG@{k}": cf_only_eval.get(f"NDCG@{k}", 0),
             f"MAP@{k}": cf_only_eval.get(f"MAP@{k}", 0)},
            {"Model": "CB Only (beta=1.0)", f"Precision@{k}": cb_only_eval.get(f"Precision@{k}", 0),
             f"Recall@{k}": cb_only_eval.get(f"Recall@{k}", 0), f"NDCG@{k}": cb_only_eval.get(f"NDCG@{k}", 0),
             f"MAP@{k}": cb_only_eval.get(f"MAP@{k}", 0)},
            {"Model": "Popularity Baseline", f"Precision@{k}": pop_eval.get(f"Precision@{k}", 0),
             f"Recall@{k}": pop_eval.get(f"Recall@{k}", 0), f"NDCG@{k}": pop_eval.get(f"NDCG@{k}", 0),
             f"MAP@{k}": pop_eval.get(f"MAP@{k}", 0)},
            {"Model": "Random Baseline", f"Precision@{k}": random_eval.get(f"Precision@{k}", 0),
             f"Recall@{k}": random_eval.get(f"Recall@{k}", 0), f"NDCG@{k}": random_eval.get(f"NDCG@{k}", 0),
             f"MAP@{k}": random_eval.get(f"MAP@{k}", 0)},
        ]

        return pd.DataFrame(rows)

"""Collaborative Filtering component supporting Implicit ALS and Surprise matrix factorization.

Provides fast vectorized all-item score computation, cold-start handling,
and model persistence.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Stable sigmoid function mapping real scores to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class CollaborativeFiltering:
    """Collaborative Filtering engine using Implicit ALS or Surprise SVD/NMF."""

    SUPPORTED_METHODS = {"implicit_als", "surprise_svd", "surprise_nmf"}

    def __init__(
        self,
        method: str = "implicit_als",
        n_factors: int = 64,
        regularization: float = 0.05,
        epochs: int = 20,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Choose from {self.SUPPORTED_METHODS}")

        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.epochs = epochs

        self.model: Any = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_interactions: Dict[str, Dict[str, float]] = {}

        # Cached latent factors (N_users x K, N_items x K)
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.item_popularity: Optional[np.ndarray] = None
        self.global_mean_rating: float = 3.0

    def fit(
        self,
        user_item_matrix: sparse.csr_matrix,
        user_ids: List[str],
        item_ids: List[str],
        user_interactions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> "CollaborativeFiltering":
        """Fit the matrix factorization model.

        Args:
            user_item_matrix: scipy sparse CSR matrix (users x items)
            user_ids: aligned user ID list matching rows
            item_ids: aligned item ID list matching columns
            user_interactions: optional dictionary mapping user_id -> {item_id: rating}
        """
        self.user_ids = [str(u) for u in user_ids]
        self.item_ids = [str(i) for i in item_ids]
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}

        if user_interactions is not None:
            self.user_interactions = user_interactions
        else:
            # Reconstruct interaction dict from sparse matrix
            self.user_interactions = {}
            cx = user_item_matrix.tocoo()
            for r, c, v in zip(cx.row, cx.col, cx.data):
                uid = self.user_ids[r]
                iid = self.item_ids[c]
                if uid not in self.user_interactions:
                    self.user_interactions[uid] = {}
                self.user_interactions[uid][iid] = float(v)

        # Compute popularity prior (normalized column sum / interactions)
        col_sums = np.array(user_item_matrix.sum(axis=0)).flatten()
        max_sum = np.max(col_sums) if len(col_sums) > 0 and np.max(col_sums) > 0 else 1.0
        self.item_popularity = (col_sums / max_sum).astype(np.float32)

        if user_item_matrix.nnz > 0:
            self.global_mean_rating = float(user_item_matrix.data.mean())

        if self.method == "implicit_als":
            self._fit_implicit(user_item_matrix)
        else:
            self._fit_surprise(user_item_matrix)

        logger.info("CF model fitted successfully using %s (%d users, %d items)", self.method, len(self.user_ids), len(self.item_ids))
        return self

    def _fit_implicit(self, matrix: sparse.csr_matrix) -> None:
        """Fit implicit ALS model."""
        from implicit.als import AlternatingLeastSquares

        # Initialize Implicit ALS
        self.model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.epochs,
            random_state=42,
        )

        # Implicit accepts user-item matrix directly
        # Confidence weighting: C_ui = 1 + alpha * r_ui
        confidence_matrix = matrix.astype(np.float32)
        self.model.fit(confidence_matrix, show_progress=False)

        # Extract factor matrices
        self.user_factors = np.array(self.model.user_factors, dtype=np.float32)
        self.item_factors = np.array(self.model.item_factors, dtype=np.float32)

    def _fit_surprise(self, matrix: sparse.csr_matrix) -> None:
        """Fit Surprise SVD/NMF model."""
        from surprise import Dataset, Reader, SVD, NMF, accuracy
        from surprise.model_selection import train_test_split

        min_val = float(matrix.data.min()) if matrix.nnz > 0 else 1.0
        max_val = float(matrix.data.max()) if matrix.nnz > 0 else 5.0
        if min_val >= max_val:
            max_val = min_val + 1.0

        reader = Reader(rating_scale=(min_val, max_val))

        cx = matrix.tocoo()
        records = [
            {"user_id": self.user_ids[r], "item_id": self.item_ids[c], "rating": float(v)}
            for r, c, v in zip(cx.row, cx.col, cx.data)
        ]
        df = pd.DataFrame(records)

        if df.empty:
            logger.warning("Empty matrix provided for Surprise CF model")
            return

        data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
        trainset = data.build_full_trainset()

        if self.method == "surprise_svd":
            self.model = SVD(
                n_factors=self.n_factors,
                reg_all=self.regularization,
                n_epochs=self.epochs,
                random_state=42,
            )
        else:
            self.model = NMF(
                n_factors=self.n_factors,
                reg_all=self.regularization,
                n_epochs=self.epochs,
                random_state=42,
            )

        self.model.fit(trainset)

        # Extract factors if available
        try:
            self.user_factors = np.array(self.model.pu, dtype=np.float32)
            self.item_factors = np.array(self.model.qi, dtype=np.float32)
        except AttributeError:
            self.user_factors = None
            self.item_factors = None

    def predict(self, user_id: str, item_id: str) -> float:
        """Predict score for a single (user, item) pair, normalized to [0, 1]."""
        u_idx = self.user_to_idx.get(str(user_id))
        i_idx = self.item_to_idx.get(str(item_id))

        if u_idx is None or i_idx is None:
            # Fallback to popularity prior if unknown
            if i_idx is not None and self.item_popularity is not None:
                return float(self.item_popularity[i_idx])
            return 0.5

        if self.method == "implicit_als" and self.user_factors is not None and self.item_factors is not None:
            raw_score = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
            return float(_sigmoid(raw_score))

        if self.method.startswith("surprise") and self.model is not None:
            pred = self.model.predict(str(user_id), str(item_id))
            # Normalize from 1-5 scale to [0, 1]
            return float(np.clip((pred.est - 1.0) / 4.0, 0.0, 1.0))

        return 0.5

    def predict_all_items(self, user_id: str) -> np.ndarray:
        """Fast vectorized calculation of CF scores for ALL catalog items.

        Returns:
            np.ndarray of shape (N_items,) with scores in [0, 1]
        """
        n_items = len(self.item_ids)
        if n_items == 0:
            return np.array([], dtype=np.float32)

        u_idx = self.user_to_idx.get(str(user_id))

        # Cold-start user: return item popularity profile
        if u_idx is None:
            if self.item_popularity is not None:
                return self.item_popularity.copy()
            return np.full(n_items, 0.5, dtype=np.float32)

        if self.method == "implicit_als" and self.user_factors is not None and self.item_factors is not None:
            user_vec = self.user_factors[u_idx]  # Shape: (K,)
            raw_scores = self.item_factors @ user_vec  # Shape: (N_items,)
            return _sigmoid(raw_scores).astype(np.float32)

        if self.method.startswith("surprise") and self.user_factors is not None and self.item_factors is not None:
            # SVD dot product + biases if available
            user_vec = self.user_factors[u_idx]
            raw_scores = self.item_factors @ user_vec
            # Scale to [0, 1]
            return _sigmoid(raw_scores).astype(np.float32)

        # Fallback elementwise prediction
        scores = np.empty(n_items, dtype=np.float32)
        for i, iid in enumerate(self.item_ids):
            scores[i] = self.predict(user_id, iid)
        return scores

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """Generate top-N recommendations for user.

        Returns:
            List of (item_id, cf_score) tuples sorted descending.
        """
        if not self.item_ids:
            return []

        scores = self.predict_all_items(user_id).copy()

        # Exclude seen items
        if exclude_seen and str(user_id) in self.user_interactions:
            seen_dict = self.user_interactions[str(user_id)]
            for seen_iid in seen_dict.keys():
                if seen_iid in self.item_to_idx:
                    scores[self.item_to_idx[seen_iid]] = -np.inf

        k = min(n, len(scores))
        if k <= 0:
            return []

        # Vectorized top-k selection
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            (self.idx_to_item[idx], float(scores[idx]))
            for idx in top_indices
            if np.isfinite(scores[idx])
        ]

    def save(self, filepath: str | Path) -> None:
        """Persist collaborative filtering model state."""
        state = {
            "method": self.method,
            "n_factors": self.n_factors,
            "regularization": self.regularization,
            "epochs": self.epochs,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
            "user_interactions": self.user_interactions,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "item_popularity": self.item_popularity,
            "global_mean_rating": self.global_mean_rating,
            "model": self.model,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved CollaborativeFiltering model to %s", filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "CollaborativeFiltering":
        """Load collaborative filtering model state."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        cf = cls(
            method=state["method"],
            n_factors=state["n_factors"],
            regularization=state["regularization"],
            epochs=state["epochs"],
        )
        cf.user_ids = state["user_ids"]
        cf.item_ids = state["item_ids"]
        cf.user_to_idx = state["user_to_idx"]
        cf.item_to_idx = state["item_to_idx"]
        cf.idx_to_item = {i: iid for iid, i in cf.item_to_idx.items()}
        cf.user_interactions = state["user_interactions"]
        cf.user_factors = state["user_factors"]
        cf.item_factors = state["item_factors"]
        cf.item_popularity = state["item_popularity"]
        cf.global_mean_rating = state["global_mean_rating"]
        cf.model = state.get("model")
        logger.info("Loaded CollaborativeFiltering model from %s", filepath)
        return cf

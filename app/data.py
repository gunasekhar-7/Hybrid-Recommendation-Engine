"""Data processing, feature extraction, and interaction matrix builder.

Handles SBERT semantic embedding generation, sparse user-item interaction matrix
construction, cold-item catalog inclusion, and user profile aggregation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes catalog items and user interactions for hybrid recommendation."""

    def __init__(
        self,
        sbert_model_name: str = "all-MiniLM-L6-v2",
        redis_client: Optional[Any] = None,
        embedding_ttl: int = 86400,
    ) -> None:
        self.sbert_model_name = sbert_model_name
        self._sbert_model: Optional[SentenceTransformer] = None
        self.redis_client = redis_client
        self.embedding_ttl = embedding_ttl

        # State attributes
        self.item_ids: List[str] = []
        self.user_ids: List[str] = []
        self.item_id_to_idx: Dict[str, int] = {}
        self.item_idx_to_id: Dict[int, str] = {}
        self.user_id_to_idx: Dict[str, int] = {}
        self.user_idx_to_user_id: Dict[int, str] = {}
        self.user_interactions: Dict[str, Dict[str, float]] = {}
        self.item_embeddings: Optional[np.ndarray] = None
        self.global_mean_embedding: Optional[np.ndarray] = None
        self.embedding_dim: int = 384

    @property
    def sbert_model(self) -> SentenceTransformer:
        """Lazy load SBERT model to speed up instantiation and testing."""
        if self._sbert_model is None:
            logger.info("Loading SBERT model: %s", self.sbert_model_name)
            self._sbert_model = SentenceTransformer(self.sbert_model_name)
        return self._sbert_model

    def fit(
        self,
        interactions_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> "DataProcessor":
        """Build user-item interaction matrix and item embeddings.

        Args:
            interactions_df: DataFrame with ['user_id', 'item_id', 'rating']
            item_features_df: DataFrame with ['item_id', 'title', optional 'description']
            batch_size: Batch size for SBERT inference
            show_progress_bar: Whether to display SBERT encoding progress bar
        """
        # Ensure string IDs for consistency
        interactions = interactions_df.copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["item_id"] = interactions["item_id"].astype(str)
        interactions["rating"] = interactions["rating"].astype(float)

        item_features = item_features_df.copy()
        item_features["item_id"] = item_features["item_id"].astype(str)

        # 1. Catalog integrity: combine all items from features and interactions
        feature_items: Set[str] = set(item_features["item_id"].unique())
        interaction_items: Set[str] = set(interactions["item_id"].unique())
        all_item_ids: Set[str] = feature_items | interaction_items

        self.item_ids = sorted(all_item_ids)
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.item_idx_to_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}

        self.user_ids = sorted(interactions["user_id"].unique())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.user_idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        # 2. Store user interaction history for quick lookup
        self.user_interactions = {
            uid: group.set_index("item_id")["rating"].to_dict()
            for uid, group in interactions.groupby("user_id")
        }

        # 3. Generate text embeddings for all catalog items
        self._generate_embeddings(
            item_features=item_features,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        logger.info(
            "DataProcessor fitted: %d users, %d catalog items (%d interacted), embeddings shape %s",
            len(self.user_ids),
            len(self.item_ids),
            len(interaction_items),
            self.item_embeddings.shape if self.item_embeddings is not None else None,
        )
        return self

    def _generate_embeddings(
        self,
        item_features: pd.DataFrame,
        batch_size: int,
        show_progress_bar: bool,
    ) -> None:
        """Generate and normalize semantic embeddings for catalog items."""
        # Create text representations
        feature_dict: Dict[str, str] = {}
        for _, row in item_features.iterrows():
            iid = str(row["item_id"])
            parts = []
            if "title" in row and pd.notna(row["title"]):
                parts.append(str(row["title"]))
            if "description" in row and pd.notna(row["description"]):
                parts.append(str(row["description"]))
            feature_dict[iid] = " ".join(parts) if parts else f"Item {iid}"

        # Align with ordered self.item_ids
        texts_to_encode: List[str] = []
        cached_embeddings: Dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, iid in enumerate(self.item_ids):
            text = feature_dict.get(iid, f"Item {iid}")
            texts_to_encode.append(text)

            cached_emb = self._get_cached_embedding(iid)
            if cached_emb is not None:
                cached_embeddings[idx] = cached_emb
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # Encode uncached texts in batches
        all_embeddings = np.zeros((len(self.item_ids), self.embedding_dim), dtype=np.float32)

        for idx, emb in cached_embeddings.items():
            all_embeddings[idx] = emb

        if uncached_texts:
            logger.info("Computing SBERT embeddings for %d items...", len(uncached_texts))
            encoded = self.sbert_model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            self.embedding_dim = encoded.shape[1]
            if all_embeddings.shape[1] != self.embedding_dim:
                all_embeddings = np.zeros((len(self.item_ids), self.embedding_dim), dtype=np.float32)
                for idx, emb in cached_embeddings.items():
                    all_embeddings[idx] = emb

            for i, idx in enumerate(uncached_indices):
                emb = encoded[i]
                all_embeddings[idx] = emb
                self._cache_embedding(self.item_ids[idx], emb)

        # L2-normalize all rows
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.item_embeddings = (all_embeddings / norms).astype(np.float32)

        if len(self.item_embeddings) > 0:
            mean_vec = np.mean(self.item_embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            self.global_mean_embedding = (mean_vec / max(norm, 1e-8)).astype(np.float32)
        else:
            self.global_mean_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

    def _cache_embedding(self, item_id: str, embedding: np.ndarray) -> None:
        """Cache single embedding into Redis with float32 bytes."""
        if not self.redis_client:
            return
        try:
            key = f"emb:{item_id}"
            self.redis_client.setex(
                name=key,
                time=self.embedding_ttl,
                value=embedding.astype(np.float32).tobytes(),
            )
        except Exception as e:
            logger.debug("Redis cache write failed for item %s: %s", item_id, e)

    def _get_cached_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Fetch cached embedding from Redis, casting back to float32."""
        if not self.redis_client:
            return None
        try:
            key = f"emb:{item_id}"
            data = self.redis_client.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.debug("Redis cache read failed for item %s: %s", item_id, e)
        return None

    def get_cf_matrix(self) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        """Construct the user-item interaction sparse matrix.

        Returns:
            Tuple of:
                - CSR sparse matrix (users x items)
                - ordered user_ids list
                - ordered item_ids list
        """
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for uid, interactions in self.user_interactions.items():
            if uid not in self.user_id_to_idx:
                continue
            u_idx = self.user_id_to_idx[uid]
            for iid, rating in interactions.items():
                if iid in self.item_id_to_idx:
                    i_idx = self.item_id_to_idx[iid]
                    rows.append(u_idx)
                    cols.append(i_idx)
                    data.append(float(rating))

        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        return matrix, self.user_ids, self.item_ids

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get precomputed normalized embedding for an item."""
        if self.item_embeddings is None or item_id not in self.item_id_to_idx:
            return None
        return self.item_embeddings[self.item_id_to_idx[item_id]]

    def get_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """Compute user profile vector as rating-weighted average of item embeddings."""
        if self.item_embeddings is None:
            return None

        interactions = self.user_interactions.get(user_id)
        if not interactions:
            return None

        item_indices: List[int] = []
        ratings: List[float] = []

        for iid, rating in interactions.items():
            if iid in self.item_id_to_idx:
                item_indices.append(self.item_id_to_idx[iid])
                ratings.append(float(rating))

        if not item_indices:
            return None

        # Vectorized weighted average
        weights = np.array(ratings, dtype=np.float32)
        selected_embs = self.item_embeddings[item_indices]  # Shape: (K, D)
        user_vector = np.average(selected_embs, axis=0, weights=weights)

        # L2 normalize
        norm = np.linalg.norm(user_vector)
        if norm > 1e-8:
            user_vector = user_vector / norm
        return user_vector.astype(np.float32)

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Return dataset interaction summary metrics."""
        if not self.user_interactions:
            return {"n_users": 0, "n_items": len(self.item_ids), "n_interactions": 0}

        counts = [len(v) for v in self.user_interactions.values()]
        ratings = [r for v in self.user_interactions.values() for r in v.values()]
        unique_interacted_items = set(
            iid for v in self.user_interactions.values() for iid in v.keys()
        )

        return {
            "n_users": len(self.user_interactions),
            "n_catalog_items": len(self.item_ids),
            "n_interacted_items": len(unique_interacted_items),
            "n_interactions": len(ratings),
            "avg_interactions_per_user": float(np.mean(counts)) if counts else 0.0,
            "sparsity": 1.0 - (len(ratings) / max(1, len(self.user_ids) * len(self.item_ids))),
            "avg_rating": float(np.mean(ratings)) if ratings else 0.0,
        }

    def save(self, filepath: str | Path) -> None:
        """Persist processor state to file."""
        state = {
            "sbert_model_name": self.sbert_model_name,
            "item_ids": self.item_ids,
            "user_ids": self.user_ids,
            "item_id_to_idx": self.item_id_to_idx,
            "user_id_to_idx": self.user_id_to_idx,
            "user_interactions": self.user_interactions,
            "item_embeddings": self.item_embeddings,
            "global_mean_embedding": self.global_mean_embedding,
            "embedding_dim": self.embedding_dim,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved DataProcessor to %s", filepath)

    @classmethod
    def load(cls, filepath: str | Path, redis_client: Optional[Any] = None) -> "DataProcessor":
        """Load processor state from file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        processor = cls(sbert_model_name=state["sbert_model_name"], redis_client=redis_client)
        processor.item_ids = state["item_ids"]
        processor.user_ids = state["user_ids"]
        processor.item_id_to_idx = state["item_id_to_idx"]
        processor.item_idx_to_id = {idx: iid for iid, idx in processor.item_id_to_idx.items()}
        processor.user_id_to_idx = state["user_id_to_idx"]
        processor.user_idx_to_user_id = {idx: uid for uid, idx in processor.user_id_to_idx.items()}
        processor.user_interactions = state["user_interactions"]
        processor.item_embeddings = state["item_embeddings"]
        processor.global_mean_embedding = state["global_mean_embedding"]
        processor.embedding_dim = state.get("embedding_dim", 384)
        logger.info("Loaded DataProcessor from %s (%d items)", filepath, len(processor.item_ids))
        return processor

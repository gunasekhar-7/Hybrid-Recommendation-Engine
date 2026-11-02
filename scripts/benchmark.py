"""Inference Latency Benchmark.

Measures single-request and batch recommendation latency across sample users.
"""

from __future__ import annotations

import time
import numpy as np

from app.hybrid import HybridRecommender
from scripts.train import generate_synthetic_data


def benchmark():
    print("Preparing dataset and training recommender for latency benchmark...")
    interactions_df, catalog_df = generate_synthetic_data(n_users=300, n_items=150, n_interactions=5000)
    recommender = HybridRecommender(cf_method="implicit_als", cf_n_factors=64, cf_epochs=15)
    recommender.fit(interactions_df, catalog_df)

    test_users = [f"u{i}" for i in range(50)]

    print(f"\nBenchmarking single-user recommendation across {len(test_users)} users (N=10)...")
    single_latencies = []
    for uid in test_users:
        t0 = time.perf_counter()
        _ = recommender.recommend(uid, n=10)
        single_latencies.append((time.perf_counter() - t0) * 1000)  # ms

    single_latencies = np.array(single_latencies)
    print(f"  P50 Latency: {np.percentile(single_latencies, 50):.2f} ms")
    print(f"  P95 Latency: {np.percentile(single_latencies, 95):.2f} ms")
    print(f"  P99 Latency: {np.percentile(single_latencies, 99):.2f} ms")
    print(f"  Mean Latency: {np.mean(single_latencies):.2f} ms")

    print("\nBenchmarking batch recommendation for 20 users...")
    batch_latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = recommender.recommend_batch(test_users[:20], n=10)
        batch_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"  Batch 20 users Mean Latency: {np.mean(batch_latencies):.2f} ms")
    print(f"  Throughput: {20 / (np.mean(batch_latencies) / 1000):.1f} recs/sec")


if __name__ == "__main__":
    benchmark()

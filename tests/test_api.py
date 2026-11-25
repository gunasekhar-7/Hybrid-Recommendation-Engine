"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi import status


def test_health_and_version(client):
    res = client.get("/health")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_fitted"] is True
    assert data["catalog_items"] > 0
    assert data["active_users"] > 0

    res_ver = client.get("/version")
    assert res_ver.status_code == status.HTTP_200_OK
    assert res_ver.json()["version"] == "1.0.0"

    res_met = client.get("/metrics")
    assert res_met.status_code == status.HTTP_200_OK
    assert res_met.json()["model_fitted"] == 1


def test_recommend_endpoint(client, fitted_recommender):
    user_id = fitted_recommender.data_processor.user_ids[0]
    payload = {
        "user_id": user_id,
        "n": 5,
        "exclude_seen": True,
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["user_id"] == user_id
    assert len(data["recommendations"]) <= 5
    assert "weights_used" in data
    assert "X-Request-ID" in res.headers
    assert "X-Process-Time" in res.headers


def test_batch_recommend_endpoint(client, fitted_recommender):
    users = fitted_recommender.data_processor.user_ids[:3]
    payload = {
        "user_ids": users,
        "n": 3,
    }
    res = client.post("/recommend/batch", json=payload)
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["users_processed"] == 3
    assert len(data["results"]) == 3


def test_similar_items_endpoint(client, fitted_recommender):
    item_id = fitted_recommender.data_processor.item_ids[0]
    payload = {
        "item_id": item_id,
        "n": 3,
    }
    res = client.post("/similar-items", json=payload)
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["item_id"] == item_id
    assert len(data["similar_items"]) <= 3


def test_validation_errors(client):
    # Invalid N (> 100)
    res = client.post("/recommend", json={"user_id": "u1", "n": 500})
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Missing user_id
    res = client.post("/recommend", json={"n": 5})
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_init_endpoint(client):
    payload = {
        "interactions": [
            {"user_id": "uA", "item_id": "iA", "rating": 5.0},
            {"user_id": "uB", "item_id": "iB", "rating": 4.0},
        ],
        "items": [
            {"item_id": "iA", "title": "Item Alpha", "description": "Tech item"},
            {"item_id": "iB", "title": "Item Beta", "description": "Book item"},
        ],
        "save_after_fit": False,
    }
    res = client.post("/init", json=payload)
    assert res.status_code == status.HTTP_201_CREATED
    data = res.json()
    assert data["status"] == "success"
    assert "fit_info" in data


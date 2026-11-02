# API Reference: Hybrid Recommendation Engine

Base URL: `http://localhost:8000`

Interactive OpenAPI Documentation: `http://localhost:8000/docs`

---

## 1. Single Recommendation
`POST /recommend`

Generate top-$N$ personalized hybrid recommendations for a single user.

### Request Body
```json
{
  "user_id": "user_42",
  "n": 10,
  "alpha": 0.5,
  "beta": 0.5,
  "exclude_seen": true
}
```

### Response (`200 OK`)
```json
{
  "user_id": "user_42",
  "recommendations": [
    {
      "item_id": "item_101",
      "hybrid_score": 0.8924,
      "cf_score": 0.9150,
      "cb_score": 0.8698
    }
  ],
  "total_evaluated": 1500,
  "weights_used": {
    "alpha": 0.5,
    "beta": 0.5
  }
}
```

---

## 2. Batch Recommendation
`POST /recommend/batch`

High-throughput multi-user recommendation endpoint.

### Request Body
```json
{
  "user_ids": ["user_1", "user_2", "user_3"],
  "n": 5,
  "exclude_seen": true
}
```

### Response (`200 OK`)
```json
{
  "results": {
    "user_1": [
      {
        "item_id": "item_88",
        "hybrid_score": 0.9321,
        "cf_score": 0.9500,
        "cb_score": 0.9142
      }
    ]
  },
  "users_processed": 3,
  "total_items_in_catalog": 1500
}
```

---

## 3. Similar Items (Item-to-Item)
`POST /similar-items`

Retrieve items semantically related to a given item using SBERT cosine similarity.

### Request Body
```json
{
  "item_id": "item_101",
  "n": 5
}
```

### Response (`200 OK`)
```json
{
  "item_id": "item_101",
  "similar_items": [
    {
      "item_id": "item_204",
      "similarity_score": 0.9412
    }
  ]
}
```

---

## 4. Train / Refit Model
`POST /init`

Fit and deploy a new model dynamically.

### Request Body
```json
{
  "interactions": [
    {"user_id": "u1", "item_id": "i1", "rating": 5.0},
    {"user_id": "u2", "item_id": "i2", "rating": 4.0}
  ],
  "items": [
    {"item_id": "i1", "title": "Mechanical Keyboard", "description": "RGB mechanical keyboard"},
    {"item_id": "i2", "title": "Wireless Mouse", "description": "Ergonomic Bluetooth mouse"}
  ],
  "save_after_fit": true
}
```

### Response (`201 Created`)
```json
{
  "status": "success",
  "message": "Model fitted successfully and deployed for inference.",
  "fit_info": {
    "is_fitted": true,
    "fit_time_seconds": 1.45,
    "cf_method": "implicit_als",
    "total_items": 2,
    "total_users": 2
  }
}
```

---

## 5. Health & Readiness Probes
`GET /health`

### Response (`200 OK`)
```json
{
  "status": "healthy",
  "is_ready": true,
  "model_fitted": true,
  "uptime_seconds": 142.3,
  "timestamp": "2026-11-02T18:30:00Z",
  "redis_connected": true,
  "catalog_items": 1500,
  "active_users": 500
}
```

# =====================================================================================
# UTILITY ENDPOINTS
# =====================================================================================
from http.client import HTTPException
from typing import List

import numpy as np
from fastapi import Query
from sklearn.metrics import euclidean_distances
from torch import cosine_similarity
from transformers import AutoTokenizer

from Backend.backend import app


@app.post("/utils/embedding-similarity")
async def calculate_embedding_similarity(
        embedding1: List[float],
        embedding2: List[float],
        metric: str = Query("euclidean", regex="^(cosine|euclidean|manhattan)$")
):
    """Calculate similarity between two embedding vectors with EUCLIDEAN as default"""
    try:
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)

        if metric == "cosine":
            similarity = float(cosine_similarity(emb1, emb2)[0][0])
            distance = float(1 - similarity)
        elif metric == "euclidean":
            distance = float(euclidean_distances(emb1, emb2)[0][0])
            similarity = float(cosine_similarity(emb1, emb2)[0][0])  # Include for reference
        elif metric == "manhattan":
            distance = float(np.sum(np.abs(emb1 - emb2)))
            similarity = float(1 / (1 + distance))

        return {
            "metric": metric,
            "similarity": similarity,
            "distance": distance
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/utils/model-compatibility")
async def check_model_compatibility(model_name: str):
    """Check if a model is compatible with the system"""
    try:
        # Try to load tokenizer to check compatibility
        test_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return {
            "compatible": True,
            "model_name": model_name,
            "tokenizer_type": type(test_tokenizer).__name__,
            "vocab_size": len(test_tokenizer) if hasattr(test_tokenizer, '__len__') else None
        }

    except Exception as e:
        return {
            "compatible": False,
            "model_name": model_name,
            "error": str(e)
        }
# =====================================================================================
# DEBUG ENDPOINTS
# =====================================================================================
from http.client import HTTPException

from sklearn.metrics import euclidean_distances
from torch import cosine_similarity

from Backend.backend import viz, app


@app.get("/debug/token-comparison/{token_index1}/{token_index2}")
async def debug_token_comparison(token_index1: int, token_index2: int):
    """Debug endpoint to verify consistency between neighbor search and direct comparison"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if token_index1 >= len(viz.tokens) or token_index2 >= len(viz.tokens):
        raise HTTPException(status_code=404, detail="Token index out of range")

    try:
        # Get embeddings using the same method
        emb1 = viz._get_raw_embedding(token_index1)
        emb2 = viz._get_raw_embedding(token_index2)

        # Calculate similarity the same way
        cosine_sim = float(cosine_similarity(emb1, emb2)[0][0])
        euclidean_dist = float(euclidean_distances(emb1, emb2)[0][0])
        cosine_dist = 1 - cosine_sim

        # Also check what neighbors search returns with EUCLIDEAN (new default)
        euclidean_neighbors = viz.find_neighbors(token_index1, n_neighbors=len(viz.tokens), metric='euclidean')
        euclidean_neighbor_entry = None
        for neighbor in euclidean_neighbors:
            if neighbor['index'] == token_index2:
                euclidean_neighbor_entry = neighbor
                break

        # Also check with cosine for comparison
        cosine_neighbors = viz.find_neighbors(token_index1, n_neighbors=len(viz.tokens), metric='cosine')
        cosine_neighbor_entry = None
        for neighbor in cosine_neighbors:
            if neighbor['index'] == token_index2:
                cosine_neighbor_entry = neighbor
                break

        # Test the new ID-based comparison
        id_comparison = viz.compare_tokens_by_id(token_index1, token_index2)

        return {
            "token1": viz.tokens[token_index1],
            "token2": viz.tokens[token_index2],
            "token1_index": token_index1,
            "token2_index": token_index2,
            "direct_comparison": {
                "cosine_similarity": cosine_sim,
                "cosine_distance": cosine_dist,
                "euclidean_distance": euclidean_dist
            },
            "id_based_comparison": id_comparison,
            "euclidean_neighbor_search": euclidean_neighbor_entry,
            "cosine_neighbor_search": cosine_neighbor_entry,
            "consistency_check": {
                "euclidean_distance_match": abs(
                    euclidean_dist - (euclidean_neighbor_entry['distance'] if euclidean_neighbor_entry else 0)) < 1e-10,
                "cosine_similarity_match": abs(
                    cosine_sim - (cosine_neighbor_entry['similarity'] if cosine_neighbor_entry else 0)) < 1e-10,
                "cosine_distance_match": abs(
                    cosine_dist - (cosine_neighbor_entry['distance'] if cosine_neighbor_entry else 0)) < 1e-10,
                "id_comparison_euclidean_match": abs(
                    euclidean_dist - id_comparison['euclidean_distance']) < 1e-10 if id_comparison else False,
                "id_comparison_cosine_match": abs(
                    cosine_sim - id_comparison['cosine_similarity']) < 1e-10 if id_comparison else False
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

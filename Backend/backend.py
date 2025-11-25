# backend.py
# FastAPI backend for Vector Embedding Visualizer v0.2

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
import re

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from Backend.embedding_visualizer import EmbeddingVisualizer
from Backend.pydantic_models import VisualizationConfig, LoadModelRequest, LoadingStatus, VisualizationData, \
    TokenDetails, TokenNeighbor, TokenWithNeighbors, SearchResult, ComparisonRequest, ComparisonResult, \
    ComparisonByIdRequest, BatchAnalysisRequest, ExportRequest

# Suppress warnings
warnings.filterwarnings("ignore")


# =====================================================================================
# FASTAPI APPLICATION
# =====================================================================================

app = FastAPI(
    title="Vector Embedding Visualizer API",
    description="API for exploring transformer model embeddings v0.2",
    version="0.2"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global visualizer instance
viz = EmbeddingVisualizer()

# File storage for exports only (no uploads)
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


# =====================================================================================
# API ENDPOINTS
# =====================================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vector Embedding Visualizer API",
        "version": "0.2",
        "session_id": viz.session_id,
        "model_loaded": viz.embeddings is not None,
        "features": [
            "Fixed some basic program bugs",
            "Added some extra information to token extraction"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loading_status = viz.get_loading_status()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": viz.embeddings is not None,
        "session_id": viz.session_id,
        "loading_status": loading_status if loading_status['is_loading'] else None
    }


@app.get("/models/available")
async def get_available_models():
    """Get list of available pre-trained models"""
    return {
        "preset_models": [
            {"name": "GPT-2 (124M)", "id": "gpt2"},
            {"name": "GPT-2 Medium (355M)", "id": "gpt2-medium"},
            {"name": "GPT-2 Large (774M)", "id": "gpt2-large"},
            {"name": "DistilGPT-2", "id": "distilgpt2"},
            {"name": "BERT Base Uncased", "id": "bert-base-uncased"},
            {"name": "DistilBERT", "id": "distilbert-base-uncased"},
            {"name": "RoBERTa Base", "id": "roberta-base"},
        ],
        "supports_huggingface": True,
        "timeout_minutes": 2,
        "note": "All models subject to 2-minute timeout with proper cancellation. Only UMAP visualization supported."
    }


async def load_model_background(model_name: str):
    """Background task for loading model with proper timeout handling"""
    success = viz.load_model(model_name)
    if success and not viz.loading_cancelled:
        # Prepare tokens after successful loading
        viz.prepare_tokens()
    return success


@app.post("/models/load")
async def load_model(request: LoadModelRequest, background_tasks: BackgroundTasks):
    """Load a model with timeout protection"""

    # Cancel any existing loading operation
    if viz.get_loading_status()['is_loading']:
        viz.cancel_loading()

    # Clear any existing model
    viz.clear_model()

    # Start loading in background
    background_tasks.add_task(load_model_background, request.model_name)

    return {
        "success": True,
        "message": f"Model loading started: {request.model_name}",
        "timeout_minutes": 2,
        "note": "Check /models/loading-status for progress. Only UMAP visualization supported."
    }


@app.get("/models/info")
async def get_model_info():
    """Get current model information"""
    return viz.get_model_info()


@app.get("/models/loading-status")
async def get_loading_status():
    """Get current loading status with proper response"""
    try:
        loading_status = viz.get_loading_status()
        return LoadingStatus(**loading_status)
    except Exception as e:
        print(f"Error getting loading status: {e}")
        # Return safe default
        return LoadingStatus(
            is_loading=False,
            model_name=None,
            progress=None,
            error="Error retrieving loading status"
        )


@app.delete("/models/unload")
async def unload_model():
    """Unload current model and free memory"""
    try:
        viz.clear_model()
        return {
            "success": True,
            "message": "Model unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualization/create")
async def create_visualization(config: VisualizationConfig):
    """Create visualization with dimension reduction using UMAP only"""
    if viz.embeddings is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Force method to be UMAP
    config.method = "umap"

    try:
        # Prepare tokens if not already done or if top_n changed
        if viz.tokens is None or len(viz.tokens) != config.top_n:
            viz.prepare_tokens(config.top_n)

        # Reduce dimensions using UMAP only
        reduction_info = viz.reduce_dimensions(config)

        # Prepare coordinates for frontend
        coordinates = viz.reduced_embeddings.tolist()

        # Count token types for statistics
        type_counts = {}
        for t in viz.token_metadata['types']:
            type_counts[t] = type_counts.get(t, 0) + 1

        statistics = {
            "total_tokens": len(viz.tokens),
            "original_dimension": viz.embeddings.shape[1],
            "reduced_dimension": config.n_components,
            "reduction_method": "umap",
            "type_distribution": type_counts,
            "reduction_info": reduction_info
        }

        return VisualizationData(
            coordinates=coordinates,
            tokens=viz.tokens,
            metadata=viz.token_metadata,
            config=config,
            statistics=statistics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tokens/{token_index}")
async def get_token_details(
        token_index: int,
        include_distances: bool = Query(False, description="Include distance calculations"),
        metric: str = Query("euclidean", regex="^(cosine|euclidean)$", description="Distance metric for calculations")
):
    """Get detailed information about a specific token with optional distance metrics"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if token_index >= len(viz.tokens):
        raise HTTPException(status_code=404, detail="Token index out of range")

    details = viz.get_token_details(token_index, include_distances=include_distances, metric=metric)
    return TokenDetails(**details)


@app.get("/tokens/{token_index}/neighbors")
async def get_token_neighbors(
        token_index: int,
        n_neighbors: int = Query(10, ge=1, le=100),
        metric: str = Query("euclidean", regex="^(cosine|euclidean)$")
):
    """Get nearest neighbors for a token using RAW embeddings with EUCLIDEAN as default"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if token_index >= len(viz.tokens):
        raise HTTPException(status_code=404, detail="Token index out of range")

    try:
        neighbors = viz.find_neighbors(token_index, n_neighbors, metric)
        return [TokenNeighbor(**neighbor) for neighbor in neighbors]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tokens/{token_index}/full")
async def get_token_with_neighbors(
        token_index: int,
        n_neighbors: int = Query(15, ge=1, le=100),
        metric: str = Query("euclidean", regex="^(cosine|euclidean)$"),
        include_embedding: bool = Query(False),
        include_distances: bool = Query(True, description="Include distance calculations in token details")
):
    """Get token details with neighbors using SAME METRIC for both token details and neighbors"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if token_index >= len(viz.tokens):
        raise HTTPException(status_code=404, detail="Token index out of range")

    try:
        # Use the same metric for both token details AND neighbors
        details = viz.get_token_details(token_index, include_distances=include_distances, metric=metric)
        neighbors = viz.find_neighbors(token_index, n_neighbors, metric)

        embedding_vector = None
        if include_embedding:
            # Get the RAW embedding
            embedding_vector = viz._get_raw_embedding(token_index)[0].tolist()

        return TokenWithNeighbors(
            details=TokenDetails(**details),
            neighbors=[TokenNeighbor(**neighbor) for neighbor in neighbors],
            embedding_vector=embedding_vector
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_tokens(
        query: str = Query(..., min_length=1),
        max_results: int = Query(50, ge=1, le=1000)
):
    """Search for tokens matching a query"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        results = viz.search_tokens(query, max_results)
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_tokens(request: ComparisonRequest):
    """Compare two tokens and their embeddings using RAW embeddings"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        result = viz.compare_tokens_by_name(request.token1, request.token2)
        if result is None:
            # Find which token wasn't found
            found_token1 = any(token == request.token1 for token in viz.tokens)
            found_token2 = any(token == request.token2 for token in viz.tokens)

            if not found_token1:
                raise HTTPException(status_code=404, detail=f"Token '{request.token1}' not found")
            else:
                raise HTTPException(status_code=404, detail=f"Token '{request.token2}' not found")

        return ComparisonResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare/by-id")
async def compare_tokens_by_id(request: ComparisonByIdRequest):
    """Compare two tokens by their indices using RAW embeddings"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        result = viz.compare_tokens_by_id(request.token1_index, request.token2_index)
        if result is None:
            # Check which index is out of range
            if request.token1_index >= len(viz.tokens) or request.token1_index < 0:
                raise HTTPException(status_code=404, detail=f"Token index {request.token1_index} out of range")
            else:
                raise HTTPException(status_code=404, detail=f"Token index {request.token2_index} out of range")

        return ComparisonResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
async def batch_analyze_tokens(request: BatchAnalysisRequest):
    """Analyze multiple tokens and return similarity matrix using RAW embeddings"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        # Find indices for all tokens
        token_indices = {}
        for token in request.tokens:
            for i, t in enumerate(viz.tokens):
                if t.lower() == token.lower():
                    token_indices[token] = i
                    break

        # Filter out tokens not found
        valid_tokens = list(token_indices.keys())
        if len(valid_tokens) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid tokens for comparison")

        # Get RAW embeddings
        embeddings = []
        for token in valid_tokens:
            selected_idx = token_indices[token]
            raw_embedding = viz._get_raw_embedding(selected_idx)[0]
            embeddings.append(raw_embedding)

        embeddings = np.array(embeddings)

        # Calculate similarity matrix using RAW embeddings
        similarity_matrix = cosine_similarity(embeddings)

        # Convert to serializable format
        result = {
            "tokens": valid_tokens,
            "similarity_matrix": similarity_matrix.tolist(),
            "not_found": [token for token in request.tokens if token not in valid_tokens]
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/token")
async def export_token_data(
        request: ExportRequest,
        metric: str = Query("euclidean", regex="^(cosine|euclidean)$", description="Distance metric for neighbors and token details")
):
    """Export comprehensive data for a specific token using SAME METRIC for details and neighbors"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    if request.token_index >= len(viz.tokens):
        raise HTTPException(status_code=404, detail="Token index out of range")

    try:
        # Get token details with distance information using SAME METRIC as neighbors
        details = viz.get_token_details(request.token_index, include_distances=True, metric=metric)

        # Get neighbors using SAME METRIC as token details
        neighbors = []
        if request.include_neighbors:
            neighbors = viz.find_neighbors(request.token_index, request.n_neighbors, metric=metric)

        # Get the RAW embedding vector
        embedding_vector = viz._get_raw_embedding(request.token_index)[0].tolist()

        # Create export data
        export_data = {
            'selected_token': details,
            'neighbors': neighbors,
            'neighbor_count': len(neighbors),
            'embedding_vector': embedding_vector,
            'distance_metric_used': metric,
            'export_timestamp': datetime.now().isoformat(),
            'model_name': viz.current_model_name,
            'session_id': viz.session_id,
            'api_version': '0.2'
        }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"token_export_{request.token_index}_n{request.n_neighbors}_{metric}_{timestamp}.json"
        filepath = EXPORT_DIR / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='application/json'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/metadata")
async def export_all_metadata(
        metric: str = Query("euclidean", regex="^(cosine|euclidean)$", description="Distance metric for metadata calculations")
):
    """Export all token metadata to CSV using specified distance metric"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        # Create comprehensive metadata using SAME metric for all tokens
        data = []
        for i in range(len(viz.tokens)):
            entry = viz.get_token_details(i, include_distances=True, metric=metric)
            data.append(entry)

        df = pd.DataFrame(data)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = re.sub(r'[^\w\-_\.]', '_', viz.current_model_name or 'unknown')
        filename = f"token_metadata_{model_name_clean}_{len(data)}tokens_{metric}_{timestamp}.csv"
        filepath = EXPORT_DIR / filename

        df.to_csv(filepath, index=False)

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type='text/csv'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/statistics")
async def get_analysis_statistics():
    """Get comprehensive statistics about the current model and tokens"""
    if viz.tokens is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        # Count token types
        type_counts = {}
        for t in viz.token_metadata['types']:
            type_counts[t] = type_counts.get(t, 0) + 1

        # Calculate embedding statistics
        embedding_norms = viz.token_metadata['embedding_norm']
        token_lengths = viz.token_metadata['lengths']

        statistics = {
            "model_info": {
                "name": viz.current_model_name,
                "vocabulary_size": viz.embeddings.shape[0],
                "embedding_dimension": viz.embeddings.shape[1],
                "tokens_loaded": len(viz.tokens),
                "api_version": "0.2"
            },
            "token_distribution": {
                "by_type": type_counts,
                "by_length": {
                    "min": min(token_lengths),
                    "max": max(token_lengths),
                    "mean": sum(token_lengths) / len(token_lengths),
                    "median": sorted(token_lengths)[len(token_lengths) // 2]
                }
            },
            "embedding_statistics": {
                "norm": {
                    "min": float(min(embedding_norms)),
                    "max": float(max(embedding_norms)),
                    "mean": float(sum(embedding_norms) / len(embedding_norms)),
                    "std": float(np.std(embedding_norms))
                }
            },
            "special_characteristics": {
                "has_special_chars": sum(viz.token_metadata['has_special_chars']),
                "is_uppercase": sum(viz.token_metadata['is_uppercase']),
                "is_digit": sum(viz.token_metadata['is_digit'])
            }
        }

        return statistics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/exports")
async def list_export_files():
    """List all available export files"""
    try:
        files = []
        for filepath in EXPORT_DIR.iterdir():
            if filepath.is_file():
                stat = filepath.stat()
                files.append({
                    "filename": filepath.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        return {
            "files": sorted(files, key=lambda x: x['created'], reverse=True),
            "total_files": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/download/{filename}")
async def download_export_file(filename: str):
    """Download a specific export file"""
    filepath = EXPORT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on extension
    if filename.endswith('.json'):
        media_type = 'application/json'
    elif filename.endswith('.csv'):
        media_type = 'text/csv'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type
    )


@app.delete("/files/cleanup")
async def cleanup_old_files(
        older_than_hours: int = Query(1, ge=1, le=168)  # Default 1 hour, max 1 week
):
    """Clean up old export files - handles session-based file cleanup properly"""
    try:
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (older_than_hours * 3600)

        deleted_files = []
        total_files_checked = 0

        # Clean export directory
        for filepath in EXPORT_DIR.iterdir():
            if filepath.is_file():
                total_files_checked += 1
                file_mtime = filepath.stat().st_mtime

                # Delete files older than cutoff time OR files from current session older than 5 minutes
                should_delete = (
                        file_mtime < cutoff_time or
                        (viz.session_id in filepath.name and (current_time - file_mtime) > 300)  # 5 minutes
                )

                if should_delete:
                    try:
                        filepath.unlink()
                        deleted_files.append(f"exports/{filepath.name}")
                    except Exception as e:
                        print(f"Error deleting {filepath}: {e}")

        return {
            "success": True,
            "deleted_files": deleted_files,
            "count": len(deleted_files),
            "total_checked": total_files_checked,
            "cleanup_criteria": f"Files older than {older_than_hours}h or session files older than 5min"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

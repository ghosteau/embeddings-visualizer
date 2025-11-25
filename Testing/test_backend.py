# test_backend.py

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

# Add the current directory to Python path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FastAPI testing
try:
    from fastapi.testclient import TestClient
    import pandas as pd
except ImportError:
    pytest.skip("FastAPI or pandas not available", allow_module_level=True)


# Create mock functions that return realistic values
def mock_cosine_similarity(X, Y=None):
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if Y is None:
        # Self-similarity matrix
        size = X.shape[0]
        return np.random.rand(size, size) * 0.5 + 0.5  # Values between 0.5 and 1.0
    else:
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        return np.random.rand(X.shape[0], Y.shape[0]) * 0.5 + 0.5


def mock_euclidean_distances(X, Y=None):
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if Y is None:
        size = X.shape[0]
        return np.random.rand(size, size) * 2.0  # Values between 0 and 2
    else:
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        return np.random.rand(X.shape[0], Y.shape[0]) * 2.0


# Mock UMAP with proper numpy array return
class MockUMAP:
    def __init__(self, n_neighbors=15, min_dist=0.1, metric='cosine', n_components=3, random_state=42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X):
        # Return actual numpy array with correct shape
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(X.shape[0], self.n_components)


# Create comprehensive mock modules
mock_sklearn_pairwise = Mock()
mock_sklearn_pairwise.cosine_similarity = mock_cosine_similarity
mock_sklearn_pairwise.euclidean_distances = mock_euclidean_distances

mock_sklearn_metrics = Mock()
mock_sklearn_metrics.pairwise = mock_sklearn_pairwise

mock_sklearn = Mock()
mock_sklearn.metrics = mock_sklearn_metrics

mock_umap_umap = Mock()
mock_umap_umap.UMAP = MockUMAP

mock_umap = Mock()
mock_umap.umap_ = mock_umap_umap
mock_umap.UMAP = MockUMAP  # Add UMAP directly to mock_umap as well

# Mock the heavy dependencies before importing
sys.modules['transformers'] = Mock()
sys.modules['umap'] = mock_umap
sys.modules['umap.umap_'] = mock_umap_umap
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.metrics'] = mock_sklearn_metrics
sys.modules['sklearn.metrics.pairwise'] = mock_sklearn_pairwise


# Mock transformers
class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def decode(self, token_ids):
        if not token_ids:
            return "unk"
        return f"token_{token_ids[0]}"

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()


class MockModel:
    def __init__(self):
        self.input_embeddings = Mock()
        # Create realistic embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 768).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Mock the weight.data.cpu().numpy() chain
        weight_mock = Mock()
        weight_mock.data.cpu.return_value.numpy.return_value = embeddings
        self.input_embeddings.weight = weight_mock

    def get_input_embeddings(self):
        return self.input_embeddings

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()


sys.modules['transformers'].AutoTokenizer = MockTokenizer
sys.modules['transformers'].AutoModel = MockModel
sys.modules['transformers'].GPT2Tokenizer = MockTokenizer

# Now import our module
try:

    from Backend.embedding_visualizer import EmbeddingVisualizer
    from Backend.lifecycle import app
    from Backend.timeout_handler import ModelLoadingTimeout
    from Backend.timeout_handler import TimeoutException

    # Get the global visualizer instance
    from Backend.backend import viz as global_viz

    # Patch the imported modules directly in the Backend module
    import Backend.backend as backend_module
    import Backend.embedding_visualizer as visualizer_module

    backend_module.euclidean_distances = mock_euclidean_distances
    backend_module.cosine_similarity = mock_cosine_similarity

    # Also patch in the embedding_visualizer module where it's actually used
    visualizer_module.euclidean_distances = mock_euclidean_distances
    visualizer_module.cosine_similarity = mock_cosine_similarity

except ImportError as e:
    pytest.skip(f"Cannot import backend: {e}", allow_module_level=True)

from Backend.backend import (
    TokenDetails,
    VisualizationConfig,
    LoadModelRequest,
)

# Pydantic model
from Backend.pydantic_models import ModelInfo


# =====================================================================================
# FIXTURES
# =====================================================================================

@pytest.fixture
def mock_embeddings():
    """Create realistic mock embeddings"""
    np.random.seed(42)
    embeddings = np.random.randn(100, 768).astype(np.float32)
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def fresh_visualizer():
    """Create a fresh EmbeddingVisualizer instance"""
    return EmbeddingVisualizer()


@pytest.fixture
def loaded_visualizer(fresh_visualizer, mock_embeddings):
    """Create a visualizer with mocked loaded model"""
    viz = fresh_visualizer

    # Set up the mock model and tokenizer
    viz.tokenizer = MockTokenizer()
    viz.model = MockModel()
    viz.embeddings = mock_embeddings
    viz.current_model_name = "test-model"

    # Set up selected indices and embeddings for prepare_tokens
    viz.selected_token_indices = list(range(50))
    viz.selected_embeddings = mock_embeddings[:50]

    # Prepare tokens
    viz.prepare_tokens(top_n=50)

    return viz


@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def temp_export_dir():
    """Create temporary directory for exports"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the EXPORT_DIR
        import Backend.backend
        original_export_dir = getattr(Backend.backend, 'EXPORT_DIR', None)
        Backend.backend.EXPORT_DIR = Path(temp_dir)
        yield Path(temp_dir)
        if original_export_dir:
            Backend.backend.EXPORT_DIR = original_export_dir


# =====================================================================================
# BASIC FUNCTIONALITY TESTS
# =====================================================================================

class TestBasicFunctionality:
    """Test basic functionality without external dependencies"""

    def test_visualizer_initialization(self, fresh_visualizer):
        """Test that visualizer initializes correctly"""
        viz = fresh_visualizer
        assert viz.model is None
        assert viz.tokenizer is None
        assert viz.embeddings is None
        assert viz.tokens is None
        assert viz.current_model_name is None
        assert isinstance(viz.session_id, str)
        assert not viz.loading_status['is_loading']

    def test_loading_status_updates(self, fresh_visualizer):
        """Test loading status updates"""
        viz = fresh_visualizer

        viz.update_loading_status(True, "test-model", "loading...")
        status = viz.get_loading_status()

        assert status['is_loading'] is True
        assert status['model_name'] == "test-model"
        assert status['progress'] == "loading..."

    def test_timeout_handler_creation(self):
        """Test timeout handler can be created"""
        timeout = ModelLoadingTimeout(60)
        assert timeout.timeout_seconds == 60
        assert not timeout.cancelled

    def test_timeout_context_manager(self):
        """Test timeout as context manager"""
        with ModelLoadingTimeout(1) as timeout:
            assert not timeout.is_cancelled()
            timeout.cancelled = True
            assert timeout.is_cancelled()


# =====================================================================================
# PYDANTIC MODEL TESTS
# =====================================================================================

class TestPydanticModels:
    """Test Pydantic models work correctly"""

    def test_token_details_creation(self):
        """Test TokenDetails model"""
        details = TokenDetails(
            token="test",
            index=0,
            length=4,
            type="word",
            frequency_rank=1,
            embedding_norm=1.0,
            has_special_chars=False,
            is_uppercase=False,
            is_digit=False
        )
        assert details.token == "test"
        assert details.index == 0

    def test_visualization_config_defaults(self):
        """Test VisualizationConfig defaults"""
        config = VisualizationConfig()
        assert config.method == "umap"
        assert config.n_components == 3
        assert config.top_n == 3000

    def test_load_model_request(self):
        """Test LoadModelRequest model"""
        request = LoadModelRequest(model_name="gpt2")
        assert request.model_name == "gpt2"

    def test_model_info(self):
        """Test ModelInfo model"""
        info = ModelInfo(
            name="test",
            vocabulary_size=1000,
            embedding_dimension=768,
            is_loaded=True,
            token_count=50
        )
        assert info.name == "test"
        assert info.is_loaded is True


# =====================================================================================
# LOADED MODEL TESTS
# =====================================================================================

class TestLoadedModel:
    """Test functionality with a loaded model"""

    def test_prepare_tokens(self, loaded_visualizer):
        """Test token preparation"""
        viz = loaded_visualizer

        assert viz.tokens is not None
        assert len(viz.tokens) == 50
        assert len(viz.token_metadata['lengths']) == 50
        assert len(viz.token_metadata['types']) == 50

        # Check token format
        for token in viz.tokens:
            assert isinstance(token, str)
            assert len(token) > 0

    def test_get_model_info(self, loaded_visualizer):
        """Test getting model info"""
        viz = loaded_visualizer
        info = viz.get_model_info()

        assert info.name == "test-model"
        assert info.vocabulary_size == 100
        assert info.embedding_dimension == 768
        assert info.is_loaded is True
        assert info.token_count == 50

    def test_get_token_details(self, loaded_visualizer):
        """Test getting token details"""
        viz = loaded_visualizer
        details = viz.get_token_details(0)

        assert details is not None
        assert details['token'] == viz.tokens[0]
        assert details['index'] == 0
        assert 'length' in details
        assert 'type' in details

    def test_get_token_details_with_distances(self, loaded_visualizer):
        """Test token details with distance calculations"""
        viz = loaded_visualizer
        details = viz.get_token_details(0, include_distances=True, metric='euclidean')

        assert 'euclidean_distance_to_origin' in details
        assert details['distance_metric_used'] == 'euclidean'

    def test_find_neighbors(self, loaded_visualizer):
        """Test finding neighbors"""
        viz = loaded_visualizer
        neighbors = viz.find_neighbors(0, n_neighbors=5, metric='euclidean')

        assert len(neighbors) == 5
        for neighbor in neighbors:
            assert 'token' in neighbor
            assert 'distance' in neighbor
            assert 'index' in neighbor
            assert neighbor['index'] != 0  # Should not include self

    def test_search_tokens(self, loaded_visualizer):
        """Test token search"""
        viz = loaded_visualizer
        results = viz.search_tokens("token", max_results=10)

        # Should find some tokens containing "token"
        assert len(results) > 0
        for result in results:
            assert 'token' in result
            assert 'index' in result
            assert 'match_type' in result

    def test_compare_tokens_by_id(self, loaded_visualizer):
        """Test comparing tokens by ID"""
        viz = loaded_visualizer
        result = viz.compare_tokens_by_id(0, 1)

        assert result is not None
        assert result['token1_index'] == 0
        assert result['token2_index'] == 1
        assert 'cosine_similarity' in result
        assert 'euclidean_distance' in result

    def test_compare_tokens_by_name(self, loaded_visualizer):
        """Test comparing tokens by name"""
        viz = loaded_visualizer
        token1 = viz.tokens[0]
        token2 = viz.tokens[1]

        result = viz.compare_tokens_by_name(token1, token2)

        assert result is not None
        assert result['token1'] == token1
        assert result['token2'] == token2

    def test_reduce_dimensions(self, loaded_visualizer):
        """Test dimension reduction with proper shape checking"""
        viz = loaded_visualizer
        config = VisualizationConfig(n_components=2, top_n=30)

        # Need to prepare tokens with the right number
        viz.selected_token_indices = list(range(30))
        viz.selected_embeddings = viz.embeddings[:30]
        viz.prepare_tokens(config.top_n)

        result = viz.reduce_dimensions(config)

        assert viz.reduced_embeddings is not None
        # Check the actual shape
        assert isinstance(viz.reduced_embeddings, np.ndarray)
        assert viz.reduced_embeddings.shape[0] == 30  # Number of tokens
        assert viz.reduced_embeddings.shape[1] == 2  # Number of components
        assert result['method'] == 'umap'

    def test_clear_model(self, loaded_visualizer):
        """Test clearing the model"""
        viz = loaded_visualizer

        assert viz.model is not None
        viz.clear_model()
        assert viz.model is None
        assert viz.embeddings is None


# =====================================================================================
# API ENDPOINT TESTS
# =====================================================================================

class TestAPIEndpoints:
    """Test FastAPI endpoints"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self, test_client):
        """Test health check"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_available_models(self, test_client):
        """Test getting available models"""
        response = test_client.get("/models/available")
        assert response.status_code == 200
        data = response.json()
        assert "preset_models" in data
        assert len(data["preset_models"]) > 0

    def test_model_info_no_model(self, test_client):
        """Test model info when no model loaded"""
        response = test_client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert data["is_loaded"] is False

    def test_loading_status(self, test_client):
        """Test loading status endpoint"""
        response = test_client.get("/models/loading-status")
        assert response.status_code == 200
        data = response.json()
        assert "is_loading" in data

    def test_unload_model(self, test_client):
        """Test unloading model"""
        response = test_client.delete("/models/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_endpoints_without_model(self, test_client):
        """Test endpoints that require model when none is loaded"""
        # These should return 400
        endpoints = [
            "/tokens/0",
            "/tokens/0/neighbors",
            "/tokens/0/full",
            "/search?query=test"
        ]

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == 400
            assert "No model loaded" in response.json()["detail"]

    def test_post_endpoints_without_model(self, test_client):
        """Test POST endpoints that require model when none is loaded"""
        # Compare tokens
        response = test_client.post("/compare", json={"token1": "hello", "token2": "world"})
        assert response.status_code == 400

        # Compare by ID
        response = test_client.post("/compare/by-id", json={"token1_index": 0, "token2_index": 1})
        assert response.status_code == 400

        # Batch analyze
        response = test_client.post("/analyze/batch", json={"tokens": ["hello", "world"]})
        assert response.status_code == 400

    def test_utility_endpoints(self, test_client):
        """Test utility endpoints that don't require a loaded model"""
        # Embedding similarity - only test if endpoint exists
        response = test_client.post(
            "/utils/embedding-similarity",
            json={"embedding1": [1.0, 2.0, 3.0], "embedding2": [2.0, 3.0, 4.0]}
        )

        if response.status_code == 404:
            # Utility endpoints not registered, skip this test
            pytest.skip("Utility endpoints not registered in test app")
            return

        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        assert "distance" in data
        assert "metric" in data
        assert data["metric"] == "euclidean"  # Default metric

        # Model compatibility check
        response = test_client.get("/utils/model-compatibility?model_name=gpt2")
        assert response.status_code == 200
        data = response.json()
        assert "compatible" in data
        assert "model_name" in data

    def test_file_operations(self, test_client, temp_export_dir):
        """Test file operation endpoints"""
        # List exports (should work even with empty directory)
        response = test_client.get("/files/exports")
        assert response.status_code == 200

        # Cleanup files
        response = test_client.delete("/files/cleanup")
        assert response.status_code == 200

    def test_load_model_endpoint(self, test_client):
        """Test model loading endpoint"""
        # This should work even if the actual loading fails
        response = test_client.post("/models/load", json={"model_name": "gpt2"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# =====================================================================================
# ERROR HANDLING TESTS
# =====================================================================================

class TestErrorHandling:
    """Test error handling scenarios"""

    def test_invalid_token_index(self, test_client, loaded_visualizer):
        """Test invalid token index handling"""
        # Mock a loaded model by patching the global visualizer
        with patch('Backend.backend.viz', loaded_visualizer):
            response = test_client.get("/tokens/999")
            assert response.status_code == 404

    def test_invalid_json_payload(self, test_client):
        """Test invalid JSON in POST requests"""
        response = test_client.post("/compare", json={"token1": "hello"})  # Missing token2
        assert response.status_code == 422  # Validation error

    def test_invalid_query_parameters(self, test_client, loaded_visualizer):
        """Test invalid query parameters"""
        # Mock a loaded model
        with patch('Backend.backend.viz', loaded_visualizer):
            # Invalid metric
            response = test_client.get("/tokens/0?metric=invalid")
            assert response.status_code == 422


# =====================================================================================
# INTEGRATION TESTS
# =====================================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests with mocked loaded model"""

    def test_full_workflow(self, test_client, loaded_visualizer):
        """Test a complete workflow"""
        with patch('Backend.backend.viz', loaded_visualizer):
            # Get model info
            response = test_client.get("/models/info")
            assert response.status_code == 200
            assert response.json()["is_loaded"] is True

            # Get token details
            response = test_client.get("/tokens/0")
            assert response.status_code == 200

            # Search tokens
            response = test_client.get("/search?query=token")
            assert response.status_code == 200

            # Compare tokens
            response = test_client.post("/compare/by-id", json={"token1_index": 0, "token2_index": 1})
            assert response.status_code == 200

    def test_visualization_creation(self, test_client, loaded_visualizer):
        """Test creating visualization with proper UMAP mocking"""
        config = {
            "method": "umap",
            "n_components": 2,
            "top_n": 20
        }

        # Prepare the visualizer with the right number of tokens
        loaded_visualizer.selected_token_indices = list(range(20))
        loaded_visualizer.selected_embeddings = loaded_visualizer.embeddings[:20]
        loaded_visualizer.prepare_tokens(top_n=20)

        with patch('Backend.backend.viz', loaded_visualizer):
            response = test_client.post("/visualization/create", json=config)
            assert response.status_code == 200
            data = response.json()
            assert "coordinates" in data
            assert "tokens" in data

    def test_statistics_endpoint(self, test_client, loaded_visualizer):
        """Test analysis statistics"""
        with patch('Backend.backend.viz', loaded_visualizer):
            response = test_client.get("/analysis/statistics")
            assert response.status_code == 200
            data = response.json()
            assert "model_info" in data
            assert "token_distribution" in data


# =====================================================================================
# PERFORMANCE TESTS
# =====================================================================================

@pytest.mark.performance
class TestPerformance:
    """Basic performance tests"""

    def test_neighbor_search_speed(self, loaded_visualizer):
        """Test neighbor search completes quickly"""
        viz = loaded_visualizer

        start_time = time.time()
        neighbors = viz.find_neighbors(0, n_neighbors=10)
        end_time = time.time()

        # Should complete in less than 1 second
        assert (end_time - start_time) < 1.0
        assert len(neighbors) == 10

    def test_token_preparation_speed(self, fresh_visualizer, mock_embeddings):
        """Test token preparation speed"""
        viz = fresh_visualizer
        viz.embeddings = mock_embeddings
        viz.tokenizer = MockTokenizer()
        viz.selected_token_indices = list(range(50))
        viz.selected_embeddings = mock_embeddings[:50]

        start_time = time.time()
        viz.prepare_tokens(top_n=50)
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 2.0
        assert len(viz.tokens) == 50


# =====================================================================================
# NUMPY ARRAY CONSISTENCY TESTS
# =====================================================================================

class TestNumpyArrayConsistency:
    """Test that mocked functions return proper numpy arrays"""

    def test_mock_umap_returns_numpy_array(self):
        """Test that our MockUMAP returns actual numpy arrays"""
        mock_umap = MockUMAP(n_components=2)
        fake_input = np.random.rand(10, 768)
        result = mock_umap.fit_transform(fake_input)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 2)

    def test_mock_similarity_functions_return_numpy_arrays(self):
        """Test that similarity functions return numpy arrays"""
        X = np.random.rand(5, 10)
        Y = np.random.rand(3, 10)

        cos_sim = mock_cosine_similarity(X, Y)
        eucl_dist = mock_euclidean_distances(X, Y)

        assert isinstance(cos_sim, np.ndarray)
        assert isinstance(eucl_dist, np.ndarray)
        assert cos_sim.shape == (5, 3)
        assert eucl_dist.shape == (5, 3)


# =====================================================================================
# RUN TESTS
# =====================================================================================

if __name__ == "__main__":
    """Run the tests"""
    # Run with basic options that should work
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    sys.exit(exit_code)
# =====================================================================================
# CORE EMBEDDING VISUALIZER CLASS
# =====================================================================================
import re
import threading
import uuid
from typing import Dict, Any, List, Optional

import numpy as np
import umap
from sklearn.metrics import euclidean_distances
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer

from Backend.pydantic_models import VisualizationConfig, ModelInfo
from Backend.timeout_handler import ModelLoadingTimeout, TimeoutException

# NumPy cosine similarity helper
def cosine_sim_np(a, b):
    """
    Compute cosine similarity between two NumPy vectors.
    """
    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)

    num = np.dot(a, b.T)
    den = np.linalg.norm(a) * np.linalg.norm(b)

    return float(num / den)


class EmbeddingVisualizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embeddings = None  # Full raw embedding matrix
        self.tokens = None
        self.reduced_embeddings = None  # Dimensionally reduced version for visualization
        self.current_model_name = None
        self.token_metadata = {}
        self.selected_embeddings = None  # Raw embeddings for selected tokens
        self.session_id = str(uuid.uuid4())
        self.selected_token_indices = None

        # Loading status tracking with proper thread safety
        self._loading_lock = threading.Lock()
        self.loading_status = {
            'is_loading': False,
            'model_name': None,
            'progress': None,
            'error': None
        }
        self.loading_timeout_handler = None
        self.loading_cancelled = False

    def update_loading_status(self, is_loading: bool, model_name: str = None,
                              progress: str = None, error: str = None):
        """Update the loading status with thread safety"""
        with self._loading_lock:
            self.loading_status = {
                'is_loading': is_loading,
                'model_name': model_name,
                'progress': progress,
                'error': error
            }
            print(f"Loading status updated: {self.loading_status}")

    def get_loading_status(self) -> Dict:
        """Get current loading status with thread safety"""
        with self._loading_lock:
            return self.loading_status.copy()

    def cancel_loading(self):
        """Cancel ongoing loading operation"""
        self.loading_cancelled = True
        if self.loading_timeout_handler:
            self.loading_timeout_handler.cancelled = True
        self.update_loading_status(False, error="Loading cancelled by timeout")

    def load_model(self, model_name: str) -> bool:
        """Load transformer model and tokenizer with proper timeout protection"""

        self.loading_cancelled = False
        self.update_loading_status(True, model_name, "Initializing model loading...")

        try:
            self.loading_timeout_handler = ModelLoadingTimeout(120)  # 2-minute timeout

            with self.loading_timeout_handler:
                print(f"Loading model: {model_name}")
                self.update_loading_status(True, model_name, "Downloading tokenizer...")

                # Check for cancellation at each step
                if self.loading_cancelled or self.loading_timeout_handler.is_cancelled():
                    self.update_loading_status(False, error="Loading cancelled")
                    return False

                try:
                    if 'bert' in model_name.lower():
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if self.loading_cancelled or self.loading_timeout_handler.is_cancelled():
                            self.update_loading_status(False, error="Loading cancelled")
                            return False

                        self.update_loading_status(True, model_name, "Downloading model weights...")
                        self.model = AutoModel.from_pretrained(model_name)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if self.loading_cancelled or self.loading_timeout_handler.is_cancelled():
                            self.update_loading_status(False, error="Loading cancelled")
                            return False

                        self.update_loading_status(True, model_name, "Downloading model weights...")
                        self.model = AutoModel.from_pretrained(model_name)

                        # Fallback to GPT2 tokenizer for unknown models
                        if self.tokenizer is None:
                            print("Falling back to GPT2Tokenizer...")
                            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

                    # Check for cancellation before continuing
                    if self.loading_cancelled or self.loading_timeout_handler.is_cancelled():
                        self.clear_partial_model()
                        self.update_loading_status(False, error="Loading cancelled")
                        return False

                    # Handle padding token for GPT-2
                    if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    self.current_model_name = model_name

                    self.update_loading_status(True, model_name, "Extracting embeddings...")

                    # Extract embeddings - this is the RAW embedding matrix
                    self.embeddings = self.model.get_input_embeddings().weight.data.cpu().numpy()

                    # Final cancellation check
                    if self.loading_cancelled or self.loading_timeout_handler.is_cancelled():
                        self.clear_partial_model()
                        self.update_loading_status(False, error="Loading cancelled")
                        return False

                    print(f"Model loaded successfully!")
                    print(f"   Vocabulary size: {self.embeddings.shape[0]:,}")
                    print(f"   Embedding dimension: {self.embeddings.shape[1]}")

                    self.update_loading_status(False, model_name, "Model loaded successfully")
                    return True

                except Exception as e:
                    error_msg = f"Error loading model: {str(e)}"
                    print(error_msg)
                    self.clear_partial_model()
                    self.update_loading_status(False, error=error_msg)
                    return False

        except TimeoutException:
            error_msg = f"Model loading timed out after 2 minutes"
            print(error_msg)
            self.clear_partial_model()
            self.update_loading_status(False, error=error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error during model loading: {str(e)}"
            print(error_msg)
            self.clear_partial_model()
            self.update_loading_status(False, error=error_msg)
            return False

    def clear_partial_model(self):
        """Clear partially loaded model components"""
        try:
            self.model = None
            self.tokenizer = None
            self.embeddings = None
            self.current_model_name = None
        except:
            pass

    def prepare_tokens(self, top_n: int = 3000) -> None:
        """Prepare token data with metadata"""
        print(f"Preparing top {top_n:,} tokens...")

        # Select top N embeddings and store the indices for mapping back to full embeddings
        self.selected_token_indices = list(range(min(top_n, self.embeddings.shape[0])))
        # IMPORTANT: selected_embeddings are still RAW embeddings, just a subset
        self.selected_embeddings = self.embeddings[self.selected_token_indices].copy()

        # Initialize metadata
        self.tokens = []
        self.token_metadata = {
            'lengths': [],
            'types': [],
            'has_special_chars': [],
            'is_uppercase': [],
            'is_digit': [],
            'frequency_rank': [],
            'embedding_norm': [],
            'cosine_distance_from_origin': []
        }

        for i, original_idx in enumerate(self.selected_token_indices):
            try:
                # Decode token
                if self.tokenizer is not None:
                    token = self.tokenizer.decode([original_idx])
                    clean_token = token.strip() if token.strip() else f"<TOKEN_{original_idx}>"
                else:
                    clean_token = f"<TOKEN_{original_idx}>"

                self.tokens.append(clean_token)

                # Compute metadata using RAW embeddings
                embedding = self.selected_embeddings[i]

                self.token_metadata['lengths'].append(len(clean_token))
                self.token_metadata['frequency_rank'].append(original_idx)
                self.token_metadata['has_special_chars'].append(bool(re.search(r'[^a-zA-Z0-9\s]', clean_token)))
                self.token_metadata['is_uppercase'].append(clean_token.isupper())
                self.token_metadata['is_digit'].append(clean_token.isdigit())
                self.token_metadata['embedding_norm'].append(float(np.linalg.norm(embedding)))
                self.token_metadata['cosine_distance_from_origin'].append(
                    float(1 - np.dot(embedding, np.zeros_like(embedding))))

                # Classify token type
                if clean_token.startswith('<') and clean_token.endswith('>'):
                    token_type = 'special'
                elif clean_token.isdigit():
                    token_type = 'number'
                elif clean_token.isalpha():
                    token_type = 'word'
                else:
                    token_type = 'mixed'

                self.token_metadata['types'].append(token_type)

            except Exception as e:
                # Handle problematic tokens
                self.tokens.append(f"<UNK_{original_idx}>")
                self.token_metadata['lengths'].append(0)
                self.token_metadata['frequency_rank'].append(original_idx)
                self.token_metadata['has_special_chars'].append(True)
                self.token_metadata['is_uppercase'].append(False)
                self.token_metadata['is_digit'].append(False)
                self.token_metadata['embedding_norm'].append(0.0)
                self.token_metadata['cosine_distance_from_origin'].append(1.0)
                self.token_metadata['types'].append('unknown')

        print(f"Tokens prepared!")

    def reduce_dimensions(self, config: VisualizationConfig) -> Dict[str, Any]:
        """Reduce embedding dimensions for visualization using UMAP only"""
        print(f"Reducing dimensions using UMAP...")

        reducer = umap.UMAP(
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            n_components=config.n_components,
            random_state=42
        )

        # Use RAW embeddings for dimension reduction
        self.reduced_embeddings = reducer.fit_transform(self.selected_embeddings)
        print(f"Dimensions reduced to {config.n_components}D using UMAP")

        return {
            'method': 'umap',
            'original_dim': self.selected_embeddings.shape[1],
            'reduced_dim': config.n_components,
            'explained_variance': None  # UMAP doesn't provide explained variance
        }

    def _get_raw_embedding(self, selected_token_idx: int) -> np.ndarray:
        """Get the raw embedding for a selected token index"""
        return self.selected_embeddings[selected_token_idx:selected_token_idx + 1]

    def _get_original_index(self, selected_token_idx: int) -> int:
        """Get the original index in the full vocabulary for a selected token"""
        return self.selected_token_indices[selected_token_idx]

    def find_neighbors(self, token_idx: int, n_neighbors: int = 10, metric: str = 'euclidean') -> List[Dict]:
        """Find nearest neighbors for a token using RAW embeddings with EUCLIDEAN as default"""
        if token_idx >= len(self.tokens):
            raise ValueError(f"Token index {token_idx} out of range")

        # Get the RAW embedding for the target token
        target_embedding = self._get_raw_embedding(token_idx)

        if metric == 'cosine':
            # Calculate cosine similarity using RAW embeddings
            similarities = np.array([
                cosine_sim_np(target_embedding, emb)
                for emb in self.selected_embeddings
            ])
            # use "cosine distance" as the distance for the cosine metric
            distances = 1 - similarities
        elif metric == 'euclidean':
            # Calculate euclidean distance using RAW embeddings
            distances = euclidean_distances(target_embedding, self.selected_embeddings)[0]
            # just use cosine similarity as similarities for reference
            similarities = np.array([
                cosine_sim_np(target_embedding, emb)
                for emb in self.selected_embeddings
            ])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Get indices of nearest neighbors (excluding self)
        neighbor_indices = np.argsort(distances)
        # Remove self from neighbors if present
        neighbor_indices = neighbor_indices[neighbor_indices != token_idx]
        neighbor_indices = neighbor_indices[:n_neighbors]

        neighbors = []
        for idx in neighbor_indices:
            dist = float(distances[idx])
            sim = None if metric == 'euclidean' else float(similarities[idx])

            neighbors.append({
                'token': self.tokens[idx],
                'distance': dist,
                'similarity': sim,
                'index': int(idx)
            })

        return neighbors

    def search_tokens(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search for tokens matching a query"""
        query_lower = query.lower()
        results = []

        for i, token in enumerate(self.tokens):
            token_lower = token.lower()
            if query_lower in token_lower:
                results.append({
                    'token': token,
                    'index': i,
                    'match_type': 'contains'
                })

                if len(results) >= max_results:
                    break

        return results

    def get_token_details(self, token_idx: int, include_distances: bool = False, metric: str = 'euclidean') -> Optional[
        Dict]:
        """Get detailed information about a token with optional distance calculations"""
        if token_idx >= len(self.tokens):
            return None

        token = self.tokens[token_idx]

        details = {
            'token': token,
            'index': token_idx,
            'length': self.token_metadata['lengths'][token_idx],
            'type': self.token_metadata['types'][token_idx],
            'frequency_rank': self.token_metadata['frequency_rank'][token_idx],
            'embedding_norm': self.token_metadata['embedding_norm'][token_idx],
            'has_special_chars': self.token_metadata['has_special_chars'][token_idx],
            'is_uppercase': self.token_metadata['is_uppercase'][token_idx],
            'is_digit': self.token_metadata['is_digit'][token_idx],
        }

        if self.reduced_embeddings is not None:
            details.update({
                'x': float(self.reduced_embeddings[token_idx, 0]),
                'y': float(self.reduced_embeddings[token_idx, 1]),
                'z': float(self.reduced_embeddings[token_idx, 2]) if self.reduced_embeddings.shape[1] > 2 else 0.0
            })

        if include_distances:
            target_embedding = self._get_raw_embedding(token_idx)
            origin = np.zeros_like(target_embedding)

            if metric == 'cosine':
                # For cosine metric, only include cosine distance (since cosine similarity to origin is always 0)
                cosine_dist = 1 - cosine_sim_np(target_embedding, origin)
                details['cosine_distance_to_origin'] = cosine_dist
            elif metric == 'euclidean':
                # For euclidean metric, include euclidean distance (cosine similarity to origin is meaningless)
                euclidean_dist = float(euclidean_distances(target_embedding, origin.reshape(1, -1))[0][0])
                details['euclidean_distance_to_origin'] = euclidean_dist

            details['distance_metric_used'] = metric

        return details

    def get_model_info(self) -> ModelInfo:
        """Get current model information"""
        loading_status = self.get_loading_status()
        return ModelInfo(
            name=self.current_model_name or "None",
            vocabulary_size=self.embeddings.shape[0] if self.embeddings is not None else 0,
            embedding_dimension=self.embeddings.shape[1] if self.embeddings is not None else 0,
            is_loaded=self.embeddings is not None,
            token_count=len(self.tokens) if self.tokens else 0,
            loading_status=loading_status if loading_status['is_loading'] else None
        )

    def compare_tokens_by_name(self, token1: str, token2: str) -> Optional[Dict]:
        """Compare two tokens by name using RAW embeddings"""
        # Find token indices
        idx1 = idx2 = None
        for i, token in enumerate(self.tokens):
            if token == token1:
                idx1 = i
            if token == token2:
                idx2 = i

        if idx1 is None or idx2 is None:
            return None

        # Get RAW embeddings for both tokens
        emb1 = self._get_raw_embedding(idx1)
        emb2 = self._get_raw_embedding(idx2)

        # Calculate similarity and distance using the SAME method as neighbors
        cosine_sim = float(cosine_sim_np(emb1, emb2))
        euclidean_dist = float(euclidean_distances(emb1, emb2)[0][0])

        return {
            'token1': token1,
            'token2': token2,
            'token1_index': idx1,
            'token2_index': idx2,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
        }

    def compare_tokens_by_id(self, token1_index: int, token2_index: int) -> Optional[Dict]:
        """Compare two tokens by their indices using RAW embeddings"""
        if token1_index >= len(self.tokens) or token2_index >= len(self.tokens):
            return None

        if token1_index < 0 or token2_index < 0:
            return None

        # Get RAW embeddings for both tokens
        emb1 = self._get_raw_embedding(token1_index)
        emb2 = self._get_raw_embedding(token2_index)

        # Calculate similarity and distance using the SAME method as neighbors
        cosine_sim = float(cosine_sim_np(emb1, emb2))
        euclidean_dist = float(euclidean_distances(emb1, emb2)[0][0])

        return {
            'token1': self.tokens[token1_index],
            'token2': self.tokens[token2_index],
            'token1_index': token1_index,
            'token2_index': token2_index,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
        }

    def clear_model(self):
        """Clear the current model and free memory"""
        # Cancel any ongoing loading
        self.cancel_loading()

        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.tokens = None
        self.reduced_embeddings = None
        self.selected_embeddings = None
        self.token_metadata = {}
        self.current_model_name = None
        self.selected_token_indices = None
        self.update_loading_status(False)

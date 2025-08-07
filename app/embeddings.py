import logging
import numpy as np
from annoy import AnnoyIndex
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Dataclass to hold search results"""
    content: str
    score: float
    metadata: dict = None

class DocumentEmbedder:
    """
    Handles document embedding and semantic search using Annoy.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document embedder with a pre-trained sentence transformer model.

        Args:
            model_name: Name of the pre-trained model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.metadata = []
        self.model_name = model_name

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks.

        Args:
            chunks: List of text chunks to embed

        Returns:
            np.ndarray: Array of embeddings
        """
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        return self.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    def build_index(self, chunks: List[str], metadatas: List[dict] = None, n_trees: int = 10):
        """
        Build an Annoy index from text chunks.

        Args:
            chunks: List of text chunks to index
            metadatas: Optional list of metadata dictionaries for each chunk
            n_trees: Number of trees for the Annoy index. More trees gives higher precision.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        self.documents = chunks
        self.metadata = metadatas or [{}] * len(chunks)

        embeddings = self.create_embeddings(chunks)

        self.index = AnnoyIndex(self.dimension, 'angular')
        for i, vector in enumerate(embeddings):
            self.index.add_item(i, vector)

        self.index.build(n_trees)
        logger.info(f"Built Annoy index with {len(chunks)} documents and {n_trees} trees")

    def search(self, query: str, k: int = 10, threshold: float = 0.3) -> List[SearchResult]:
        """
        Search for similar documents to the query.

        Args:
            query: The search query
            k: Number of results to return
            threshold: Minimum similarity score (0-1) for results

        Returns:
            List[SearchResult]: List of search results with content and scores
        """
        if self.index is None or not self.documents:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.model.encode([query], show_progress_bar=False)[0]

        indices, distances = self.index.get_nns_by_vector(
            query_embedding, k, include_distances=True
        )

        results: List[SearchResult] = []
        for idx, distance in zip(indices, distances):
            # Annoy's angular distance is sqrt(2 * (1 - cosine_similarity))
            # So, cosine_similarity = 1 - (distance^2) / 2
            score = 1 - (distance ** 2) / 2

            results.append(SearchResult(
                content=self.documents[idx],
                score=score,
                metadata=self.metadata[idx]
            ))

        
        results.sort(key=lambda x: x.score, reverse=True)
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        # Apply threshold filter
        filtered = [r for r in results if r.score >= threshold]
        # Fallback: if none meet threshold, return top-k anyway
        if not filtered:
            filtered = results
        return filtered[:k]

    def save_index(self, index_path: str):
        """Save the Annoy index and document data to disk."""
        if not self.index:
            raise ValueError("No index to save")

        # Create directory if it doesn't exist
        dir_name = os.path.dirname(index_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Save Annoy index
        self.index.save(f"{index_path}.ann")

        # Save document data and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'dimension': self.dimension
        }

        with open(f"{index_path}.json", 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved Annoy index and data to {index_path}.ann and {index_path}.json")

    @classmethod
    def load_index(cls, index_path: str):
        """Load a saved Annoy index and document data from disk."""
        # Load document data and metadata
        with open(f"{index_path}.json", 'r') as f:
            data = json.load(f)

        # Create new DocumentEmbedder instance
        embedder = cls(model_name=data['model_name'])
        embedder.documents = data['documents']
        embedder.metadata = data['metadata']
        
        # Load Annoy index
        embedder.index = AnnoyIndex(embedder.dimension, 'angular')
        embedder.index.load(f"{index_path}.ann")
        logger.info(f"Loaded Annoy index and data from {index_path}.ann and {index_path}.json")

        return embedder

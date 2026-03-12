"""Embedding model runner for similarity scoring."""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.common.logging import log


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr at the file descriptor level."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)

    # Open /dev/null and redirect
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


class EmbeddingRunner:
    """Runner for computing text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: Name of sentence-transformers model to use.
        """
        log(f"Loading embedding model: {model_name}")

        # Suppress "LOAD REPORT" noise printed at file descriptor level
        with suppress_stdout_stderr():
            self.model = SentenceTransformer(model_name)

        self.model_name = model_name

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> NDArray[np.float32]:
        """Compute embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Array of shape (embedding_dim,).
        """
        return self.model.encode([text], convert_to_numpy=True)[0]

    def similarity(self, text: str, reference: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text: Text to compare.
            reference: Reference text.

        Returns:
            Cosine similarity in range [0, 1] (clamped from [-1, 1]).
        """
        embeddings = self.embed([text, reference])
        # Cosine similarity
        dot = np.dot(embeddings[0], embeddings[1])
        norm = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        sim = dot / norm if norm > 0 else 0.0
        # Clamp to [0, 1] for scoring purposes
        return float(max(0.0, min(1.0, (sim + 1) / 2)))

    def similarities(
        self,
        text: str,
        references: list[str],
    ) -> list[float]:
        """Compute cosine similarities between text and multiple references.

        Args:
            text: Text to compare.
            references: List of reference texts.

        Returns:
            List of cosine similarities in range [0, 1].
        """
        if not references:
            return []

        all_texts = [text] + references
        embeddings = self.embed(all_texts)
        text_emb = embeddings[0]
        ref_embs = embeddings[1:]

        similarities = []
        text_norm = np.linalg.norm(text_emb)
        for ref_emb in ref_embs:
            dot = np.dot(text_emb, ref_emb)
            norm = text_norm * np.linalg.norm(ref_emb)
            sim = dot / norm if norm > 0 else 0.0
            # Clamp to [0, 1]
            similarities.append(float(max(0.0, min(1.0, (sim + 1) / 2))))

        return similarities

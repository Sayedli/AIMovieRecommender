"""
src/text_embed.py

Handles text preprocessing for movies and computing transformer embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any


def build_movie_text(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a text field from title + genres for embedding.

    Args:
        movies: DataFrame with at least [movieId, title, genres].

    Returns:
        DataFrame with an extra column "text".
    """
    # Replace missing values
    movies = movies.copy()
    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    # Genres like "Action|Comedy" → "Action Comedy"
    genres_clean = movies["genres"].str.replace("|", " ", regex=False)

    # Combine into one text string
    movies["text"] = movies["title"] + " [GENRES] " + genres_clean

    return movies


def compute_embeddings(
    texts: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute L2-normalized embeddings for given texts using SentenceTransformers.

    Args:
        texts: Series of strings.
        model_name: Hugging Face model name.
        batch_size: Batch size for encoding.

    Returns:
        np.ndarray [num_texts, embedding_dim]
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)
"""
src/data.py

Handles loading MovieLens data and splitting into train/validation.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_movielens(root: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens ratings and movies CSVs.

    Args:
        root: Folder containing ratings.csv and movies.csv

    Returns:
        ratings: DataFrame with [userId, movieId, rating, timestamp]
        movies:  DataFrame with [movieId, title, genres]
    """
    root = Path(root)

    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")

    return ratings, movies


def train_val_split(ratings: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and validation by taking the most recent ratings per user.

    Args:
        ratings: Ratings DataFrame.
        val_ratio: Approximate fraction of ratings to hold out for validation.
        seed: Random seed.

    Returns:
        train, val: DataFrames.
    """
    # Sort ratings per user by timestamp
    ratings = ratings.sort_values(["userId", "timestamp"])

    # Group by user and pick the last N ratings for validation
    val_idx = ratings.groupby("userId").tail(int(len(ratings) * val_ratio / ratings["userId"].nunique())).index
    val = ratings.loc[val_idx]
    train = ratings.drop(val_idx)

    return train, val
import pandas as pd
from src.data import load_movielens, train_val_split
from src.text_embed import build_movie_text, compute_embeddings
from src.mf import train_mf
from src.hybrid import HybridRecommender

# Load data
ratings, movies = load_movielens("data")
train, val = train_val_split(ratings)

# Build text & embeddings
movies = build_movie_text(movies)
emb = compute_embeddings(movies["text"])  # limit to 500 for speed test

# Train MF
mf_model, u2idx, i2idx = train_mf(train, movies, epochs=2, dim=32)

# Build user history dict
user_hist = train.groupby("userId")["movieId"].apply(list).to_dict()

# Hybrid recommender
hr = HybridRecommender(mf_model, u2idx, i2idx, emb, alpha=0.6)

# Pick a random user
some_user = int(train["userId"].sample(1).iloc[0])
top_idx, scores = hr.recommend_for_user(some_user, topk=5, exclude_seen=user_hist.get(some_user, []))

# Map back to titles
inv_i = {v: k for k, v in i2idx.items()}
rec_movie_ids = [inv_i[i] for i in top_idx]
print("User:", some_user)
print(movies[movies["movieId"].isin(rec_movie_ids)][["movieId", "title"]])
# Hybrid Movie Recommender (NLP + CF)

A small, production-ish starter:
- Content: Transformer embeddings (Hugging Face).
- Collaborative filtering: PyTorch MF.
- Hybrid blend with tunable alpha.
- Flask UI.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put MovieLens CSVs in ./data (ratings.csv, movies.csv, tags.csv optional)
export PYTHONPATH=.
python app.py  # http://localhost:5000
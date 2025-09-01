"""
Flask UI for the hybrid movie recommender (username-based).

Run:
    export PYTHONPATH=.
    python app.py
Open http://localhost:5000
"""

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_movielens
from src.text_embed import build_movie_text, compute_embeddings
from src.mf import train_mf
from src.hybrid import HybridRecommender

# ----------------------------
# Paths / Config
# ----------------------------
DATA_DIR = Path("data")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMB_PATH = MODELS_DIR / "movie_embeddings.npy"
USERS_PATH = DATA_DIR / "users.json"  # username -> internal numeric id
TOPK_DEFAULT = 10
ALPHA_DEFAULT = 0.6  # 1.0 = history/CF, 0.0 = content

# ----------------------------
# Data & Artifacts
# ----------------------------
print("Loading MovieLens data...")
ratings, movies = load_movielens(DATA_DIR.as_posix())
movies = build_movie_text(movies)

if EMB_PATH.exists():
    print(f"Loading cached embeddings: {EMB_PATH}")
    movie_embeddings = np.load(EMB_PATH)
else:
    print("Computing embeddings (first run ~1–2 min on CPU)...")
    movie_embeddings = compute_embeddings(movies["text"])
    np.save(EMB_PATH, movie_embeddings)

print("Training MF model...")
mf_model, u2idx, i2idx = train_mf(
    train_df=ratings,
    movies_df=movies,
    dim=64,
    epochs=2,
    bs=4096,
    lr=1e-2,
    device="cpu",
)
inv_i = {v: k for k, v in i2idx.items()}  # idx -> movieId
user_hist_raw: Dict[int, List[int]] = ratings.groupby("userId")["movieId"].apply(list).to_dict()

# Popularity (fallback)
popularity = (
    ratings.groupby("movieId")["rating"]
    .agg(["count", "mean"])
    .rename(columns={"count": "n_ratings", "mean": "avg"})
    .reset_index()
).sort_values(["n_ratings", "avg"], ascending=[False, False])

# Title index for search
movies_idx = movies.assign(title_lc=movies["title"].str.lower())

# Hybrid recommender
HR = HybridRecommender(mf_model, u2idx, i2idx, movie_embeddings, alpha=ALPHA_DEFAULT, device="cpu")

# ----------------------------
# Minimal user store (username -> internal userId)
# ----------------------------
def load_users() -> Dict[str, int]:
    if USERS_PATH.exists():
        try:
            with open(USERS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(d: Dict[str, int]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_PATH, "w") as f:
        json.dump(d, f, indent=2)

users_map: Dict[str, int] = load_users()
max_existing_uid = int(max(ratings["userId"])) if not ratings.empty else 0
next_user_id = max_existing_uid + 1

# In-memory feedback (username-based)
feedback_events: List[dict] = []  # {"username": str, "movie_id": int, "rating": float}

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>🎬 Movie Recommender</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { --bg:#0b0c10; --card:#14161a; --text:#e5e7eb; --muted:#94a3b8; --accent:#60a5fa; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
    header { padding:24px; border-bottom:1px solid #1f2937; text-align:center; position:relative; }
    a { color: var(--accent); text-decoration: none; }
    main { padding:24px; max-width:1000px; margin:0 auto; }
    h1 { margin:0 0 4px; font-size:24px; }
    p.muted { margin:0; color:var(--muted); }
    .tabs { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; justify-content:center; }
    .tab { padding:8px 12px; border:1px solid #1f2937; border-radius:10px; background:var(--card); color:var(--text); cursor:pointer; }
    .tab[aria-selected="true"] { outline:2px solid var(--accent); }
    .card { background:var(--card); border:1px solid #1f2937; border-radius:14px; padding:16px; margin-top:16px; }
    label { display:block; margin:8px 0 4px; color:var(--muted); }
    input[type="number"], input[type="text"] { width: 240px; background:#0f1115; color:var(--text); border:1px solid #1f2937; border-radius:10px; padding:8px 10px; }
    button { background:var(--accent); color:#0b1220; border:none; border-radius:10px; padding:8px 12px; cursor:pointer; font-weight:600; }
    button.ghost { background:transparent; color:var(--text); border:1px solid #334155; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    ol { margin:0; padding-left:18px; }
    li { margin:6px 0; }
    .hint { color:var(--muted); font-size:13px; }
    .small { font-size:13px; color:var(--muted); }
    .pill { display:inline-flex; align-items:center; gap:8px; background:#0f1115; border:1px solid #1f2937; border-radius:999px; padding:6px 10px; }
    nav { position:absolute; right:24px; top:24px; }
  </style>
</head>
<body>
<header>
  <h1 style="margin:0 0 4px;">FilmMatcher 🎬</h1>
  <p class="muted">Blend your <b>history</b> with <b>content similarity</b> to get personalized picks.</p>
  <div class="tabs" role="tablist" aria-label="Modes">
    <button class="tab" role="tab" data-tab="cold-start" aria-selected="true">Cold Start (Pick Favorites)</button>
    <button class="tab" role="tab" data-tab="by-user" aria-selected="false">Existing User</button>
    <button class="tab" role="tab" data-tab="search" aria-selected="false">Search Titles</button>
  </div>
  <nav><a href="/users">Users</a></nav>
</header>

<main>
  <!-- Existing User -->
  <section class="card" id="tab-by-user" role="tabpanel" hidden>
    <div class="row">
      <div>
        <label>Username</label>
        <input id="username" type="text" placeholder="e.g., hassan"/>
      </div>
      <div>
        <label>How many recommendations?</label>
        <input id="topK" type="number" min="1" max="50" value="10"/>
      </div>
      <div>
        <label>Blend (History ↔ Content)</label>
        <input id="alpha" type="number" min="0" max="1" step="0.05" value="0.6"/>
        <div class="small">Higher = more based on similar users’ ratings. Lower = more based on movie content/genres.</div>
      </div>
      <button id="btnUserRecs" type="button">Get Recommendations</button>
    </div>
    <p class="hint">Tip: create a user on the <a href="/users">Users</a> page, then star a few movies to improve results.</p>
    <div id="userRecs"></div>
  </section>

  <!-- Cold Start -->
  <section class="card" id="tab-cold-start" role="tabpanel">
    <p class="hint">Pick <b>exactly 10 movies</b> you like. We’ll use them to build your profile.</p>
    <div id="genreGrid"></div>
    <div class="row" style="margin-top:12px;">
      <div class="small">Selected: <span id="selCount">0</span>/10</div>
      <button id="btnColdStart" type="button" disabled>Get Recommendations</button>
    </div>
    <div id="coldRecs"></div>
  </section>

  <!-- Search -->
  <section class="card" id="tab-search" role="tabpanel" hidden>
    <div class="row">
      <div>
        <label>Search by Title</label>
        <input id="searchQuery" type="text" placeholder="e.g., Toy Story"/>
      </div>
      <button id="btnSearch" class="ghost" type="button">Search</button>
    </div>
    <div id="searchResults"></div>
  </section>
</main>

<script>
  document.addEventListener('DOMContentLoaded', () => {
    // --- helpers to avoid Safari issues ---
    function $(id){ return document.getElementById(id); }
    function on(el, ev, fn){ if(el) el.addEventListener(ev, fn); }

    // Tab controller
    var tabButtons = Array.prototype.slice.call(document.querySelectorAll('.tab'));
    var sections = {
      'by-user': $('tab-by-user'),
      'cold-start': $('tab-cold-start'),
      'search': $('tab-search')
    };

    function showTab(key) {
      for (var k in sections) {
        if (sections[k]) sections[k].hidden = (k !== key);
      }
      tabButtons.forEach(function(btn){
        btn.setAttribute('aria-selected', btn.getAttribute('data-tab') === key ? 'true' : 'false');
      });
      if (key === 'cold-start') {
        loadGenreSeeds();
      }
    }

    tabButtons.forEach(function(btn){
      on(btn, 'click', function(){ showTab(btn.getAttribute('data-tab')); });
    });

    // default tab
    showTab('cold-start');

    // Render helper
    function renderList(container, items) {
      if (!container) return;
      if (!items || items.length === 0) {
        container.innerHTML = "<p class='hint'>No results.</p>";
        return;
      }
      var html = ["<ol>"];
      items.forEach(function(r){
        html.push("<li><b>"+r.title+"</b> <span class='small'>(movieId="+r.movieId+")</span> " +
                  "<button class='ghost' type='button' onclick='rate(null, "+r.movieId+", 5)'>★5</button> " +
                  "<button class='ghost' type='button' onclick='rate(null, "+r.movieId+", 4)'>★4</button>" +
                  "</li>");
      });
      html.push("</ol>");
      container.innerHTML = html.join("");
    }

    // Existing User actions
    on($('btnUserRecs'), 'click', async function(){
      var username = ($('username') && $('username').value ? $('username').value : '').trim();
      var k = Number($('topK') && $('topK').value ? $('topK').value : 10);
      var alpha = Number($('alpha') && $('alpha').value ? $('alpha').value : 0.6);
      try {
        const res = await fetch("/api/recommend?username="+encodeURIComponent(username)+"&k="+k+"&alpha="+alpha);
        const data = await res.json();
        renderList($('userRecs'), data.recs || []);
      } catch(e){ console.error(e); }
    });

    // Search
    on($('btnSearch'), 'click', async function(){
      var q = ($('searchQuery') && $('searchQuery').value) || '';
      try {
        const res = await fetch("/api/search?q="+encodeURIComponent(q));
        const data = await res.json();
        renderList($('searchResults'), data.results || []);
      } catch(e){ console.error(e); }
    });

    // Ratings helper (Safari-safe)
    window.rate = async function(username, movieId, rating){
      var raw = username || ($('username') && $('username').value) || '';
      var u = String(raw).trim() || null;
      try {
        await fetch('/rate', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ username: u, movie_id: movieId, rating: rating })
        });
      } catch(e){ console.error(e); }
    };

    // ----- Cold Start: pick exactly 10 -----
    var csSelected = new Set();
    var genreData = []; // [{name, movies:[{movieId,title}]}]

    function updateSelectedUI(){
      if ($('selCount')) $('selCount').textContent = String(csSelected.size);
      if ($('btnColdStart')) $('btnColdStart').disabled = (csSelected.size !== 10);
    }

    function genreBlockHTML(g){
      var items = (g.movies || []).map(function(m){
        var checked = csSelected.has(m.movieId) ? 'checked' : '';
        return (
          "<label style='display:block;margin:6px 0;'>" +
          "<input type='checkbox' data-mid='"+m.movieId+"' "+checked+" /> " +
          m.title + " <span class='small'>(movieId="+m.movieId+")</span>" +
          "</label>"
        );
      }).join('');
      return (
        "<div class='card genre-block' data-genre='"+g.name+"' style='margin-top:12px;'>" +
          "<div class='row' style='justify-content:space-between;align-items:center;'>" +
            "<h3 style='margin:0 0 6px;'>"+g.name+"</h3>" +
            "<button class='ghost' type='button' data-shuffle='"+g.name+"'>🔄 Shuffle</button>" +
          "</div>" +
          "<div class='genre-items'>"+items+"</div>" +
        "</div>"
      );
    }

    async function loadGenreSeeds(){
      var container = $('genreGrid');
      if (!container) return;
      container.innerHTML = "<p class='small'>Loading genres…</p>";
      try {
        const res = await fetch('/api/coldstart_seed');
        const data = await res.json();
        genreData = data.genres || [];
        if (genreData.length === 0) {
          container.innerHTML = "<p class='hint'>No genres found in the dataset.</p>";
          return;
        }
        container.innerHTML = genreData.map(genreBlockHTML).join('');
        updateSelectedUI();
      } catch(e){
        console.error(e);
        container.innerHTML = "<p class='hint'>Failed to load genres.</p>";
      }
    }

    // Event delegation for checkboxes & shuffle
    on($('genreGrid'), 'change', function(e){
      var el = e.target;
      if (!el || el.tagName !== 'INPUT' || el.type !== 'checkbox' || !el.getAttribute('data-mid')) return;
      var mid = Number(el.getAttribute('data-mid'));
      if (el.checked) {
        if (csSelected.size >= 10) { el.checked = false; return; }
        csSelected.add(mid);
      } else {
        csSelected.delete(mid);
      }
      updateSelectedUI();
    });

    on($('genreGrid'), 'click', async function(e){
      var target = e.target;
      while (target && target !== this && !(target.tagName === 'BUTTON' && target.getAttribute('data-shuffle'))) {
        target = target.parentNode;
      }
      if (!target || !(target.tagName === 'BUTTON')) return;
      var genreName = target.getAttribute('data-shuffle');
      if (!genreName) return;

      target.disabled = True = true;
      try {
        const res = await fetch('/api/coldstart_seed_one?genre='+encodeURIComponent(genreName));
        const data = await res.json();
        var idx = -1;
        for (var i=0;i<genreData.length;i++) if (genreData[i].name === genreName) { idx = i; break; }
        if (idx >= 0) genreData[idx].movies = data.movies || [];
        var blocks = document.getElementsByClassName('genre-block');
        for (var j=0;j<blocks.length;j++){
          if (blocks[j].getAttribute('data-genre') === genreName){
            blocks[j].outerHTML = genreBlockHTML(genreData[idx]);
            break;
          }
        }
      } catch(e){ console.error(e); }
      finally { target.disabled = false; }
      updateSelectedUI();
    });

    on($('btnColdStart'), 'click', async function(){
      if (csSelected.size !== 10) return;
      var ids = Array.from(csSelected.values());
      try {
        const res = await fetch('/api/coldstart', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ liked_movie_ids: ids, k: 10 })
        });
        const data = await res.json();
        renderList($('coldRecs'), data.recs || []);
      } catch(e){ console.error(e); }
    });
  });
</script>
</body>
</html>
"""

USERS_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Users</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; padding:24px; }
    .card { border:1px solid #ddd; border-radius:12px; padding:16px; max-width:600px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    input { padding:8px 10px; border:1px solid #ddd; border-radius:10px; }
    button { padding:8px 12px; border:none; border-radius:10px; background:#60a5fa; color:#0b1220; font-weight:600; cursor:pointer; }
    ul { padding-left:18px; }
    a { text-decoration:none; color:#2563eb; }
  </style>
</head>
<body>
  <h1>Users</h1>
  <div class="card">
    <h3>Create user</h3>
    <form method="post" action="/users">
      <div class="row">
        <input name="username" placeholder="e.g., hassan" required />
        <button type="submit">Create</button>
        <a href="/">← Back</a>
      </div>
    </form>
  </div>
  <div class="card" style="margin-top:12px;">
    <h3>Existing users</h3>
    {% if items %}
      <ul>
        {% for u, uid in items %}
          <li><b>{{ u }}</b></li>
        {% endfor %}
      </ul>
    {% else %}
      <p>No users yet.</p>
    {% endif %}
</body>
</html>
"""

# ----------------------------
# Helpers
# ----------------------------
def username_to_internal_id(username: str) -> Tuple[int | None, bool]:
    """Return (internal_id, exists_in_cf) for a username. exists_in_cf tells if MF knows this user."""
    if not username:
        return None, False
    if username in users_map:
        uid = users_map[username]
        return uid, (uid in u2idx)  # MF trained on original ratings only
    return None, False

def content_profile_from_feedback(username: str) -> np.ndarray | None:
    """Average embeddings of movies the user rated (in this session)."""
    user_items = [e["movie_id"] for e in feedback_events if e.get("username") == username]
    idxs = [i2idx[m] for m in user_items if m in i2idx]
    if not idxs: return None
    prof = movie_embeddings[idxs].mean(axis=0, keepdims=True)
    return prof

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return render_template_string(TEMPLATE)

@app.route("/users", methods=["GET", "POST"])
def users_page():
    global next_user_id
    if request.method == "POST":
        uname = (request.form.get("username") or "").strip()
        if uname and uname not in users_map:
            users_map[uname] = next_user_id
            next_user_id += 1
            save_users(users_map)
        return redirect(url_for("users_page"))
    items = sorted(users_map.items(), key=lambda x: x[0].lower())
    return render_template_string(USERS_TEMPLATE, items=items)

@app.get("/api/coldstart_seed")
def api_coldstart_seed():
    """Return 10 genres, each with up to 10 random movies (only those present in i2idx)."""
    candidate_genres = [
        "Action","Adventure","Animation","Comedy","Crime",
        "Documentary","Drama","Fantasy","Horror","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]
    out = []
    for g in candidate_genres:
        # non-capturing groups to avoid pandas "match groups" warning
        pattern = rf"(?:^|\|){re.escape(g)}(?:\||$)"
        subset = movies[movies["genres"].str.contains(pattern, regex=True, na=False, case=False)]
        subset = subset[subset["movieId"].isin(i2idx.keys())]
        if subset.empty:
            continue
        take = min(10, len(subset))
        sample = subset.sample(take, random_state=np.random.randint(0, 1_000_000))
        out.append({"name": g, "movies": sample[["movieId","title"]].to_dict("records")})
        if len(out) >= 10:
            break
    return jsonify({"genres": out})

@app.get("/api/coldstart_seed_one")
def api_coldstart_seed_one():
    """Return up to 10 random movies for a single genre."""
    genre = (request.args.get("genre") or "").strip()
    if not genre:
        return jsonify({"genre": genre, "movies": []})
    pattern = rf"(?:^|\|){re.escape(genre)}(?:\||$)"
    subset = movies[movies["genres"].str.contains(pattern, regex=True, na=False, case=False)]
    subset = subset[subset["movieId"].isin(i2idx.keys())]
    if subset.empty:
        return jsonify({"genre": genre, "movies": []})
    take = min(10, len(subset))
    sample = subset.sample(take, random_state=np.random.randint(0, 1_000_000))
    return jsonify({"genre": genre, "movies": sample[["movieId","title"]].to_dict("records")})

@app.get("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip().lower()
    if not q: return jsonify({"results": []})
    m = movies_idx[movies_idx["title_lc"].str.contains(q, na=False)]
    m = m[["movieId", "title"]].head(20)
    return jsonify({"results": m.to_dict("records")})

@app.get("/api/recommend")
def api_recommend():
    username = (request.args.get("username") or "").strip()
    try:
        k = int(request.args.get("k", TOPK_DEFAULT))
    except ValueError:
        k = TOPK_DEFAULT
    try:
        alpha = float(request.args.get("alpha", ALPHA_DEFAULT))
    except ValueError:
        alpha = ALPHA_DEFAULT

    HR.alpha = alpha
    uid, known_to_cf = username_to_internal_id(username)

    # If known CF user (original MovieLens numeric users only), use hybrid CF
    if known_to_cf and uid is not None:
        seen = user_hist_raw.get(uid, [])
        top_idx, _ = HR.recommend_for_user(uid, topk=k, exclude_seen=seen)
        if top_idx.size == 0: return jsonify({"username": username, "recs": []})
        rec_movie_ids = [inv_i[i] for i in top_idx]
        recs = movies[movies["movieId"].isin(rec_movie_ids)][["movieId","title"]].to_dict("records")
        return jsonify({"username": username, "alpha": alpha, "recs": recs})

    # New username (no CF history): try content from any feedback (stars)
    if username:
        prof = content_profile_from_feedback(username)
        if prof is not None:
            sims = cosine_similarity(prof, movie_embeddings).ravel()
            rated = {e["movie_id"] for e in feedback_events if e.get("username") == username}
            for m in rated:
                if m in i2idx: sims[i2idx[m]] = -1e9
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            rec_movie_ids = [inv_i[i] for i in top_idx]
            recs = movies[movies["movieId"].isin(rec_movie_ids)][["movieId","title"]].to_dict("records")
            return jsonify({"username": username, "alpha": alpha, "recs": recs, "cold_start": True})

    # Otherwise, pure popularity fallback
    popular_ids = popularity["movieId"].tolist()
    out = []
    seen = set()
    for mid in popular_ids:
        if mid not in seen:
            out.append(mid); seen.add(mid)
        if len(out) >= k: break
    recs = movies[movies["movieId"].isin(out)][["movieId","title"]].to_dict("records")
    return jsonify({"username": username, "alpha": alpha, "recs": recs, "cold_start": True})

@app.post("/api/coldstart")
def api_coldstart():
    payload = request.get_json(force=True, silent=True) or {}
    liked: List[int] = payload.get("liked_movie_ids") or []
    k = int(payload.get("k", TOPK_DEFAULT))
    valid_idx = [i2idx[m] for m in liked if m in i2idx]
    if not valid_idx:
        return jsonify({"recs": []})
    profile = movie_embeddings[valid_idx].mean(axis=0, keepdims=True)
    sims = cosine_similarity(profile, movie_embeddings).ravel()
    for m in liked:
        if m in i2idx: sims[i2idx[m]] = -1e9
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    rec_movie_ids = [inv_i[i] for i in top_idx]
    recs = movies[movies["movieId"].isin(rec_movie_ids)][["movieId","title"]].to_dict("records")
    return jsonify({"recs": recs})

@app.post("/rate")
def rate():
    payload = request.get_json(force=True, silent=True) or {}
    username = (payload.get("username") or "").strip()
    movie_id = payload.get("movie_id")
    rating = payload.get("rating")
    if movie_id is None or rating is None:
        return jsonify({"ok": False, "error": "movie_id and rating required"}), 400
    feedback_events.append(
        {"username": username if username else None, "movie_id": int(movie_id), "rating": float(rating)}
    )
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
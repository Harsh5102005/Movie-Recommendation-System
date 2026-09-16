"""
Movie Recommender – FastAPI Backend
====================================
Provides REST endpoints consumed by the Streamlit frontend:

  GET /health               – liveness check
  GET /home                 – TMDB home-feed (trending / popular / …)
  GET /tmdb/search          – raw TMDB keyword search (for autocomplete + grid)
  GET /movie/id/{tmdb_id}  – full movie details from TMDB
  GET /recommend/genre      – genre-based recommendations via TMDB Discover
  GET /recommend/tfidf      – TF-IDF recommendations from local dataset (debug)
  GET /movie/search         – bundle: details + TF-IDF recs + genre recs

Environment variables (see .env.example):
  TMDB_API_KEY  – required, TMDB v3 API key
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------
load_dotenv()

TMDB_API_KEY: Optional[str] = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Add it to a .env file as: TMDB_API_KEY=your_key_here"
    )

# Pickle files are stored one level up in data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DF_PATH = os.path.join(DATA_DIR, "df.pkl")
INDICES_PATH = os.path.join(DATA_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(DATA_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.pkl")


# ---------------------------------------------------------------------------
# In-memory globals (populated at startup via lifespan)
# ---------------------------------------------------------------------------
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class TMDBMovieCard(BaseModel):
    """Lightweight movie card returned by list endpoints (home feed, search grid)."""

    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    """Full movie detail returned by /movie/id/{tmdb_id}."""

    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    """A single TF-IDF recommendation with optional TMDB metadata."""

    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    """Bundle returned by /movie/search: details + TF-IDF recs + genre recs."""

    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# ---------------------------------------------------------------------------
# Lifespan: load pickles once at startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts into memory when the server starts up."""
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # TF-IDF matrix is typically a scipy sparse matrix
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # TF-IDF vectorizer (retained for potential transform calls)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    TITLE_TO_IDX = _build_title_to_idx_map(indices_obj)

    yield  # application runs here

    # Cleanup (optional – clear large objects on shutdown)
    df = indices_obj = tfidf_matrix = tfidf_obj = TITLE_TO_IDX = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Movie Recommender API",
    version="3.0",
    description=__doc__,
    lifespan=lifespan,
)

# CORS — allow_credentials must be False when allow_origins=["*"]
# If you need cookies/auth, replace "*" with your specific front-end origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _norm_title(t: str) -> str:
    """Normalise a movie title to lowercase strip for consistent dict lookups."""
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    """Convert a TMDB relative image path to an absolute w500 URL."""
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform an authenticated GET request against the TMDB v3 API.

    Raises HTTPException 502 on network errors or non-200 TMDB responses
    so that callers never need to handle raw httpx exceptions.
    """
    query = dict(params)
    query["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=query)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(exc).__name__} | {repr(exc)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB error {r.status_code}: {r.text}",
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    """Convert a raw TMDB results list into a list of TMDBMovieCard objects."""
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    """Fetch full movie details for a given TMDB movie ID."""
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Keyword search against TMDB /search/movie.

    Returns the raw TMDB response dict (contains a 'results' list) so that
    the Streamlit frontend can use it for both dropdown suggestions and the
    poster grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    """Return the single best TMDB match for a title query, or None."""
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------
def _build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Build a normalised title → row-index dict from the loaded indices pickle.

    Supports both:
    - ``dict`` mapping title → index
    - pandas ``Series`` with index=title, values=index
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or any mapping with .items()
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as exc:
        raise RuntimeError(
            "indices.pkl must be a dict or pandas Series-like (with .items())"
        ) from exc


def _get_local_idx_by_title(title: str) -> int:
    """
    Look up the DataFrame row index for a movie title.

    Raises HTTPException 404 if the title is not in the local dataset,
    and 500 if the index map was never initialised.
    """
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialised")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Compute cosine-similarity recommendations using the pre-loaded TF-IDF matrix.

    Args:
        query_title: Title to find similar movies for.
        top_n:       Maximum number of recommendations to return.

    Returns:
        List of (title, similarity_score) tuples, sorted descending by score.
    """
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = _get_local_idx_by_title(query_title)

    # Dot product of sparse query vector with full matrix → cosine similarities
    query_vec = tfidf_matrix[idx]
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()

    order = np.argsort(-scores)  # descending

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue  # skip the query movie itself
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def _attach_tmdb_card(title: str) -> Optional[TMDBMovieCard]:
    """
    Search TMDB by title and return a TMDBMovieCard for the best match.

    Returns None (never raises) so that a missing poster does not crash
    the recommendations endpoint.
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Utility"])
def health():
    """Liveness check – returns {status: ok}."""
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard], tags=["TMDB"])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home-feed endpoint that returns a list of movie cards.

    **category** options:
    - ``trending``   – trending movies today
    - ``popular``    – currently popular
    - ``top_rated``  – all-time top rated
    - ``now_playing``– in cinemas now
    - ``upcoming``   – releasing soon
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        valid = {"popular", "top_rated", "upcoming", "now_playing"}
        if category not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category '{category}'. Choose from: {sorted(valid)}",
            )

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Home route failed: {exc}") from exc


@app.get("/tmdb/search", tags=["TMDB"])
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Keyword search proxied directly from TMDB /search/movie.

    Returns the raw TMDB response shape ``{"results": [...], "total_pages": N, ...}``
    so that the Streamlit frontend can build both dropdown suggestions and the
    poster grid from a single request.
    """
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails, tags=["TMDB"])
async def movie_details_route(tmdb_id: int):
    """Fetch full TMDB movie details by numeric TMDB ID."""
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard], tags=["Recommendations"])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Genre-based recommendations via TMDB Discover.

    Fetches details for ``tmdb_id``, extracts the first genre, then returns
    popular movies in that genre (excluding the query movie itself).
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    # Remove the query movie from recommendations
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommend/tfidf", tags=["Recommendations"])
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """
    TF-IDF only recommendations from the local dataset (useful for debugging).

    Returns a list of ``{title, score}`` dicts without TMDB poster metadata.
    For the full enriched bundle use /movie/search instead.
    """
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse, tags=["Recommendations"])
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    All-in-one bundle for the movie details page.

    Given a movie title query, returns:
    1. **movie_details** – full TMDB details for the best match
    2. **tfidf_recommendations** – up to ``tfidf_top_n`` similar movies from
       the local TF-IDF model, each enriched with a TMDB poster
    3. **genre_recommendations** – up to ``genre_limit`` popular movies in the
       same genre from TMDB Discover

    If the title is not in the local dataset (e.g. a new release), TF-IDF recs
    gracefully fall back to an empty list rather than returning a 404.

    For multiple search matches use /tmdb/search instead.
    """
    # Resolve the best TMDB match for the user's query
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: '{query}'"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash the endpoint)
    raw_recs: List[Tuple[str, float]] = []
    try:
        raw_recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            # Fallback: try the raw user query string
            raw_recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            raw_recs = []

    tfidf_items: List[TFIDFRecItem] = []
    for title, score in raw_recs:
        card = await _attach_tmdb_card(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations via TMDB Discover
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )

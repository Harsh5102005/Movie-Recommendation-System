# 🎬 Movie Recommender

A full-stack movie recommendation web app powered by **TF-IDF content-based filtering** (local dataset) and the **TMDB API** for live metadata and posters.

| Layer | Technology |
|---|---|
| Backend API | FastAPI + httpx |
| Frontend UI | Streamlit |
| Recommendations | scikit-learn TF-IDF + cosine similarity |
| Movie metadata | TMDB v3 API |
| Deployment | Render (backend) + Streamlit Community Cloud (frontend) |

---

## Features

- 🏠 **Home feed** — trending, popular, top-rated, now-playing, and upcoming movies via TMDB
- 🔍 **Keyword search** — live autocomplete dropdown + matching poster grid
- 📄 **Movie details** — poster, backdrop, overview, genres, release date
- 🤖 **TF-IDF recommendations** — content-based similar movies from the local dataset
- 🎭 **Genre recommendations** — popular movies in the same genre via TMDB Discover

---

## Project Structure

```
movie-rec-main/
├── backend/
│   └── main.py          # FastAPI REST API
├── frontend/
│   └── app.py           # Streamlit web app
├── data/                # ⚠ git-ignored — store locally
│   ├── df.pkl           # Processed movie DataFrame
│   ├── indices.pkl      # Title → DataFrame index map
│   ├── tfidf.pkl        # TF-IDF vectorizer
│   ├── tfidf_matrix.pkl # TF-IDF feature matrix (scipy sparse)
│   └── movies_metadata.csv
├── notebooks/
│   └── movies.ipynb     # Data processing & model training
├── .env.example         # Environment variable template
├── .gitignore
├── requirements.txt
└── runtime.txt          # Python version for Render
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 2. Clone & install

```bash
git clone https://github.com/harsh5102005/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in your TMDB_API_KEY
```

### 4. Data files

The pickle files are **not** committed to git (they are large binary files).  
Generate them by running the training notebook:

```bash
jupyter notebook notebooks/movies.ipynb
```

This will create the `data/` directory with the required `.pkl` files.

### 5. Run the backend

```bash
uvicorn backend.main:app --reload
# API docs available at http://127.0.0.1:8000/docs
```

### 6. Run the frontend

```bash
# In a second terminal:
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/home` | Home feed (category: trending, popular, …) |
| `GET` | `/tmdb/search` | Keyword search (raw TMDB results) |
| `GET` | `/movie/id/{tmdb_id}` | Full movie details |
| `GET` | `/recommend/genre` | Genre-based recommendations |
| `GET` | `/recommend/tfidf` | TF-IDF recommendations (debug) |
| `GET` | `/movie/search` | Bundle: details + TF-IDF + genre recs |

Interactive docs: [`/docs`](http://127.0.0.1:8000/docs) (Swagger UI)

---

## Deployment

### Backend — Render

1. Push to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `TMDB_API_KEY=your_key`

> **Note:** The `data/` pickle files must be present at runtime. Either include them in a separate build step or use Render's persistent disk.

### Frontend — Streamlit Community Cloud

1. Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io)
2. Set **Main file path**: `frontend/app.py`
3. Add secret: `API_BASE=https://your-render-backend.onrender.com`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TMDB_API_KEY` | ✅ | — | TMDB v3 API key |
| `API_BASE` | ❌ | `http://127.0.0.1:8000` | Backend URL (frontend only) |

---

## License

MIT — see [LICENSE](LICENSE) for details.

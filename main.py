"""
main.py — FastAPI application entry point.

Endpoints
─────────
GET /health                              → {"status": "ok"}
GET /popular?k=10                        → top-k globally popular items
GET /recommendations?user_id=u1&k=10    → personalized recs (falls back to popular)

Usage
─────
    uvicorn main:app --reload
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_data
from recommender import build_recommenders, PopularityRecommender, ItemBasedRecommender
from schemas import HealthResponse, PopularResponse, RecommendationResponse, ItemResult


# ── App state (populated at startup) ──────────────────────────────────────────

class AppState:
    popularity_rec: PopularityRecommender = None
    cf_rec: ItemBasedRecommender = None
    loaded: bool = False
    load_error: str = None


state = AppState()


# ── Lifespan: load data & build recommenders once at startup ──────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and index all data before accepting requests."""
    data_dir = os.environ.get("DATA_DIR", "data")
    try:
        users_df, items_df, events_df = load_data(data_dir=data_dir)
        state.popularity_rec, state.cf_rec = build_recommenders(users_df, items_df, events_df)
        state.loaded = True
        print("[main] App startup complete. Ready to serve recommendations.")
    except FileNotFoundError as e:
        state.load_error = str(e)
        print(f"[main] STARTUP ERROR (missing file): {e}")
    except ValueError as e:
        state.load_error = str(e)
        print(f"[main] STARTUP ERROR (data validation): {e}")
    except Exception as e:
        state.load_error = f"Unexpected error during startup: {e}"
        print(f"[main] STARTUP ERROR: {e}")

    yield  # App runs here

    print("[main] App shutting down.")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="Streaming Recommendation API",
    description=(
        "Recommendation engine for a streaming platform. "
        "Provides global popularity rankings and personalised item-based CF recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Dependency: guard all recommendation endpoints ────────────────────────────

def require_data():
    """Raise a clear HTTP 503 if data failed to load at startup."""
    if not state.loaded:
        detail = state.load_error or "Data not loaded. Check server logs."
        raise HTTPException(status_code=503, detail=detail)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health() -> HealthResponse:
    """Returns service status. Always returns 200 even if data failed to load."""
    return HealthResponse(status="ok")


@app.get(
    "/popular",
    response_model=PopularResponse,
    summary="Global popular recommendations",
    tags=["Recommendations"],
)
def popular(
    k: Annotated[
        int,
        Query(ge=1, le=10000, description="Number of top items to return (if k > catalog size, returns all items)"),
    ] = 10,
) -> PopularResponse:
    """
    Returns the top-k globally popular items ranked by a weighted combination
    of total watch time (60%) and event count (40%).
    """
    require_data()
    items = state.popularity_rec.recommend(k=k)
    return PopularResponse(
        k=k,
        items=[ItemResult(**item) for item in items],
    )


@app.get(
    "/recommendations",
    response_model=RecommendationResponse,
    summary="Personalized recommendations for a user",
    tags=["Recommendations"],
)
def recommendations(
    user_id: Annotated[
        str,
        Query(min_length=1, description="The user ID to generate recommendations for"),
    ],
    k: Annotated[
        int,
        Query(ge=1, le=10000, description="Number of recommended items to return (if k > catalog, returns all)"),
    ] = 10,
) -> RecommendationResponse:
    """
    Returns personalized recommendations for the given user using item-based
    collaborative filtering.

    - Items the user has already watched heavily (≥600 seconds) are excluded.
    - Falls back to global popularity if the user is unknown or has no history.
    - `fallback_used: true` indicates the popularity fallback was applied.
    """
    require_data()

    items_raw, fallback_used = state.cf_rec.recommend(user_id=user_id.strip(), k=k)

    return RecommendationResponse(
        user_id=user_id,
        k=k,
        items=[ItemResult(**item) for item in items_raw],
        fallback_used=fallback_used,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
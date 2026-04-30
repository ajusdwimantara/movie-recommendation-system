"""
recommender.py — Core recommendation logic.

Two strategies:
1. PopularityRecommender   — global popularity score per item
2. ItemBasedRecommender    — item-based collaborative filtering (cosine similarity)
   Falls back to popularity for cold-start / unknown users.

Design notes
────────────
Popularity score
    score_i = 0.6 * norm(total_watch_seconds_i) + 0.4 * norm(event_count_i)
    Normalised to [0, 1] using min-max scaling so both signals are on the same
    footing regardless of absolute magnitude.

Item-based CF
    • Build a user–item matrix M where M[u, i] = total watch_seconds.
    • Apply log1p transform to compress heavy-tail distribution of watch time.
    • Compute cosine similarity between every pair of items (item vectors = columns).
    • For a target user u, score candidate item j as:
          score(j) = Σ_i  sim(i, j) * M[u, i]   for all i ∈ user's history
    • Exclude "heavily watched" items (watch_seconds ≥ HEAVY_WATCH_THRESHOLD).

HEAVY_WATCH_THRESHOLD defaults to 600 seconds (10 minutes). Items watched less
than that threshold are NOT excluded — they may still appear as recommendations
because the user hasn't committed to them yet.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any


# ── Constants ──────────────────────────────────────────────────────────────────

HEAVY_WATCH_THRESHOLD: int = 600   # seconds; items >= this are "already watched"
POPULARITY_WATCH_WEIGHT: float = 0.6
POPULARITY_COUNT_WEIGHT: float = 0.4


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def _rows_to_dicts(
    items_df: pd.DataFrame,
    item_ids: List[str],
    reasons: Dict[str, str] = None,
) -> List[Dict[str, Any]]:
    """Convert a list of item_ids to enriched dicts using items_df lookup."""
    lookup = items_df.set_index("item_id")
    reasons = reasons or {}
    results = []
    for iid in item_ids:
        if iid in lookup.index:
            row = lookup.loc[iid]
            results.append({
                "item_id":      iid,
                "title":        row.get("title", ""),
                "content_type": row.get("content_type", None),
                "genre":        row.get("genre", None),
                "reason":       reasons.get(iid, None),
            })
        else:
            results.append({
                "item_id":      iid,
                "title":        "Unknown",
                "content_type": None,
                "genre":        None,
                "reason":       reasons.get(iid, None),
            })
    return results


# ── 1. Popularity Recommender ──────────────────────────────────────────────────

class PopularityRecommender:
    """
    Ranks items by a weighted combination of total watch time and event count.
    Scores are pre-computed at init time and cached.
    """

    def __init__(self, items_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
        self._items_df = items_df
        self._scores   = self._compute_scores(items_df, events_df)

    @staticmethod
    def _compute_scores(items_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        # Aggregate per item
        agg = (
            events_df
            .groupby("item_id", as_index=False)
            .agg(
                total_watch_seconds=("watch_seconds", "sum"),
                total_events=("event_count", "sum"),
            )
        )

        # Ensure every item appears (even those with zero events)
        all_items = items_df[["item_id"]].copy()
        agg = all_items.merge(agg, on="item_id", how="left").fillna(0)

        # Normalise both signals and combine
        agg["norm_watch"] = _minmax_norm(agg["total_watch_seconds"])
        agg["norm_count"] = _minmax_norm(agg["total_events"])
        agg["score"] = (
            POPULARITY_WATCH_WEIGHT * agg["norm_watch"] +
            POPULARITY_COUNT_WEIGHT * agg["norm_count"]
        )

        return agg.sort_values("score", ascending=False).reset_index(drop=True)

    def recommend(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k items by global popularity."""
        top_k = self._scores.head(max(k, 1)).reset_index(drop=True)

        reasons = {}
        for rank_idx, row in top_k.iterrows():
            total_secs  = int(row["total_watch_seconds"])
            total_plays = int(row["total_events"])
            minutes     = round(total_secs / 60, 1)
            reasons[row["item_id"]] = (
                f"#{rank_idx + 1} most popular — "
                f"{total_plays} play{'s' if total_plays != 1 else ''}, "
                f"{minutes} min total watch time"
            )

        return _rows_to_dicts(self._items_df, top_k["item_id"].tolist(), reasons)[:k]


# ── 2. Item-Based Collaborative Filtering Recommender ─────────────────────────

class ItemBasedRecommender:
    """
    Item-based CF using cosine similarity on a log-scaled user–item matrix.

    Attributes
    ----------
    _user_item  : DataFrame (users × items), values = log1p(watch_seconds)
    _sim_matrix : ndarray of shape (n_items, n_items), cosine similarities
    _item_index : maps item_id → column index in _user_item
    """

    def __init__(
        self,
        items_df:  pd.DataFrame,
        events_df: pd.DataFrame,
        popularity_recommender: PopularityRecommender,
    ) -> None:
        self._items_df  = items_df
        self._fallback  = popularity_recommender
        self._user_item, self._item_index = self._build_matrix(events_df)
        self._sim_matrix = self._compute_similarity()

    # ── Matrix construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_matrix(events_df: pd.DataFrame):
        """
        Build a user–item matrix (users as rows, items as columns).
        Values are log1p(watch_seconds) to compress heavy tails.
        """
        pivot = events_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="watch_seconds",
            aggfunc="sum",
            fill_value=0,
        )
        # Log-scale implicit feedback
        log_pivot = np.log1p(pivot)
        item_index = {item_id: idx for idx, item_id in enumerate(log_pivot.columns)}
        return log_pivot, item_index

    def _compute_similarity(self) -> np.ndarray:
        """
        Compute item–item cosine similarity.
        Items are represented as column vectors (i.e. how all users rated them).
        Returns shape (n_items, n_items).
        """
        item_matrix = self._user_item.values.T  # shape: (n_items, n_users)
        if item_matrix.shape[0] == 0:
            return np.array([])
        sim = cosine_similarity(item_matrix)
        # Zero out self-similarity on diagonal
        np.fill_diagonal(sim, 0.0)
        return sim

    # ── Recommendation logic ──────────────────────────────────────────────────

    def _is_known_user(self, user_id: str) -> bool:
        return user_id in self._user_item.index

    def _get_user_vector(self, user_id: str) -> np.ndarray:
        """Return the log-scaled interaction vector for a user."""
        return self._user_item.loc[user_id].values  # shape: (n_items,)

    def _build_cf_reasons(
        self,
        recommended_item_ids: List[str],
        user_vec: np.ndarray,
        top_n_drivers: int = 2,
    ) -> Dict[str, str]:
        """
        For each recommended item, find the top contributing items from the
        user's watch history that drove the recommendation score.

        score(j) = Σ_i  sim(i, j) * user_vec[i]

        The contribution of history item i toward recommended item j is:
            contribution(i, j) = sim(i, j) * user_vec[i]

        We surface the top `top_n_drivers` history items with the highest
        positive contribution and name them in the reason string.
        """
        item_ids   = list(self._user_item.columns)
        title_lookup = self._items_df.set_index("item_id")["title"].to_dict()
        reasons    = {}

        for rec_id in recommended_item_ids:
            if rec_id not in self._item_index:
                reasons[rec_id] = "Recommended based on platform trends"
                continue

            j_idx = self._item_index[rec_id]
            # contribution of every history item toward this recommendation
            contributions = self._sim_matrix[:, j_idx] * user_vec  # shape: (n_items,)

            # Sort by contribution descending, keep only positive drivers
            driver_indices = np.argsort(contributions)[::-1]
            drivers = [
                item_ids[i]
                for i in driver_indices
                if contributions[i] > 0 and item_ids[i] != rec_id
            ][:top_n_drivers]

            if not drivers:
                reasons[rec_id] = "Recommended based on your watch history"
                continue

            driver_titles = [
                f'"{title_lookup.get(d, d)}"' for d in drivers
            ]

            if len(driver_titles) == 1:
                reasons[rec_id] = f"Because you watched {driver_titles[0]}"
            else:
                reasons[rec_id] = (
                    f"Because you watched {', '.join(driver_titles[:-1])} "
                    f"and {driver_titles[-1]}"
                )

        return reasons

    def _get_heavily_watched(self, user_id: str) -> set:
        """
        Items the user has watched more than HEAVY_WATCH_THRESHOLD seconds.
        Uses the user–item matrix (which is already aggregated).
        """
        row = self._user_item.loc[user_id]
        threshold_log = np.log1p(HEAVY_WATCH_THRESHOLD)
        return set(row[row >= threshold_log].index.tolist())

    def recommend(self, user_id: str, k: int = 10) -> tuple[List[Dict[str, Any]], bool]:
        """
        Return (recommendations, fallback_used).

        If the user is unknown or has no history, falls back to popularity.
        """
        # ── Cold start / unknown user ─────────────────────────────────────────
        if not self._is_known_user(user_id):
            return self._fallback.recommend(k), True

        user_vec = self._get_user_vector(user_id)

        # ── User has no watch history at all ─────────────────────────────────
        if user_vec.sum() == 0:
            return self._fallback.recommend(k), True

        # ── Score all items ───────────────────────────────────────────────────
        # score(j) = Σ_i  sim(i,j) * interaction(u, i)
        # Matrix form: scores = sim_matrix.T @ user_vec  (or sim_matrix @ user_vec,
        # since sim is symmetric)
        scores = self._sim_matrix @ user_vec   # shape: (n_items,)

        # ── Exclude heavily watched items ─────────────────────────────────────
        heavily_watched = self._get_heavily_watched(user_id)
        item_ids = list(self._user_item.columns)

        # Build a mask: True = keep this item as a candidate
        candidate_mask = np.array([
            item_id not in heavily_watched for item_id in item_ids
        ], dtype=bool)

        # Also exclude items not in our items_df (shouldn't happen after data_loader,
        # but defensive check)
        valid_items = set(self._items_df["item_id"])
        candidate_mask &= np.array([item_id in valid_items for item_id in item_ids])

        filtered_scores = np.where(candidate_mask, scores, -np.inf)

        # ── Pick top-k ────────────────────────────────────────────────────────
        top_indices = np.argsort(filtered_scores)[::-1]
        top_item_ids = [item_ids[i] for i in top_indices if filtered_scores[i] > -np.inf][:k]

        # ── Handle case where not enough CF candidates exist ──────────────────
        if len(top_item_ids) == 0:
            return self._fallback.recommend(k), True

        # Pad with popular items if CF returns fewer than k results
        fallback_used = False
        if len(top_item_ids) < k:
            popular = self._fallback.recommend(k)
            seen = set(top_item_ids)
            for p in popular:
                if p["item_id"] not in seen and p["item_id"] not in heavily_watched:
                    top_item_ids.append(p["item_id"])
                    seen.add(p["item_id"])
                if len(top_item_ids) >= k:
                    break
            # Note: not setting fallback_used=True here — we did produce CF results,
            # we just padded. Only set True when we fully replaced CF.

        reasons = self._build_cf_reasons(top_item_ids[:k], user_vec)
        return _rows_to_dicts(self._items_df, top_item_ids[:k], reasons), fallback_used


# ── Factory: build both recommenders from DataFrames ──────────────────────────

def build_recommenders(
    users_df:  pd.DataFrame,
    items_df:  pd.DataFrame,
    events_df: pd.DataFrame,
) -> tuple[PopularityRecommender, ItemBasedRecommender]:
    """
    Instantiate and return (popularity_rec, cf_rec).
    Call once at app startup.
    """
    pop_rec = PopularityRecommender(items_df, events_df)
    cf_rec  = ItemBasedRecommender(items_df, events_df, pop_rec)
    print(
        f"[recommender] Ready. "
        f"Popularity: {len(items_df)} items scored. "
        f"CF matrix: {cf_rec._user_item.shape[0]} users × {cf_rec._user_item.shape[1]} items."
    )
    return pop_rec, cf_rec
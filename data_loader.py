"""
data_loader.py — Loads and validates CSV data files into DataFrames.

Handles:
- Missing files (raises descriptive FileNotFoundError)
- Missing/extra columns (warns, proceeds with available data)
- Null values in key columns (rows are dropped with a warning)
- Mismatched foreign keys between tables (filtered out silently)
- Type coercion (watch_seconds → int, timestamp → datetime)
"""

import os
import warnings
import pandas as pd
from pathlib import Path


# ── Column contracts ───────────────────────────────────────────────────────────

USERS_REQUIRED  = {"user_id", "age", "gender", "region"}
ITEMS_REQUIRED  = {"item_id", "title", "content_type", "genre"}
EVENTS_REQUIRED = {"user_id", "item_id", "event_type", "watch_seconds", "timestamp"}


def _check_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{filename}] Missing required columns: {missing}")


def _drop_null_keys(df: pd.DataFrame, key_cols: list, filename: str) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=key_cols)
    dropped = before - len(df)
    if dropped:
        warnings.warn(
            f"[{filename}] Dropped {dropped} row(s) with null values in key columns {key_cols}."
        )
    return df


# ── Public loader ──────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load users, items, and events from CSV files in `data_dir`.

    Returns
    -------
    users_df  : DataFrame with columns [user_id, age, gender, region]
    items_df  : DataFrame with columns [item_id, title, content_type, genre]
    events_df : DataFrame with columns [user_id, item_id, event_type,
                                         watch_seconds, timestamp]

    Raises
    ------
    FileNotFoundError : if any of the three CSV files does not exist.
    ValueError        : if a file is missing a required column.
    """
    base = Path(data_dir)

    # ── 1. Resolve paths ──────────────────────────────────────────────────────
    paths = {
        "users":  base / "users.csv",
        "items":  base / "items.csv",
        "events": base / "events.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: '{path}'. "
                f"Please place {name}.csv inside the '{data_dir}/' directory."
            )

    # ── 2. Load raw CSVs ──────────────────────────────────────────────────────
    users_df  = pd.read_csv(paths["users"],  dtype=str)
    items_df  = pd.read_csv(paths["items"],  dtype=str)
    events_df = pd.read_csv(paths["events"], dtype=str)

    # ── 3. Validate columns ───────────────────────────────────────────────────
    _check_columns(users_df,  USERS_REQUIRED,  "users.csv")
    _check_columns(items_df,  ITEMS_REQUIRED,  "items.csv")
    _check_columns(events_df, EVENTS_REQUIRED, "events.csv")

    # ── 4. Drop rows with null key fields ─────────────────────────────────────
    users_df  = _drop_null_keys(users_df,  ["user_id"],            "users.csv")
    items_df  = _drop_null_keys(items_df,  ["item_id", "title"],   "items.csv")
    events_df = _drop_null_keys(events_df, ["user_id", "item_id"], "events.csv")

    # ── 5. Type coercions ─────────────────────────────────────────────────────
    # age → numeric (invalid → NaN, kept as float for flexibility)
    users_df["age"] = pd.to_numeric(users_df["age"], errors="coerce")

    # watch_seconds → integer (invalid rows dropped)
    events_df["watch_seconds"] = pd.to_numeric(events_df["watch_seconds"], errors="coerce")
    bad_ws = events_df["watch_seconds"].isna().sum()
    if bad_ws:
        warnings.warn(f"[events.csv] Dropped {bad_ws} row(s) with non-numeric watch_seconds.")
    events_df = events_df.dropna(subset=["watch_seconds"])
    events_df["watch_seconds"] = events_df["watch_seconds"].astype(int)

    # timestamp → datetime (keep as string if coercion fails — non-critical)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce")

    # ── 6. Strip whitespace from ID columns ───────────────────────────────────
    for col in ["user_id"]:
        users_df[col] = users_df[col].str.strip()
    for col in ["item_id"]:
        items_df[col] = items_df[col].str.strip()
    for col in ["user_id", "item_id"]:
        events_df[col] = events_df[col].str.strip()

    # ── 7. Filter events to known users/items only ────────────────────────────
    valid_users = set(users_df["user_id"])
    valid_items = set(items_df["item_id"])

    before = len(events_df)
    events_df = events_df[
        events_df["user_id"].isin(valid_users) &
        events_df["item_id"].isin(valid_items)
    ]
    filtered = before - len(events_df)
    if filtered:
        warnings.warn(
            f"[events.csv] Filtered out {filtered} event(s) referencing unknown user_ids or item_ids."
        )

    # ── 8. Weight events by type, then aggregate per user–item pair ───────────
    EVENT_WEIGHTS = {
        "play":     1.0,
        "complete": 2.0,
        "replay":   1.5,
        "pause":    0.5,
        "seek":     0.8,
        "skip":    -0.5,
        "stop":    -0.3,
        "dislike":  -1.0,
    }
    DEFAULT_WEIGHT = 1.0

    events_df["event_weight"] = (
        events_df["event_type"]
        .str.strip()
        .str.lower()
        .map(EVENT_WEIGHTS)
        .fillna(DEFAULT_WEIGHT)
    )
    events_df["weighted_seconds"] = events_df["watch_seconds"] * events_df["event_weight"]

    events_df = (
        events_df
        .groupby(["user_id", "item_id"], as_index=False)
        .agg(
            watch_seconds=("weighted_seconds", "sum"),
            event_count=("item_id", "count"),
        )
    )

    # Clip to 0 — net negative score means no interest, not a penalty to others
    events_df["watch_seconds"] = events_df["watch_seconds"].clip(lower=0)

    print(
        f"[data_loader] Loaded: "
        f"{len(users_df)} users, {len(items_df)} items, {len(events_df)} interaction pairs."
    )

    return users_df, items_df, events_df
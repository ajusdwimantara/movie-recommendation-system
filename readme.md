# Streaming Recommendation API

A FastAPI-based recommendation engine for a streaming platform (TV, Movies, Series, Microdrama).

## Project Structure

```
recommender/
├── main.py           # FastAPI app & endpoints
├── data_loader.py    # CSV ingestion with error handling
├── recommender.py    # Popularity + personalized CF logic
├── schemas.py        # Pydantic response models
├── requirements.txt  # Dependencies
└── data/
    ├── users.csv
    ├── items.csv
    └── events.csv
```

## Running

```bash
docker compose up --build
```

## Test API

open on browser:
```bash
localhost:8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/popular?k=10` | Top-k globally popular items |
| GET | `/recommendations?user_id=u1&k=10` | Personalized recommendations |

## Model Details

### Popularity
Score = 0.6 × normalized(total_watch_seconds) + 0.4 × normalized(event_count)

### Personalized (Item-Based CF)
- Builds a user–item matrix with `watch_seconds` as implicit feedback
- Computes item–item cosine similarity
- Scores unseen items by weighted similarity to user's watch history
- Falls back to popularity for unknown/cold-start users
- "Heavily watched" threshold: ≥ 600 seconds (configurable)
"""
schemas.py — Pydantic models for request/response validation.
"""

from pydantic import BaseModel
from typing import List, Optional


class ItemResult(BaseModel):
    """A single recommended item returned in a response."""
    item_id: str
    title: str
    content_type: Optional[str] = None
    genre: Optional[str] = None
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class PopularResponse(BaseModel):
    k: int
    items: List[ItemResult]


class RecommendationResponse(BaseModel):
    user_id: str
    k: int
    items: List[ItemResult]
    fallback_used: bool
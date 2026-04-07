from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmailInput(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    sender: Optional[str] = None
    timestamp: Optional[str] = None
    thread_id: Optional[str] = None
    has_reply_prefix: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchEmailInput(BaseModel):
    emails: List[EmailInput]


class EmailPrediction(BaseModel):
    predicted_category: str
    confidence_score: float
    priority_score: int
    priority_band: str
    explanation: List[str]
    extracted_signals: Dict[str, Any]


class BatchPrediction(BaseModel):
    results: List[EmailPrediction]




from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from typing import Dict, List

from app.config import GMAIL_QUERY, GMAIL_USER_ID
from integrations.gmail_service import get_gmail_service
from models.predict import predict_email
from storage.db import insert_email

_TAG_RE = re.compile(r"<[^>]+>")


def _decode_body(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_body(payload: Dict) -> str:
    if not payload:
        return ""
    if payload.get("body", {}).get("data"):
        return _decode_body(payload["body"]["data"])

    parts = payload.get("parts", [])
    for part in parts:
        mime = part.get("mimeType", "")
        if mime == "text/plain" and part.get("body", {}).get("data"):
            return _decode_body(part["body"]["data"])
    for part in parts:
        mime = part.get("mimeType", "")
        if mime == "text/html" and part.get("body", {}).get("data"):
            raw = _decode_body(part["body"]["data"])
            return _TAG_RE.sub(" ", raw)
    return ""


def _header_value(headers: List[Dict], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def fetch_message_detail(service, msg_id: str) -> Dict:
    return (
        service.users()
        .messages()
        .get(userId=GMAIL_USER_ID, id=msg_id, format="full")
        .execute()
    )


def list_message_ids(service, query: str) -> List[str]:
    msg_ids: List[str] = []
    request = service.users().messages().list(userId=GMAIL_USER_ID, q=query)
    while request is not None:
        response = request.execute()
        msg_ids.extend([m["id"] for m in response.get("messages", [])])
        request = service.users().messages().list_next(request, response)
    return msg_ids


def process_message(message: Dict) -> Dict:
    headers = message.get("payload", {}).get("headers", [])
    subject = _header_value(headers, "Subject")
    sender = _header_value(headers, "From")
    timestamp = _header_value(headers, "Date")
    body = _extract_body(message.get("payload", {}))

    result = predict_email(subject=subject or "", body=body or "", sender=sender or "")

    record = {
        "gmail_id": message.get("id"),
        "thread_id": message.get("threadId"),
        "sender": sender,
        "subject": subject,
        "body": body,
        "snippet": message.get("snippet"),
        "timestamp": timestamp,
        "internal_date": int(message.get("internalDate", 0)),
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence_score"],
        "priority_score": result["priority_score"],
        "priority_band": result["priority_band"],
        "explanation": result["explanation"],
        "extracted_signals": result["extracted_signals"],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return record


def poll_gmail_once(query: str) -> int:
    logger = logging.getLogger(__name__)
    service = get_gmail_service()
    msg_ids = list_message_ids(service, query)
    processed = 0
    for msg_id in msg_ids:
        try:
            message = fetch_message_detail(service, msg_id)
            record = process_message(message)
            insert_email(record)
            processed += 1
        except FileNotFoundError as exc:
            logger.error("Model artifact missing: %s", exc)
            break
        except Exception as exc:
            logger.exception("Failed processing message %s: %s", msg_id, exc)
    logger.info("Processed %s new messages", processed)
    return processed


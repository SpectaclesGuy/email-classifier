from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.logging_config import setup_logging
from app.schemas import BatchEmailInput, BatchPrediction, EmailInput, EmailPrediction
from models.predict import predict_batch, predict_email
from services.gmail_worker import GmailPollingWorker
from storage.db import init_db, list_emails

setup_logging()

app = FastAPI(title="Email Classification and Priority Scoring")
worker = GmailPollingWorker()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    worker.stop()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/emails")
def get_emails(limit: int = 50, offset: int = 0) -> dict:
    return {"items": list_emails(limit=limit, offset=offset)}


@app.get("/portal", response_class=HTMLResponse)
def portal() -> str:
    items = list_emails(limit=100, offset=0)
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td>{item.get('timestamp','')}</td>"
            f"<td>{item.get('sender','')}</td>"
            f"<td>{item.get('subject','')}</td>"
            f"<td>{item.get('predicted_category','')}</td>"
            f"<td>{item.get('priority_score','')}</td>"
            f"<td>{item.get('priority_band','')}</td>"
            "</tr>"
        )
    table_rows = "\n".join(rows)
    html = f"""
    <html>
      <head>
        <title>Email Triage Portal</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; }}
          th {{ background: #f5f5f5; text-align: left; }}
        </style>
      </head>
      <body>
        <h2>Email Triage Portal</h2>
        <p>Latest classified emails</p>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Sender</th>
              <th>Subject</th>
              <th>Category</th>
              <th>Priority</th>
              <th>Band</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </body>
    </html>
    """
    return html


@app.post("/classify-email", response_model=EmailPrediction)
def classify_email(payload: EmailInput) -> EmailPrediction:
    try:
        result = predict_email(
            subject=payload.subject,
            body=payload.body,
            sender=payload.sender,
            timestamp=payload.timestamp,
            thread_id=payload.thread_id,
            has_reply_prefix=payload.has_reply_prefix,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EmailPrediction(**result)


@app.post("/classify-batch", response_model=BatchPrediction)
def classify_batch(payload: BatchEmailInput) -> BatchPrediction:
    try:
        results = predict_batch([email.model_dump() for email in payload.emails])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BatchPrediction(results=[EmailPrediction(**result) for result in results])


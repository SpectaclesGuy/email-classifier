from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.logging_config import setup_logging
from app.schemas import BatchEmailInput, BatchPrediction, EmailInput, EmailPrediction
from models.predict import predict_batch, predict_email
from services.gmail_worker import GmailPollingWorker
from storage.db import init_db, list_emails

setup_logging()

app = FastAPI(title="Email Classification and Priority Scoring")
worker = GmailPollingWorker()

# Serve static assets (background image)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    worker.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    worker.stop()


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/portal")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/emails")
def get_emails(limit: int = 50, offset: int = 0) -> dict:
    return {"items": list_emails(limit=limit, offset=offset)}


@app.get("/portal", response_class=HTMLResponse)
def portal() -> str:
    items = list_emails(limit=200, offset=0)
    rows = []
    for item in items:
        band = (item.get("priority_band") or "low").lower()
        cat = (item.get("predicted_category") or "").lower()
        rows.append(
            "<tr>"
            f"<td class='cell-date'>{item.get('timestamp','')}</td>"
            f"<td class='cell-sender'>{item.get('sender','')}</td>"
            f"<td class='cell-subject'>{item.get('subject','')}</td>"
            f"<td class='cell-category'><span class='badge badge-{cat}'>"
            f"{item.get('predicted_category','')}</span></td>"
            f"<td class='cell-priority'>{item.get('priority_score','')}</td>"
            f"<td class='cell-band'><span class='pill pill-{band}'>"
            f"{item.get('priority_band','')}</span></td>"
            "</tr>"
        )
    table_rows = "\n".join(rows)
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>Email Triage Portal</title>
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
          :root {{
            --text: #f8fafc;
            --muted: #e2e8f0;
            --border: rgba(148, 163, 184, 0.25);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: 'Space Grotesk', system-ui, -apple-system, sans-serif;
            color: var(--text);
            background: url('/static/pexels-pixabay-33545.jpg') center/cover fixed no-repeat;
            min-height: 100vh;
          }}
          .overlay {{
            min-height: 100vh;
            background: linear-gradient(180deg, rgba(2,6,23,0.55), rgba(2,6,23,0.75));
            padding: 32px 20px 48px;
          }}
          .page {{
            max-width: 1200px;
            margin: 0 auto;
          }}
          .header {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
          }}
          .title {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 0.3px;
          }}
          .subtitle {{
            color: var(--muted);
            font-size: 14px;
            margin-top: 6px;
          }}
          .status {{
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 12px;
            backdrop-filter: blur(8px);
          }}
          .glass {{
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            backdrop-filter: blur(14px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.35);
          }}
          .table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
          }}
          .table th {{
            text-align: left;
            color: var(--muted);
            font-weight: 600;
            padding: 12px 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
          }}
          .table td {{
            padding: 12px 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            vertical-align: top;
          }}
          .cell-subject {{
            max-width: 420px;
            word-break: break-word;
          }}
          .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid transparent;
            text-transform: capitalize;
          }}
          .badge-urgent {{ background: rgba(239, 68, 68, 0.2); color: #fecaca; border-color: rgba(239, 68, 68, 0.35); }}
          .badge-follow_up {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; border-color: rgba(245, 158, 11, 0.35); }}
          .badge-spam {{ background: rgba(148, 163, 184, 0.2); color: #e2e8f0; border-color: rgba(148, 163, 184, 0.35); }}
          .badge-informational {{ background: rgba(56, 189, 248, 0.2); color: #bae6fd; border-color: rgba(56, 189, 248, 0.35); }}
          .badge- {{ background: rgba(148, 163, 184, 0.2); color: var(--muted); border-color: rgba(148, 163, 184, 0.35); }}
          .pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: capitalize;
          }}
          .pill-high {{ background: rgba(34, 197, 94, 0.2); color: #bbf7d0; }}
          .pill-medium {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; }}
          .pill-low {{ background: rgba(148, 163, 184, 0.2); color: #e2e8f0; }}
          .empty {{
            padding: 24px;
            color: var(--muted);
            text-align: center;
          }}
          @media (max-width: 860px) {{
            .table thead {{ display: none; }}
            .table tr {{ display: grid; gap: 6px; padding: 12px 0; }}
            .table td {{ border: none; padding: 0; }}
          }}
        </style>
      </head>
      <body>
        <div class='overlay'>
          <div class='page'>
            <div class='header'>
              <div>
                <div class='title'>Email Triage Portal</div>
                <div class='subtitle'>Latest classified emails with priority scoring</div>
              </div>
              <div class='status'>Auto-refresh every poll cycle</div>
            </div>

            <div class='glass'>
              <table class='table'>
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
                  {table_rows if table_rows else '<tr><td colspan="6" class="empty">No emails processed yet.</td></tr>'}
                </tbody>
              </table>
            </div>
          </div>
        </div>
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


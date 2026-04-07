from __future__ import annotations

from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.config import GMAIL_CLIENT_SECRET_PATH, GMAIL_TOKEN_PATH

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_credentials() -> Credentials:
    creds: Optional[Credentials] = None
    if GMAIL_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds:
        if not GMAIL_CLIENT_SECRET_PATH.exists():
            raise FileNotFoundError(
                f"Gmail client secret not found at {GMAIL_CLIENT_SECRET_PATH}."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(GMAIL_CLIENT_SECRET_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        GMAIL_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GMAIL_TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    return creds


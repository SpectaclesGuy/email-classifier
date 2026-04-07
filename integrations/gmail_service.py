from __future__ import annotations

from googleapiclient.discovery import build

from integrations.gmail_auth import get_credentials


def get_gmail_service():
    creds = get_credentials()
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


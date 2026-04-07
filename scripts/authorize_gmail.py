from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from integrations.gmail_auth import get_credentials


def main() -> None:
    creds = get_credentials()
    print("Authorized Gmail account:", creds.client_id)


if __name__ == "__main__":
    main()


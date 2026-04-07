from __future__ import annotations

import json
from pathlib import Path


SAMPLES = [
    {
        "subject": "Urgent: client issue needs resolution today",
        "body": "Please resolve this before EOD. The client is waiting.",
        "sender": "manager@company.com",
        "timestamp": "2026-04-06T10:30:00",
    },
    {
        "subject": "Re: Q2 planning",
        "body": "Just checking in on the updated forecast.",
        "sender": "analyst@company.com",
    },
]


def main() -> None:
    path = Path("sample_requests.json")
    path.write_text(json.dumps({"emails": SAMPLES}, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()




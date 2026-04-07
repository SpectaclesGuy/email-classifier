from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


import json

from models.predict import predict_email


def main() -> None:
    sample = {
        "subject": "Urgent: client issue needs resolution today",
        "body": "Please resolve this before EOD. The client is waiting.",
        "sender": "manager@company.com",
    }
    result = predict_email(**sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()





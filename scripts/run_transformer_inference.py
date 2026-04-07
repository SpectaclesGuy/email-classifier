from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models.transformer_predict import load_default_predictor


def main() -> None:
    predictor = load_default_predictor()
    sample = {
        "subject": "Urgent: client issue needs resolution today",
        "body": "Please resolve this before EOD. The client is waiting.",
        "sender": "manager@company.com",
    }
    result = predictor.predict(**sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


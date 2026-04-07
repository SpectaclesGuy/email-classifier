from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from datasets.build_dataset import main as build_dataset
from models.evaluate import main as evaluate
from models.train import main as train


def main() -> None:
    build_dataset()
    train()
    evaluate()


if __name__ == "__main__":
    main()





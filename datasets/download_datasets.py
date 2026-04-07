from __future__ import annotations

import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.config import RAW_DIR


DATASETS = {
    "trec07": "bayes2003/emails-for-spam-or-ham-classification-trec-2007",
    "enron": "wcukierski/enron-email-dataset",
}


def _extract_archive(src: Path, dest: Path) -> bool:
    if src.suffix == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest)
        return True
    if src.suffix in {".tgz", ".gz"} or src.name.endswith(".tar.gz"):
        with tarfile.open(src, "r:gz") as tf:
            tf.extractall(dest)
        return True
    return False


def _copy_file(src: Path, dest: Path) -> None:
    try:
        shutil.copy2(src, dest)
    except OSError:
        with src.open("rb") as fsrc, dest.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)


def _copy_dataset(src: str, dest: str) -> None:
    src_path = Path(src)
    dest_path = RAW_DIR / dest
    dest_path.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        target = dest_path / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            if _extract_archive(item, dest_path):
                continue
            _copy_file(item, target)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, dataset_id in DATASETS.items():
        print(f"Downloading {name} from KaggleHub: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        print("Downloaded to:", path)
        _copy_dataset(path, name)

    print("Datasets copied into data/raw/<dataset>")


if __name__ == "__main__":
    main()


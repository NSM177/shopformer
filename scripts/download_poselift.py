"""Download PoseLift JSON pose data and ground-truth masks from Google Drive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import urllib.request

import sys
import gdown
import yaml

# Allow listing folders with more than 50 files
sys.modules["gdown.download_folder"].MAX_NUMBER_FILES = 500

DRIVE_FOLDERS: dict[str, str] = {
    "gt_mask_dir": "1oKA8p5AIoHkvxErI7GpzO0VfJzSIUDTX",
    "pose_test_dir": "1Cr5wAwWV2OhLyg1sY0J_3_G418xye3z5",
    "pose_train_dir": "1xtbihtbbfiGkzatF7PNGsh12et0l_t9d",
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download_single_file(file_id: str, dest_path: Path) -> Path:
    """Download a single Google Drive file by ID."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    dest_path.write_bytes(data)
    return dest_path


def download_poselift(config_path: Path, project_root: Path) -> dict[str, int]:
    """Download PoseLift train/test JSONs and ground-truth frame masks concurrently."""
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    counts: dict[str, int] = {}

    for key, folder_id in DRIVE_FOLDERS.items():
        target_dir = project_root / paths_cfg[key]
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = ".npy" if key == "gt_mask_dir" else ".json"
        existing = list(target_dir.glob(f"*{ext}"))
        min_expected = 40 if key != "pose_train_dir" else 45
        if len(existing) >= min_expected:
            print(f"[SKIP] {key} ({target_dir}) already has {len(existing)} {ext} files.")
            counts[key] = len(existing)
            continue

        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"[LIST] Listing files in {key} from {url} ...")
        items = gdown.download_folder(
            url=url,
            output=str(target_dir),
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
        print(f"[DOWNLOAD] Downloading {len(items)} files into {target_dir} ...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [
                pool.submit(_download_single_file, item.id, Path(item.local_path))
                for item in items
            ]
            for fut in as_completed(futures):
                fut.result()

        downloaded = list(target_dir.glob(f"*{ext}"))
        counts[key] = len(downloaded)
        print(f"[DONE] {key}: {counts[key]} {ext} files in {target_dir}.")

    return counts


def main() -> None:
    """CLI entrypoint for downloading PoseLift dataset."""
    parser = argparse.ArgumentParser(description="Download PoseLift dataset from Google Drive.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    counts = download_poselift(config_path=config_path, project_root=project_root)
    print("PoseLift dataset summary:", counts)


if __name__ == "__main__":
    main()

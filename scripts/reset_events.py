#!/usr/bin/env python3
"""Script dọn sạch các sự kiện cảnh báo cũ trong SQLite (`data/events.db`) và thư mục clip (`data/clips`)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shoplift_iot.config import load_app_config  # noqa: E402
from shoplift_iot.dashboard.db import EventDatabase  # noqa: E402


def reset_all_events(
    config_path: Path | str | None = None,
    db_path_override: Path | str | None = None,
    clips_dir_override: Path | str | None = None,
    delete_clips: bool = True,
) -> dict[str, int]:
    """Xóa toàn bộ bản ghi trong bảng `events` của SQLite và (tùy chọn) xóa các file `.mp4` cũ trong `clips_dir`."""
    cfg = load_app_config(config_path, project_root=PROJECT_ROOT)
    db_path = Path(db_path_override) if db_path_override is not None else cfg.resolve_path(cfg.paths["db_path"])
    clips_dir = (
        Path(clips_dir_override) if clips_dir_override is not None else cfg.resolve_path(cfg.paths["clips_dir"])
    )

    db = EventDatabase(db_path)
    deleted_events = db.clear_events()

    deleted_clips = 0
    if delete_clips and clips_dir.exists():
        for mp4_file in clips_dir.glob("*.mp4"):
            try:
                mp4_file.unlink()
                deleted_clips += 1
            except OSError:
                pass

    return {
        "deleted_events": deleted_events,
        "deleted_clips": deleted_clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Xóa toàn bộ sự kiện cảnh báo cũ khỏi SQLite và thư mục clips.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Đường dẫn file YAML cấu hình.")
    parser.add_argument("--db-path", type=str, default=None, help="Ghi đè đường dẫn file SQLite.")
    parser.add_argument("--clips-dir", type=str, default=None, help="Ghi đè đường dẫn thư mục clips.")
    parser.add_argument("--keep-clips", action="store_true", help="Giữ lại các file .mp4 trong thư mục clips.")
    args = parser.parse_args()

    res = reset_all_events(
        config_path=args.config,
        db_path_override=args.db_path,
        clips_dir_override=args.clips_dir,
        delete_clips=not args.keep_clips,
    )
    print(
        f"[reset_events] Đã xóa {res['deleted_events']} sự kiện trong SQLite "
        f"và {res['deleted_clips']} file clip (.mp4)."
    )


if __name__ == "__main__":
    main()


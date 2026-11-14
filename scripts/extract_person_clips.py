"""Offline script to cut per-person clips from recorded videos for VideoMAE training (shares real-time code)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.config import load_app_config  # noqa: E402
from shoplift_iot.vision.person_clips import (  # noqa: E402
    PersonClipExtractor,
    save_person_clip_mp4,
)
from videomae_lite import VIDEO_EXTS  # noqa: E402


def extract_clips_from_video_file(
    video_path: Path,
    out_dir: Path,
    config_path: Path | None = None,
    yolo_instance: Any | None = None,
    clip_sec_override: float | None = None,
    hop_sec_override: float | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Run `PersonClipExtractor` on `video_path` and save cropped person `.mp4` files into `out_dir`."""
    cfg = load_app_config(config_path, project_root=PROJECT_ROOT)
    extractor = PersonClipExtractor.from_config(
        cfg,
        yolo_instance=yolo_instance,
        clip_sec_override=clip_sec_override,
        hop_sec_override=hop_sec_override,
    )
    target_fps = float(cfg.video["target_fps"])
    windows = extractor.extract_from_video(
        video_path=video_path,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    saved_paths: list[Path] = []
    per_track_counter: dict[int, int] = {}

    for win in windows:
        seq_idx = per_track_counter.get(win.track_id, 0) + 1
        per_track_counter[win.track_id] = seq_idx
        # Preserve person prefix (e.g. `an_001` -> `an_001_tid1_001.mp4`) so train_videomae.ipynb splits by person
        out_name = f"{stem}_tid{win.track_id}_{seq_idx:03d}.mp4"
        out_file = save_person_clip_mp4(win, out_dir / out_name, fps=target_fps)
        saved_paths.append(out_file)

    return saved_paths


def extract_dataset_folder(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    clip_sec_override: float | None = None,
    hop_sec_override: float | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract person clips from either a single video file or a directory of videos (`normal/`, `shoplifting/`)."""
    if input_path.is_file():
        return extract_clips_from_video_file(
            video_path=input_path,
            out_dir=output_dir,
            config_path=config_path,
            clip_sec_override=clip_sec_override,
            hop_sec_override=hop_sec_override,
            max_frames=max_frames,
        )

    all_saved: list[Path] = []
    video_files = sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    for vid in video_files:
        rel_parent = vid.parent.relative_to(input_path)
        target_sub = output_dir / rel_parent
        saved = extract_clips_from_video_file(
            video_path=vid,
            out_dir=target_sub,
            config_path=config_path,
            clip_sec_override=clip_sec_override,
            hop_sec_override=hop_sec_override,
            max_frames=max_frames,
        )
        all_saved.extend(saved)
        print(f"[extract_person_clips] {vid.name} -> {len(saved)} person clip(s) in {target_sub}")

    return all_saved


def main() -> None:
    """CLI entrypoint for offline person clip extraction."""
    parser = argparse.ArgumentParser(
        description="Cut per-person 16-frame clips from recorded videos using YOLO + ByteTrack."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file or directory (e.g. raw_videos/ containing normal/ and shoplifting/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clips_person"),
        help="Output directory for cropped person clips.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--clip-sec", type=float, default=None, help="Override clip duration (seconds).")
    parser.add_argument("--hop-sec", type=float, default=None, help="Override hop interval (seconds).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max downsampled frames per video.")
    args = parser.parse_args()

    saved = extract_dataset_folder(
        input_path=args.input,
        output_dir=args.output,
        config_path=args.config,
        clip_sec_override=args.clip_sec,
        hop_sec_override=args.hop_sec,
        max_frames=args.max_frames,
    )
    print(f"[extract_person_clips] Done. Total person clips saved: {len(saved)}")


if __name__ == "__main__":
    main()

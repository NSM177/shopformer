"""Extract human pose sequences from videos into PoseLift-compatible JSON (and optional PKL) files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.capture.source import VideoSource  # noqa: E402
from shoplift_iot.vision.pose_track import PoseTracker, load_tracker_from_config  # noqa: E402
from shoplift_iot.vision.preprocess import (  # noqa: E402
    interpolate_missing_frames,
    smooth_pose_sequence,
)


def _interpolate_contiguous_segments(
    frame_to_kps: dict[int, np.ndarray],
    max_gap: int = 2,
    smoothing_window: int = 8,
) -> dict[int, np.ndarray]:
    """Interpolate gaps <= max_gap and apply 8-frame temporal smoothing per contiguous segment."""
    if not frame_to_kps:
        return {}

    sorted_frames = sorted(frame_to_kps.keys())
    segments: list[list[int]] = [[sorted_frames[0]]]
    for f_idx in sorted_frames[1:]:
        gap = f_idx - segments[-1][-1] - 1
        if gap <= max_gap:
            segments[-1].append(f_idx)
        else:
            segments.append([f_idx])

    processed: dict[int, np.ndarray] = {}
    for seg in segments:
        s_frame, e_frame = seg[0], seg[-1]
        sub_map = {k: frame_to_kps[k] for k in seg}
        dense = interpolate_missing_frames(
            sub_map,
            start_frame=s_frame,
            end_frame=e_frame,
            max_missing_frames=max_gap * max(1, len(seg)),
        )
        if dense is None:
            continue
        smoothed = smooth_pose_sequence(dense, window_size=smoothing_window)
        for offset in range(smoothed.shape[0]):
            processed[s_frame + offset] = smoothed[offset]

    return processed


def extract_video_poses_to_poselift_dict(
    video_path: Path,
    tracker: PoseTracker,
    target_fps: float = 15.0,
    max_missing_frames: int = 2,
    smoothing_window: int = 8,
    swap_xy_to_yx_for_poselift: bool = True,
    max_frames: int | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[int, dict[int, list[Any]]]]:
    """Run Steps 1-5 on a video and return (poselift_json_dict, poselift_pkl_dict)."""
    per_track_kps: dict[int, dict[int, np.ndarray]] = {}
    per_track_bboxes: dict[int, dict[int, list[int]]] = {}
    total_frames = 0

    with VideoSource(source=video_path, target_fps=target_fps) as reader:
        for pkt in reader:
            total_frames = pkt.frame_idx + 1
            poses = tracker.process_frame(pkt.frame)
            for p in poses:
                tid = int(p.track_id)
                per_track_kps.setdefault(tid, {})[pkt.frame_idx] = p.keypoints_xyc
                per_track_bboxes.setdefault(tid, {})[pkt.frame_idx] = [
                    int(round(v)) for v in p.bbox_xywh.tolist()
                ]
            if max_frames is not None and total_frames >= max_frames:
                break

    json_dict: dict[str, dict[str, dict[str, Any]]] = {}
    pkl_dict: dict[int, dict[int, list[Any]]] = {f: {} for f in range(total_frames)}

    for tid in sorted(per_track_kps.keys()):
        smoothed_track = _interpolate_contiguous_segments(
            per_track_kps[tid],
            max_gap=max_missing_frames,
            smoothing_window=smoothing_window,
        )
        if not smoothed_track:
            continue

        person_str = str(tid)
        json_dict[person_str] = {}
        for f_idx in sorted(smoothed_track.keys()):
            kps = smoothed_track[f_idx].copy()  # (17, 3) in [X, Y, C]
            if swap_xy_to_yx_for_poselift:
                # Match raw PoseLift JSON convention [Y, X, C]
                kps = kps[:, [1, 0, 2]]
            flat_kps = [float(val) for val in kps.reshape(-1).tolist()]
            json_dict[person_str][str(f_idx)] = {
                "keypoints": flat_kps,
                "scores": None,
            }
            if f_idx in pkl_dict:
                bbox = per_track_bboxes.get(tid, {}).get(f_idx, [0, 0, 0, 0])
                pkl_dict[f_idx][tid] = [bbox, kps.astype(np.float16)]

    return json_dict, pkl_dict


def extract_poses(
    input_path: Path,
    output_dir: Path,
    config_path: Path,
    scene_id: int = 1,
    start_clip_id: int = 900,
    save_pkl: bool = False,
    max_frames: int | None = None,
) -> list[Path]:
    """Extract poses from a video file or directory of videos into PoseLift JSON files."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tracker = load_tracker_from_config(config_path)
    target_fps = float(cfg["video"]["target_fps"])
    swap_yx = bool(cfg["video"].get("swap_yx_from_poselift_json", True))
    max_missing = int(cfg["preprocess"]["max_missing_frames"])
    smooth_win = int(cfg["preprocess"]["smoothing_window"])

    if input_path.is_dir():
        video_files = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
            ]
        )
    else:
        video_files = [input_path]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for idx, vid_file in enumerate(video_files):
        clip_id = start_clip_id + idx
        json_dict, pkl_dict = extract_video_poses_to_poselift_dict(
            video_path=vid_file,
            tracker=tracker,
            target_fps=target_fps,
            max_missing_frames=max_missing,
            smoothing_window=smooth_win,
            swap_xy_to_yx_for_poselift=swap_yx,
            max_frames=max_frames,
        )
        json_name = f"{scene_id:02d}_{clip_id:04d}_alphapose_tracked_person.json"
        json_out = output_dir / json_name
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(json_dict, jf)
        saved_files.append(json_out)
        print(
            f"[extract_poses] Saved {json_out} ({len(json_dict)} tracks) from '{vid_file.name}'"
        )

        if save_pkl:
            pkl_out = output_dir / f"{scene_id}_{clip_id}.pkl"
            with open(pkl_out, "wb") as pf:
                pickle.dump(pkl_dict, pf)

    return saved_files


def main() -> None:
    """CLI entrypoint for PoseLift-compatible pose extraction."""
    parser = argparse.ArgumentParser(
        description="Extract PoseLift-formatted JSON pose files from video(s)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Video file or directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for PoseLift JSON files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--scene-id", type=int, default=1, help="Camera/scene ID prefix.")
    parser.add_argument("--clip-id", type=int, default=900, help="Starting clip ID.")
    parser.add_argument("--save-pkl", action="store_true", help="Also save PoseLift .pkl file.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames limit.")
    args = parser.parse_args()

    extract_poses(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        scene_id=args.scene_id,
        start_clip_id=args.clip_id,
        save_pkl=args.save_pkl,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""score_video_bundle.py: Chạy CHUẨN offline theo đúng quy trình lúc train VideoMAE:
YOLO tracking -> Cửa sổ theo thời gian (timestamp) -> Cắt vuông & mask người khác (crops_lite) -> Predictor -> Điểm thô theo từng track và từng giây.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "models" / "videomae_bundle")):
    if p not in sys.path:
        sys.path.insert(0, p)

from crops_lite import YoloTracker, make_windows, render_windows, sample_16_frames_in_time
from videomae_lite import Predictor


def run_score_video_bundle(
    video_path: Path | str,
    bundle_dir: Path | str = "models/videomae_bundle",
    out_csv: Path | str | None = None,
    save_first_grids: int = 0,
    grids_dir: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Chạy đánh giá chuẩn offline trên video và trả về danh sách cửa sổ + tensor 16 khung."""
    vpath = Path(video_path)
    bpath = Path(bundle_dir)
    prep = json.loads((bpath / "preprocess.json").read_text(encoding="utf-8"))
    cal = json.loads((bpath / "calibration.json").read_text(encoding="utf-8"))

    clip_sec = float(prep.get("clip_sec", 3.0))
    hop_sec = float(prep.get("hop_sec", 1.0))
    margin = float(prep.get("crop_margin", 0.18))
    mask_others = bool(prep.get("mask_others", True))
    size = int(prep.get("size", 224))
    det_cfg = prep.get("detector", {})

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video: {vpath}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames_bgr: list[np.ndarray] = []
    timestamps: list[float] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames_bgr.append(frame)
        timestamps.append(round(idx / fps, 4))
        idx += 1
    cap.release()

    total_frames = len(frames_bgr)
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # 1. Tracking
    w_path = PROJECT_ROOT / det_cfg.get("weights", "models/yolov8s.pt")
    if not w_path.exists():
        w_path = PROJECT_ROOT / "models" / det_cfg.get("weights", "yolov8s.pt")
    tracker = YoloTracker(
        weights=str(w_path),
        conf=float(det_cfg.get("conf", 0.15)),
        classes=det_cfg.get("classes", [0]),
        tracker=det_cfg.get("tracker", "bytetrack.yaml"),
    )
    tracker.reset()

    # Lưu track records: {track_id: [(ts, target_bbox, other_bboxes), ...]}
    track_records: dict[int, list[tuple[float, list[float], list[list[float]]]]] = {}

    for f_idx, frame in enumerate(frames_bgr):
        ts = timestamps[f_idx]
        tracks = tracker.track_frame(frame)
        all_bboxes = [t[1] for t in tracks]
        for tid, bbox, conf in tracks:
            other_bboxes = [b for b in all_bboxes if b != bbox]
            track_records.setdefault(tid, []).append((ts, bbox, other_bboxes))

    # 2. Make windows (clip_sec=3.0s, hop_sec=1.0s, min_cover=0.9)
    windows = make_windows(
        track_records=track_records,
        duration_sec=duration_sec,
        clip_sec=clip_sec,
        hop_sec=hop_sec,
        min_cover=0.9,
    )

    # 3. Predictor
    predictor = Predictor(bpath)

    results: list[dict[str, Any]] = []
    saved_grids_tensors: list[np.ndarray] = []

    if grids_dir:
        Path(grids_dir).mkdir(parents=True, exist_ok=True)

    for w_idx, win in enumerate(windows):
        tid = win["track_id"]
        t0 = win["t0"]
        t1 = win["t1"]
        records = win["records"]

        frames_rgb_16 = sample_16_frames_in_time(
            records=records,
            frames_bgr_by_idx=frames_bgr,
            timestamps=timestamps,
            t0=t0,
            t1=t1,
            margin=margin,
            mask_others=mask_others,
            size=size,
        )

        p = float(predictor.predict(frames_rgb_16))
        results.append({
            "track_id": tid,
            "t0": round(t0, 2),
            "p": round(p, 6),
        })

        if w_idx < save_first_grids:
            saved_grids_tensors.append(frames_rgb_16)
            if grids_dir:
                g_path = Path(grids_dir) / f"ref_win_{w_idx}_tid{tid}_t0_{t0:.2f}.png"
                render_windows(frames_rgb_16, g_path, title=f"REF Win {w_idx} | Track #{tid} | t0={t0:.2f}s | p={p:.4f}")

    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "t0", "p"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    return results, saved_grids_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy chuẩn offline VideoMAE bundle.")
    parser.add_argument("--video", type=Path, required=True, help="Đường dẫn file video.")
    parser.add_argument("--bundle", type=Path, default=Path("models/videomae_bundle"), help="Thư mục bundle.")
    parser.add_argument("--out-csv", type=Path, default=None, help="File CSV đầu ra (track_id, t0, p).")
    parser.add_argument("--save-grids", type=int, default=3, help="Số cửa sổ đầu tiên lưu lưới ảnh.")
    parser.add_argument("--grids-dir", type=Path, default=Path("data/diagnostics/grids_ref"), help="Thư mục lưu lưới ảnh.")
    args = parser.parse_args()

    results, _ = run_score_video_bundle(
        video_path=args.video,
        bundle_dir=args.bundle,
        out_csv=args.out_csv,
        save_first_grids=args.save_grids,
        grids_dir=args.grids_dir,
    )
    print(f"[score_video_bundle] Hoàn thành {len(results)} cửa sổ cho {args.video.name}.")
    for r in results:
        print(f"  Track #{r['track_id']} | t0={r['t0']:5.2f}s | p={r['p']:.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic script: Run video(s) through the VideoMAE pipeline in:
  (a) full_frame  : 16 frames sampled uniformly from the whole video frame
  (b) person_crop : YOLO + ByteTrack (persist=True, classes=[0]) -> per-track 18% margin squared crop -> 16 frames

Also supports:
  --run-suite : Execute full diagnostic suite:
    1) Sweep clip_sec = [3.0, 5.0, 8.0, 10.87] in person_crop mode on theft_overhead_cctv.mp4
    2) Baseline on >= 3 normal overhead CCTV videos in data/normal/
    3) Compare Before (yolov8n, no coasting) vs After (yolov8s, conf=0.15, Kalman bbox coasting)
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shoplift_iot.config import load_app_config  # noqa: E402
from shoplift_iot.contracts import PersonPose  # noqa: E402
from shoplift_iot.vision.person_clips import (  # noqa: E402
    PersonClipBuffer,
    build_videomae_clip_from_crops,
)
from shoplift_iot.vision.pose_track import PoseTracker, load_tracker_from_config  # noqa: E402

try:
    from shoplift_iot.model.videomae_lite import Predictor  # noqa: E402
except ImportError:
    from videomae_lite import Predictor  # type: ignore[no-redef]  # noqa: E402


def max_consecutive_above(rows: list[dict[str, Any]], threshold: float) -> int:
    """Compute maximum consecutive windows with score >= threshold per track."""
    by_track: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_track.setdefault(int(r["track_id"]), []).append(r)
    best = 0
    for _, t_rows in by_track.items():
        cur = 0
        for r in t_rows:
            if float(r["score"]) >= threshold:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
    return best


def extract_poses_sequence(
    frames_bgr: list[np.ndarray],
    fps: float,
    tracker: PoseTracker,
) -> tuple[list[tuple[float, list[PersonPose]]], dict[int, list[tuple[int, float, float, list[float], bool]]]]:
    """Run tracker across all frames once and return per-frame poses + detailed per-track log."""
    if hasattr(tracker, "reset"):
        tracker.reset()
    seq: list[tuple[float, list[PersonPose]]] = []
    track_logs: dict[int, list[tuple[int, float, float, list[float], bool]]] = {}

    for idx, frame in enumerate(frames_bgr):
        ts = idx / fps
        poses = tracker.process_frame(frame)
        coasted_ids = getattr(tracker, "last_coasted_ids", set())
        seq.append((ts, poses))
        for p in poses:
            is_coasted = int(p.track_id) in coasted_ids
            track_logs.setdefault(int(p.track_id), []).append(
                (
                    idx,
                    round(ts, 2),
                    round(float(p.score), 3),
                    [round(float(v), 1) for v in p.bbox_xywh],
                    is_coasted,
                )
            )
    return seq, track_logs


def score_poses_with_clip_sec(
    frames_bgr: list[np.ndarray],
    poses_seq: list[tuple[float, list[PersonPose]]],
    predictor: Any,
    clip_sec: float = 3.0,
    hop_sec: float = 0.5,
    margin_ratio: float = 0.18,
    square: bool = True,
    min_span_ratio: float = 0.8,
    min_frames: int = 16,
    stale_sec: float = 5.0,
    adapt_short_video: bool = True,
) -> list[dict[str, Any]]:
    """Score pre-extracted poses using PersonClipBuffer with a given clip_sec."""
    duration_sec = poses_seq[-1][0] if poses_seq else 0.0
    eff_min_span = min_span_ratio
    if adapt_short_video and duration_sec > 0 and clip_sec * min_span_ratio > duration_sec * 0.55:
        # Allow sliding evaluation starting once track has >= 2.4s of data while keeping up to clip_sec in buffer
        eff_min_span = max(0.22, min(min_span_ratio, 2.40 / clip_sec))

    clip_buf = PersonClipBuffer(
        clip_sec=clip_sec,
        hop_sec=hop_sec,
        n_frames=int(predictor.n_frames),
        size=int(predictor.size),
        margin_ratio=margin_ratio,
        min_span_ratio=eff_min_span,
        min_frames=min_frames,
        stale_sec=stale_sec,
        square=square,
    )

    rows: list[dict[str, Any]] = []
    for idx, (ts, poses) in enumerate(poses_seq):
        windows = clip_buf.update(ts, frames_bgr[idx], poses)
        for win in windows:
            score = float(predictor.predict(win.frames_rgb))
            rows.append(
                {
                    "time_sec": round(ts, 2),
                    "track_id": int(win.track_id),
                    "score": round(score, 4),
                    "mode": "person_crop",
                    "clip_sec": round(clip_sec, 2),
                    "start_sec": round(win.start_ts, 2),
                    "end_sec": round(win.end_ts, 2),
                    "n_buffered_frames": len(win.raw_crops_bgr),
                }
            )
    return rows


def run_full_frame_mode(
    frames_bgr: list[np.ndarray],
    fps: float,
    predictor: Any,
    clip_sec: float = 3.0,
    hop_sec: float = 0.5,
    min_span_ratio: float = 0.8,
    min_frames: int = 16,
) -> list[dict[str, Any]]:
    """Mode (a): Score 16 uniformly sampled full frames per sliding window."""
    buf: deque[tuple[float, np.ndarray]] = deque()
    next_eval: float | None = None
    rows: list[dict[str, Any]] = []

    for idx, frame in enumerate(frames_bgr):
        ts = idx / fps
        buf.append((ts, frame))
        while buf and (ts - buf[0][0]) > clip_sec + 1e-6:
            buf.popleft()

        if next_eval is None:
            next_eval = buf[0][0] + clip_sec * min_span_ratio

        span = buf[-1][0] - buf[0][0]
        if (
            len(buf) >= min_frames
            and span + 1e-6 >= clip_sec * min_span_ratio
            and ts + 1e-6 >= next_eval
        ):
            next_eval = ts + hop_sec
            raw_frames = [item[1] for item in buf]
            clip_rgb = build_videomae_clip_from_crops(
                raw_frames,
                n_frames=int(predictor.n_frames),
                size=int(predictor.size),
            )
            score = float(predictor.predict(clip_rgb))
            rows.append(
                {
                    "time_sec": round(ts, 2),
                    "track_id": 0,
                    "score": round(score, 4),
                    "mode": "full_frame",
                    "clip_sec": round(clip_sec, 2),
                    "start_sec": round(buf[0][0], 2),
                    "end_sec": round(buf[-1][0], 2),
                    "n_buffered_frames": len(buf),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write diagnostic score rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_sec",
        "track_id",
        "score",
        "mode",
        "clip_sec",
        "start_sec",
        "end_sec",
        "n_buffered_frames",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def render_comparison_plot(
    full_rows: list[dict[str, Any]],
    crop_rows: list[dict[str, Any]],
    threshold: float,
    video_name: str,
    duration_sec: float,
    out_png: Path,
) -> None:
    """Draw a high-resolution comparison chart of score vs time for (a) full_frame and (b) person_crop."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        if full_rows:
            t_full = [r["time_sec"] for r in full_rows]
            s_full = [r["score"] for r in full_rows]
            ax.plot(
                t_full,
                s_full,
                marker="o",
                linewidth=2.2,
                color="#94a3b8",
                label="(a) Cả khung hình (full_frame)",
            )
            peak_f = max(full_rows, key=lambda x: x["score"])
            ax.annotate(
                f"Max (a): {peak_f['score']:.4f} @ {peak_f['time_sec']:.2f}s",
                xy=(peak_f["time_sec"], peak_f["score"]),
                xytext=(peak_f["time_sec"], min(0.96, peak_f["score"] + 0.07)),
                color="#cbd5e1",
                fontsize=9,
                fontweight="bold",
                ha="center",
                arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.2},
            )

        by_track: dict[int, list[dict[str, Any]]] = {}
        for r in crop_rows:
            by_track.setdefault(int(r["track_id"]), []).append(r)

        palette = ["#38bdf8", "#22c55e", "#f59e0b", "#a855f7", "#ec4899"]
        for idx_t, (tid, t_rows) in enumerate(sorted(by_track.items())):
            color = palette[idx_t % len(palette)]
            t_crop = [r["time_sec"] for r in t_rows]
            s_crop = [r["score"] for r in t_rows]
            ax.plot(
                t_crop,
                s_crop,
                marker="s",
                linewidth=2.5,
                color=color,
                label=f"(b) Cắt quanh người — Track #{tid} (square+18% margin)",
            )

        if crop_rows:
            peak_c = max(crop_rows, key=lambda x: x["score"])
            ax.annotate(
                f"Max (b) Track #{peak_c['track_id']}: {peak_c['score']:.4f} @ {peak_c['time_sec']:.2f}s",
                xy=(peak_c["time_sec"], peak_c["score"]),
                xytext=(peak_c["time_sec"], min(0.96, peak_c["score"] + 0.09)),
                color="#38bdf8",
                fontsize=9,
                fontweight="bold",
                ha="center",
                arrowprops={"arrowstyle": "->", "color": "#38bdf8", "lw": 1.4},
            )

        ax.axhline(
            threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=1.6,
            label=f"Ngưỡng cảnh báo ({threshold:.3f})",
        )

        ax.set_xlim(0.0, max(duration_sec, 1.0))
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Thời gian video (giây)", color="#e2e8f0", fontsize=10)
        ax.set_ylabel("Xác suất nghi trộm VideoMAE p(shoplifting)", color="#e2e8f0", fontsize=10)
        ax.set_title(
            f"So sánh điểm VideoMAE theo thời gian — {video_name}\n(a) Cả khung hình vs (b) Cắt vuông quanh người (18% lề)",
            color="#f8fafc",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.35, color="#64748b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        leg = ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#475569", fontsize=9)
        for text in leg.get_texts():
            text.set_color("#f1f5f9")

        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close(fig)
    except Exception:
        pass


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Read all BGR frames and FPS from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frm = cap.read()
        if not ok or frm is None:
            break
        frames_bgr.append(frm)
    cap.release()
    return frames_bgr, fps


def run_diagnostic_suite(predictor: Any, threshold: float, out_dir: Path) -> None:
    """Execute Steps 1, 2, 3 requested by the user and print structured tables."""
    test_video = PROJECT_ROOT / "data" / "theft_overhead_cctv.mp4"
    frames_bgr, fps = load_video_frames(test_video)
    total_frames = len(frames_bgr)
    duration_sec = total_frames / fps

    print("\n" + "=" * 86, flush=True)
    print("PHẦN 3 & 1: SO SÁNH TRƯỚC / SAU KHI ĐỔI SANG YOLOv8s + GIỮ BBOX DỰ ĐOÁN KALMAN", flush=True)
    print("=" * 86, flush=True)

    # 1. BEFORE: yolov8n.pt, conf=0.20, max_coast_frames=0
    tracker_before = PoseTracker(
        model_name="models/yolov8n.pt",
        tracker_config="configs/bytetrack_cctv.yaml",
        conf_threshold=0.20,
        iou_threshold=0.45,
        device="cuda:0",
        max_coast_frames=0,
    )
    seq_before, logs_before = extract_poses_sequence(frames_bgr, fps, tracker_before)

    # 2. AFTER: yolov8s.pt, conf=0.15, max_coast_frames=15 (Kalman coasting)
    tracker_after = PoseTracker(
        model_name="models/yolov8s.pt",
        tracker_config="configs/bytetrack_cctv.yaml",
        conf_threshold=0.15,
        iou_threshold=0.45,
        device="cuda:0",
        max_coast_frames=15,
    )
    seq_after, logs_after = extract_poses_sequence(frames_bgr, fps, tracker_after)

    def _summarize_track_drops(logs: dict[int, list[tuple[int, float, float, list[float], bool]]]) -> dict[str, Any]:
        if not logs:
            return {"main_tid": None, "f_start": 0, "f_end": 0, "span_frames": 0, "present": 0, "dropped": total_frames, "coasted": 0}
        main_tid, hist = max(logs.items(), key=lambda kv: len(kv[1]))
        f_start = hist[0][0]
        f_end = hist[-1][0]
        span_frames = f_end - f_start + 1
        present = len(hist)
        coasted = sum(1 for x in hist if x[4])
        dropped_in_span = span_frames - present
        dropped_total = total_frames - present
        return {
            "main_tid": main_tid,
            "f_start": f_start,
            "f_end": f_end,
            "t_start": hist[0][1],
            "t_end": hist[-1][1],
            "span_frames": span_frames,
            "present": present,
            "coasted": coasted,
            "dropped_in_span": dropped_in_span,
            "dropped_total": dropped_total,
        }

    stat_before = _summarize_track_drops(logs_before)
    stat_after = _summarize_track_drops(logs_after)

    print(
        f"[TRƯỚC - yolov8n, conf=0.20, coast=0] Track chính #{stat_before['main_tid']}: "
        f"khung {stat_before['f_start']}..{stat_before['f_end']} ({stat_before['t_start']:.2f}s..{stat_before['t_end']:.2f}s) | "
        f"Có mặt: {stat_before['present']}/{stat_before['span_frames']} khung trong đoạn "
        f"(Rơi trong đoạn: {stat_before['dropped_in_span']} khung, Không có bbox toàn video: {stat_before['dropped_total']}/{total_frames} khung)",
        flush=True,
    )
    print(
        f"[SAU   - yolov8s, conf=0.15, coast=15] Track chính #{stat_after['main_tid']}: "
        f"khung {stat_after['f_start']}..{stat_after['f_end']} ({stat_after['t_start']:.2f}s..{stat_after['t_end']:.2f}s) | "
        f"Có mặt: {stat_after['present']}/{stat_after['span_frames']} khung trong đoạn "
        f"(Kalman bù: {stat_after['coasted']} khung | Rơi trong đoạn: {stat_after['dropped_in_span']} khung, "
        f"Không có bbox toàn video: {stat_after['dropped_total']}/{total_frames} khung)",
        flush=True,
    )

    clip_secs = [3.0, 5.0, 8.0, round(float(predictor.clip_sec), 2)]

    print("\n" + "=" * 86, flush=True)
    print("BẢNG 1A: QUÉT clip_sec TRƯỚC KHI BÙ BBOX (yolov8n, conf=0.20, coast=0)", flush=True)
    print("clip_sec | peak_score | peak_time (window)      | mean_score | max_consec >= 0.5863 | n_windows", flush=True)
    print("-" * 86, flush=True)
    rows_before_3s: list[dict[str, Any]] = []
    for cs in clip_secs:
        r_list = score_poses_with_clip_sec(frames_bgr, seq_before, predictor, clip_sec=cs, hop_sec=0.5)
        if cs == 3.0:
            rows_before_3s = r_list
        if r_list:
            pk = max(r_list, key=lambda x: x["score"])
            mn = sum(x["score"] for x in r_list) / len(r_list)
            mc = max_consecutive_above(r_list, threshold)
            print(
                f"{cs:8.2f} | {pk['score']:10.4f} | {pk['time_sec']:5.2f}s ({pk['start_sec']:4.2f}s..{pk['end_sec']:4.2f}s) | "
                f"{mn:10.4f} | {mc:20d} | {len(r_list):9d}",
                flush=True,
            )
            for r in r_list:
                print(f"   -> t={r['time_sec']:4.2f}s [{r['start_sec']:4.2f}..{r['end_sec']:4.2f}s] tid={r['track_id']} score={r['score']:.4f}", flush=True)
        else:
            print(f"{cs:8.2f} |       N/A  | N/A                     |       N/A  |                    0 |         0", flush=True)

    print("\n" + "=" * 86, flush=True)
    print("BẢNG 1B: QUÉT clip_sec SAU KHI DÙNG YOLOv8s + GIỮ BBOX DỰ ĐOÁN (conf=0.15, coast=15)", flush=True)
    print("clip_sec | peak_score | peak_time (window)      | mean_score | max_consec >= 0.5863 | n_windows", flush=True)
    print("-" * 86, flush=True)
    rows_after_3s: list[dict[str, Any]] = []
    for cs in clip_secs:
        r_list = score_poses_with_clip_sec(frames_bgr, seq_after, predictor, clip_sec=cs, hop_sec=0.5)
        if cs == 3.0:
            rows_after_3s = r_list
        if r_list:
            pk = max(r_list, key=lambda x: x["score"])
            mn = sum(x["score"] for x in r_list) / len(r_list)
            mc = max_consecutive_above(r_list, threshold)
            print(
                f"{cs:8.2f} | {pk['score']:10.4f} | {pk['time_sec']:5.2f}s ({pk['start_sec']:4.2f}s..{pk['end_sec']:4.2f}s) | "
                f"{mn:10.4f} | {mc:20d} | {len(r_list):9d}",
                flush=True,
            )
            for r in r_list:
                print(f"   -> t={r['time_sec']:4.2f}s [{r['start_sec']:4.2f}..{r['end_sec']:4.2f}s] tid={r['track_id']} score={r['score']:.4f}", flush=True)
        else:
            print(f"{cs:8.2f} |       N/A  | N/A                     |       N/A  |                    0 |         0", flush=True)

    # Save updated CSV and plot for theft_overhead_cctv.mp4 using yolov8s + coasting
    full_rows = run_full_frame_mode(frames_bgr, fps, predictor, clip_sec=3.0, hop_sec=0.5)
    write_csv(full_rows + rows_after_3s, out_dir / "theft_overhead_cctv_scores_comparison.csv")
    render_comparison_plot(
        full_rows=full_rows,
        crop_rows=rows_after_3s,
        threshold=threshold,
        video_name="theft_overhead_cctv.mp4 (yolov8s + coasting)",
        duration_sec=duration_sec,
        out_png=out_dir / "theft_overhead_cctv_scores_plot.png",
    )

    print("\n" + "=" * 86, flush=True)
    print("PHẦN 2: ĐƯỜNG CƠ SỞ (BASELINE) TRÊN CÁC VIDEO BÌNH THƯỜNG CÙNG GÓC CAMERA TRÊN CAO", flush=True)
    print("video_name               | dur_s | clip_sec | peak_score | peak_time | mean_score | max_consec>=0.5863 | n_win", flush=True)
    print("-" * 86, flush=True)

    normal_videos = sorted((PROJECT_ROOT / "data" / "normal").glob("*.mp4"))
    for nv in normal_videos:
        n_frames_bgr, n_fps = load_video_frames(nv)
        n_dur = len(n_frames_bgr) / n_fps if n_fps > 0 else 0.0
        n_seq, _ = extract_poses_sequence(n_frames_bgr, n_fps, tracker_after)
        for cs in (3.0, 5.0):
            n_rows = score_poses_with_clip_sec(n_frames_bgr, n_seq, predictor, clip_sec=cs, hop_sec=0.5)
            if cs == 3.0:
                write_csv(n_rows, out_dir / f"{nv.stem}_person_crop.csv")
            if n_rows:
                pk = max(n_rows, key=lambda x: x["score"])
                mn = sum(x["score"] for x in n_rows) / len(n_rows)
                mc = max_consecutive_above(n_rows, threshold)
                print(
                    f"{nv.name:24s} | {n_dur:5.2f} | {cs:8.2f} | {pk['score']:10.4f} | "
                    f"{pk['time_sec']:8.2f}s | {mn:10.4f} | {mc:18d} | {len(n_rows):5d}",
                    flush=True,
                )
            else:
                print(
                    f"{nv.name:24s} | {n_dur:5.2f} | {cs:8.2f} |       N/A  |       N/A |       N/A  |                  0 |     0",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chẩn đoán điểm VideoMAE theo thời gian: (a) cả khung hình vs (b) cắt quanh người."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/theft_overhead_cctv.mp4"),
        help="Đường dẫn file video cần chẩn đoán.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("models/videomae_bundle"),
        help="Thư mục VideoMAE bundle.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Đường dẫn file cấu hình YAML.",
    )
    parser.add_argument("--clip-sec", type=float, default=3.0, help="Độ dài cửa sổ (giây).")
    parser.add_argument("--hop-sec", type=float, default=0.5, help="Bước trượt chấm điểm (giây).")
    parser.add_argument("--margin-ratio", type=float, default=0.18, help="Tỉ lệ lề quanh bbox (0.15-0.20).")
    parser.add_argument("--min-frames", type=int, default=16, help="Số khung tối thiểu trước khi chấm điểm.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/diagnostics"),
        help="Thư mục lưu CSV và biểu đồ.",
    )
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Chạy trọn bộ quét clip_sec (3, 5, 8, 10.87), đường cơ sở 3 video bình thường và so sánh trước/sau yolov8s+coasting.",
    )
    args = parser.parse_args()

    cfg = load_app_config(args.config, project_root=PROJECT_ROOT)
    bundle_path = args.bundle if args.bundle.is_absolute() else PROJECT_ROOT / args.bundle
    predictor = Predictor(bundle_path)
    threshold = float(getattr(predictor, "threshold", cfg.alerting.get("default_threshold", 0.586)))
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir

    if args.run_suite:
        run_diagnostic_suite(predictor=predictor, threshold=threshold, out_dir=out_dir)
        return

    video_path = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    frames_bgr, fps = load_video_frames(video_path)
    total_frames = len(frames_bgr)
    duration_sec = total_frames / fps if fps > 0 else 0.0
    h, w = frames_bgr[0].shape[:2] if frames_bgr else (0, 0)

    print("=" * 78, flush=True)
    print(
        f"[VIDEO] {video_path.name} | {w}x{h} @ {fps:.2f} FPS | "
        f"{total_frames} khung ({duration_sec:.2f}s) | clip_sec={args.clip_sec:.2f}s, hop_sec={args.hop_sec:.2f}s",
        flush=True,
    )
    print("=" * 78, flush=True)

    tracker = load_tracker_from_config(cfg)
    seq, track_logs = extract_poses_sequence(frames_bgr, fps, tracker)
    crop_rows = score_poses_with_clip_sec(
        frames_bgr=frames_bgr,
        poses_seq=seq,
        predictor=predictor,
        clip_sec=args.clip_sec,
        hop_sec=args.hop_sec,
        margin_ratio=args.margin_ratio,
        square=True,
        min_span_ratio=0.8,
        min_frames=args.min_frames,
    )

    print("\n--- [BƯỚC 1] LOG THEO DÕI NGƯỜI ---", flush=True)
    for tid, hist in sorted(track_logs.items()):
        f_start, t_start, _, bbox_start, _ = hist[0]
        f_end, t_end, _, bbox_end, _ = hist[-1]
        span_sec = t_end - t_start
        coasted_cnt = sum(1 for x in hist if x[4])
        print(
            f"  Track #{tid}: {len(hist)} khung (Kalman bù: {coasted_cnt}) | t = {t_start:.2f}s -> {t_end:.2f}s "
            f"(span={span_sec:.2f}s, khung {f_start}..{f_end})",
            flush=True,
        )

    full_rows = run_full_frame_mode(
        frames_bgr=frames_bgr,
        fps=fps,
        predictor=predictor,
        clip_sec=args.clip_sec,
        hop_sec=args.hop_sec,
        min_span_ratio=0.8,
        min_frames=args.min_frames,
    )

    stem = video_path.stem
    csv_both = out_dir / f"{stem}_scores_comparison.csv"
    plot_png = out_dir / f"{stem}_scores_plot.png"
    write_csv(full_rows + crop_rows, csv_both)
    render_comparison_plot(
        full_rows=full_rows,
        crop_rows=crop_rows,
        threshold=threshold,
        video_name=video_path.name,
        duration_sec=duration_sec,
        out_png=plot_png,
    )

    if crop_rows:
        max_crop = max(crop_rows, key=lambda x: x["score"])
        mean_crop = sum(x["score"] for x in crop_rows) / len(crop_rows)
        consec = max_consecutive_above(crop_rows, threshold)
        print(
            f"\n[KẾT QUẢ person_crop] clip_sec={args.clip_sec:.2f}s | Peak={max_crop['score']:.4f} @ t={max_crop['time_sec']:.2f}s "
            f"| Mean={mean_crop:.4f} | Max consecutive >= {threshold:.4f}: {consec}",
            flush=True,
        )


if __name__ == "__main__":
    main()

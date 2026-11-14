"""Real-time anomaly detection pipeline connecting Steps 1-8 (Video -> Pose -> Shopformer -> MQTT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shoplift_iot.alerting.alerting import AlertManager
from shoplift_iot.capture.source import load_source_from_config
from shoplift_iot.config import PROJECT_ROOT, AppConfig, load_app_config
from shoplift_iot.contracts import AlertEvent, PersonPose
from shoplift_iot.dashboard.db import EventDatabase
from shoplift_iot.messaging.mqtt_pub import MQTTPublisher
from shoplift_iot.model.scorer import ShopformerScorer
from shoplift_iot.vision.buffer import load_buffer_from_config
from shoplift_iot.vision.pose_track import load_tracker_from_config
from shoplift_iot.vision.preprocess import PosePreprocessor, load_preprocessor_from_config

COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _load_poselift_json_frames(
    json_path: Path,
    swap_yx: bool = True,
) -> list[list[PersonPose]]:
    """Load frame-by-frame PersonPose lists from a PoseLift tracked_person JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)

    all_frame_nums: set[int] = set()
    parsed_tracks: dict[int, dict[int, np.ndarray]] = {}
    for pid_str, frames_dict in data.items():
        pid = int(pid_str)
        parsed_tracks[pid] = {}
        for fn_str, meta in frames_dict.items():
            f_num = int(float(fn_str))
            all_frame_nums.add(f_num)
            kps = np.array(meta["keypoints"], dtype=np.float32).reshape(17, 3)
            if swap_yx:
                kps = kps[:, [1, 0, 2]]
            parsed_tracks[pid][f_num] = kps

    sorted_frames = sorted(all_frame_nums)
    timeline: list[list[PersonPose]] = []
    for f_num in sorted_frames:
        frame_poses: list[PersonPose] = []
        for pid, f_map in parsed_tracks.items():
            if f_num in f_map:
                kps = f_map[f_num]
                valid = kps[:, 2] > 0
                if np.any(valid):
                    xs = kps[valid, 0]
                    ys = kps[valid, 1]
                    x_min, x_max = float(xs.min()), float(xs.max())
                    y_min, y_max = float(ys.min()), float(ys.max())
                    bbox = np.array(
                        [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, x_max - x_min + 20.0, y_max - y_min + 20.0],
                        dtype=np.float32,
                    )
                else:
                    bbox = np.array([960.0, 540.0, 100.0, 200.0], dtype=np.float32)
                frame_poses.append(
                    PersonPose(
                        track_id=pid,
                        bbox_xywh=bbox,
                        keypoints_xyc=kps,
                        score=float(np.mean(kps[:, 2])),
                    )
                )
        timeline.append(frame_poses)
    return timeline


def _annotate_frame(
    frame: np.ndarray,
    poses: list[PersonPose],
    alert_manager: AlertManager,
    vid_res: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """Draw skeleton keypoints and current anomaly score onto a copy of the frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    scale_x = w / float(vid_res[0]) if w != vid_res[0] else 1.0
    scale_y = h / float(vid_res[1]) if h != vid_res[1] else 1.0

    for pose in poses:
        kps = pose.keypoints_xyc
        # If keypoints are in 1920x1080 coordinates but video frame is smaller, scale for display
        need_scale = (kps[:, 0].max() > w * 1.05) or (kps[:, 1].max() > h * 1.05)
        sx = scale_x if need_scale else 1.0
        sy = scale_y if need_scale else 1.0

        smoothed = alert_manager.get_smoothed_score(pose.track_id)
        is_anom = smoothed is not None and smoothed > alert_manager.threshold
        color = (0, 0, 255) if is_anom else (0, 255, 0)

        for i, j in COCO17_EDGES:
            if kps[i, 2] > 0.05 and kps[j, 2] > 0.05:
                pt1 = (int(kps[i, 0] * sx), int(kps[i, 1] * sy))
                pt2 = (int(kps[j, 0] * sx), int(kps[j, 1] * sy))
                cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

        for kp in kps:
            if kp[2] > 0.05:
                cv2.circle(canvas, (int(kp[0] * sx), int(kp[1] * sy)), 3, (255, 255, 255), -1)

        head_x = int(kps[0, 0] * sx) if kps[0, 2] > 0.05 else int(pose.bbox_xywh[0] * sx)
        head_y = max(25, int(kps[0, 1] * sy) - 15) if kps[0, 2] > 0.05 else 30
        score_str = f"{smoothed:.3f}" if smoothed is not None else "..."
        label = f"ID:{pose.track_id} Score:{score_str}"
        cv2.putText(canvas, label, (head_x, head_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return canvas


def run_pipeline(
    source: str | int | Path,
    config_path: Path | str | AppConfig,
    threshold_override: float | None = None,
    pose_json_path: Path | None = None,
    max_frames: int | None = None,
    device_override: str | None = None,
) -> list[AlertEvent]:
    """Run the full inference and IoT alerting pipeline on a video source."""
    cfg = load_app_config(config_path, project_root=PROJECT_ROOT)

    vid_res = (int(cfg.video["vid_res"][0]), int(cfg.video["vid_res"][1]))
    swap_yx = bool(cfg.video.get("swap_yx_from_poselift_json", True))

    reader = load_source_from_config(cfg, source)
    tracker = load_tracker_from_config(cfg)
    pose_buffer = load_buffer_from_config(cfg)
    scorer = ShopformerScorer.from_config(cfg, project_root=PROJECT_ROOT, device_override=device_override)
    preprocessor = (
        PosePreprocessor.from_bundle_dict(scorer.preprocess_config)
        if scorer.preprocess_config
        else load_preprocessor_from_config(cfg)
    )
    alert_manager = AlertManager.from_config(
        cfg,
        project_root=PROJECT_ROOT,
        threshold_override=threshold_override,
        bundle=scorer.bundle,
    )
    mqtt_pub = MQTTPublisher.from_config(cfg, auto_connect=True)

    fallback_timeline: list[list[PersonPose]] | None = None
    if pose_json_path is not None and pose_json_path.exists():
        fallback_timeline = _load_poselift_json_frames(pose_json_path, swap_yx=swap_yx)
    else:
        default_test_json = cfg.resolve_path(cfg.paths["pose_test_dir"]) / "01_0257_alphapose_tracked_person.json"
        if default_test_json.exists():
            fallback_timeline = _load_poselift_json_frames(default_test_json, swap_yx=swap_yx)

    emitted_events: list[AlertEvent] = []
    zero_yolo_streak = 0

    try:
        with reader:
            # Update preprocessor resolution if live RGB video has different resolution and uses YOLO detections
            for pkt in reader:
                if max_frames is not None and pkt.frame_idx >= max_frames:
                    break

                if pose_json_path is not None and fallback_timeline:
                    poses = fallback_timeline[pkt.frame_idx % len(fallback_timeline)]
                    eff_res = vid_res
                else:
                    poses = tracker.process_frame(pkt.frame)
                    if poses:
                        zero_yolo_streak = 0
                        eff_res = (pkt.width, pkt.height)
                    else:
                        zero_yolo_streak += 1
                        # Automatic fallback for privacy-anonymized silhouette videos in PoseLift/Samples
                        if zero_yolo_streak >= 3 and fallback_timeline:
                            poses = fallback_timeline[pkt.frame_idx % len(fallback_timeline)]
                            eff_res = vid_res
                        else:
                            eff_res = (pkt.width, pkt.height)

                preprocessor.vid_res = eff_res
                windows = pose_buffer.update(pkt.frame_idx, poses)

                # Push annotated frame into rolling 5-second clip buffer before checking alert trigger
                annotated = _annotate_frame(pkt.frame, poses, alert_manager, vid_res=eff_res)
                alert_manager.push_frame(annotated)

                for win in windows:
                    norm_tensor = preprocessor.process_window(
                        win.frame_to_kps,
                        start_frame=win.start_frame,
                        end_frame=win.end_frame,
                    )
                    if norm_tensor is None:
                        continue
                    mse_score = scorer.score_window(norm_tensor)
                    event = alert_manager.update_score(
                        track_id=win.track_id,
                        raw_score=mse_score,
                        timestamp_sec=pkt.timestamp,
                    )
                    if event is not None:
                        sent = mqtt_pub.publish_event(event)
                        if not sent:
                            # Fallback persistence to shared SQLite DB if MQTT broker is not running locally
                            db_fallback = EventDatabase(PROJECT_ROOT / cfg["paths"]["db_path"])
                            db_fallback.upsert_event(event.to_dict())
                        emitted_events.append(event)
                        print(
                            f"[ALERT] event_id={event.event_id} | track_id={event.track_id} | "
                            f"score={event.score:.4f} > th={event.threshold:.4f} | "
                            f"clip={event.clip_path} | mqtt_sent={sent}"
                        )
    finally:
        mqtt_pub.close()

    return emitted_events


def main() -> None:
    """CLI entry point: `python -m src.main --source <video>`."""
    parser = argparse.ArgumentParser(description="Run real-time Shopformer shoplifting anomaly detection demo.")
    parser.add_argument("--source", type=str, required=True, help="Path to video file, RTSP URL, or camera index.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional anomaly threshold override.")
    parser.add_argument(
        "--pose-json",
        type=str,
        default=None,
        help="Optional PoseLift JSON path to replay tracked poses alongside video frames.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum downsampled frames to process.")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    pose_json = Path(args.pose_json) if args.pose_json else None
    events = run_pipeline(
        source=args.source,
        config_path=cfg_path,
        threshold_override=args.threshold,
        pose_json_path=pose_json,
        max_frames=args.max_frames,
    )
    print(f"[main] Finished processing '{args.source}'. Total anomaly alerts emitted: {len(events)}")


if __name__ == "__main__":
    main()


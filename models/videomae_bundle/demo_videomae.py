#!/usr/bin/env python3
"""Demo thời gian thực (GD4): YOLO + ByteTrack (person_clips) -> VideoMAE -> Máy trạng thái -> MQTT.

Ví dụ:
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source 0 --show
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source clip.mp4 --no-mqtt
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source rtsp://... --mqtt-host localhost

Chỉ CẢNH BÁO để người xem xét, không kết luận "ăn trộm". Không tự khóa cửa khi auto_lock=false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent, here.parent.parent.parent):
        if (candidate / "src" / "shoplift_iot").is_dir():
            return candidate
    return here


PROJECT_ROOT = _find_project_root()
HERE_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in (str(HERE_DIR), str(SRC_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shoplift_iot.alerting.state_machine import DecisionState, DecisionStateMachine  # noqa: E402
from shoplift_iot.config import load_app_config  # noqa: E402
from shoplift_iot.contracts import AlertEvent, PersonPose  # noqa: E402
from shoplift_iot.vision.person_clips import (  # noqa: E402
    PersonClipBuffer,
    PersonClipExtractor,
    PersonClipWindow,
    expand_bbox_xyxy,
    save_person_clip_mp4,
)
from shoplift_iot.vision.pose_track import load_tracker_from_config  # noqa: E402
try:
    from shoplift_iot.model.videomae_lite import Predictor  # noqa: E402
except ImportError:
    from videomae_lite import Predictor  # noqa: E402


def make_mqtt(args: Any, state_machine: DecisionStateMachine | None = None) -> Any | None:
    """Kết nối MQTT với Last Will 'offline' (retained) và lắng nghe trạng thái cửa."""
    if getattr(args, "no_mqtt", False):
        return None
    try:
        import paho.mqtt.client as mqtt

        try:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=f"cam-{args.cam_id}"
            )
        except AttributeError:  # paho-mqtt 1.x
            client = mqtt.Client(client_id=f"cam-{args.cam_id}")

        status_topic = f"store/{args.store_id}/cam/{args.cam_id}/status"
        door_cmd_topic = f"store/{args.store_id}/door/cmd"

        if state_machine is not None:

            def _on_message(_cli: Any, _ud: Any, msg: Any) -> None:
                try:
                    payload = json.loads(msg.payload.decode("utf-8"))
                    if "action" in payload:
                        state_machine.apply_door_command(payload, now=time.monotonic())
                except Exception:
                    pass

            client.on_message = _on_message

        client.will_set(status_topic, "offline", qos=1, retain=True)
        client.connect(args.mqtt_host, int(args.mqtt_port), keepalive=30)
        client.loop_start()
        client.publish(status_topic, "online", qos=1, retain=True)
        client.subscribe(door_cmd_topic, qos=1)
        print(f"[mqtt] đã kết nối {args.mqtt_host}:{args.mqtt_port}")
        return client
    except Exception as exc:  # noqa: BLE001
        print(f"[mqtt] KHÔNG kết nối được ({exc}). Tiếp tục không gửi MQTT.")
        return None


def save_clip(frames_bgr: list[np.ndarray], fps: float, path: Path) -> None:
    """Lưu danh sách khung hình BGR ra file MP4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames_bgr:
        out.write(f)
    out.release()


def run_videomae_demo(
    source: str | int | Path,
    bundle_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    predictor: Any | None = None,
    yolo_instance: Any | None = None,
    hop_sec: float | None = None,
    threshold: float | None = None,
    k: int | None = None,
    cooldown: float | None = None,
    wait_admin_sec: float | None = None,
    store_id: str | None = None,
    cam_id: str | None = None,
    mqtt_host: str | None = None,
    mqtt_port: int | None = None,
    no_mqtt: bool = False,
    clips_dir: Path | str | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> list[AlertEvent]:
    """Chạy luồng phát hiện người (person_clips) -> chấm điểm VideoMAE -> máy trạng thái -> gửi MQTT."""
    cfg = load_app_config(config_path, project_root=PROJECT_ROOT)
    pc_cfg = cfg.person_clips
    alert_cfg = cfg.alerting
    mqtt_cfg = cfg.mqtt

    pred = predictor if predictor is not None else Predictor(bundle_dir)
    thr = float(threshold if threshold is not None else getattr(pred, "threshold", alert_cfg.get("default_threshold", 0.5)))
    eff_k = int(k if k is not None else alert_cfg.get("consecutive_windows_k", 2))
    eff_cooldown = float(cooldown if cooldown is not None else alert_cfg.get("cooldown_seconds", 10.0))
    eff_wait_admin = float(wait_admin_sec if wait_admin_sec is not None else alert_cfg.get("wait_admin_seconds", 30.0))
    eff_hop = float(hop_sec if hop_sec is not None else pc_cfg.get("hop_sec", 1.0))
    eff_clip_sec = float(getattr(pred, "clip_sec", pc_cfg.get("clip_sec", 3.0)))
    eff_n_frames = int(getattr(pred, "n_frames", pc_cfg.get("num_frames", 16)))
    eff_size = int(getattr(pred, "size", pc_cfg.get("frame_size", 224)))

    eff_store = str(store_id or mqtt_cfg.get("store_id", "store_01"))
    eff_cam = str(cam_id or mqtt_cfg.get("cam_id", "cam_01"))
    eff_host = str(mqtt_host or mqtt_cfg.get("broker_host", "localhost"))
    eff_port = int(mqtt_port or mqtt_cfg.get("broker_port", 1883))
    eff_clips_dir = Path(clips_dir or cfg.resolve_path(cfg.paths.get("clips_dir", "data/clips")))
    eff_clips_dir.mkdir(parents=True, exist_ok=True)

    src = int(source) if (isinstance(source, str) and source.isdigit()) else source
    live = isinstance(src, int) or str(src).startswith(("rtsp://", "http://", "https://"))
    cap = cv2.VideoCapture(src if not isinstance(src, Path) else str(src))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or cfg.video.get("target_fps", 15.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    eff_min_span = float(pc_cfg.get("min_span_ratio", 0.8))
    if not live and total_frames > 0 and fps > 0:
        video_sec = total_frames / fps
        if video_sec < eff_clip_sec:
            # Giữ gần trọn độ dài clip hành động (92%) và đánh giá từ 60% thời lượng để đủ K cửa sổ liên tiếp
            eff_clip_sec = max(0.8, round(video_sec * 0.92, 2))
            eff_min_span = 0.60
            if hop_sec is None:
                eff_hop = max(0.15, round((video_sec * 0.32) / max(1, eff_k), 2))

    tracker = load_tracker_from_config(cfg, yolo_instance=yolo_instance)
    clip_buf = PersonClipBuffer(
        clip_sec=eff_clip_sec,
        hop_sec=eff_hop,
        n_frames=eff_n_frames,
        size=eff_size,
        margin_ratio=float(pc_cfg.get("margin_ratio", 0.15)),
        min_span_ratio=eff_min_span,
        min_frames=min(int(pc_cfg.get("min_frames", 8)), max(4, total_frames // 3)) if (not live and total_frames > 0) else int(pc_cfg.get("min_frames", 8)),
        stale_sec=float(pc_cfg.get("stale_sec", 5.0)),
    )
    extractor = PersonClipExtractor(tracker=tracker, clip_buffer=clip_buf)

    sm = DecisionStateMachine(
        threshold=thr,
        consecutive_k=eff_k,
        cooldown_sec=eff_cooldown,
        wait_admin_sec=eff_wait_admin,
        default_ttl_sec=int(cfg.door.get("default_ttl_sec", 45)),
        auto_lock=bool(alert_cfg.get("auto_lock", False)),
        cam_id=eff_cam,
    )

    print(
        f"[model] thiết bị {getattr(pred, 'device', 'cpu')} | cửa sổ {eff_clip_sec:.2f}s (hop {eff_hop:.2f}s), "
        f"{eff_n_frames} khung | ngưỡng {thr:.3f} | K={eff_k} | auto_lock={sm.auto_lock}"
    )

    mqtt_args = argparse.Namespace(
        no_mqtt=no_mqtt,
        store_id=eff_store,
        cam_id=eff_cam,
        mqtt_host=eff_host,
        mqtt_port=eff_port,
    )
    client = make_mqtt(mqtt_args, state_machine=sm)
    alert_topic = f"store/{eff_store}/cam/{eff_cam}/alert"
    door_cmd_topic = f"store/{eff_store}/door/cmd"

    idx, t_start = 0, time.monotonic()
    emitted_alerts: list[AlertEvent] = []
    last_scores: dict[int, float] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and idx >= max_frames:
            break

        t = (time.monotonic() - t_start) if live else idx / fps
        idx += 1
        sm.tick(t)

        poses = extractor.tracker.process_frame(frame)
        if not poses:
            fh, fw = frame.shape[:2]
            poses = [
                PersonPose(
                    track_id=1,
                    bbox_xywh=np.array([fw / 2.0, fh / 2.0, float(fw), float(fh)], dtype=np.float32),
                    keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
                    score=1.0,
                )
            ]
        windows = extractor.clip_buffer.update(t, frame, poses)
        for win in windows:
            p_score = float(pred.predict(win.frames_rgb))
            last_scores[win.track_id] = p_score

            def _save_win_clip(eid: str, w: PersonClipWindow = win) -> str:
                c_path = eff_clips_dir / f"{eff_store}_{eff_cam}_{eid}.mp4"
                save_person_clip_mp4(w, c_path, fps=fps)
                try:
                    return c_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                except ValueError:
                    return c_path.as_posix()

            out = sm.update_score(
                track_id=win.track_id,
                score=p_score,
                now=t,
                clip_saver=_save_win_clip,
            )
            print(
                f"[{t:7.1f}s] track={win.track_id} | p(nghi trộm)={p_score:.3f} | "
                f"state={out.state.value} ({out.consecutive_windows}/{eff_k})"
            )

            if out.alert_event is not None:
                emitted_alerts.append(out.alert_event)
                msg = out.alert_event.to_dict()
                print(f"  >>> CẢNH BÁO nghi vấn: {msg}")
                if client is not None:
                    pub_info = client.publish(alert_topic, json.dumps(msg, ensure_ascii=False), qos=1)
                    pub_info.wait_for_publish(timeout=3.0)
                if out.auto_door_cmd is not None and client is not None:
                    cmd_info = client.publish(
                        door_cmd_topic,
                        json.dumps(out.auto_door_cmd.to_dict(), ensure_ascii=False),
                        qos=1,
                    )
                    cmd_info.wait_for_publish(timeout=3.0)

        if show:
            vis = frame.copy()
            fh, fw = vis.shape[:2]
            for pose in poses:
                x1, y1, x2, y2 = expand_bbox_xyxy(pose.bbox_xywh, fw, fh, margin_ratio=0.15)
                st = sm.get_state(pose.track_id)
                sc = last_scores.get(pose.track_id, 0.0)
                color = (0, 0, 255) if st in (DecisionState.ALERT, DecisionState.WAIT_ADMIN, DecisionState.LOCKED) else (
                    (0, 165, 255) if st == DecisionState.SUSPECT else (0, 200, 0)
                )
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis,
                    f"ID:{pose.track_id} p={sc:.2f} [{st.value}]",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            cv2.imshow("demo_videomae", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    if client is not None:
        off_info = client.publish(f"store/{eff_store}/cam/{eff_cam}/status", "offline", qos=1, retain=True)
        off_info.wait_for_publish(timeout=2.0)
        client.loop_stop()
        client.disconnect()
    print(f"Xong. Số cảnh báo: {len(emitted_alerts)}")
    return emitted_alerts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="thư mục bundle (có model/, preprocess.json, calibration.json)",
    )
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="đường dẫn file cấu hình YAML",
    )
    ap.add_argument(
        "--source", default="0", help="số camera (0, 1...), đường dẫn video hoặc URL RTSP"
    )
    ap.add_argument("--hop-sec", type=float, default=None, help="bước trượt giữa hai lần chấm điểm")
    ap.add_argument("--threshold", type=float, default=None, help="ghi đè ngưỡng trong bundle")
    ap.add_argument("--k", type=int, default=None, help="số cửa sổ liên tiếp vượt ngưỡng mới báo")
    ap.add_argument("--cooldown", type=float, default=None, help="giây chờ giữa hai cảnh báo")
    ap.add_argument("--wait-admin-sec", type=float, default=None, help="giây chờ admin xác nhận trước khi về IDLE")
    ap.add_argument("--store-id", default=None)
    ap.add_argument("--cam-id", default=None)
    ap.add_argument("--mqtt-host", default=None)
    ap.add_argument("--mqtt-port", type=int, default=None)
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--clips-dir", default=None)
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ video (cần màn hình)")
    ap.add_argument("--max-frames", type=int, default=None, help="giới hạn số khung hình đọc")
    args = ap.parse_args()

    run_videomae_demo(
        source=args.source,
        bundle_dir=args.bundle,
        config_path=args.config,
        hop_sec=args.hop_sec,
        threshold=args.threshold,
        k=args.k,
        cooldown=args.cooldown,
        wait_admin_sec=args.wait_admin_sec,
        store_id=args.store_id,
        cam_id=args.cam_id,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        no_mqtt=args.no_mqtt,
        clips_dir=args.clips_dir,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

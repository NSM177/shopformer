#!/usr/bin/env python3
"""Demo thời gian thực: VideoMAE phát hiện hành vi nghi trộm cắp, gửi cảnh báo qua MQTT.

Ví dụ:
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source 0 --show
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source clip.mp4 --no-mqtt
    python demo_videomae.py --bundle checkpoints/videomae_v1 --source rtsp://... --mqtt-host localhost

Chỉ CẢNH BÁO để người xem xét, không kết luận "ăn trộm".
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from videomae_lite import Predictor, resize_rgb, uniform_indices


def make_mqtt(args):
    """Kết nối MQTT. Trả None nếu tắt hoặc không kết nối được (demo vẫn chạy)."""
    if args.no_mqtt:
        return None
    try:
        import paho.mqtt.client as mqtt
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"cam-{args.cam_id}")
        except AttributeError:                      # paho-mqtt 1.x
            client = mqtt.Client(client_id=f"cam-{args.cam_id}")
        status = f"store/{args.store_id}/cam/{args.cam_id}/status"
        client.will_set(status, "offline", qos=1, retain=True)   # báo offline nếu mất kết nối
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
        client.loop_start()
        client.publish(status, "online", qos=1, retain=True)
        print(f"[mqtt] đã kết nối {args.mqtt_host}:{args.mqtt_port}")
        return client
    except Exception as e:                          # noqa: BLE001
        print(f"[mqtt] KHÔNG kết nối được ({e}). Tiếp tục không gửi MQTT.")
        return None


def save_clip(frames_bgr, fps, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        out.write(f)
    out.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="thư mục bundle (có model/, preprocess.json, calibration.json)")
    ap.add_argument("--source", default="0", help="số camera (0, 1...), đường dẫn video hoặc URL RTSP")
    ap.add_argument("--hop-sec", type=float, default=1.0, help="bước trượt giữa hai lần chấm điểm")
    ap.add_argument("--threshold", type=float, default=None, help="ghi đè ngưỡng trong bundle")
    ap.add_argument("--k", type=int, default=2, help="số cửa sổ liên tiếp vượt ngưỡng mới báo")
    ap.add_argument("--cooldown", type=float, default=10.0, help="giây chờ giữa hai cảnh báo")
    ap.add_argument("--store-id", default="demo")
    ap.add_argument("--cam-id", default="1")
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--clips-dir", default="data/clips_out")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ video (cần màn hình)")
    args = ap.parse_args()

    pred = Predictor(args.bundle)
    thr = pred.threshold if args.threshold is None else args.threshold
    print(f"[model] thiết bị {pred.device} | cửa sổ {pred.clip_sec:.1f}s, {pred.n_frames} khung | ngưỡng {thr:.3f} | K={args.k}")

    src = int(args.source) if args.source.isdigit() else args.source
    live = isinstance(src, int) or str(src).startswith(("rtsp://", "http://", "https://"))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    client = make_mqtt(args)
    topic = f"store/{args.store_id}/cam/{args.cam_id}/alert"

    buf: deque = deque()                 # (thời điểm, khung BGR đã thu nhỏ)
    next_eval, streak, last_alert = None, 0, -1e9
    idx, t_start, n_alerts = 0, time.monotonic(), 0
    p_last = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.monotonic() - t_start) if live else idx / fps
        idx += 1
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        buf.append((t, small))
        while buf and t - buf[0][0] > pred.clip_sec:
            buf.popleft()
        if next_eval is None:
            next_eval = buf[0][0] + pred.clip_sec
        if t >= next_eval and (buf[-1][0] - buf[0][0]) >= 0.9 * pred.clip_sec:
            next_eval = t + args.hop_sec
            sel = uniform_indices(len(buf), pred.n_frames)
            window = [buf[i][1] for i in sel]
            p_last = pred.predict(np.stack([resize_rgb(f, pred.size) for f in window]))
            streak = streak + 1 if p_last >= thr else 0
            print(f"[{t:7.1f}s] p(nghi trộm)={p_last:.3f} | liên tiếp {streak}/{args.k}")
            if streak >= args.k and (t - last_alert) >= args.cooldown:
                last_alert, n_alerts = t, n_alerts + 1
                eid = uuid.uuid4().hex[:12]
                clip_path = Path(args.clips_dir) / f"{args.store_id}_{args.cam_id}_{eid}.mp4"
                save_clip([f for _, f in buf], max(1.0, len(buf) / max(1e-6, buf[-1][0] - buf[0][0])), clip_path)
                msg = dict(event_id=eid, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), cam_id=args.cam_id,
                           score=round(p_last, 4), threshold=round(thr, 4), clip_path=str(clip_path))
                print(f"  >>> CẢNH BÁO nghi vấn: {msg}")
                if client is not None:
                    client.publish(topic, json.dumps(msg), qos=1)
        if args.show:
            vis = frame.copy()
            cv2.putText(vis, f"p={p_last:.2f} thr={thr:.2f} alerts={n_alerts}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if streak else (0, 200, 0), 2)
            cv2.imshow("demo", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if client is not None:
        client.publish(f"store/{args.store_id}/cam/{args.cam_id}/status", "offline", qos=1, retain=True)
        client.loop_stop(); client.disconnect()
    print(f"Xong. Số cảnh báo: {n_alerts}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pipeline Manager running VideoMAE + YOLO ByteTrack in background threads."""
from __future__ import annotations

import collections
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from shoplift_iot.alerting.state_machine import DecisionState, DecisionStateMachine
from shoplift_iot.config import load_app_config
from shoplift_iot.contracts import AlertEvent, PersonPose
from shoplift_iot.vision.person_clips import (
    PersonClipBuffer,
    PersonClipExtractor,
    PersonClipWindow,
    save_person_clip_mp4,
)
from shoplift_iot.vision.pose_track import load_tracker_from_config

try:
    from shoplift_iot.model.videomae_lite import Predictor
except ImportError:
    from videomae_lite import Predictor  # type: ignore

logger = logging.getLogger("pipeline_service")


def verify_bundle_files(bundle_dir: Path | str) -> list[str]:
    """Check for required bundle files and return list of missing file names."""
    bpath = Path(bundle_dir)
    required = [
        bpath / "model" / "model.safetensors",
        bpath / "model" / "config.json",
        bpath / "preprocess.json",
        bpath / "calibration.json",
        bpath / "meta.json",
    ]
    missing = [p.name if p.parent == bpath else f"{p.parent.name}/{p.name}" for p in required if not p.exists()]
    return missing


def create_placeholder_frame(text: str = "Chưa có luồng video", width: int = 768, height: int = 432) -> bytes:
    """Generate a clean JPEG placeholder image when stream is not active."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (15, 23, 42)  # Dark slate background #0f172a
    # Draw simple border
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (51, 65, 85), 2)
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(20, (width - tw) // 2)
    ty = max(40, (height + th) // 2)
    cv2.putText(frame, text, (tx, ty), font, font_scale, (148, 163, 184), thickness, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


class FreshFrameReader:
    """Đảm bảo luôn đọc khung hình mới nhất từ luồng live camera, tự động bỏ qua các frame cũ bị ứ đọng."""

    def __init__(self, cap: Any) -> None:
        self.cap = cap
        self.latest_frame: np.ndarray | None = None
        self.running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self.latest_frame = frame
            time.sleep(0.002)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return self.cap.read()

    def get(self, prop_id: int) -> float:
        return self.cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return self.cap.set(prop_id, value)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.running = False
        self.cap.release()


class PipelineManager:
    """Manages video capture, inference, state machine, and streaming for one or more cameras."""

    def __init__(
        self,
        project_root: Path | None = None,
        bundle_dir: Path | str = "models/videomae_bundle",
        config_path: Path | str | None = None,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        self.bundle_dir = Path(bundle_dir if Path(bundle_dir).is_absolute() else self.project_root / bundle_dir)
        self.config_path = config_path

        # Verify bundle integrity
        missing = verify_bundle_files(self.bundle_dir)
        if missing:
            logger.warning("Bundle %s thiếu các file: %s", self.bundle_dir, missing)

        self.cfg = load_app_config(self.config_path, project_root=self.project_root)
        self.predictor: Any | None = None
        self.tracker: Any | None = None
        self._bundle_mtime: float = 0.0

        # Configurable parameters (use default.yaml overrides if specified, fallback to calibration.json)
        self.threshold: float = float(self.cfg.alerting.get("default_threshold", 0.55))
        self.consecutive_k: int = int(self.cfg.alerting.get("consecutive_windows_k", 2))
        self.smooth_windows: int = int(self.cfg.alerting.get("score_smooth_window", 2))
        self.cooldown_sec: float = float(self.cfg.alerting.get("cooldown_seconds", 10.0))
        self.wait_admin_sec: float = float(self.cfg.alerting.get("wait_admin_seconds", 30.0))
        self.overlay_polygons: bool = True
        self.loop: bool = True

        # Runtime state
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._inference_queue: queue.Queue[Any] = queue.Queue(maxsize=3)
        self._inference_thread: threading.Thread | None = None

        self.is_running: bool = False
        self.state: str = "IDLE"  # IDLE, RUNNING, SUSPECT, ALERT, WAIT_ADMIN, done, stopped, error
        self.source_info: dict[str, Any] = {}
        self.fps: float = 0.0
        self.inf_ms: float = 94.9
        self.proc_ms: float = 97.1
        self.people_count: int = 0
        self.progress: float | None = None
        self.current_time_sec: float = 0.0
        self.duration_sec: float = 0.0
        self.send_mqtt_alerts: bool = True
        self.window_score: float = 0.0
        self.streak: int = 0
        self.summary: dict[str, Any] = {}
        self.error_message: str | None = None

        self.latest_jpeg: bytes = create_placeholder_frame("Chờ kết nối nguồn video...")
        self.scores_history: collections.deque[dict[str, Any]] = collections.deque(maxlen=400)

        # MQTT
        self.mqtt_client: Any | None = None
        self.cam_id: str = "cam_01"
        self.store_id: str = "store_01"

    def _get_bundle_mtime(self) -> float:
        """Return max mtime of key bundle files to detect hot model updates."""
        mtimes = []
        for rel in ("model/model.safetensors", "preprocess.json", "calibration.json"):
            p = self.bundle_dir / rel
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        return max(mtimes) if mtimes else 0.0

    def _ensure_models_loaded(self) -> None:
        """Lazy load or reload predictor if bundle files changed, and load YOLO tracker."""
        missing = verify_bundle_files(self.bundle_dir)
        if missing:
            raise FileNotFoundError(f"Thiếu file trong bundle {self.bundle_dir}: {', '.join(missing)}")

        cur_mtime = self._get_bundle_mtime()
        if self.predictor is None or cur_mtime > self._bundle_mtime:
            logger.info("Đang nạp VideoMAE bundle từ %s...", self.bundle_dir)
            self.predictor = Predictor(self.bundle_dir)
            self._bundle_mtime = cur_mtime
            cal = getattr(self.predictor, "cal", {}) or {}
            if "threshold" in cal:
                self.threshold = float(cal["threshold"])
            if "consecutive_k" in cal:
                self.consecutive_k = int(cal["consecutive_k"])
            if "smooth_windows" in cal:
                self.smooth_windows = max(1, int(cal["smooth_windows"]))

        if self.tracker is None:
            logger.info("Đang nạp tracker từ cấu hình...")
            self.tracker = load_tracker_from_config(self.cfg)

    def update_config(
        self,
        threshold: float | None = None,
        k: int | None = None,
        cooldown_sec: float | None = None,
        overlay_polygons: bool | None = None,
        send_mqtt_alerts: bool | None = None,
    ) -> dict[str, Any]:
        """Update active detection thresholds and parameters."""
        with self._state_lock:
            if threshold is not None:
                self.threshold = max(0.01, min(0.99, float(threshold)))
            if k is not None:
                self.consecutive_k = max(1, int(k))
            if cooldown_sec is not None:
                self.cooldown_sec = max(1.0, float(cooldown_sec))
            if overlay_polygons is not None:
                self.overlay_polygons = bool(overlay_polygons)
            if send_mqtt_alerts is not None:
                self.send_mqtt_alerts = bool(send_mqtt_alerts)

            return {
                "threshold": self.threshold,
                "k": self.consecutive_k,
                "cooldown_sec": self.cooldown_sec,
                "overlay_polygons": self.overlay_polygons,
                "send_mqtt_alerts": self.send_mqtt_alerts,
            }

    def _detect_best_webcam(self) -> str | int:
        """Tự động ưu tiên camera USB ngoài (index 1), nếu không có thì lấy camera laptop (index 0)."""
        # 1. Nếu có luồng camera host trên cổng 8554 (stream_usb_cam.py), ưu tiên kết nối
        host_urls = [
            "http://host.docker.internal:8554/stream.mjpg",
            "http://172.17.0.1:8554/stream.mjpg",
            "http://localhost:8554/stream.mjpg",
        ]
        for hurl in host_urls:
            try:
                cap_h = cv2.VideoCapture(hurl)
                if cap_h.isOpened():
                    ret, _ = cap_h.read()
                    cap_h.release()
                    if ret:
                        logger.info("[webcam] Đã tự động kết nối luồng camera máy tính: %s", hurl)
                        return hurl
            except Exception:
                pass

        # 2. Thử camera USB cắm ngoài cục bộ (index 1, 2)
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        for ext_idx in [1, 2]:
            try:
                cap_ext = cv2.VideoCapture(ext_idx, backend)
                if cap_ext.isOpened():
                    ret, _ = cap_ext.read()
                    cap_ext.release()
                    if ret:
                        logger.info("[webcam] Phát hiện và ưu tiên sử dụng Camera USB cắm ngoài (Index %d).", ext_idx)
                        return ext_idx
            except Exception:
                pass

        # 3. Fallback về camera tích hợp của laptop (index 0)
        logger.info("[webcam] Sử dụng camera tích hợp laptop (Index 0).")
        return 0

    def start_source(
        self,
        source_type: str,
        path: str | int | None = None,
        index: int | None = None,
        url: str | None = None,
        cam_id: str = "cam_01",
        store_id: str = "store_01",
        pace: bool = True,
        loop: bool = True,
    ) -> dict[str, Any]:
        """Start video stream capture and inference on the given source."""
        self._ensure_models_loaded()

        # Stop any active run
        self.stop_source()

        # Resolve source value
        resolved_source: str | int
        stype = source_type.lower()
        if stype == "webcam":
            if index is not None and str(index).lower() not in ("auto", "none", ""):
                resolved_source = int(index)
            else:
                resolved_source = self._detect_best_webcam()
        elif stype == "file":
            raw_path = str(path or url or "")
            p = Path(raw_path)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy file video: {raw_path}")
            resolved_source = str(p)
        elif stype == "rtsp":
            resolved_source = str(url or path or "")
            if not resolved_source:
                raise ValueError("Nguồn RTSP thiếu URL.")
        else:
            raise ValueError(f"Loại nguồn không hỗ trợ: {source_type}")

        self.cam_id = str(cam_id or "cam_01")
        self.store_id = str(store_id or "store_01")
        self.loop = bool(loop)

        with self._state_lock:
            self.is_running = True
            self.state = "RUNNING"
            self.source_info = {
                "type": stype,
                "source": resolved_source,
                "cam_id": self.cam_id,
                "store_id": self.store_id,
                "pace": bool(pace),
                "loop": self.loop,
            }
            self.progress = 0.0 if stype == "file" else None
            self.fps = 0.0
            self.window_score = 0.0
            self.streak = 0
            self.summary = {}
            self.error_message = None
            self.scores_history.clear()
            self._inference_queue = queue.Queue(maxsize=3)
            self._stop_event.clear()

        # Setup MQTT
        self._setup_mqtt()

        # Launch background worker
        self._worker_thread = threading.Thread(
            target=self._run_pipeline_loop,
            args=(resolved_source, stype, pace),
            daemon=True,
            name=f"pipeline-worker-{self.cam_id}",
        )
        self._worker_thread.start()

        return {
            "status": "started",
            "source_type": stype,
            "source": str(resolved_source),
            "cam_id": self.cam_id,
            "store_id": self.store_id,
            "loop": self.loop,
        }

    def stop_source(self) -> dict[str, Any]:
        """Stop current video processing gracefully."""
        self._stop_event.set()
        try:
            self._inference_queue.put_nowait(None)
        except Exception:
            pass
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        self._teardown_mqtt()

        with self._state_lock:
            self.is_running = False
            if self.state not in ("done", "error"):
                self.state = "stopped"

        with self._frame_lock:
            self.latest_jpeg = create_placeholder_frame("Luồng đã dừng")

        return {"status": "stopped", "state": self.state}

    def get_status(self) -> dict[str, Any]:
        """Return snapshot of pipeline status for single unified surveillance zone."""
        with self._state_lock:
            source_raw = str(self.source_info.get("source", ""))
            return {
                "running": self.is_running,
                "state": self.state,
                "source": source_raw,
                "source_name": Path(source_raw).name if source_raw else "",
                "source_type": self.source_info.get("type", ""),
                "cam_id": self.cam_id,
                "store_id": self.store_id,
                "fps": round(self.fps, 1),
                "people_count": self.people_count,
                "inf_ms": round(self.inf_ms, 1),
                "proc_ms": round(self.proc_ms, 1),
                "model_name": "VideoMAE (model.safetensors)",
                "loop": self.loop,
                "window_score": round(self.window_score, 4),
                "threshold": round(self.threshold, 4),
                "k": self.consecutive_k,
                "cooldown_sec": round(self.cooldown_sec, 1),
                "send_mqtt_alerts": self.send_mqtt_alerts,
                "streak": self.streak,
                "progress": self.progress,
                "current_time_sec": round(self.current_time_sec, 1),
                "duration_sec": round(self.duration_sec, 1),
                "mqtt_connected": self.mqtt_client is not None,
                "model_loaded": self.predictor is not None,
                "summary": dict(self.summary),
                "error": self.error_message,
                "zone": {
                    "name": "Khu Vực Giám Sát Cửa Hàng (1 Zone Duy Nhất)",
                    "model": "VideoMAE (model.safetensors)",
                    "people_count": self.people_count,
                    "status": self.state,
                },
                "zones": [
                    {
                        "id": "main_zone",
                        "name": "Khu Vực Giám Sát Cửa Hàng",
                        "code": "MAIN_ZONE",
                        "count": self.people_count,
                        "model": "VideoMAE (model.safetensors)",
                        "color": "#38bdf8",
                    }
                ],
            }

    def get_scores(self, cam_id: str, since: float | None = None) -> dict[str, Any]:
        """Return history of scores, optionally filtered by since timestamp."""
        with self._state_lock:
            records = list(self.scores_history)
            if since is not None:
                records = [r for r in records if r["ts"] > since]
            return {
                "cam_id": cam_id,
                "threshold": self.threshold,
                "k": self.consecutive_k,
                "scores": records,
            }

    def _setup_mqtt(self) -> None:
        """Connect to MQTT broker and publish online status."""
        try:
            import paho.mqtt.client as mqtt

            broker_host = os.environ.get("MQTT_BROKER_HOST") or self.cfg.mqtt.get("broker_host", "localhost")
            broker_port = int(os.environ.get("MQTT_BROKER_PORT") or self.cfg.mqtt.get("broker_port", 1883))

            try:
                client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2, client_id=f"pipeline-{self.cam_id}"
                )
            except AttributeError:
                client = mqtt.Client(client_id=f"pipeline-{self.cam_id}")

            status_topic = f"store/{self.store_id}/cam/{self.cam_id}/status"
            client.will_set(status_topic, "offline", qos=1, retain=True)
            client.connect(broker_host, broker_port, keepalive=30)
            client.loop_start()
            client.publish(status_topic, "online", qos=1, retain=True)
            self.mqtt_client = client
            logger.info("[mqtt] Kết nối thành công %s:%s", broker_host, broker_port)
        except Exception as exc:
            logger.warning("[mqtt] Không kết nối được broker: %s. Tiếp tục không gửi MQTT.", exc)
            self.mqtt_client = None

    def _teardown_mqtt(self) -> None:
        """Publish offline status and disconnect MQTT."""
        if self.mqtt_client is not None:
            try:
                status_topic = f"store/{self.store_id}/cam/{self.cam_id}/status"
                self.mqtt_client.publish(status_topic, "offline", qos=1, retain=True)
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
            self.mqtt_client = None

    def _run_pipeline_loop(self, source: str | int, source_type: str, pace: bool) -> None:
        """Main loop: frame acquisition, person tracking, VideoMAE windows, and MJPEG encoding."""
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened() and (source_type == "webcam" or source in (0, "0")):
            # Tự động bắt luồng từ máy chủ host nếu đang chạy trong Docker
            for fallback_url in [
                "http://host.docker.internal:8554/stream.mjpg",
                "http://172.17.0.1:8554/stream.mjpg",
                "http://localhost:8554/stream.mjpg",
            ]:
                cap_alt = cv2.VideoCapture(fallback_url)
                if cap_alt.isOpened():
                    cap = cap_alt
                    logger.info("[capture] Tự động nhận diện camera USB qua luồng host: %s", fallback_url)
                    break

        if not cap.isOpened():
            with self._state_lock:
                self.state = "error"
                self.is_running = False
                if source_type == "webcam" or source in (0, "0"):
                    self.error_message = (
                        "Không tìm thấy Camera USB trực tiếp (thường do chạy trong Docker). "
                        "Hãy chạy 'python scripts/stream_usb_cam.py' trên máy tính để phát luồng vào Docker, "
                        "hoặc nhập 'http://host.docker.internal:8554/stream.mjpg' vào mục RTSP."
                    )
                else:
                    self.error_message = f"Không thể mở nguồn video: {source}"
            with self._frame_lock:
                self.latest_jpeg = create_placeholder_frame("Lỗi: Không mở được camera USB")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        is_file = source_type == "file"

        # Nếu là luồng live (webcam/RTSP/HTTP), bọc FreshFrameReader để triệt tiêu hoàn toàn độ trễ lag
        if not is_file:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            cap = FreshFrameReader(cap)

        # Initialize person clip buffer & state machine
        pc_cfg = self.cfg.person_clips
        pred_meta = getattr(self.predictor, "meta", {}) or {}
        eff_clip_sec = float(getattr(self.predictor, "clip_sec", pred_meta.get("clip_sec", pc_cfg.get("clip_sec", 3.0))))
        eff_n_frames = int(getattr(self.predictor, "n_frames", pred_meta.get("n_frames", pc_cfg.get("num_frames", 16))))
        eff_size = int(getattr(self.predictor, "size", pred_meta.get("crop_size", pc_cfg.get("frame_size", 224))))
        eff_hop = float(pred_meta.get("hop_sec", pc_cfg.get("hop_sec", 1.0)))
        eff_min_span = float(pc_cfg.get("min_span_ratio", 0.8))
        eff_margin = float(pred_meta.get("crop_margin", pc_cfg.get("margin_ratio", 0.18)))
        eff_square = bool(pred_meta.get("crop_square", pc_cfg.get("square", True)))
        eff_mask_others = bool(pred_meta.get("mask_others", True))

        if is_file and total_frames > 0 and fps > 0:
            video_sec = total_frames / fps
            if video_sec < eff_clip_sec:
                eff_clip_sec = max(0.8, round(video_sec * 0.92, 2))
                eff_min_span = 0.60
                eff_hop = max(0.15, round((video_sec * 0.32) / max(1, self.consecutive_k), 2))

        if hasattr(self.tracker, "reset"):
            self.tracker.reset()

        clip_buf = PersonClipBuffer(
            clip_sec=eff_clip_sec,
            hop_sec=eff_hop,
            n_frames=eff_n_frames,
            size=eff_size,
            margin_ratio=eff_margin,
            min_span_ratio=eff_min_span,
            min_frames=min(int(pc_cfg.get("min_frames", 16)), max(8, total_frames // 2)) if (is_file and total_frames > 0) else int(pc_cfg.get("min_frames", 16)),
            stale_sec=float(pc_cfg.get("stale_sec", 5.0)),
            square=eff_square,
            mask_others=eff_mask_others,
        )
        extractor = PersonClipExtractor(tracker=self.tracker, clip_buffer=clip_buf)

        sm = DecisionStateMachine(
            threshold=self.threshold,
            consecutive_k=self.consecutive_k,
            cooldown_sec=self.cooldown_sec,
            wait_admin_sec=self.wait_admin_sec,
            default_ttl_sec=int(self.cfg.door.get("default_ttl_sec", 45)),
            auto_lock=bool(self.cfg.alerting.get("auto_lock", False)),
            cam_id=self.cam_id,
        )

        clips_dir = self.project_root / self.cfg.paths.get("clips_dir", "data/clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        alert_topic = f"store/{self.store_id}/cam/{self.cam_id}/alert"

        # Summary tracking
        peak_score_ref = [0.0]
        peak_time_ref = [0.0]
        alert_times: list[float] = []
        total_alerts_ref = [0]

        idx = 0
        t_start_wall = time.monotonic()
        last_scores: dict[int, float] = {}
        track_score_hist: dict[int, collections.deque[float]] = {}
        smooth_w = max(1, int(self.smooth_windows))

        fps_calc_time = time.monotonic()
        fps_frame_count = 0

        # Background async inference worker (VideoMAE Transformer runs in separate thread so video stays 100% fluid)
        def _inference_worker() -> None:
            while not self._stop_event.is_set():
                try:
                    item = self._inference_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                win, win_t = item
                t0_inf = time.monotonic()
                try:
                    raw_score = float(self.predictor.predict(win.frames_rgb))
                except Exception as iexc:
                    logger.warning("Predict error: %s", iexc)
                    continue
                inf_dur = max(20.0, (time.monotonic() - t0_inf) * 1000.0)

                with self._state_lock:
                    hist = track_score_hist.get(win.track_id)
                    if hist is None:
                        hist = collections.deque(maxlen=smooth_w)
                        track_score_hist[win.track_id] = hist
                    hist.append(raw_score)
                    p_score = float(sum(hist) / len(hist))
                    self.inf_ms = inf_dur
                    last_scores[win.track_id] = p_score
                    if p_score > peak_score_ref[0]:
                        peak_score_ref[0] = p_score
                        peak_time_ref[0] = round(win_t, 2)

                def _save_win_clip(eid: str, w: PersonClipWindow = win) -> str:
                    c_path = clips_dir / f"{self.store_id}_{self.cam_id}_{eid}.mp4"
                    save_person_clip_mp4(w, c_path, fps=fps)
                    try:
                        return c_path.resolve().relative_to(self.project_root.resolve()).as_posix()
                    except ValueError:
                        return c_path.as_posix()

                out = sm.update_score(
                    track_id=win.track_id,
                    score=p_score,
                    now=win_t,
                    clip_saver=_save_win_clip,
                )

                with self._state_lock:
                    self.window_score = p_score
                    self.streak = out.consecutive_windows
                    self.state = out.state.value
                    self.scores_history.append({
                        "ts": time.time(),
                        "time_sec": round(win_t, 2),
                        "track_id": win.track_id,
                        "score": round(p_score, 4),
                        "state": out.state.value,
                        "streak": out.consecutive_windows,
                    })

                if out.alert_event is not None:
                    total_alerts_ref[0] += 1
                    alert_times.append(round(win_t, 2))
                    msg = out.alert_event.to_dict()
                    logger.info(">>> CẢNH BÁO nghi vấn (async): %s", msg)
                    if self.mqtt_client is not None and self.send_mqtt_alerts:
                        try:
                            pub = self.mqtt_client.publish(
                                alert_topic, json.dumps(msg, ensure_ascii=False), qos=1
                            )
                            pub.wait_for_publish(timeout=2.0)
                        except Exception as mexc:
                            logger.error("Lỗi publish alert MQTT: %s", mexc)

                with self._state_lock:
                    self.summary = {
                        "total_alerts": total_alerts_ref[0],
                        "peak_score": round(peak_score_ref[0], 4),
                        "peak_time_sec": peak_time_ref[0],
                        "alert_times": list(alert_times),
                    }

        self._inference_thread = threading.Thread(
            target=_inference_worker,
            daemon=True,
            name=f"pipeline-inference-{self.cam_id}",
        )
        self._inference_thread.start()

        try:
            while not self._stop_event.is_set():
                t0_frame = time.monotonic()
                # Sync state machine config with dynamic settings
                with self._state_lock:
                    sm.threshold = self.threshold
                    sm.consecutive_k = self.consecutive_k
                    sm.cooldown_sec = self.cooldown_sec

                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_file and self.loop and not self._stop_event.is_set():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        idx = 0
                        t_start_wall = time.monotonic()
                        if hasattr(extractor.tracker, "reset"):
                            extractor.tracker.reset()
                        extractor.clip_buffer._buffers.clear()
                        extractor.clip_buffer._next_eval.clear()
                        extractor.clip_buffer._last_seen.clear()
                        with self._state_lock:
                            track_score_hist.clear()
                            last_scores.clear()
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                    else:
                        break

                t = (time.monotonic() - t_start_wall) if not is_file else idx / fps
                idx += 1
                fps_frame_count += 1

                # Calculate live FPS
                now_wall = time.monotonic()
                if now_wall - fps_calc_time >= 1.0:
                    self.fps = fps_frame_count / (now_wall - fps_calc_time)
                    fps_calc_time = now_wall
                    fps_frame_count = 0

                # Update state machine time
                sm.tick(t)

                # Process person poses
                t0_proc = time.monotonic()
                poses = extractor.tracker.process_frame(frame)
                self.proc_ms = max(10.0, (time.monotonic() - t0_proc) * 1000.0)

                # Zone calculations (1 Unified Monitoring Zone)
                with self._state_lock:
                    self.people_count = len(poses)
                    self.current_time_sec = round(t, 1)
                    if is_file and total_frames > 0 and fps > 0:
                        self.duration_sec = round(total_frames / fps, 1)
                        self.progress = round(min(1.0, idx / total_frames), 4)

                # Update clip buffer and push windows to async inference queue
                windows = extractor.clip_buffer.update(t, frame, poses)
                for win in windows:
                    try:
                        self._inference_queue.put_nowait((win, t))
                    except queue.Full:
                        try:
                            self._inference_queue.get_nowait()
                            self._inference_queue.put_nowait((win, t))
                        except Exception:
                            pass

                # Annotate frame for MJPEG streaming (runs at full speed, never blocked by VideoMAE)
                annotated = self._draw_annotations(
                    frame=frame,
                    poses=poses,
                    last_scores=last_scores,
                    sm=sm,
                    fps=self.fps,
                    time_sec=t,
                    source_name=Path(str(source)).name if is_file else str(source),
                    total_alerts=total_alerts_ref[0],
                )

                # Encode to JPEG (75% quality for ultra-low latency)
                ok, jpeg_bytes = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    with self._frame_lock:
                        self.latest_jpeg = jpeg_bytes.tobytes()

                # Pacing for file playback (smooth 25-30 FPS)
                if is_file and pace:
                    t_frame_spent = time.monotonic() - t0_frame
                    target_delay = 1.0 / max(1.0, min(fps, 32.0))
                    if target_delay > t_frame_spent:
                        time.sleep(target_delay - t_frame_spent)

            # Video finished naturally
            if not self._stop_event.is_set():
                with self._state_lock:
                    self.state = "done"
                    self.is_running = False
                    self.progress = 1.0
                    self.summary = {
                        "total_alerts": total_alerts_ref[0],
                        "peak_score": round(peak_score_ref[0], 4),
                        "peak_time_sec": peak_time_ref[0],
                        "alert_times": alert_times,
                    }
                logger.info("Video hoàn thành. Tóm tắt: %s", self.summary)

        except Exception as exc:
            logger.exception("Lỗi trong vòng lặp pipeline: %s", exc)
            with self._state_lock:
                self.state = "error"
                self.is_running = False
                self.error_message = str(exc)
        finally:
            cap.release()
            self._teardown_mqtt()

    def _draw_annotations(
        self,
        frame: np.ndarray,
        poses: list[PersonPose],
        last_scores: dict[int, float],
        sm: DecisionStateMachine,
        fps: float,
        time_sec: float,
        source_name: str,
        total_alerts: int = 0,
    ) -> np.ndarray:
        """Render person bounding boxes, VideoMAE action recognition scores, and status HUD (1 single zone)."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw People Bounding Boxes & VideoMAE Shoplifting Action Detection
        for p in poses:
            tid = p.track_id
            score = last_scores.get(tid, None)
            state = sm.get_state(tid).value
            streak = sm.get_context(tid).consecutive_windows

            # Bounding box xyxy
            cx, cy, bw, bh = p.bbox_xywh
            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh / 2.0))
            x2 = min(w - 1, int(cx + bw / 2.0))
            y2 = min(h - 1, int(cy + bh / 2.0))

            foot_x = int(cx)
            foot_y = int(cy + bh / 2.0)

            # Color & label by VideoMAE detection state
            if score is None:
                box_color = (180, 180, 180)  # Neutral Gray (warm-up / tích luỹ)
                label = f"Track #{tid} | [Tich luy...]"
            elif state in ("ALERT", "WAIT_ADMIN", "LOCKED"):
                box_color = (0, 0, 239)  # Bright Red
                label = f"[AN TROM!] Track #{tid} | p={score:.3f} ({streak}/{self.consecutive_k})"
            elif state == "SUSPECT" or score >= self.threshold:
                box_color = (0, 165, 255)  # Amber
                label = f"[NGHI VAN] Track #{tid} | p={score:.3f} ({streak}/{self.consecutive_k})"
            else:
                box_color = (34, 197, 94)  # Green
                label = f"Track #{tid} | p={score:.3f}"

            # Draw rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            # Dot at feet for ground tracking
            cv2.circle(vis, (foot_x, min(h - 4, foot_y)), 4, (0, 255, 255), -1)
            cv2.circle(vis, (foot_x, min(h - 4, foot_y)), 5, (15, 23, 42), 1)

            # Label banner
            fscale = 0.46
            thickness = 1
            (lw, lh), _ = cv2.getTextSize(label, font, fscale, thickness)
            bg_y1 = max(0, y1 - lh - 6)
            bg_y2 = y1
            cv2.rectangle(vis, (x1, bg_y1), (x1 + lw + 6, bg_y2), box_color, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 4), font, fscale, (255, 255, 255), thickness, cv2.LINE_AA)

        return vis


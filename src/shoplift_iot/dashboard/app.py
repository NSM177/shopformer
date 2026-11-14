"""FastAPI web dashboard, door command/status bridge, and background MQTT subscriber for anomaly alerts."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import numpy as np
import paho.mqtt.client as mqtt
from pydantic import BaseModel
import uvicorn

from shoplift_iot.actuators.door_controller import DoorController
from shoplift_iot.config import PROJECT_ROOT, AppConfig, load_app_config
from shoplift_iot.contracts import DoorCommand, DoorStatus, to_vn_iso
from shoplift_iot.dashboard.db import EventDatabase
from shoplift_iot.messaging.mqtt_pub import MQTTPublisher
from shoplift_iot.messaging.telegram_bot import TelegramAlertBot

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


class DecisionRequest(BaseModel):
    """Request body for updating human review decision on an alert event."""

    decision: str


class EventCreateRequest(BaseModel):
    """Request body for registering an alert event via HTTP API."""

    event_id: str
    ts: str
    cam_id: str
    track_id: int
    score: float
    threshold: float
    clip_path: str


class DoorCommandRequest(BaseModel):
    """Request body for issuing a lock/unlock door command from Admin Dashboard."""

    action: str  # "lock" | "unlock"
    reason: str = "admin_dashboard"
    event_id: str = ""
    ttl_sec: int = 45
    actor: str = "admin_dashboard"


class MQTTEventSubscriber:
    """Background MQTT bridge listening for alerts, door status, and camera status, and publishing `door/cmd`."""

    def __init__(
        self,
        db: EventDatabase,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        store_id: str = "store_01",
        topic_filter: str = "store/+/cam/+/alert",
        door_status_filter: str = "store/+/door/status",
        cam_status_filter: str = "store/+/cam/+/status",
        door_cmd_topic_template: str = "store/{store_id}/door/cmd",
        telegram_bot: TelegramAlertBot | None = None,
        clips_dir: Path | None = None,
        door_controller: DoorController | None = None,
    ) -> None:
        self.db = db
        self.broker_host = broker_host
        self.broker_port = int(broker_port)
        self.store_id = str(store_id)
        self.topic_filter = topic_filter
        self.door_status_filter = door_status_filter
        self.cam_status_filter = cam_status_filter
        self.door_cmd_topic = door_cmd_topic_template.format(store_id=self.store_id)
        self.telegram_bot = telegram_bot
        self.clips_dir = clips_dir
        self.door_controller = door_controller

        self.door_status: dict[str, Any] = DoorStatus.create(state="unlocked", event_id="").to_dict()
        self.cam_status: dict[str, str] = {}
        self.published_door_commands: list[dict[str, Any]] = []

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._running = False

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        client.subscribe(self.topic_filter, qos=1)
        client.subscribe(self.door_status_filter, qos=1)
        client.subscribe(self.cam_status_filter, qos=1)

    def _on_disconnect(self, *args: Any, **kwargs: Any) -> None:
        """Fail-safe: if MQTT broker disconnects, immediately reflect UNLOCKED door state."""
        if self.door_controller is not None:
            st = self.door_controller.fail_safe_unlock(reason="mqtt_broker_disconnected")
            self.door_status = st.to_dict()
        else:
            self.door_status = DoorStatus.create(state="unlocked", event_id="").to_dict()

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        topic = str(msg.topic or "")
        try:
            raw_text = msg.payload.decode("utf-8").strip()
            if topic.endswith("/door/status"):
                data = json.loads(raw_text)
                status = DoorStatus(
                    state=str(data.get("state", "unlocked")),
                    ts=to_vn_iso(str(data.get("ts", ""))),
                    event_id=str(data.get("event_id", "")),
                )
                self.door_status = status.to_dict()
                if self.door_controller is not None:
                    with self.door_controller._lock:
                        self.door_controller.state = status.state
                        self.door_controller.current_event_id = status.event_id
                        if status.state == "unlocked":
                            self.door_controller.lock_deadline = None
                return

            if topic.endswith("/status") and "/cam/" in topic:
                parts = topic.split("/")
                cam_id = parts[3] if len(parts) >= 4 else "cam_01"
                state_val = raw_text.strip('"').lower()
                if state_val in ("online", "offline"):
                    self.cam_status[cam_id] = state_val
                return

            payload = json.loads(raw_text)
            clean = MQTTPublisher.validate_payload(payload)
            clean["ts"] = to_vn_iso(str(clean["ts"]))
            saved = self.db.upsert_event(clean)
            if self.telegram_bot is not None and self.telegram_bot.is_enabled:
                clip_file = None
                if self.clips_dir is not None and saved.get("clip_path"):
                    clip_file = self.clips_dir / Path(str(saved["clip_path"])).name
                self.telegram_bot.send_alert(saved, clip_file=clip_file)
        except Exception as exc:
            print(f"[MQTTEventSubscriber] Invalid message on {topic}: {exc}")

    def publish_door_command(self, cmd: DoorCommand) -> dict[str, Any]:
        """Publish `DoorCommand` JSON to `store/{store_id}/door/cmd` with QoS 1."""
        payload_dict = cmd.to_dict()
        self.published_door_commands.append(payload_dict)
        if self._running:
            try:
                self._client.publish(
                    self.door_cmd_topic,
                    payload=json.dumps(payload_dict, ensure_ascii=False),
                    qos=1,
                    retain=False,
                )
            except Exception as exc:
                print(f"[MQTTEventSubscriber] Warning: Failed to publish door/cmd ({exc})")
        return payload_dict

    def start(self) -> bool:
        """Connect to broker and start background network loop."""
        try:
            self._client.connect_async(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            self._running = True
            return True
        except Exception as exc:
            print(f"[MQTTEventSubscriber] Warning: Could not start subscriber ({exc})")
            return False

    def stop(self) -> None:
        """Stop background MQTT subscriber loop."""
        if self._running:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._running = False


def _ensure_webm_clip(mp4_file: Path) -> Path | None:
    """Convert an OpenCV `mp4v` clip into a browser-native `VP80` `.webm` file if not already cached."""
    if not mp4_file.exists():
        return None
    webm_file = mp4_file.with_suffix(".webm")
    if webm_file.exists() and webm_file.stat().st_size > 0:
        return webm_file

    cap = cv2.VideoCapture(str(mp4_file))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 224)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 224)
    writer = cv2.VideoWriter(str(webm_file), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return None
    frames_written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        frames_written += 1
    cap.release()
    writer.release()
    return webm_file if (frames_written > 0 and webm_file.exists()) else None


def _stream_mp4_as_mjpeg(clip_file: Path) -> Iterator[bytes]:
    """Read an MP4 clip with OpenCV and yield MJPEG multipart frames for universal browser playback."""
    if not clip_file.exists():
        blank = np.zeros((270, 480, 3), dtype=np.uint8)
        cv2.putText(blank, "Clip not found", (130, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    cap = cv2.VideoCapture(str(clip_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if (fps and fps > 1.0) else (1.0 / 15.0)
    frames_jpg: list[bytes] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            frames_jpg.append(buf.tobytes())
    cap.release()

    if not frames_jpg:
        blank = np.zeros((270, 480, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    for jpg in frames_jpg:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(min(delay, 0.08))


def create_app(
    config_path: Path | str | AppConfig | Mapping[str, Any] | None = None,
    db_path_override: Path | None = None,
    clips_dir_override: Path | None = None,
    audit_log_override: Path | None = None,
    start_mqtt: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    cfg = load_app_config(config_path, project_root=PROJECT_ROOT)

    db_path = db_path_override or cfg.resolve_path(cfg.paths["db_path"])
    clips_dir = clips_dir_override or cfg.resolve_path(cfg.paths["clips_dir"])
    clips_dir.mkdir(parents=True, exist_ok=True)

    door_cfg = cfg.door
    audit_log_path = audit_log_override or cfg.resolve_path(
        door_cfg.get("audit_log_path", "data/door_audit.log")
    )
    default_ttl_sec = int(door_cfg.get("default_ttl_sec", 45))
    max_ttl_sec = int(door_cfg.get("max_ttl_sec", 120))
    auto_lock = bool(cfg.alerting.get("auto_lock", False))

    broker_host = os.environ.get("MQTT_BROKER_HOST", str(cfg.mqtt["broker_host"]))
    broker_port = int(os.environ.get("MQTT_BROKER_PORT", cfg.mqtt["broker_port"]))
    store_id = os.environ.get("STORE_ID", str(cfg.mqtt.get("store_id", "store_01")))

    db = EventDatabase(db_path)

    # Local fail-safe DoorController mirror for TTL enforcement, audit logging, and offline/test mode
    door_controller = DoorController(
        store_id=store_id,
        broker_host=broker_host,
        broker_port=broker_port,
        default_ttl_sec=default_ttl_sec,
        max_ttl_sec=max_ttl_sec,
        audit_log_path=audit_log_path,
        auto_connect=False,
    )

    telegram_bot = TelegramAlertBot(
        default_ttl_sec=default_ttl_sec,
        max_ttl_sec=max_ttl_sec,
        audit_log_path=audit_log_path,
    )

    subscriber = MQTTEventSubscriber(
        db=db,
        broker_host=broker_host,
        broker_port=broker_port,
        store_id=store_id,
        topic_filter="store/+/cam/+/alert",
        telegram_bot=telegram_bot,
        clips_dir=clips_dir,
        door_controller=door_controller,
    )

    def _execute_door_command(cmd: DoorCommand, actor: str = "admin_dashboard") -> dict[str, Any]:
        log_row = db.log_door_command(
            action=cmd.action,
            reason=cmd.reason,
            event_id=cmd.event_id,
            ttl_sec=cmd.ttl_sec,
            actor=actor,
        )
        raw_cmd = cmd.to_dict()
        raw_cmd["actor"] = actor
        status = door_controller.handle_command(raw_cmd)
        subscriber.door_status = status.to_dict()
        subscriber.publish_door_command(cmd)
        return {
            "command": cmd.to_dict(),
            "door_status": subscriber.door_status,
            "remaining_ttl_sec": round(door_controller.remaining_ttl(), 1),
            "log": log_row,
        }

    telegram_bot.on_door_command = _execute_door_command

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_mqtt:
            subscriber.start()
        try:
            yield
        finally:
            subscriber.stop()

    app = FastAPI(title="Shopformer IoT Alert Dashboard", lifespan=lifespan)
    app.state.db = db
    app.state.clips_dir = clips_dir
    app.state.subscriber = subscriber
    app.state.door_controller = door_controller
    app.state.telegram_bot = telegram_bot

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_content = TEMPLATE_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)

    @app.get("/api/events")
    def list_events(limit: int = 100) -> dict[str, Any]:
        events = db.list_events(limit=limit)
        for ev in events:
            ev["ts"] = to_vn_iso(str(ev.get("ts", "")))
            if ev.get("decided_at"):
                ev["decided_at"] = to_vn_iso(str(ev["decided_at"]))
        return {"events": events}

    @app.post("/api/events")
    def create_event(req: EventCreateRequest) -> dict[str, Any]:
        data = req.model_dump()
        data["ts"] = to_vn_iso(str(data["ts"]))
        saved = db.upsert_event(data)
        if telegram_bot.is_enabled:
            clip_file = clips_dir / Path(str(saved["clip_path"])).name
            telegram_bot.send_alert(saved, clip_file=clip_file)
        return {"event": saved}

    @app.delete("/api/events")
    @app.post("/api/events/reset")
    def reset_events(delete_clips: bool = True) -> dict[str, int]:
        deleted_events = db.clear_events()
        deleted_clips = 0
        if delete_clips and clips_dir.exists():
            for clip_file in list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.webm")):
                try:
                    clip_file.unlink()
                    deleted_clips += 1
                except OSError:
                    pass
        return {
            "deleted_events": deleted_events,
            "deleted_clips": deleted_clips,
        }

    @app.post("/api/events/{event_id}/decision")
    def update_decision(event_id: str, req: DecisionRequest) -> dict[str, Any]:
        if req.decision not in ("confirmed", "ignored"):
            raise HTTPException(status_code=400, detail="Decision must be 'confirmed' or 'ignored'")
        updated = db.update_decision(event_id, req.decision)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
        return {"event": updated}

    @app.get("/api/door/status")
    def get_door_status(limit_logs: int = 20) -> dict[str, Any]:
        expired = door_controller.tick()
        if expired is not None:
            subscriber.door_status = expired.to_dict()
        subscriber.door_status["ts"] = to_vn_iso(str(subscriber.door_status.get("ts", "")))
        return {
            "door_status": subscriber.door_status,
            "remaining_ttl_sec": round(door_controller.remaining_ttl(), 1),
            "cam_status": subscriber.cam_status,
            "default_ttl_sec": default_ttl_sec,
            "max_ttl_sec": max_ttl_sec,
            "auto_lock": auto_lock,
            "logs": db.list_door_logs(limit=limit_logs),
        }

    @app.post("/api/door/cmd")
    def send_door_command(req: DoorCommandRequest) -> dict[str, Any]:
        action = str(req.action or "").strip().lower()
        if action not in ("lock", "unlock"):
            raise HTTPException(status_code=400, detail="Action must be 'lock' or 'unlock'")

        ttl_sec = int(req.ttl_sec) if action == "lock" else 0
        if action == "lock":
            if ttl_sec <= 0:
                raise HTTPException(status_code=400, detail="ttl_sec must be positive for 'lock'")
            ttl_sec = min(ttl_sec, max_ttl_sec)

        cmd = DoorCommand(
            action=action,
            reason=str(req.reason or f"admin_dashboard:{action}"),
            event_id=str(req.event_id or "manual_dashboard"),
            ttl_sec=ttl_sec if action == "lock" else 1,
        )
        # For unlock, preserve ttl_sec=0 in MQTT JSON contract
        if action == "unlock":
            object.__setattr__(cmd, "ttl_sec", 0)

        actor = str(req.actor or "admin_dashboard")
        return _execute_door_command(cmd, actor=actor)

    @app.get("/clips/{clip_name}")
    @app.get("/data/clips/{clip_name}")
    def get_clip_file(clip_name: str) -> FileResponse:
        safe_name = Path(clip_name).name
        target = clips_dir / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Clip file not found")
        media_type = "video/webm" if safe_name.endswith(".webm") else "video/mp4"
        return FileResponse(path=str(target), media_type=media_type, filename=safe_name)

    @app.get("/api/clips/{clip_name}/webm")
    def get_clip_webm(clip_name: str) -> FileResponse:
        safe_name = Path(clip_name).name
        target_mp4 = clips_dir / safe_name
        if not target_mp4.exists():
            raise HTTPException(status_code=404, detail="Clip file not found")
        webm_path = _ensure_webm_clip(target_mp4)
        if webm_path is None or not webm_path.exists():
            return FileResponse(path=str(target_mp4), media_type="video/mp4", filename=safe_name)
        return FileResponse(path=str(webm_path), media_type="video/webm", filename=webm_path.name)

    @app.get("/api/clips/{clip_name}/mjpeg")
    def stream_clip_mjpeg(clip_name: str) -> StreamingResponse:
        safe_name = Path(clip_name).name
        target = clips_dir / safe_name
        return StreamingResponse(
            _stream_mp4_as_mjpeg(target),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main() -> None:
    """Run the FastAPI dashboard server."""
    parser = argparse.ArgumentParser(description="Start Shopformer IoT Dashboard server.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--host", type=str, default=None, help="Bind host override.")
    parser.add_argument("--port", type=int, default=None, help="Bind port override.")
    args = parser.parse_args()

    cfg = load_app_config(args.config, project_root=PROJECT_ROOT)

    host = args.host or str(cfg.dashboard["host"])
    port = int(args.port or cfg.dashboard["port"])
    app = create_app(config_path=cfg, start_mqtt=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

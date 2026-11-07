"""FastAPI web dashboard and background MQTT subscriber for anomaly alerts."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import numpy as np
import paho.mqtt.client as mqtt
from pydantic import BaseModel
import uvicorn
import yaml

from shoplift_iot.dashboard.db import EventDatabase
from shoplift_iot.messaging.mqtt_pub import MQTTPublisher

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


class MQTTEventSubscriber:
    """Background MQTT subscriber listening on `store/+/cam/+/alert` and saving to SQLite."""

    def __init__(
        self,
        db: EventDatabase,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_filter: str = "store/+/cam/+/alert",
    ) -> None:
        self.db = db
        self.broker_host = broker_host
        self.broker_port = int(broker_port)
        self.topic_filter = topic_filter
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
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

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            clean = MQTTPublisher.validate_payload(payload)
            self.db.upsert_event(clean)
        except Exception as exc:
            print(f"[MQTTEventSubscriber] Invalid message on {msg.topic}: {exc}")

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
    config_path: Path | None = None,
    db_path_override: Path | None = None,
    clips_dir_override: Path | None = None,
    start_mqtt: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    cfg_file = config_path or (PROJECT_ROOT / "configs" / "default.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    db_path = db_path_override or (PROJECT_ROOT / cfg["paths"]["db_path"])
    clips_dir = clips_dir_override or (PROJECT_ROOT / cfg["paths"]["clips_dir"])
    clips_dir.mkdir(parents=True, exist_ok=True)

    broker_host = os.environ.get("MQTT_BROKER_HOST", str(cfg["mqtt"]["broker_host"]))
    broker_port = int(os.environ.get("MQTT_BROKER_PORT", cfg["mqtt"]["broker_port"]))

    db = EventDatabase(db_path)
    subscriber = MQTTEventSubscriber(
        db=db,
        broker_host=broker_host,
        broker_port=broker_port,
        topic_filter="store/+/cam/+/alert",
    )

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

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_content = TEMPLATE_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)

    @app.get("/api/events")
    def list_events(limit: int = 100) -> dict[str, Any]:
        return {"events": db.list_events(limit=limit)}

    @app.post("/api/events")
    def create_event(req: EventCreateRequest) -> dict[str, Any]:
        saved = db.upsert_event(req.model_dump())
        return {"event": saved}

    @app.post("/api/events/{event_id}/decision")
    def update_decision(event_id: str, req: DecisionRequest) -> dict[str, Any]:
        if req.decision not in ("confirmed", "ignored"):
            raise HTTPException(status_code=400, detail="Decision must be 'confirmed' or 'ignored'")
        updated = db.update_decision(event_id, req.decision)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
        return {"event": updated}

    @app.get("/clips/{clip_name}")
    def get_clip_file(clip_name: str) -> FileResponse:
        safe_name = Path(clip_name).name
        target = clips_dir / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Clip file not found")
        return FileResponse(path=str(target), media_type="video/mp4", filename=safe_name)

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

    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    host = args.host or str(cfg["dashboard"]["host"])
    port = int(args.port or cfg["dashboard"]["port"])
    app = create_app(config_path=cfg_path, start_mqtt=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


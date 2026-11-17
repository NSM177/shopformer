#!/usr/bin/env python3
"""FastAPI Service for Video Analysis Pipeline (VideoMAE + YOLO ByteTrack) on port 8001."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from shoplift_iot.pipeline_service.service import PipelineManager, verify_bundle_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("pipeline_app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class StartSourceRequest(BaseModel):
    type: str = Field(..., description="Nguon: 'webcam', 'file', hoac 'rtsp'")
    path: str | None = Field(None, description="Duong dan file video")
    index: int | str | None = Field(None, description="Index webcam hoặc 'auto'")
    url: str | None = Field(None, description="URL RTSP")
    cam_id: str = Field("cam_01", description="ID camera")
    store_id: str = Field("store_01", description="ID cua hang")
    pace: bool = Field(True, description="Pace theo FPS video thuc hay chay toc do toi da")
    loop: bool = Field(True, description="Phat lap lai video nhu mot camera")


class UpdateConfigRequest(BaseModel):
    threshold: float | None = Field(None, description="Nguong phat hien VideoMAE (0.01 - 0.99)")
    k: int | None = Field(None, description="So cua so lien tiep de kich hoat ALERT (K >= 1)")
    cooldown_sec: float | None = Field(None, description="Thoi gian cooldown sau canh bao (giay)")
    overlay_polygons: bool | None = Field(None, description="Bat/tat lop phu Polygon cac khu vuc")
    send_mqtt_alerts: bool | None = Field(None, description="Bat/tat gui canh bao len MQTT")


async def mjpeg_frame_generator(manager: PipelineManager, max_frames: int | None = None):
    """Generate MJPEG multipart chunks from latest_jpeg in manager."""
    count = 0
    while True:
        with manager._frame_lock:
            jpeg_bytes = manager.latest_jpeg
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        await asyncio.sleep(0.04)  # ~25 FPS stream delivery


def create_app(bundle_dir: str = "models/videomae_bundle", config_path: str | None = None) -> FastAPI:
    """Create and configure FastAPI instance with PipelineManager."""
    manager = PipelineManager(
        project_root=PROJECT_ROOT,
        bundle_dir=bundle_dir,
        config_path=config_path,
    )
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Khoi dong Pipeline Service tren port 8001...")
        missing = verify_bundle_files(manager.bundle_dir)
        if missing:
            logger.warning("Canh bao: Bundle %s thieu file: %s", manager.bundle_dir, missing)
        else:
            logger.info("Bundle %s day du. San sang suy luan.", manager.bundle_dir)
        yield
        logger.info("Dang tat Pipeline Service...")
        manager.stop_source()

    app = FastAPI(
        title="VideoMAE Shoplifting Pipeline Service",
        description="FastAPI service controlling YOLO ByteTrack + VideoMAE live detection & streaming.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.manager = manager
    app.state.uploads_dir = uploads_dir

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Return runtime state, progress, score, and summary."""
        return manager.get_status()

    @app.post("/api/source/start")
    async def start_source(req: StartSourceRequest) -> dict[str, Any]:
        """Start video capture and inference pipeline."""
        try:
            res = manager.start_source(
                source_type=req.type,
                path=req.path,
                index=req.index,
                url=req.url,
                cam_id=req.cam_id,
                store_id=req.store_id,
                pace=req.pace,
                loop=req.loop,
            )
            return res
        except FileNotFoundError as fnf:
            raise HTTPException(status_code=400, detail=str(fnf)) from fnf
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as exc:
            logger.exception("Loi khi bat dau nguon: %s", exc)
            raise HTTPException(status_code=500, detail=f"Loi khoi dong pipeline: {exc}") from exc

    @app.post("/api/source/stop")
    async def stop_source() -> dict[str, Any]:
        """Stop active video capture and inference pipeline."""
        return manager.stop_source()

    @app.get("/api/scores/{cam_id}")
    async def get_scores(cam_id: str, since: float | None = Query(None)) -> dict[str, Any]:
        """Get time series of anomaly scores for plotting."""
        return manager.get_scores(cam_id=cam_id, since=since)

    @app.post("/api/config")
    async def update_config(req: UpdateConfigRequest) -> dict[str, Any]:
        """Update threshold, K, and cooldown dynamically."""
        return manager.update_config(
            threshold=req.threshold,
            k=req.k,
            cooldown_sec=req.cooldown_sec,
            overlay_polygons=req.overlay_polygons,
            send_mqtt_alerts=req.send_mqtt_alerts,
        )

    @app.post("/api/upload")
    async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload a video file to data/uploads/ for pipeline processing."""
        filename = Path(file.filename or "uploaded_video.mp4").name
        ext = Path(filename).suffix.lower()
        if ext not in (".mp4", ".avi", ".mov", ".mkv"):
            raise HTTPException(
                status_code=400,
                detail=f"Dinh dang file '{ext}' khong ho tro. Chi chap nhan .mp4, .avi, .mov, .mkv",
            )

        # Sanitize filename with timestamp to avoid collision
        stem = Path(filename).stem
        safe_name = f"{stem}_{int(time.time())}{ext}"
        target_path = uploads_dir / safe_name

        try:
            contents = await file.read()
            with open(target_path, "wb") as f:
                f.write(contents)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Loi luu file upload: {exc}") from exc

        rel_path = f"data/uploads/{safe_name}"
        logger.info("Da luu file upload: %s (%d bytes)", rel_path, len(contents))
        return {
            "status": "uploaded",
            "filename": safe_name,
            "path": rel_path,
            "size": len(contents),
        }

    @app.get("/stream/{cam_id}.mjpg")
    async def stream_mjpeg(cam_id: str) -> StreamingResponse:
        """Stream live MJPEG frames with annotated bounding boxes and scores."""
        return StreamingResponse(
            mjpeg_frame_generator(manager),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chay Pipeline Service VideoMAE tren port 8001")
    parser.add_argument("--host", default="0.0.0.0", help="Host bind (mac dinh 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port (mac dinh 8001)")
    parser.add_argument("--bundle", default="models/videomae_bundle", help="Thu muc bundle VideoMAE")
    parser.add_argument("--config", default="configs/default.yaml", help="File cau hinh")
    args = parser.parse_args()

    app_instance = create_app(bundle_dir=args.bundle, config_path=args.config)
    uvicorn.run(app_instance, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from shoplift_iot.pipeline_service.app import create_app
from shoplift_iot.pipeline_service.service import create_placeholder_frame, verify_bundle_files


@pytest.fixture
def client():
    app = create_app(bundle_dir="models/videomae_bundle", config_path="configs/default.yaml")
    return TestClient(app)


def test_verify_bundle_files_success():
    missing = verify_bundle_files("models/videomae_bundle")
    assert missing == [], f"Missing files in real bundle: {missing}"


def test_verify_bundle_files_missing(tmp_path):
    missing = verify_bundle_files(tmp_path)
    assert len(missing) == 5


def test_create_placeholder_frame():
    jpeg = create_placeholder_frame("Test")
    assert isinstance(jpeg, bytes)
    assert jpeg.startswith(b"\xff\xd8")  # JPEG SOI marker


def test_get_status_initial(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data
    assert "state" in data
    assert "threshold" in data
    assert "k" in data
    assert data["running"] is False


def test_update_config(client):
    res = client.post("/api/config", json={"threshold": 0.65, "k": 4, "cooldown_sec": 15.0})
    assert res.status_code == 200
    data = res.json()
    assert data["threshold"] == 0.65
    assert data["k"] == 4
    assert data["cooldown_sec"] == 15.0

    # Verify status reflects updated config
    status = client.get("/api/status").json()
    assert status["threshold"] == 0.65
    assert status["k"] == 4


def test_upload_invalid_file_extension(client):
    file_content = b"dummy text"
    res = client.post(
        "/api/upload",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert res.status_code == 400
    assert "khong ho tro" in res.json()["detail"]


def test_upload_valid_video(client):
    file_content = b"\x00\x00\x00\x18ftypmp42" + b"A" * 100
    res = client.post(
        "/api/upload",
        files={"file": ("sample_test.mp4", io.BytesIO(file_content), "video/mp4")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "uploaded"
    assert "sample_test" in data["filename"]
    assert Path(data["path"]).exists()
    # Clean up uploaded test file
    Path(data["path"]).unlink(missing_ok=True)


def test_start_source_invalid_file(client):
    res = client.post(
        "/api/source/start",
        json={"type": "file", "path": "data/non_existent_video_12345.mp4"},
    )
    assert res.status_code == 400
    assert "Không tìm thấy file" in res.json()["detail"]


def test_start_source_invalid_type(client):
    res = client.post(
        "/api/source/start",
        json={"type": "unknown_type"},
    )
    assert res.status_code == 400


def test_stop_source(client):
    res = client.post("/api/source/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"


def test_get_scores_empty(client):
    res = client.get("/api/scores/cam_01")
    assert res.status_code == 200
    data = res.json()
    assert data["cam_id"] == "cam_01"
    assert isinstance(data["scores"], list)


@pytest.mark.asyncio
async def test_mjpeg_stream_generator():
    from shoplift_iot.pipeline_service.app import mjpeg_frame_generator
    from shoplift_iot.pipeline_service.service import PipelineManager

    mgr = PipelineManager(bundle_dir="models/videomae_bundle", config_path="configs/default.yaml")
    chunks = []
    async for chunk in mjpeg_frame_generator(mgr, max_frames=2):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert b"--frame\r\n" in chunks[0]
    assert b"Content-Type: image/jpeg\r\n" in chunks[0]

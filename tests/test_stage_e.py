"""Unit and integration tests for Stage E: alerting, MQTT publisher, dashboard API/SQLite, and main pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerting import AlertManager  # noqa: E402
from src.dashboard.app import create_app  # noqa: E402
from src.main import run_pipeline  # noqa: E402
from src.mqtt_pub import MQTTPublisher, REQUIRED_KEYS  # noqa: E402


def test_alert_manager_k_consecutive_and_cooldown(tmp_path: Path) -> None:
    """Verify temporal smoothing, K=3 consecutive windows requirement, cooldown, and MP4 clip creation."""
    mgr = AlertManager(
        threshold=0.5,
        clips_dir=tmp_path / "clips",
        cam_id="cam_test",
        score_smooth_window=3,
        consecutive_windows_k=3,
        cooldown_seconds=10.0,
        clip_duration_seconds=2.0,
        target_fps=15.0,
        project_root=tmp_path,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # Window 1 & 2 above 0.5 -> should not fire yet (K=3)
    mgr.push_frame(frame)
    assert mgr.update_score(track_id=7, raw_score=0.8, timestamp_sec=1.0) is None
    mgr.push_frame(frame)
    assert mgr.update_score(track_id=7, raw_score=0.9, timestamp_sec=2.0) is None

    # Window 3 above 0.5 -> 3rd consecutive window -> MUST fire!
    mgr.push_frame(frame)
    ev1 = mgr.update_score(track_id=7, raw_score=1.0, timestamp_sec=3.0)
    assert ev1 is not None
    payload = ev1.to_dict()
    assert tuple(payload.keys()) == REQUIRED_KEYS
    assert payload["track_id"] == 7
    assert (tmp_path / ev1.clip_path).exists()

    # Next 3 windows at t=4, 5, 6s (within 10s cooldown from t=3.0) -> must be suppressed by cooldown
    for t in (4.0, 5.0, 6.0):
        mgr.push_frame(frame)
        assert mgr.update_score(track_id=7, raw_score=1.0, timestamp_sec=t) is None

    # At t=14.0s (> 3.0 + 10.0s cooldown), should fire again
    mgr.push_frame(frame)
    ev2 = mgr.update_score(track_id=7, raw_score=1.0, timestamp_sec=14.0)
    assert ev2 is not None
    assert ev2.event_id != ev1.event_id


def test_mqtt_publisher_payload_validation() -> None:
    """Verify MQTTPublisher formats topic and validates strict JSON schema without raw media."""
    pub = MQTTPublisher(
        broker_host="127.0.0.1",
        broker_port=1883,
        store_id="store_01",
        cam_id="cam_01",
        auto_connect=False,
    )
    assert pub.topic == "store/store_01/cam/cam_01/alert"

    sample = {
        "event_id": "evt_test_123",
        "ts": "2026-11-07T16:00:00+00:00",
        "cam_id": "cam_01",
        "track_id": 2,
        "score": 0.91,
        "threshold": 0.25,
        "clip_path": "data/clips/evt_test_123.mp4",
        "extra_forbidden_raw_frame": "should_be_stripped",
    }
    cleaned = pub.validate_payload(sample)
    assert set(cleaned.keys()) == set(REQUIRED_KEYS)
    assert "extra_forbidden_raw_frame" not in cleaned


def test_dashboard_api_and_sqlite_decisions(tmp_path: Path) -> None:
    """Verify FastAPI dashboard lists events and persists 'confirmed' / 'ignored' decisions in SQLite."""
    db_file = tmp_path / "events_test.db"
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
        db_path_override=db_file,
        clips_dir_override=clips_dir,
        start_mqtt=False,
    )
    client = TestClient(app)

    # 1. Create event
    ev_payload = {
        "event_id": "evt_ui_001",
        "ts": "2026-11-07T16:10:00+00:00",
        "cam_id": "cam_01",
        "track_id": 3,
        "score": 1.25,
        "threshold": 0.25,
        "clip_path": "data/clips/evt_ui_001.mp4",
    }
    res_post = client.post("/api/events", json=ev_payload)
    assert res_post.status_code == 200
    assert res_post.json()["event"]["decision"] == "pending"

    # 2. Check HTML index and GET /api/events
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert "Xác nhận" in res_html.text and "Bỏ qua" in res_html.text

    res_list = client.get("/api/events")
    assert res_list.status_code == 200
    assert len(res_list.json()["events"]) == 1

    # 3. Click 'Xác nhận' (confirmed) -> persisted in SQLite
    res_conf = client.post("/api/events/evt_ui_001/decision", json={"decision": "confirmed"})
    assert res_conf.status_code == 200
    assert res_conf.json()["event"]["decision"] == "confirmed"
    assert res_conf.json()["event"]["decided_at"] is not None

    # 4. Click 'Bỏ qua' (ignored) -> persisted in SQLite
    res_ign = client.post("/api/events/evt_ui_001/decision", json={"decision": "ignored"})
    assert res_ign.status_code == 200
    assert res_ign.json()["event"]["decision"] == "ignored"


def test_main_pipeline_emits_alert_on_poselift_clip() -> None:
    """Verify full 8-step pipeline (`src/main.py`) emits an anomaly alert and saves a 5s clip."""
    sample_videos = sorted((PROJECT_ROOT / "third_party" / "PoseLift" / "Samples").glob("*.mp4"))
    assert len(sample_videos) > 0, "Expected sample MP4 in third_party/PoseLift/Samples"

    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    events = run_pipeline(
        source=sample_videos[0],
        config_path=cfg_path,
        threshold_override=0.25,
        max_frames=25,
        device_override="cpu",
    )
    assert len(events) >= 1
    first_clip = PROJECT_ROOT / events[0].clip_path
    assert first_clip.exists() and first_clip.stat().st_size > 0


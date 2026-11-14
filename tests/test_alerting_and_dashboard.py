"""Unit and integration tests for Stage E: alerting, MQTT publisher, dashboard API/SQLite, and main pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.alerting.alerting import AlertManager  # noqa: E402
from shoplift_iot.dashboard.app import create_app  # noqa: E402
from shoplift_iot.main import run_pipeline  # noqa: E402
from shoplift_iot.messaging.mqtt_pub import MQTTPublisher, REQUIRED_KEYS  # noqa: E402
from shoplift_iot.messaging.telegram_bot import TelegramAlertBot  # noqa: E402


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


def test_dashboard_door_control_and_audit_log(tmp_path: Path) -> None:
    """Verify GD6 door status/command endpoints, confirmation modal HTML, unlock priority, and audit logs."""
    db_file = tmp_path / "events_door.db"
    clips_dir = tmp_path / "clips"
    audit_log = tmp_path / "door_audit.log"

    app = create_app(
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
        db_path_override=db_file,
        clips_dir_override=clips_dir,
        audit_log_override=audit_log,
        start_mqtt=False,
    )
    client = TestClient(app)

    # 1. HTML includes confirmation modal and Lock/Unlock controls
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert "lock-confirm-modal" in res_html.text
    assert "Xác nhận Khóa cửa" in res_html.text
    assert "Mở cửa ngay" in res_html.text

    # 2. Initial door status is fail-safe UNLOCKED and auto_lock is False
    res_st = client.get("/api/door/status")
    assert res_st.status_code == 200
    st_json = res_st.json()
    assert st_json["door_status"]["state"] == "unlocked"
    assert st_json["auto_lock"] is False
    assert st_json["default_ttl_sec"] == 45

    # 3. Admin issues LOCK command with TTL=45s
    res_lock = client.post(
        "/api/door/cmd",
        json={
            "action": "lock",
            "reason": "admin_dashboard:confirm_lock",
            "event_id": "evt_door_01",
            "ttl_sec": 45,
            "actor": "admin_dashboard",
        },
    )
    assert res_lock.status_code == 200
    lock_data = res_lock.json()
    assert lock_data["door_status"]["state"] == "locked"
    assert lock_data["door_status"]["event_id"] == "evt_door_01"
    assert lock_data["command"] == {
        "action": "lock",
        "reason": "admin_dashboard:confirm_lock",
        "event_id": "evt_door_01",
        "ttl_sec": 45,
    }
    assert len(app.state.subscriber.published_door_commands) == 1

    # 4. Admin issues priority UNLOCK command -> immediately unlocked
    res_unlock = client.post(
        "/api/door/cmd",
        json={
            "action": "unlock",
            "reason": "admin_dashboard:priority_unlock",
            "event_id": "evt_door_01",
            "ttl_sec": 0,
            "actor": "admin_dashboard",
        },
    )
    assert res_unlock.status_code == 200
    assert res_unlock.json()["door_status"]["state"] == "unlocked"

    # 5. Check SQLite door_logs and file audit_log both have 2 entries
    res_st_after = client.get("/api/door/status").json()
    assert len(res_st_after["logs"]) == 2
    assert res_st_after["logs"][0]["action"] == "unlock"
    assert res_st_after["logs"][1]["action"] == "lock"
    assert audit_log.exists()
    assert len(audit_log.read_text(encoding="utf-8").strip().splitlines()) == 2

    # 6. Invalid action or non-positive ttl_sec for lock returns 400
    assert client.post("/api/door/cmd", json={"action": "invalid"}).status_code == 400
    assert client.post("/api/door/cmd", json={"action": "lock", "ttl_sec": 0}).status_code == 400


def test_mqtt_subscriber_door_and_cam_status_sync(tmp_path: Path) -> None:
    """Verify MQTTEventSubscriber updates real-time door_status and cam_status from MQTT messages."""
    import json

    app = create_app(
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
        db_path_override=tmp_path / "sync.db",
        clips_dir_override=tmp_path / "clips",
        audit_log_override=tmp_path / "audit.log",
        start_mqtt=False,
    )
    sub = app.state.subscriber

    class DummyMsg:
        def __init__(self, topic: str, payload: bytes) -> None:
            self.topic = topic
            self.payload = payload

    sub._on_message(
        None,
        None,
        DummyMsg(
            "store/store_01/door/status",
            json.dumps({"state": "locked", "ts": "2026-11-14T10:00:00+00:00", "event_id": "evt_99"}).encode("utf-8"),
        ),
    )
    sub._on_message(
        None,
        None,
        DummyMsg("store/store_01/cam/cam_01/status", b"online"),
    )

    client = TestClient(app)
    data = client.get("/api/door/status").json()
    assert data["door_status"]["state"] == "locked"
    assert data["door_status"]["event_id"] == "evt_99"
    assert data["cam_status"]["cam_01"] == "online"


def test_telegram_bot_alert_and_admin_commands(tmp_path: Path, monkeypatch) -> None:
    """Verify TelegramAlertBot env-var reading, alert formatting, and authorized `/lock` & `/unlock` commands."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    disabled_bot = TelegramAlertBot()
    assert not disabled_bot.is_enabled
    assert not disabled_bot.send_alert({"event_id": "evt_1"})

    sent_payloads: list[dict] = []
    emitted_cmds: list[tuple] = []
    audit_path = tmp_path / "tg_audit.log"

    bot = TelegramAlertBot(
        bot_token="123456:TEST_TOKEN",
        chat_id="998877",
        default_ttl_sec=45,
        audit_log_path=audit_path,
        on_door_command=lambda cmd, actor: emitted_cmds.append((cmd, actor)),
        http_sender=lambda method, payload: sent_payloads.append({"method": method, "payload": payload}) or {"ok": True},
    )
    assert bot.is_enabled

    ok = bot.send_alert(
        {
            "event_id": "evt_tg_01",
            "ts": "2026-11-14T10:05:00+00:00",
            "cam_id": "cam_01",
            "track_id": 4,
            "score": 0.88,
            "threshold": 0.50,
            "clip_path": "data/clips/evt_tg_01.mp4",
        }
    )
    assert ok
    assert len(sent_payloads) == 1
    assert "/lock evt_tg_01 45" in sent_payloads[0]["payload"]["text"]

    # Unauthorized chat_id MUST be rejected (Safety Constraint #1)
    unauth = bot.handle_update(
        {"message": {"chat": {"id": "000000"}, "from": {"username": "stranger"}, "text": "/lock evt_tg_01 45"}}
    )
    assert unauth is None
    assert len(emitted_cmds) == 0

    # Authorized admin chat_id issues /lock evt_tg_01 60
    cmd_lock = bot.handle_update(
        {"message": {"chat": {"id": "998877"}, "from": {"username": "admin_user"}, "text": "/lock evt_tg_01 60"}}
    )
    assert cmd_lock is not None
    assert cmd_lock.to_dict() == {
        "action": "lock",
        "reason": "admin_telegram:lock",
        "event_id": "evt_tg_01",
        "ttl_sec": 60,
    }

    # Authorized admin chat_id issues /unlock
    cmd_unlock = bot.handle_update(
        {"message": {"chat": {"id": "998877"}, "from": {"username": "admin_user"}, "text": "/unlock evt_tg_01"}}
    )
    assert cmd_unlock is not None
    assert cmd_unlock.action == "unlock"
    assert len(emitted_cmds) == 2
    assert audit_path.exists()


def test_gd7_docker_compose_and_readme_contracts() -> None:
    """Verify GD7 docker-compose.yml defines all 4 services and README documents safety constraints."""
    import yaml

    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists()
    compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = compose_data.get("services", {})
    for required_service in ("mosquitto", "dashboard", "fake_door", "pipeline"):
        assert required_service in services, f"Missing '{required_service}' in docker-compose.yml"

    assert (PROJECT_ROOT / "docker" / "Dockerfile.dashboard").exists()
    assert (PROJECT_ROOT / "docker" / "Dockerfile.pipeline").exists()

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "store/{store_id}/cam/{cam_id}/alert" in readme_text
    assert "store/{store_id}/door/cmd" in readme_text
    assert "store/{store_id}/door/status" in readme_text
    assert "auto_lock" in readme_text
    assert "ttl_sec" in readme_text




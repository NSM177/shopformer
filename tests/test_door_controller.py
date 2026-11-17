"""Unit and integration tests for GD5: `DoorController`, fail-safe behavior, TTL, and MQTT JSON contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.actuators.door_controller import DoorController  # noqa: E402
from shoplift_iot.contracts import DoorCommand  # noqa: E402


class _MockMQTTClient:
    """In-memory mock MQTT client recording subscriptions, Last Will, and published messages."""

    def __init__(self) -> None:
        self.on_connect: Any = None
        self.on_disconnect: Any = None
        self.on_message: Any = None
        self.will: dict[str, Any] | None = None
        self.subscriptions: list[tuple[str, int]] = []
        self.published: list[dict[str, Any]] = []

    def will_set(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.will = {"topic": topic, "payload": json.loads(payload), "qos": qos, "retain": retain}

    def connect(self, host: str, port: int, keepalive: int = 60) -> None:
        pass

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def subscribe(self, topic: str, qos: int = 0) -> None:
        self.subscriptions.append((topic, qos))

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.published.append(
            {
                "topic": topic,
                "payload": json.loads(payload),
                "qos": qos,
                "retain": retain,
            }
        )


def test_door_controller_lock_ttl_unlock_priority_and_contracts(tmp_path: Path) -> None:
    """Verify door lock with TTL, auto-unlock on TTL expiry, unlock priority, and strict MQTT JSON contracts."""
    mock_mqtt = _MockMQTTClient()
    audit_file = tmp_path / "door_audit.log"

    ctrl = DoorController(
        store_id="store_01",
        default_ttl_sec=45,
        max_ttl_sec=120,
        audit_log_path=audit_file,
        mqtt_client=mock_mqtt,
        auto_connect=True,
    )

    # 1. Verify topics, Last Will, and initial fail-safe unlocked state
    assert ctrl.cmd_topic == "store/store_01/door/cmd"
    assert ctrl.status_topic == "store/store_01/door/status"
    assert mock_mqtt.will is not None
    assert mock_mqtt.will["topic"] == "store/store_01/door/status"
    assert mock_mqtt.will["retain"] is True
    assert mock_mqtt.will["payload"]["state"] == "unlocked"
    assert ("store/store_01/door/cmd", 1) in mock_mqtt.subscriptions
    assert ctrl.state == "unlocked"

    # 2. Send lock command (TTL=45s) at t=100.0
    cmd_lock = DoorCommand(
        action="lock",
        reason="admin_dashboard:confirmed",
        event_id="evt-001",
        ttl_sec=45,
    )
    assert set(cmd_lock.to_dict().keys()) == {"action", "reason", "event_id", "ttl_sec"}

    st_locked = ctrl.handle_command(cmd_lock.to_dict(), now=100.0)
    assert ctrl.state == "locked"
    assert set(st_locked.to_dict().keys()) == {"state", "ts", "event_id"}
    assert st_locked.state == "locked"
    assert st_locked.event_id == "evt-001"
    assert mock_mqtt.published[-1]["topic"] == "store/store_01/door/status"
    assert mock_mqtt.published[-1]["retain"] is True
    assert mock_mqtt.published[-1]["payload"]["state"] == "locked"
    assert ctrl.remaining_ttl(now=120.0) == 25.0

    # 3. Before TTL expires (t=144.0), door stays locked
    assert ctrl.tick(now=144.0) is None
    assert ctrl.state == "locked"

    # 4. At t=145.0 (45s elapsed), door automatically unlocks and publishes retained status
    st_auto_open = ctrl.tick(now=145.0)
    assert st_auto_open is not None
    assert ctrl.state == "unlocked"
    assert st_auto_open.state == "unlocked"
    assert mock_mqtt.published[-1]["payload"]["state"] == "unlocked"
    assert mock_mqtt.published[-1]["retain"] is True

    # 5. Lock again at t=200.0 (TTL=60s), then verify unlock command immediately overrides lock
    ctrl.handle_command(
        {"action": "lock", "reason": "admin_telegram:lock", "event_id": "evt-002", "ttl_sec": 60},
        now=200.0,
    )
    assert ctrl.state == "locked"

    st_manual_open = ctrl.handle_command(
        {"action": "unlock", "reason": "admin_dashboard:manual_unlock", "event_id": "evt-002", "ttl_sec": 0},
        now=205.0,
    )
    assert ctrl.state == "unlocked"
    assert st_manual_open.state == "unlocked"
    assert ctrl.remaining_ttl(now=205.0) == 0.0

    # 6. Verify audit log records timestamp, actor, and event_id for every command
    log_lines = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(log_lines) >= 4
    actors = [entry["actor"] for entry in log_lines]
    assert "admin_dashboard" in actors
    assert "ttl_timer" in actors
    assert "admin_telegram" in actors
    for entry in log_lines:
        assert "ts" in entry and "actor" in entry and "event_id" in entry


def test_door_controller_fail_safe_on_disconnect_and_invalid_payload(tmp_path: Path) -> None:
    """Verify Safety Constraint #3: MQTT disconnect or bad payload forces door to UNLOCKED."""
    mock_mqtt = _MockMQTTClient()
    ctrl = DoorController(
        store_id="store_01",
        audit_log_path=tmp_path / "audit.log",
        mqtt_client=mock_mqtt,
        auto_connect=True,
    )

    # Lock door first
    ctrl.handle_command(
        {"action": "lock", "reason": "admin:lock", "event_id": "evt-99", "ttl_sec": 45},
        now=10.0,
    )
    assert ctrl.state == "locked"

    # Simulate unexpected MQTT disconnect -> MUST immediately unlock (fail-safe)
    ctrl._on_disconnect()
    assert ctrl.state == "unlocked"
    assert ctrl.remaining_ttl(now=11.0) == 0.0

    # Lock door again, then send corrupted JSON payload -> MUST immediately unlock (fail-safe)
    ctrl.handle_command(
        {"action": "lock", "reason": "admin:lock", "event_id": "evt-100", "ttl_sec": 45},
        now=20.0,
    )
    assert ctrl.state == "locked"

    ctrl.handle_command(b"{corrupted_json_payload", now=21.0)
    assert ctrl.state == "unlocked"


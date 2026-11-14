"""GD5: Fail-safe IoT door controller subscribing to `door/cmd` and publishing retained `door/status`."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import paho.mqtt.client as mqtt

from shoplift_iot.config import PROJECT_ROOT, AppConfig, load_app_config
from shoplift_iot.contracts import DoorCommand, DoorStatus


class DoorController:
    """Manage fail-safe door state ('unlocked' by default), TTL countdown, audit logging, and MQTT sync."""

    def __init__(
        self,
        store_id: str = "store_01",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        default_ttl_sec: int = 45,
        max_ttl_sec: int = 120,
        cmd_topic_template: str = "store/{store_id}/door/cmd",
        status_topic_template: str = "store/{store_id}/door/status",
        audit_log_path: Path | str | None = None,
        mqtt_client: Any | None = None,
        auto_connect: bool = False,
        on_state_change: Callable[[DoorStatus, str], None] | None = None,
    ) -> None:
        self.store_id = str(store_id)
        self.broker_host = str(broker_host)
        self.broker_port = int(broker_port)
        self.default_ttl_sec = max(1, int(default_ttl_sec))
        self.max_ttl_sec = max(self.default_ttl_sec, int(max_ttl_sec))
        self.cmd_topic = cmd_topic_template.format(store_id=self.store_id)
        self.status_topic = status_topic_template.format(store_id=self.store_id)

        self.audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else (PROJECT_ROOT / "data" / "door_audit.log")
        )
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_state_change = on_state_change

        # Safety Constraint #3: Fail-safe default state is ALWAYS "unlocked"
        self.state: str = "unlocked"
        self.current_event_id: str = ""
        self.lock_deadline: float | None = None
        self._lock = threading.RLock()

        self.audit_entries: list[dict[str, Any]] = []
        self.status_history: list[dict[str, Any]] = []

        self._client = mqtt_client or mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"door-{self.store_id}",
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        # Configure MQTT Last Will: if door controller loses connection/power, broker sets status to "unlocked"
        lwt_payload = DoorStatus.create(state="unlocked", event_id="").to_dict()
        try:
            self._client.will_set(
                self.status_topic,
                payload=json.dumps(lwt_payload, ensure_ascii=False),
                qos=1,
                retain=True,
            )
        except Exception:
            pass

        self._connected = False
        if auto_connect:
            self.connect()

    @classmethod
    def from_config(
        cls,
        config_or_path: Path | str | AppConfig | Mapping[str, Any] | None = None,
        mqtt_client: Any | None = None,
        auto_connect: bool = False,
        on_state_change: Callable[[DoorStatus, str], None] | None = None,
    ) -> DoorController:
        """Instantiate `DoorController` from `configs/default.yaml` and environment variables."""
        cfg = load_app_config(config_or_path, project_root=PROJECT_ROOT)
        door_cfg = cfg.door
        mqtt_cfg = cfg.mqtt

        store_id = os.environ.get("STORE_ID", str(mqtt_cfg.get("store_id", "store_01")))
        broker_host = os.environ.get("MQTT_BROKER_HOST", str(mqtt_cfg.get("broker_host", "localhost")))
        broker_port = int(os.environ.get("MQTT_BROKER_PORT", mqtt_cfg.get("broker_port", 1883)))
        audit_rel = door_cfg.get("audit_log_path", "data/door_audit.log")

        return cls(
            store_id=store_id,
            broker_host=broker_host,
            broker_port=broker_port,
            default_ttl_sec=int(door_cfg.get("default_ttl_sec", 45)),
            max_ttl_sec=int(door_cfg.get("max_ttl_sec", 120)),
            cmd_topic_template=str(door_cfg.get("cmd_topic_template", "store/{store_id}/door/cmd")),
            status_topic_template=str(door_cfg.get("status_topic_template", "store/{store_id}/door/status")),
            audit_log_path=cfg.resolve_path(audit_rel),
            mqtt_client=mqtt_client,
            auto_connect=auto_connect,
            on_state_change=on_state_change,
        )

    def connect(self) -> bool:
        """Connect to MQTT broker, subscribe to `door/cmd`, and publish initial `unlocked` status."""
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=30)
            self._client.loop_start()
            self._connected = True
            self._client.subscribe(self.cmd_topic, qos=1)
            self.publish_status()
            return True
        except Exception as exc:
            self._connected = False
            self.fail_safe_unlock(reason=f"mqtt_connect_error:{exc}")
            return False

    def close(self) -> None:
        """Fail-safe unlock door, publish `unlocked` status, and disconnect cleanly."""
        self.fail_safe_unlock(reason="controller_shutdown")
        if self._connected:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._connected = False

    def _on_connect(
        self,
        client: Any,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        self._connected = True
        client.subscribe(self.cmd_topic, qos=1)
        self.publish_status()

    def _on_disconnect(self, *args: Any, **kwargs: Any) -> None:
        """Safety Constraint #3: On MQTT disconnect, immediately return door to UNLOCKED."""
        self._connected = False
        self.fail_safe_unlock(reason="mqtt_disconnected")

    def _on_message(self, _client: Any, _userdata: Any, msg: Any) -> None:
        self.handle_command(msg.payload)

    @staticmethod
    def _extract_actor(reason: str) -> str:
        """Extract issuer identity from `reason` (e.g., 'admin:dashboard' -> 'admin:dashboard')."""
        r_clean = str(reason or "").strip()
        if not r_clean:
            return "unknown"
        if ":" in r_clean:
            return r_clean.split(":", 1)[0].strip()
        if "telegram" in r_clean.lower():
            return "admin_telegram"
        if "admin" in r_clean.lower() or "dashboard" in r_clean.lower():
            return "admin_dashboard"
        return r_clean

    def _write_audit_log(
        self,
        actor: str,
        action: str,
        reason: str,
        event_id: str,
        ttl_sec: int,
        resulting_state: str,
    ) -> dict[str, Any]:
        """Safety Constraint #4: Every door command/transition is logged with timestamp, actor, and event_id."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "store_id": self.store_id,
            "actor": str(actor),
            "action": str(action),
            "reason": str(reason),
            "event_id": str(event_id),
            "ttl_sec": int(ttl_sec),
            "resulting_state": str(resulting_state),
        }
        self.audit_entries.append(entry)
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return entry

    def publish_status(self) -> DoorStatus:
        """Publish current `DoorStatus` (retained) to `store/{store_id}/door/status`."""
        status_obj = DoorStatus.create(state=self.state, event_id=self.current_event_id)
        payload_dict = status_obj.to_dict()
        self.status_history.append(payload_dict)

        if self._connected or self._client is not None:
            try:
                self._client.publish(
                    self.status_topic,
                    payload=json.dumps(payload_dict, ensure_ascii=False),
                    qos=1,
                    retain=True,
                )
            except Exception:
                pass
        return status_obj

    def fail_safe_unlock(
        self,
        reason: str = "fail_safe",
        now: float | None = None,
    ) -> DoorStatus:
        """Force the door into 'unlocked' state immediately (Safety Constraint #3)."""
        with self._lock:
            prev_state = self.state
            self.state = "unlocked"
            self.lock_deadline = None
            self._write_audit_log(
                actor="fail_safe",
                action="unlock",
                reason=reason,
                event_id=self.current_event_id,
                ttl_sec=0,
                resulting_state=self.state,
            )
            status = self.publish_status()
            if self.on_state_change is not None and prev_state != self.state:
                self.on_state_change(status, reason)
            return status

    def handle_command(
        self,
        payload: Mapping[str, Any] | DoorCommand | str | bytes,
        now: float | None = None,
    ) -> DoorStatus:
        """Process a `door/cmd` message (`unlock` always has priority; errors trigger fail-safe unlock)."""
        ts = time.monotonic() if now is None else float(now)
        with self._lock:
            try:
                if isinstance(payload, DoorCommand):
                    raw = payload.to_dict()
                elif isinstance(payload, (str, bytes)):
                    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
                    raw = json.loads(text)
                elif isinstance(payload, Mapping):
                    raw = dict(payload)
                else:
                    raise TypeError(f"Unsupported door command payload type: {type(payload)}")

                action = str(raw["action"]).strip().lower()
                if action not in ("lock", "unlock"):
                    raise ValueError(f"Invalid door action: {action}")

                reason = str(raw.get("reason", "admin_command"))
                event_id = str(raw.get("event_id", ""))
                actor = str(raw.get("actor") or self._extract_actor(reason))

                # Safety Constraint #2: Unlock always takes priority and opens the door immediately
                if action == "unlock":
                    self.state = "unlocked"
                    self.lock_deadline = None
                    if event_id:
                        self.current_event_id = event_id
                    self._write_audit_log(
                        actor=actor,
                        action="unlock",
                        reason=reason,
                        event_id=self.current_event_id,
                        ttl_sec=0,
                        resulting_state=self.state,
                    )
                    status = self.publish_status()
                    if self.on_state_change is not None:
                        self.on_state_change(status, reason)
                    return status

                # Action == "lock": must have positive ttl_sec (default 45, clamped to max_ttl_sec)
                raw_ttl = raw.get("ttl_sec", self.default_ttl_sec)
                ttl_sec = int(raw_ttl) if raw_ttl is not None else self.default_ttl_sec
                if ttl_sec <= 0:
                    raise ValueError("ttl_sec must be > 0 for lock command")
                ttl_sec = min(ttl_sec, self.max_ttl_sec)

                self.state = "locked"
                self.current_event_id = event_id
                self.lock_deadline = ts + float(ttl_sec)
                self._write_audit_log(
                    actor=actor,
                    action="lock",
                    reason=reason,
                    event_id=self.current_event_id,
                    ttl_sec=ttl_sec,
                    resulting_state=self.state,
                )
                status = self.publish_status()
                if self.on_state_change is not None:
                    self.on_state_change(status, f"{reason} (ttl={ttl_sec}s)")
                return status

            except Exception as exc:
                # Safety Constraint #3: Software/validation error forces door to fail-safe UNLOCKED state
                return self.fail_safe_unlock(reason=f"command_error:{exc}", now=ts)

    def tick(self, now: float | None = None) -> DoorStatus | None:
        """Check TTL expiration and automatically unlock door when `ttl_sec` elapses."""
        ts = time.monotonic() if now is None else float(now)
        with self._lock:
            if (
                self.state == "locked"
                and self.lock_deadline is not None
                and ts >= self.lock_deadline
            ):
                self.state = "unlocked"
                self.lock_deadline = None
                self._write_audit_log(
                    actor="ttl_timer",
                    action="unlock",
                    reason="ttl_expired",
                    event_id=self.current_event_id,
                    ttl_sec=0,
                    resulting_state=self.state,
                )
                status = self.publish_status()
                if self.on_state_change is not None:
                    self.on_state_change(status, "ttl_expired")
                return status
        return None

    def remaining_ttl(self, now: float | None = None) -> float:
        """Return remaining seconds before automatic unlock (0.0 if unlocked)."""
        ts = time.monotonic() if now is None else float(now)
        with self._lock:
            if self.state != "locked" or self.lock_deadline is None:
                return 0.0
            return max(0.0, self.lock_deadline - ts)

"""GD4: Decision state machine (`IDLE -> SUSPECT -> ALERT -> WAIT_ADMIN -> LOCKED -> RELEASED`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

from shoplift_iot.config import AppConfig, load_app_config
from shoplift_iot.contracts import AlertEvent, DoorCommand


class DecisionState(str, Enum):
    """Per-track / per-camera decision states."""

    IDLE = "IDLE"
    SUSPECT = "SUSPECT"
    ALERT = "ALERT"
    WAIT_ADMIN = "WAIT_ADMIN"
    LOCKED = "LOCKED"
    RELEASED = "RELEASED"


@dataclass(slots=True)
class DecisionOutput:
    """Result of updating the decision state machine on a new window score."""

    track_id: int
    state: DecisionState
    consecutive_windows: int
    score: float
    threshold: float
    alert_event: AlertEvent | None = None
    auto_door_cmd: DoorCommand | None = None


@dataclass(slots=True)
class TrackDecisionContext:
    """Mutable state context for a single tracked person (`track_id`)."""

    track_id: int
    state: DecisionState = DecisionState.IDLE
    consecutive_windows: int = 0
    last_score: float = 0.0
    active_event_id: str = ""
    wait_admin_deadline: float | None = None
    lock_deadline: float | None = None
    cooldown_until: float = -1e9
    history: list[DecisionState] = field(default_factory=lambda: [DecisionState.IDLE])

    def transition_to(self, new_state: DecisionState) -> None:
        """Transition to `new_state` and append to transition history if changed."""
        if self.state != new_state:
            self.state = new_state
            self.history.append(new_state)


class DecisionStateMachine:
    """State machine managing `IDLE -> SUSPECT -> ALERT -> WAIT_ADMIN -> LOCKED -> RELEASED` per track."""

    def __init__(
        self,
        threshold: float = 0.5,
        consecutive_k: int = 3,
        cooldown_sec: float = 10.0,
        wait_admin_sec: float = 30.0,
        default_ttl_sec: int = 45,
        auto_lock: bool = False,
        cam_id: str = "cam_01",
    ) -> None:
        if consecutive_k <= 0:
            raise ValueError("consecutive_k must be >= 1.")
        if default_ttl_sec <= 0:
            raise ValueError("default_ttl_sec must be > 0.")

        self.threshold = float(threshold)
        self.consecutive_k = int(consecutive_k)
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.wait_admin_sec = max(1.0, float(wait_admin_sec))
        self.default_ttl_sec = int(default_ttl_sec)
        self.auto_lock = bool(auto_lock)  # Safety constraint #1: MUST default to False
        self.cam_id = str(cam_id)

        self._contexts: dict[int, TrackDecisionContext] = {}

    @classmethod
    def from_config(
        cls,
        config_or_path: Path | str | AppConfig | Mapping[str, Any] | None = None,
        threshold_override: float | None = None,
        k_override: int | None = None,
        cooldown_override: float | None = None,
        wait_admin_override: float | None = None,
        cam_id_override: str | None = None,
    ) -> DecisionStateMachine:
        """Create `DecisionStateMachine` from `configs/default.yaml`."""
        cfg = load_app_config(config_or_path)
        alert_cfg = cfg.alerting
        door_cfg = cfg.door
        mqtt_cfg = cfg.mqtt

        return cls(
            threshold=float(
                threshold_override
                if threshold_override is not None
                else alert_cfg.get("default_threshold", 0.30)
            ),
            consecutive_k=int(
                k_override
                if k_override is not None
                else alert_cfg.get("consecutive_windows_k", 3)
            ),
            cooldown_sec=float(
                cooldown_override
                if cooldown_override is not None
                else alert_cfg.get("cooldown_seconds", 10.0)
            ),
            wait_admin_sec=float(
                wait_admin_override
                if wait_admin_override is not None
                else alert_cfg.get("wait_admin_seconds", 30.0)
            ),
            default_ttl_sec=int(door_cfg.get("default_ttl_sec", 45)),
            auto_lock=bool(alert_cfg.get("auto_lock", False)),
            cam_id=str(cam_id_override or mqtt_cfg.get("cam_id", "cam_01")),
        )

    def get_context(self, track_id: int) -> TrackDecisionContext:
        """Return or create state context for `track_id`."""
        tid = int(track_id)
        if tid not in self._contexts:
            self._contexts[tid] = TrackDecisionContext(track_id=tid)
        return self._contexts[tid]

    def get_state(self, track_id: int) -> DecisionState:
        """Return current `DecisionState` for `track_id`."""
        return self.get_context(track_id).state

    def tick(self, now: float) -> None:
        """Advance time-driven transitions (`ALERT -> WAIT_ADMIN`, `WAIT_ADMIN` timeout, `LOCKED` TTL, `RELEASED -> IDLE`)."""
        ts = float(now)
        for ctx in self._contexts.values():
            if ctx.state == DecisionState.ALERT:
                ctx.transition_to(DecisionState.WAIT_ADMIN)
                if ctx.wait_admin_deadline is None:
                    ctx.wait_admin_deadline = ts + self.wait_admin_sec

            if (
                ctx.state == DecisionState.WAIT_ADMIN
                and ctx.wait_admin_deadline is not None
                and ts >= ctx.wait_admin_deadline
            ):
                # WAIT_ADMIN expired after N seconds without admin lock -> return to IDLE with cooldown
                ctx.wait_admin_deadline = None
                ctx.consecutive_windows = 0
                ctx.cooldown_until = ts + self.cooldown_sec
                ctx.transition_to(DecisionState.IDLE)

            if (
                ctx.state == DecisionState.LOCKED
                and ctx.lock_deadline is not None
                and ts >= ctx.lock_deadline
            ):
                # Door lock TTL expired -> automatically unlock to RELEASED and start cooldown
                ctx.lock_deadline = None
                ctx.consecutive_windows = 0
                ctx.cooldown_until = ts + self.cooldown_sec
                ctx.transition_to(DecisionState.RELEASED)

            if ctx.state == DecisionState.RELEASED and ts >= ctx.cooldown_until:
                ctx.active_event_id = ""
                ctx.transition_to(DecisionState.IDLE)

    def update_score(
        self,
        track_id: int,
        score: float,
        now: float,
        clip_saver: Callable[[str], str] | None = None,
    ) -> DecisionOutput:
        """Process a window anomaly score for `track_id` at time `now`."""
        ts = float(now)
        val = float(score)
        ctx = self.get_context(track_id)
        ctx.last_score = val

        # Advance any pending timers first
        self.tick(ts)

        # While waiting for admin, locked, or in RELEASED/cooldown, do not emit duplicate alerts
        if ctx.state in (DecisionState.WAIT_ADMIN, DecisionState.LOCKED, DecisionState.RELEASED):
            return DecisionOutput(
                track_id=ctx.track_id,
                state=ctx.state,
                consecutive_windows=ctx.consecutive_windows,
                score=val,
                threshold=self.threshold,
            )

        # In IDLE, check if cooldown is still active (e.g., after WAIT_ADMIN timeout)
        if ts < ctx.cooldown_until:
            ctx.consecutive_windows = 0
            return DecisionOutput(
                track_id=ctx.track_id,
                state=ctx.state,
                consecutive_windows=0,
                score=val,
                threshold=self.threshold,
            )

        if val >= self.threshold:
            ctx.consecutive_windows += 1
            if ctx.state == DecisionState.IDLE:
                ctx.transition_to(DecisionState.SUSPECT)

            if ctx.consecutive_windows >= self.consecutive_k:
                ctx.transition_to(DecisionState.ALERT)
                event_id = str(uuid.uuid4())
                ctx.active_event_id = event_id
                ctx.wait_admin_deadline = ts + self.wait_admin_sec

                clip_path = clip_saver(event_id) if clip_saver is not None else f"data/clips/{event_id}.mp4"
                alert_ev = AlertEvent.create(
                    event_id=event_id,
                    cam_id=self.cam_id,
                    track_id=ctx.track_id,
                    score=val,
                    threshold=self.threshold,
                    clip_path=clip_path,
                )

                auto_cmd: DoorCommand | None = None
                if self.auto_lock:
                    ctx.transition_to(DecisionState.WAIT_ADMIN)
                    ctx.transition_to(DecisionState.LOCKED)
                    ctx.wait_admin_deadline = None
                    ctx.lock_deadline = ts + float(self.default_ttl_sec)
                    auto_cmd = DoorCommand(
                        action="lock",
                        reason="auto_lock",
                        event_id=event_id,
                        ttl_sec=self.default_ttl_sec,
                    )

                return DecisionOutput(
                    track_id=ctx.track_id,
                    state=ctx.state,
                    consecutive_windows=ctx.consecutive_windows,
                    score=val,
                    threshold=self.threshold,
                    alert_event=alert_ev,
                    auto_door_cmd=auto_cmd,
                )
        else:
            ctx.consecutive_windows = 0
            if ctx.state == DecisionState.SUSPECT:
                ctx.transition_to(DecisionState.IDLE)

        return DecisionOutput(
            track_id=ctx.track_id,
            state=ctx.state,
            consecutive_windows=ctx.consecutive_windows,
            score=val,
            threshold=self.threshold,
        )

    def on_admin_lock(
        self,
        track_id: int,
        now: float,
        ttl_sec: int | None = None,
        event_id: str | None = None,
    ) -> DoorCommand:
        """Transition track to `LOCKED` when admin confirms lock (with mandatory `ttl_sec`)."""
        ts = float(now)
        ctx = self.get_context(track_id)
        eff_ttl = int(ttl_sec if ttl_sec is not None else self.default_ttl_sec)
        if eff_ttl <= 0:
            raise ValueError("ttl_sec must be positive.")

        if ctx.state == DecisionState.ALERT:
            ctx.transition_to(DecisionState.WAIT_ADMIN)

        ctx.wait_admin_deadline = None
        ctx.lock_deadline = ts + float(eff_ttl)
        if event_id:
            ctx.active_event_id = str(event_id)
        ctx.transition_to(DecisionState.LOCKED)

        return DoorCommand(
            action="lock",
            reason="admin_confirmed",
            event_id=ctx.active_event_id or str(event_id or ""),
            ttl_sec=eff_ttl,
        )

    def on_admin_unlock(
        self,
        track_id: int,
        now: float,
        reason: str = "admin_unlock",
        event_id: str | None = None,
    ) -> DoorCommand:
        """Unlock has strict priority over lock: immediately transition to `RELEASED` and enter cooldown."""
        ts = float(now)
        ctx = self.get_context(track_id)
        ctx.wait_admin_deadline = None
        ctx.lock_deadline = None
        ctx.consecutive_windows = 0
        ctx.cooldown_until = ts + self.cooldown_sec
        ctx.transition_to(DecisionState.RELEASED)

        return DoorCommand(
            action="unlock",
            reason=reason,
            event_id=str(event_id or ctx.active_event_id or ""),
            ttl_sec=0,
        )

    def apply_door_command(
        self,
        cmd: DoorCommand | Mapping[str, Any],
        now: float,
        track_id: int | None = None,
    ) -> DecisionState:
        """Apply an incoming `DoorCommand` (`unlock` always overrides `lock`)."""
        action = cmd.action if isinstance(cmd, DoorCommand) else str(cmd["action"])
        event_id = cmd.event_id if isinstance(cmd, DoorCommand) else str(cmd.get("event_id", ""))
        ttl_sec = cmd.ttl_sec if isinstance(cmd, DoorCommand) else int(cmd.get("ttl_sec", self.default_ttl_sec))

        target_tid = track_id
        if target_tid is None:
            for tid, c in self._contexts.items():
                if event_id and c.active_event_id == event_id:
                    target_tid = tid
                    break
        if target_tid is None and self._contexts:
            target_tid = next(iter(self._contexts.keys()))
        if target_tid is None:
            target_tid = 0

        if action == "unlock":
            self.on_admin_unlock(target_tid, now=now, event_id=event_id)
        elif action == "lock":
            self.on_admin_lock(target_tid, now=now, ttl_sec=ttl_sec, event_id=event_id)
        else:
            raise ValueError(f"Unsupported door action: {action}")
        return self.get_state(target_tid)

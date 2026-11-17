"""Unit and integration tests for GD4: `DecisionStateMachine` and `demo_videomae.py`."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_videomae import run_videomae_demo  # noqa: E402
from shoplift_iot.alerting.state_machine import DecisionState, DecisionStateMachine  # noqa: E402
from shoplift_iot.messaging.mqtt_pub import REQUIRED_KEYS  # noqa: E402


class _MockSinglePersonYOLO:
    """Mock YOLO tracker emitting a single persistent person (`track_id=5`)."""

    def track(self, source: np.ndarray, **kwargs: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            id=torch.tensor([5], dtype=torch.int32),
            xywh=torch.tensor([[160.0, 120.0, 80.0, 160.0]], dtype=torch.float32),
            conf=torch.tensor([0.95], dtype=torch.float32),
            cls=torch.tensor([0], dtype=torch.int32),
        )
        keypoints = SimpleNamespace(data=torch.ones((1, 17, 3), dtype=torch.float32))
        return [SimpleNamespace(boxes=boxes, keypoints=keypoints)]


class _MockVideoMAEPredictor:
    """Mock VideoMAE Predictor returning high probability across windows of a single event."""

    device = "cpu"
    threshold = 0.6
    clip_sec = 1.5
    n_frames = 16
    size = 224

    def predict(self, frames_rgb: np.ndarray) -> float:
        assert frames_rgb.shape == (16, 224, 224, 3)
        assert frames_rgb.dtype == np.uint8
        return 0.85


def test_state_machine_full_lifecycle_ttl_cooldown_and_unlock_priority() -> None:
    """Verify IDLE -> SUSPECT -> ALERT -> WAIT_ADMIN -> LOCKED -> RELEASED -> IDLE, TTL, cooldown, and unlock priority."""
    sm = DecisionStateMachine.from_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        threshold_override=0.6,
        k_override=3,
        cooldown_override=10.0,
        wait_admin_override=20.0,
    )
    # Safety constraint #1: auto_lock MUST default to False
    assert sm.auto_lock is False
    assert sm.get_state(1) == DecisionState.IDLE

    # 1. Window 1 >= 0.6 -> SUSPECT (1/3)
    out1 = sm.update_score(track_id=1, score=0.75, now=1.0)
    assert out1.state == DecisionState.SUSPECT
    assert out1.alert_event is None

    # Drop below threshold -> resets back to IDLE
    out_reset = sm.update_score(track_id=1, score=0.20, now=2.0)
    assert out_reset.state == DecisionState.IDLE
    assert out_reset.consecutive_windows == 0

    # 2. K=3 consecutive windows >= 0.6 -> SUSPECT -> ALERT
    assert sm.update_score(track_id=1, score=0.70, now=3.0).state == DecisionState.SUSPECT
    assert sm.update_score(track_id=1, score=0.80, now=4.0).state == DecisionState.SUSPECT
    out_alert = sm.update_score(track_id=1, score=0.90, now=5.0)
    assert out_alert.state == DecisionState.ALERT
    assert out_alert.alert_event is not None
    assert out_alert.auto_door_cmd is None  # auto_lock is False!
    assert tuple(out_alert.alert_event.to_dict().keys()) == REQUIRED_KEYS

    # 3. Next tick transitions ALERT -> WAIT_ADMIN and suppresses duplicate alerts
    out_wait = sm.update_score(track_id=1, score=0.95, now=6.0)
    assert out_wait.state == DecisionState.WAIT_ADMIN
    assert out_wait.alert_event is None

    # 4. Admin locks door with ttl_sec=45 -> LOCKED
    lock_cmd = sm.on_admin_lock(track_id=1, now=10.0, ttl_sec=45)
    assert sm.get_state(1) == DecisionState.LOCKED
    assert lock_cmd.to_dict() == {
        "action": "lock",
        "reason": "admin_confirmed",
        "event_id": out_alert.alert_event.event_id,
        "ttl_sec": 45,
    }

    # Before TTL expires (t=54.0 < 10.0 + 45.0), still LOCKED
    sm.tick(54.0)
    assert sm.get_state(1) == DecisionState.LOCKED

    # 5. At t=55.0 (TTL=45s expired) -> automatically transitions LOCKED -> RELEASED and starts 10s cooldown
    sm.tick(55.0)
    assert sm.get_state(1) == DecisionState.RELEASED

    # During cooldown (t=60.0 < 55.0 + 10.0), stays in RELEASED and ignores high scores
    out_cool = sm.update_score(track_id=1, score=0.99, now=60.0)
    assert out_cool.state == DecisionState.RELEASED
    assert out_cool.alert_event is None

    # 6. After cooldown expires (t=65.1 >= 65.0) -> returns to IDLE
    sm.tick(65.1)
    assert sm.get_state(1) == DecisionState.IDLE

    ctx = sm.get_context(1)
    assert DecisionState.IDLE in ctx.history
    assert DecisionState.SUSPECT in ctx.history
    assert DecisionState.ALERT in ctx.history
    assert DecisionState.WAIT_ADMIN in ctx.history
    assert DecisionState.LOCKED in ctx.history
    assert DecisionState.RELEASED in ctx.history


def test_wait_admin_timeout_and_unlock_priority() -> None:
    """Verify WAIT_ADMIN returns to IDLE after N seconds and unlock always overrides lock."""
    sm = DecisionStateMachine(
        threshold=0.5,
        consecutive_k=2,
        cooldown_sec=5.0,
        wait_admin_sec=15.0,
        default_ttl_sec=45,
        auto_lock=False,
    )
    sm.update_score(track_id=2, score=0.8, now=1.0)
    sm.update_score(track_id=2, score=0.8, now=2.0)  # ALERT at t=2.0, wait_admin_deadline=17.0
    sm.tick(3.0)
    assert sm.get_state(2) == DecisionState.WAIT_ADMIN

    # Timeout after 15s (t=17.5) without admin action -> returns to IDLE
    sm.tick(17.5)
    assert sm.get_state(2) == DecisionState.IDLE

    # Trigger another alert on track 3 and lock it, then verify unlock overrides remaining TTL immediately
    sm.update_score(track_id=3, score=0.9, now=20.0)
    sm.update_score(track_id=3, score=0.9, now=21.0)
    sm.on_admin_lock(track_id=3, now=22.0, ttl_sec=60)
    assert sm.get_state(3) == DecisionState.LOCKED

    unlock_cmd = sm.on_admin_unlock(track_id=3, now=25.0)
    assert unlock_cmd.action == "unlock"
    assert sm.get_state(3) == DecisionState.RELEASED


def test_demo_videomae_with_person_clips_emits_single_alert(tmp_path: Path) -> None:
    """Verify `demo_videomae.py` connected with `person_clips` emits exactly 1 alert for 1 event (no duplicates)."""
    vid_path = tmp_path / "suspect_scene.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    for i in range(90):  # 6.0 seconds at 15 FPS -> multiple overlapping windows
        frm = np.full((240, 320, 3), fill_value=(i * 3) % 255, dtype=np.uint8)
        writer.write(frm)
    writer.release()

    clips_out = tmp_path / "clips_out"
    alerts = run_videomae_demo(
        source=vid_path,
        predictor=_MockVideoMAEPredictor(),
        yolo_instance=_MockSinglePersonYOLO(),
        hop_sec=0.5,
        threshold=0.6,
        k=2,
        cooldown=10.0,
        wait_admin_sec=30.0,
        no_mqtt=True,
        clips_dir=clips_out,
    )
    # Even though 9+ windows are evaluated across the 6s video, state machine + WAIT_ADMIN + cooldown
    # ensures 1 event produces EXACTLY 1 alert (no duplicate spam)!
    assert len(alerts) == 1
    ev = alerts[0]
    assert ev.track_id == 5
    payload = ev.to_dict()
    assert tuple(payload.keys()) == REQUIRED_KEYS
    assert len(list(clips_out.glob("*.mp4"))) == 1


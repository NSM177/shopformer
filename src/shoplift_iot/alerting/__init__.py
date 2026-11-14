"""Alerting subpackage: state machine, temporal score smoothing, K-consecutive threshold, cooldown, and clip recorder."""

from shoplift_iot.alerting.alerting import AlertEvent, AlertManager
from shoplift_iot.alerting.clip_recorder import ClipRecorder
from shoplift_iot.alerting.state_machine import (
    DecisionOutput,
    DecisionState,
    DecisionStateMachine,
    TrackDecisionContext,
)

__all__ = [
    "AlertEvent",
    "AlertManager",
    "ClipRecorder",
    "DecisionOutput",
    "DecisionState",
    "DecisionStateMachine",
    "TrackDecisionContext",
]

"""Unit tests for Stage D: ShopformerScorer, evaluation metrics, and threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_threshold import (  # noqa: E402
    calibrate_threshold,
    calibrate_threshold_from_scores,
)
from shoplift_iot.model.scorer import ShopformerScorer  # noqa: E402


def test_shopformer_scorer_single_and_batch() -> None:
    """Verify ShopformerScorer loads checkpoints and returns non-negative MSE scores."""
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    scorer = ShopformerScorer.from_config(cfg_path, project_root=PROJECT_ROOT, device_override="cpu")

    single_win = np.random.randn(2, 12, 18).astype(np.float32)
    single_score = scorer.score_window(single_win)
    assert isinstance(single_score, float)
    assert single_score >= 0.0

    batch_win = np.random.randn(5, 2, 12, 18).astype(np.float32)
    batch_scores = scorer.score_batch(batch_win)
    assert batch_scores.shape == (5,)
    assert np.all(batch_scores >= 0.0)


def test_calibrate_threshold_from_scores_math() -> None:
    """Verify percentile threshold calculation on synthetic score distribution."""
    scores = np.linspace(0.01, 1.00, 100, dtype=np.float64)
    stats = calibrate_threshold_from_scores(scores, percentile=99.0)
    assert stats["num_windows"] == 100
    assert np.isclose(stats["threshold"], np.percentile(scores, 99.0))
    assert stats["p50"] < stats["p90"] < stats["p95"] < stats["p99"]


def test_calibrate_threshold_end_to_end_with_extra_dir() -> None:
    """Verify calibrate_threshold runs on PoseLift train set and writes threshold.json."""
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    extra_dir = PROJECT_ROOT / "data" / "extracted_samples"
    stats = calibrate_threshold(
        config_path=cfg_path,
        project_root=PROJECT_ROOT,
        percentile_override=99.0,
        extra_normal_dir=extra_dir if extra_dir.exists() else None,
    )
    th_path = PROJECT_ROOT / "checkpoints" / "threshold.json"
    assert th_path.exists()
    saved = json.loads(th_path.read_text(encoding="utf-8"))
    assert saved["percentile"] == 99.0
    assert saved["threshold"] == stats["threshold"]
    assert saved["num_windows"] > 1000


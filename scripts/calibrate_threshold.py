"""Calibrate anomaly score threshold using a configurable percentile (default 99) on normal clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shoplift_iot.model.scorer import PoseSegDataset, ShopformerScorer  # noqa: E402


def compute_normal_scores(
    scorer: ShopformerScorer,
    pose_dir: Path,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Compute window-level MSE scores for all normal pose sequences in `pose_dir`."""
    prep_cfg = cfg["preprocess"]
    vid_cfg = cfg["video"]
    ds = PoseSegDataset(
        str(pose_dir),
        normalize_pose_segs=True,
        evaluate=True,
        seg_stride=int(prep_cfg["window_stride_train"]),
        trans_list=None,
        headless=bool(cfg["model"].get("headless", False)),
        seg_len=int(prep_cfg["window_size"]),
        dataset="Poselift",
        train_seg_conf_th=0.0,
        vid_res=list(vid_cfg["vid_res"]),
        symm_range=bool(prep_cfg.get("symm_range", False)),
        swap_yx=bool(vid_cfg.get("swap_yx_from_poselift_json", True)),
        relative=False,
        traj=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    all_scores: list[np.ndarray] = []
    for batch in loader:
        data = batch[0][:, 0:2, :, :]
        scores = scorer.score_batch(data)
        all_scores.append(scores)

    if not all_scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(all_scores, axis=0)


def calibrate_threshold_from_scores(
    scores: np.ndarray,
    percentile: float = 99.0,
) -> dict[str, float | int]:
    """Compute percentile threshold and summary statistics from an array of normal scores."""
    if scores.size == 0:
        raise ValueError("Cannot calibrate threshold on empty score array.")
    if not (0.0 < percentile <= 100.0):
        raise ValueError("percentile must be in (0, 100].")

    threshold = float(np.percentile(scores, percentile))
    return {
        "percentile": float(percentile),
        "threshold": threshold,
        "num_windows": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p50": float(np.percentile(scores, 50.0)),
        "p90": float(np.percentile(scores, 90.0)),
        "p95": float(np.percentile(scores, 95.0)),
        "p99": float(np.percentile(scores, 99.0)),
        "max": float(np.max(scores)),
    }


def calibrate_threshold(
    config_path: Path,
    project_root: Path | None = None,
    percentile_override: float | None = None,
    extra_normal_dir: Path | None = None,
) -> dict[str, float | int]:
    """Run Shopformer on normal clips and save calibrated threshold to JSON."""
    root = project_root or PROJECT_ROOT
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    percentile = (
        float(percentile_override)
        if percentile_override is not None
        else float(cfg["alerting"]["calibration_percentile"])
    )
    scorer = ShopformerScorer.from_config(config_path, project_root=root)

    normal_dir = root / cfg["paths"]["pose_train_dir"]
    scores_list = [compute_normal_scores(scorer, normal_dir, cfg)]

    if extra_normal_dir is not None and extra_normal_dir.exists():
        extra_scores = compute_normal_scores(scorer, extra_normal_dir, cfg)
        if extra_scores.size > 0:
            scores_list.append(extra_scores)

    all_normal_scores = np.concatenate(scores_list, axis=0)
    stats = calibrate_threshold_from_scores(all_normal_scores, percentile=percentile)

    out_path = root / cfg["paths"]["threshold_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Threshold Calibration Complete (Percentile = {percentile:.1f}%)")
    print("=" * 60)
    print(f"Normal windows evaluated : {stats['num_windows']}")
    print(f"Score Mean ± Std         : {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"Score Range [Min, Max]   : [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(
        f"P50 / P90 / P95 / P99    : {stats['p50']:.6f} / {stats['p90']:.6f} / {stats['p95']:.6f} / {stats['p99']:.6f}"
    )
    print(f"Selected Threshold       : {stats['threshold']:.6f}")
    print(f"Saved to                 : {out_path}")
    print("=" * 60)
    return stats


def main() -> None:
    """CLI entrypoint for threshold calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate Shopformer anomaly score threshold on normal pose sequences."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Override percentile (default from config: 99.0).",
    )
    parser.add_argument(
        "--extra-normal-dir",
        type=Path,
        default=None,
        help="Optional directory with additional normal PoseLift JSON files.",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    calibrate_threshold(
        config_path=cfg_path,
        project_root=PROJECT_ROOT,
        percentile_override=args.percentile,
        extra_normal_dir=args.extra_normal_dir,
    )


if __name__ == "__main__":
    main()

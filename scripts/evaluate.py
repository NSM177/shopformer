"""Evaluate Shopformer on the PoseLift test dataset (AUC-ROC, AUC-PR, EER)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scorer import (  # noqa: E402
    PoseSegDataset,
    ShopformerScorer,
    score_dataset,
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_test_dataset(cfg: dict[str, Any], project_root: Path) -> PoseSegDataset:
    """Create the PoseLift test dataset from YAML configuration."""
    paths_cfg = cfg["paths"]
    prep_cfg = cfg["preprocess"]
    vid_cfg = cfg["video"]
    test_json_dir = project_root / paths_cfg["pose_test_dir"]
    return PoseSegDataset(
        str(test_json_dir),
        normalize_pose_segs=True,
        evaluate=True,
        seg_stride=int(prep_cfg["window_stride_eval"]),
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


def evaluate_poselift(
    config_path: Path,
    project_root: Path,
    device_override: str | None = None,
) -> dict[str, float]:
    """Run Shopformer evaluation on PoseLift test set and return metrics."""
    cfg = load_config(config_path)
    set_seed(int(cfg["training"]["seed"]))

    scorer = ShopformerScorer.from_config(
        config_path=config_path,
        project_root=project_root,
        device_override=device_override,
    )
    print(f"[INFO] Loaded ShopformerScorer on device '{scorer.device}'")

    test_ds = build_test_dataset(cfg, project_root)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    all_scores: list[np.ndarray] = []
    for batch in test_loader:
        data = batch[0][:, 0:2, :, :]
        all_scores.append(scorer.score_batch(data))
    scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=np.float64)

    eval_args = argparse.Namespace(
        mask_root=str(project_root / cfg["paths"]["gt_mask_dir"]),
        dataset="Poselift",
        seg_len=int(cfg["preprocess"]["window_size"]),
        save_results=False,
        save_results_dir="",
    )

    auc_roc, auc_pr, eer, eer_th, fpr_10er, th_10er = score_dataset(
        scores, test_ds.metadata, args=eval_args
    )

    paper = cfg["evaluation"]["paper_metrics"]
    results = {
        "auc_roc": float(auc_roc * 100.0),
        "auc_pr": float(auc_pr * 100.0),
        "eer": float(eer),
        "eer_threshold": float(eer_th),
        "fpr_at_10fnr": float(fpr_10er),
        "threshold_at_10fnr": float(th_10er),
        "paper_auc_roc": float(paper["auc_roc"]),
        "paper_auc_pr": float(paper["auc_pr"]),
        "paper_eer": float(paper["eer"]),
    }

    print("\n" + "=" * 68)
    print("PoseLift Test Set Evaluation Results (Shopformer)")
    print("=" * 68)
    print(f"{'Metric':<18} | {'Ours (Measured)':<18} | {'Paper (CVPR 2025)':<18}")
    print("-" * 68)
    print(
        f"{'AUC-ROC (%)':<18} | {results['auc_roc']:<18.2f} | {results['paper_auc_roc']:<18.2f}"
    )
    print(f"{'AUC-PR (%)':<18} | {results['auc_pr']:<18.2f} | {results['paper_auc_pr']:<18.2f}")
    print(f"{'EER':<18} | {results['eer']:<18.4f} | {results['paper_eer']:<18.2f}")
    print("-" * 68)
    print(f"EER Threshold: {results['eer_threshold']:.6f}")
    print("=" * 68)

    metrics_out = project_root / cfg["paths"]["checkpoints_dir"] / "eval_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    """CLI entrypoint for PoseLift evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Shopformer on PoseLift test set.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda:0).")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    evaluate_poselift(
        config_path=config_path, project_root=PROJECT_ROOT, device_override=args.device
    )


if __name__ == "__main__":
    main()

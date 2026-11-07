"""Unit tests for Stage B: environment, PoseLift dataset, and Shopformer evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import build_test_dataset, load_config  # noqa: E402
from src.preprocess import keypoints17_to_coco18, normalize_pose  # noqa: E402
from src.scorer import (  # noqa: E402
    Encoder_T,
    build_Models_from_config,
    score_auc,
    tokenize_batch,
)


def test_config_and_poselift_files_exist() -> None:
    """Verify default YAML config and PoseLift files exist on disk."""
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    train_dir = PROJECT_ROOT / cfg["paths"]["pose_train_dir"]
    test_dir = PROJECT_ROOT / cfg["paths"]["pose_test_dir"]
    gt_dir = PROJECT_ROOT / cfg["paths"]["gt_mask_dir"]

    assert len(list(train_dir.glob("*.json"))) >= 100
    assert len(list(test_dir.glob("*.json"))) >= 40
    assert len(list(gt_dir.glob("*.npy"))) >= 40


def test_keypoints17_to_coco18_and_normalize() -> None:
    """Verify COCO17 -> OpenPose COCO18 conversion and Shopformer normalization."""
    dummy_17 = np.random.uniform(100.0, 800.0, size=(4, 12, 17, 3)).astype(np.float32)
    dummy_17[..., 2] = 0.9
    coco18 = keypoints17_to_coco18(dummy_17)
    assert coco18.shape == (4, 12, 18, 3)
    expected_neck = 0.5 * (dummy_17[..., 5, :] + dummy_17[..., 6, :])
    np.testing.assert_allclose(coco18[..., 1, :], expected_neck, rtol=1e-5)

    normed = normalize_pose(coco18.copy(), vid_res=[1920, 1080], symm_range=False)
    assert normed.shape == (4, 12, 18, 3)
    assert not np.isnan(normed).any()


def test_shopformer_gcae_and_transformer_shapes() -> None:
    """Verify GCAE produces 2 tokens of dim 144 and Transformer reconstructs them."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    gcae, transformer = build_Models_from_config(cfg, device="cpu")
    gcae.eval()
    transformer.eval()
    encoder = Encoder_T(gcae)

    dummy_input = torch.randn(3, 2, 12, 18)
    with torch.no_grad():
        reco_pose = gcae(dummy_input)
        if reco_pose.ndim == 5 and reco_pose.shape[-1] == 1:
            reco_pose = reco_pose.squeeze(-1)
        assert reco_pose.shape == (3, 2, 12, 18)

        tokens = tokenize_batch(encoder, dummy_input)
        assert tokens.shape == (3, 2, 144)

        recon_tokens = transformer(tokens, tokens)
        assert recon_tokens.shape == (3, 2, 144)


def test_poselift_test_dataset_and_score_auc() -> None:
    """Verify PoseLift test dataset loads windows of shape (3, 12, 18) and metrics compute."""
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    test_ds = build_test_dataset(cfg, PROJECT_ROOT)
    assert len(test_ds) > 100
    sample = test_ds[0][0]
    assert sample.shape == (3, 12, 18)

    gt = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85], dtype=np.float64)
    auc_roc, auc_pr, eer, _, _, _ = score_auc(scores, gt)
    assert 0.99 <= auc_roc <= 1.0
    assert 0.99 <= auc_pr <= 1.0
    assert 0.0 <= eer <= 0.01

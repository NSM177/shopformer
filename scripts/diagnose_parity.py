#!/usr/bin/env python3
"""Script kiem tra parity giua offline reference (score_video_bundle.py)
va live pipeline simulation (service.py).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "models" / "videomae_bundle")):
    if p not in sys.path:
        sys.path.insert(0, p)

from crops_lite import render_windows
from score_video_bundle import run_score_video_bundle
from shoplift_iot.config import load_app_config
from shoplift_iot.vision.pose_track import PoseTracker
from shoplift_iot.vision.person_clips import PersonClipBuffer, PersonClipExtractor
from videomae_lite import Predictor, to_tensor


def run_live_simulation(
    video_path: Path | str,
    bundle_dir: Path | str = "models/videomae_bundle",
    out_csv: Path | str | None = None,
    save_first_grids: int = 3,
    grids_dir: Path | str | None = None,
) -> tuple[list[dict], list[np.ndarray], dict]:
    """Mo phong chinh xac cach live pipeline (service.py) xu ly video."""
    vpath = Path(video_path)
    bpath = Path(bundle_dir)
    cfg = load_app_config(PROJECT_ROOT / "configs" / "default.yaml")

    predictor = Predictor(bpath, device="cpu")
    pred_meta = getattr(predictor, "meta", {}) or {}
    pc_cfg = cfg.person_clips

    # Configs as used in service.py
    eff_clip_sec = float(getattr(predictor, "clip_sec", pc_cfg.get("clip_sec", 4.0)))
    eff_n_frames = int(getattr(predictor, "n_frames", pc_cfg.get("num_frames", 16)))
    eff_size = int(getattr(predictor, "size", pc_cfg.get("frame_size", 224)))
    eff_hop = float(pc_cfg.get("hop_sec", pred_meta.get("hop_sec", 0.5)))
    eff_min_span = float(pc_cfg.get("min_span_ratio", 0.8))
    eff_margin = float(pred_meta.get("crop_margin", pc_cfg.get("margin_ratio", 0.18)))
    eff_square = bool(pred_meta.get("crop_square", pc_cfg.get("square", True)))

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong the mo video: {vpath}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames > 0 and fps > 0:
        video_sec = total_frames / fps
        if video_sec < eff_clip_sec:
            eff_clip_sec = max(0.8, round(video_sec * 0.92, 2))
            eff_min_span = 0.60
            eff_hop = max(0.15, round((video_sec * 0.32) / 2.0, 2))

    # PoseTracker as used in service.py
    pt_cfg = cfg.pose_track
    tracker = PoseTracker(
        model_name=pt_cfg.get("yolo_model", "models/yolov8s.pt"),
        tracker_config=pt_cfg.get("tracker_config", "configs/bytetrack_cctv.yaml"),
        conf_threshold=float(pt_cfg.get("conf_threshold", 0.15)),
        iou_threshold=float(pt_cfg.get("iou_threshold", 0.45)),
        device="cpu",
    )
    tracker.reset()

    clip_buf = PersonClipBuffer(
        clip_sec=eff_clip_sec,
        hop_sec=eff_hop,
        n_frames=eff_n_frames,
        size=eff_size,
        margin_ratio=eff_margin,
        min_span_ratio=eff_min_span,
        min_frames=min(int(pc_cfg.get("min_frames", 16)), max(8, total_frames // 2)) if total_frames > 0 else int(pc_cfg.get("min_frames", 16)),
        stale_sec=float(pc_cfg.get("stale_sec", 5.0)),
        square=eff_square,
    )
    extractor = PersonClipExtractor(tracker=tracker, clip_buffer=clip_buf)

    if grids_dir:
        Path(grids_dir).mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    saved_grids: list[np.ndarray] = []
    first_window_debug: dict = {}

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t = idx / fps
        idx += 1

        poses = extractor.tracker.process_frame(frame)
        windows = extractor.clip_buffer.update(t, frame, poses)

        for win in windows:
            # Predict raw p
            raw_p = float(predictor.predict(win.frames_rgb))
            res = {
                "track_id": win.track_id,
                "t0": round(win.start_ts, 2),
                "t1": round(win.end_ts, 2),
                "p": round(raw_p, 6),
            }
            results.append(res)

            if len(saved_grids) < save_first_grids:
                saved_grids.append(win.frames_rgb)
                if grids_dir:
                    g_path = Path(grids_dir) / f"live_win_{len(saved_grids)-1}_tid{win.track_id}_t0_{win.start_ts:.2f}.png"
                    render_windows(
                        win.frames_rgb,
                        g_path,
                        title=f"LIVE Win {len(saved_grids)-1} | Track #{win.track_id} | t0={win.start_ts:.2f}s | p={raw_p:.4f}",
                    )

            if not first_window_debug:
                with torch.no_grad():
                    tx = to_tensor(win.frames_rgb, predictor.mean, predictor.std).unsqueeze(0).to(predictor.device)
                    logits = predictor.model(pixel_values=tx).logits[0].cpu().numpy().tolist()
                    first_window_debug = {
                        "tensor_min": float(tx.min()),
                        "tensor_max": float(tx.max()),
                        "tensor_mean": float(tx.mean()),
                        "tensor_std": float(tx.std()),
                        "logits": logits,
                        "raw_p": raw_p,
                    }
    cap.release()

    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "t0", "p"])
            writer.writeheader()
            for r in results:
                writer.writerow({"track_id": r["track_id"], "t0": r["t0"], "p": r["p"]})

    return results, saved_grids, first_window_debug


def get_first_window_ref_debug(bundle_dir: Path | str, frames_rgb: np.ndarray) -> dict:
    predictor = Predictor(bundle_dir, device="cpu")
    with torch.no_grad():
        tx = to_tensor(frames_rgb, predictor.mean, predictor.std).unsqueeze(0).to(predictor.device)
        logits = predictor.model(pixel_values=tx).logits[0].cpu().numpy().tolist()
        raw_p = float(torch.softmax(torch.tensor([logits]), -1)[0, predictor.pos])
        return {
            "tensor_min": float(tx.min()),
            "tensor_max": float(tx.max()),
            "tensor_mean": float(tx.mean()),
            "tensor_std": float(tx.std()),
            "logits": logits,
            "raw_p": raw_p,
        }


def compare_scores(ref_csv: Path, live_csv: Path) -> dict:
    with open(ref_csv, encoding="utf-8") as f:
        ref_rows = list(csv.DictReader(f))
    with open(live_csv, encoding="utf-8") as f:
        live_rows = list(csv.DictReader(f))

    # Match by track_id and nearest t0
    deltas = []
    comparisons = []
    for r in ref_rows:
        rtid = int(r["track_id"])
        rt0 = float(r["t0"])
        rp = float(r["p"])

        # find best match in live
        matching_live = [
            l for l in live_rows
            if int(l["track_id"]) == rtid and abs(float(l["t0"]) - rt0) <= 0.6
        ]
        if matching_live:
            best_l = min(matching_live, key=lambda l: abs(float(l["t0"]) - rt0))
            lp = float(best_l["p"])
            lt0 = float(best_l["t0"])
            delta = abs(rp - lp)
            deltas.append(delta)
            comparisons.append({
                "track_id": rtid,
                "ref_t0": rt0,
                "live_t0": lt0,
                "ref_p": rp,
                "live_p": lp,
                "delta": delta,
            })
        else:
            comparisons.append({
                "track_id": rtid,
                "ref_t0": rt0,
                "live_t0": None,
                "ref_p": rp,
                "live_p": None,
                "delta": None,
            })

    comparisons.sort(key=lambda x: (x["delta"] is None, -(x["delta"] or 0.0)))
    return {
        "n_ref": len(ref_rows),
        "n_live": len(live_rows),
        "n_matched": len(deltas),
        "max_delta": max(deltas) if deltas else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "comparisons": comparisons,
    }


def main():
    bundle_dir = PROJECT_ROOT / "models" / "videomae_bundle"
    diag_dir = PROJECT_ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        ("theft_overhead_cctv", PROJECT_ROOT / "data" / "theft_overhead_cctv.mp4"),
        ("demo", PROJECT_ROOT / "data" / "demo.mp4"),
    ]

    for name, vpath in videos:
        print(f"\n========================================================")
        print(f"DANH GIA PARITY CHO: {name} ({vpath.name})")
        print(f"========================================================")

        ref_csv = diag_dir / f"{name}_ref_scores.csv"
        live_csv = diag_dir / f"{name}_live_scores.csv"
        ref_grids_dir = diag_dir / f"grids_{name}_ref"
        live_grids_dir = diag_dir / f"grids_{name}_live"

        print("--> Chay Offline Reference (score_video_bundle.py)...")
        ref_results, ref_grids = run_score_video_bundle(
            video_path=vpath,
            bundle_dir=bundle_dir,
            out_csv=ref_csv,
            save_first_grids=3,
            grids_dir=ref_grids_dir,
        )

        print("--> Chay Live Simulation (service.py logic)...")
        live_results, live_grids, live_dbg = run_live_simulation(
            video_path=vpath,
            bundle_dir=bundle_dir,
            out_csv=live_csv,
            save_first_grids=3,
            grids_dir=live_grids_dir,
        )

        cmp_res = compare_scores(ref_csv, live_csv)
        print(f"\n[Ket qua so sanh {name}]")
        print(f"  So cua so Reference: {cmp_res['n_ref']}")
        print(f"  So cua so Live     : {cmp_res['n_live']}")
        print(f"  So cua so khop     : {cmp_res['n_matched']}")
        print(f"  Chenh lech p lon nhat: {cmp_res['max_delta']:.6f}")
        print(f"  Chenh lech p trung binh: {cmp_res['mean_delta']:.6f}")

        print("\n  Cac cua so lech nhieu nhat:")
        for row in cmp_res["comparisons"][:5]:
            print(f"    Track #{row['track_id']} | ref t0={row['ref_t0']:.2f}s (p={row['ref_p']:.4f}) vs live t0={row['live_t0']} (p={row['live_p']}) -> Delta: {row['delta']}")

        # Pixel difference for first 3 grids
        print("\n  Sai khac pixel trung binh (Mean Absolute Error) tren 3 cua so dau:")
        n_common_grids = min(len(ref_grids), len(live_grids))
        for g_i in range(n_common_grids):
            rg = ref_grids[g_i].astype(np.float32)
            lg = live_grids[g_i].astype(np.float32)
            mae = float(np.mean(np.abs(rg - lg)))
            print(f"    Grid {g_i}: MAE = {mae:.2f} / 255 (shape: {rg.shape})")

        # First window debug stats
        if ref_grids:
            ref_dbg = get_first_window_ref_debug(bundle_dir, ref_grids[0])
            print("\n  [Debug Cua so dau tien - Window 0]")
            print(f"    Reference Tensor: min={ref_dbg['tensor_min']:.4f}, max={ref_dbg['tensor_max']:.4f}, mean={ref_dbg['tensor_mean']:.4f}, std={ref_dbg['tensor_std']:.4f}")
            print(f"    Reference Logits: {ref_dbg['logits']} -> p={ref_dbg['raw_p']:.6f}")
            print(f"    Live Tensor     : min={live_dbg.get('tensor_min', 0):.4f}, max={live_dbg.get('tensor_max', 0):.4f}, mean={live_dbg.get('tensor_mean', 0):.4f}, std={live_dbg.get('tensor_std', 0):.4f}")
            print(f"    Live Logits     : {live_dbg.get('logits', [])} -> p={live_dbg.get('raw_p', 0):.6f}")


if __name__ == "__main__":
    main()

"""
Verify training dataset có bị hallucination case nào không.

Dataset: /Users/admin/workspace/diffusion-model-hallucination/simple-datasets/simple-shapes-16x16

Usage:
    python verify_dataset.py
    python verify_dataset.py --data_dir /path/to/dataset --out_dir dataset_verify
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from hallucination_detector import (
    analyze_image, summarize, COLUMN_NAMES, COLUMN_SLICES,
)

DEFAULT_DATA_DIR = os.path.join(
    REPO_ROOT, "..", "diffusion-model-hallucination",
    "simple-datasets", "simple-shapes-16x16",
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_output", "dataset_verify")

SCALE = 8   # 16→128px để nhìn rõ


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """→ (16, 16, 3) uint8 RGB."""
    return np.array(Image.open(path).convert("RGB"))


def add_red_dividers(img_uint8: np.ndarray, scale: int = SCALE) -> torch.Tensor:
    """(16,16,3) uint8 → [3, H*scale, W*scale] float [0,1] với cột đỏ."""
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
    for x in [5 * scale, 10 * scale]:
        t[0, :, x:x+2] = 1.0
        t[1, :, x:x+2] = 0.0
        t[2, :, x:x+2] = 0.0
    return t


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load all images ──
    png_files = sorted([
        f for f in os.listdir(args.data_dir) if f.lower().endswith(".png")
    ])
    if not png_files:
        print(f"Không tìm thấy file PNG trong {args.data_dir}")
        sys.exit(1)

    n_total = len(png_files)
    print(f"Dataset: {args.data_dir}")
    print(f"Total  : {n_total} images")
    print(f"Scanning", end="", flush=True)

    images   : list[np.ndarray] = []
    results  : list[dict]       = []
    filenames: list[str]        = []

    for i, fname in enumerate(png_files):
        img = load_image(os.path.join(args.data_dir, fname))
        r   = analyze_image(img)
        images.append(img)
        results.append(r)
        filenames.append(fname)
        if (i + 1) % 500 == 0:
            print(".", end="", flush=True)

    print(f" done.\n")

    # ── Summary ──
    s = summarize(results)

    print("=" * 55)
    print(f"DATASET HALLUCINATION REPORT")
    print("=" * 55)
    print(f"Total images   : {s['n_total']}")
    print(f"Hallucinations : {s['n_hall']}  ({100*s['hall_rate']:.3f}%)")
    print(f"  ├─ empty image (0 shapes)  : {s['n_empty']}")
    print(f"  └─ double col (2+ in 1 col): {s['n_double_col']}")
    print(f"Normal         : {s['n_normal']}  ({100*(1-s['hall_rate']):.3f}%)")
    print()
    print("Per-column blob distribution:")
    print(f"  {'column':10s}  {'0 shapes':>10}  {'1 shape':>10}  {'2+ shapes':>10}")
    for name in COLUMN_NAMES:
        cc = s["col_counts"][name]
        print(f"  {name:10s}  {cc['0']:>10}  {cc['1']:>10}  {cc['2+']:>10}")
    print("=" * 55)

    # ── Save stats ──
    stats_path = os.path.join(args.out_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Dataset : {args.data_dir}\n")
        f.write(f"Total   : {s['n_total']}\n")
        f.write(f"Hall.   : {s['n_hall']}  ({100*s['hall_rate']:.3f}%)\n")
        f.write(f"  empty      : {s['n_empty']}\n")
        f.write(f"  double_col : {s['n_double_col']}\n")
        f.write(f"Normal  : {s['n_normal']}\n\n")
        f.write("Per-column:\n")
        for name in COLUMN_NAMES:
            cc = s["col_counts"][name]
            f.write(f"  {name:10s}: 0={cc['0']}  1={cc['1']}  2+={cc['2+']}\n")
        if s["hall_indices"]:
            f.write(f"\nHallucination files:\n")
            for idx in s["hall_indices"]:
                r = results[idx]
                f.write(f"  {filenames[idx]:20s}  type={r['hall_type']:12s}  "
                        f"blobs={r['col_blobs']}\n")

    print(f"\nStats saved → {stats_path}")

    # ── Nếu không có hallucination thì xong ──
    if s["n_hall"] == 0:
        print("\n✓ Dataset sạch — không có hallucination case nào.")
        return

    # ── Save hallucination images ──
    print(f"\nSaving hallucination images...")

    def save_grid_png(indices, filename, title_tag):
        imgs_t = [add_red_dividers(images[i]) for i in indices[:200]]
        grid   = make_grid(torch.stack(imgs_t), nrow=10, padding=2, pad_value=0.5)
        path   = os.path.join(args.out_dir, filename)
        save_image(grid, path)
        print(f"  {filename}  ({len(imgs_t)} images)")

    if s["empty_indices"]:
        save_grid_png(s["empty_indices"], "hallucinations_empty.png", "empty")

    if s["double_col_indices"]:
        save_grid_png(s["double_col_indices"], "hallucinations_double_col.png", "double_col")

    if s["n_hall"] > 0:
        save_grid_png(s["hall_indices"], "hallucinations_all.png", "all")

    # ── In danh sách file bị hallucination ──
    print(f"\nHallucination cases ({s['n_hall']} total):")
    for idx in s["hall_indices"]:
        r = results[idx]
        print(f"  [{idx:5d}] {filenames[idx]:20s}  "
              f"type={r['hall_type']:12s}  blobs={r['col_blobs']}")

    print(f"\nOutput → {args.out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description="Verify training dataset for hallucinations"
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir",  default=DEFAULT_OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

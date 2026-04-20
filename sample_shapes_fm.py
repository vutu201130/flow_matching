"""
Sampling + Hallucination Analysis cho Flow Matching UNet (simple-shapes-16x16).

Flow Matching generate là DETERMINISTIC:
  ODESolver tích phân ODE từ t=0→1 bằng Euler, không inject noise ở mỗi bước.
  => cùng x_init luôn cho cùng ảnh => lưu x_init là đủ để trace lại.

Image layout (3 cột shape + 1 cột padding):
  col 0 [x 0:5]  : triangle
  col 1 [x 5:10] : square
  col 2 [x 10:15]: pentagon
  col 3 [x 15:16]: padding (ignored)

Hallucination: >= 2 shapes trong 1 cột

Usage:
    python sample_shapes_fm.py                         # defaults
    python sample_shapes_fm.py --n_total 30000 --steps 100
    python sample_shapes_fm.py --ckpt shapes_fm_output/checkpoints/unet_epoch0300.pt
"""

import argparse
import glob
import os
import sys
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.solver import ODESolver                             # noqa: E402
from flow_matching.utils import ModelWrapper                           # noqa: E402
from models.unet import UNetModel                                      # noqa: E402
from hallucination_detector import (                                   # noqa: E402
    analyze_batch, summarize, COLUMN_NAMES,
)

# ── Output dirs ───────────────────────────────────────────────────────────────
CKPT_DIR   = os.path.join(REPO_ROOT, "shapes_fm_output", "checkpoints")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "shapes_fm_output", "hallucination_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_unet() -> UNetModel:
    return UNetModel(
        in_channels=3, model_channels=64, out_channels=3,
        num_res_blocks=3, attention_resolutions=(2,), dropout=0.1,
        channel_mult=(1, 2, 2), conv_resample=True, dims=2,
        num_classes=None, use_checkpoint=False, num_heads=1,
        num_head_channels=-1, num_heads_upsample=-1,
        use_scale_shift_norm=True, resblock_updown=False,
        use_new_attention_order=True, with_fourier_features=False,
    )


def load_model(ckpt_path: str, device: torch.device) -> UNetModel:
    model = build_unet().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss", float("nan"))
    print(f"Loaded {os.path.basename(ckpt_path)}  (epoch={epoch}, loss={loss:.5f})")
    model.eval()
    return model


class UNetVelocityWrapper(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


# ─────────────────────────────────────────────────────────────────────────────
# ODE sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_batch(wrapper: UNetVelocityWrapper, x_init: torch.Tensor,
                 steps: int, return_intermediates: bool = False):
    """
    Deterministic ODE integration (Euler).
    return_intermediates=True  →  shape [steps+1, B, 3, 16, 16]  (all timesteps)
    return_intermediates=False →  shape [B, 3, 16, 16]            (final only)
    """
    device    = x_init.device
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    solver    = ODESolver(velocity_model=wrapper)
    out = solver.sample(
        x_init=x_init,
        step_size=None,
        method="euler",
        time_grid=time_grid,
        return_intermediates=return_intermediates,
    )
    return out   # already on device


def decode(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] → [0,1] float, clipped."""
    return (t.clamp(-1, 1) + 1) / 2


def to_uint8_numpy(img_01: torch.Tensor) -> np.ndarray:
    """[3,16,16] float [0,1] → (16,16,3) uint8."""
    return (img_01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# Hallucination detection — imported from hallucination_detector.py


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

SCALE = 8   # 16→128 px per image

def add_red_dividers(img_tensor_01: torch.Tensor, scale: int = SCALE) -> torch.Tensor:
    """
    img_tensor_01: [3, H, W] float [0,1].
    Scale up and paint 2 red vertical columns (at x=5*scale and x=10*scale).
    Returns [3, H*scale, W*scale].
    """
    up = F.interpolate(img_tensor_01.unsqueeze(0), scale_factor=scale, mode="nearest").squeeze(0)
    # Red dividers: R=1, G=0, B=0, width 2px
    for x in [5 * scale, 10 * scale]:
        up[0, :, x:x+2] = 1.0   # R
        up[1, :, x:x+2] = 0.0   # G
        up[2, :, x:x+2] = 0.0   # B
    return up


def make_labeled_grid(imgs_01: list[torch.Tensor], nrow: int = 10) -> torch.Tensor:
    """imgs_01: list of [3,16,16]. Returns grid tensor with red dividers."""
    scaled = [add_red_dividers(im) for im in imgs_01]
    return make_grid(torch.stack(scaled), nrow=nrow, padding=2, pad_value=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Print channel-0 numpy array across ODE steps
# ─────────────────────────────────────────────────────────────────────────────

def print_channel0_steps(intermediates: torch.Tensor, n_print: int = 5):
    """
    intermediates: [steps+1, 3, 16, 16] (single sample, on cpu).
    Prints channel 0 (raw, before decode) at evenly-spaced step indices.
    """
    T      = intermediates.shape[0]
    idxs   = np.linspace(0, T - 1, n_print, dtype=int)
    t_vals = np.linspace(0.0, 1.0, T)

    print("\n  ── Channel-0 evolution (raw latent, before [-1,1]→[0,1] decode) ──")
    for idx in idxs:
        arr = intermediates[idx, 0].cpu().numpy()   # (16, 16)
        print(f"\n  step={idx:3d}  t={t_vals[idx]:.2f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
        # Format as 16×16 grid of 5-char floats
        rows = []
        for row in arr:
            rows.append("  " + " ".join(f"{v:+.2f}" for v in row))
        print("\n".join(rows))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── dirs ──
    hall_dir  = os.path.join(ANALYSIS_DIR, "hallucinations")
    norm_dir  = os.path.join(ANALYSIS_DIR, "normal")
    trace_dir = os.path.join(ANALYSIS_DIR, "traces")
    for d in [hall_dir, norm_dir, trace_dir]:
        os.makedirs(d, exist_ok=True)

    # ── model ──
    ckpt_path = args.ckpt or _latest_ckpt()
    model     = load_model(ckpt_path, device)
    wrapper   = UNetVelocityWrapper(model)

    # ─────────────────────────────────────────────────────────────
    # PASS 1 — bulk sampling, detect hallucinations
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 1: sampling {args.n_total} images (batch={args.batch_size}, steps={args.steps})")
    print(f"{'='*60}")

    all_noises   : list[torch.Tensor] = []  # each: [B, 3, 16, 16]
    all_finals   : list[torch.Tensor] = []  # decoded [0,1]
    all_analyses : list[dict]         = []

    n_done = 0
    batch_id = 0
    while n_done < args.n_total:
        B = min(args.batch_size, args.n_total - n_done)
        x_init = torch.randn(B, 3, 16, 16, device=device)

        x_final = sample_batch(wrapper, x_init, steps=args.steps, return_intermediates=False)
        x_01    = decode(x_final)   # [B, 3, 16, 16] in [0,1]

        imgs_np = np.stack([to_uint8_numpy(x_01[i]) for i in range(B)])  # (B,16,16,3)
        analyses = analyze_batch(imgs_np)

        all_noises.append(x_init.cpu())
        all_finals.append(x_01.cpu())
        all_analyses.extend(analyses)

        n_done  += B
        batch_id += 1
        n_hall_batch = sum(1 for a in analyses if a["is_hallucination"])
        if batch_id % 5 == 0 or n_done == args.n_total:
            print(f"  [{n_done:6d}/{args.n_total}]  batch hall: {n_hall_batch}/{B}")

    all_noises_t = torch.cat(all_noises, dim=0)   # [N, 3, 16, 16]
    all_finals_t = torch.cat(all_finals, dim=0)   # [N, 3, 16, 16]

    s = summarize(all_analyses)
    hall_indices   = s["hall_indices"]
    norm_indices   = s["normal_indices"] if "normal_indices" in s else [
        i for i, a in enumerate(all_analyses) if not a["is_hallucination"]
    ]
    empty_indices  = s["empty_indices"]
    double_indices = s["double_col_indices"]
    n_hall, n_norm = s["n_hall"], s["n_normal"]
    n_empty, n_double = s["n_empty"], s["n_double_col"]

    print(f"\nResults: {n_hall} hallucinations / {args.n_total} total  ({100*n_hall/args.n_total:.2f}%)")
    print(f"  ├─ empty image (0 shapes)  : {n_empty}")
    print(f"  └─ double col (2+ in 1 col): {n_double}")
    for name in COLUMN_NAMES:
        print(f"       {name:10s}: {s['col_counts'][name]['2+']} images with 2+ shapes")

    # ─────────────────────────────────────────────────────────────
    # Save hallucination grids — tách riêng 2 loại
    # ─────────────────────────────────────────────────────────────
    print(f"\n── Saving hallucination grids ──")

    if empty_indices:
        imgs = [all_finals_t[i] for i in empty_indices[:200]]
        save_image(make_labeled_grid(imgs, nrow=10),
                   os.path.join(hall_dir, "grid_empty_images.png"))
        print(f"  grid_empty_images.png  ({len(imgs)} images)")

    if double_indices:
        imgs = [all_finals_t[i] for i in double_indices[:200]]
        save_image(make_labeled_grid(imgs, nrow=10),
                   os.path.join(hall_dir, "grid_double_col.png"))
        print(f"  grid_double_col.png  ({len(imgs)} images)")

    n_show_hall = min(n_hall, 200)
    if n_show_hall > 0:
        hall_imgs = [all_finals_t[i] for i in hall_indices[:n_show_hall]]
        grid = make_labeled_grid(hall_imgs, nrow=10)
        save_image(grid, os.path.join(hall_dir, "grid_all_hallucinations.png"))
        print(f"  grid_all_hallucinations.png  ({n_show_hall} images)")

    # ─────────────────────────────────────────────────────────────
    # Save normal grid
    # ─────────────────────────────────────────────────────────────
    print(f"\n── Saving normal grid ──")
    n_show_norm = min(n_norm, 200)
    if n_show_norm > 0:
        norm_imgs = [all_finals_t[i] for i in norm_indices[:n_show_norm]]
        grid = make_labeled_grid(norm_imgs, nrow=10)
        save_image(grid, os.path.join(norm_dir, "grid_normal.png"))
        print(f"  grid_normal.png  ({n_show_norm} images)")

    # ─────────────────────────────────────────────────────────────
    # PASS 2 — re-run hallucination cases with intermediates
    # ─────────────────────────────────────────────────────────────
    n_trace = min(n_hall, args.n_trace)
    if n_trace == 0:
        print("\nNo hallucinations to trace.")
    else:
        print(f"\n{'='*60}")
        print(f"PASS 2: tracing {n_trace} hallucination cases with full ODE intermediates")
        print(f"{'='*60}")

        for case_num, global_idx in enumerate(hall_indices[:n_trace]):
            x_init_single = all_noises_t[global_idx].unsqueeze(0).to(device)  # [1,3,16,16]
            analysis      = all_analyses[global_idx]

            print(f"\n[Case {case_num+1:03d} / {n_trace}]  sample_idx={global_idx}  "
                  f"score={analysis['score']}  "
                  f"blobs={analysis['col_blobs']}")

            # Re-run with intermediates (deterministic → same result)
            intermediates = sample_batch(
                wrapper, x_init_single, steps=args.steps,
                return_intermediates=True,
            )  # [steps+1, 1, 3, 16, 16]
            intermediates_sq = intermediates[:, 0]  # [steps+1, 3, 16, 16]

            # ── Save case dir ──
            case_dir = os.path.join(trace_dir, f"case_{case_num+1:04d}_idx{global_idx}")
            os.makedirs(case_dir, exist_ok=True)

            # 1) initial noise
            torch.save(x_init_single.cpu(), os.path.join(case_dir, "noise_init.pt"))

            # 2) intermediates tensor
            torch.save(intermediates_sq.cpu(), os.path.join(case_dir, "intermediates.pt"))

            # 3) final image with red dividers
            final_img = decode(intermediates_sq[-1])   # [3,16,16] in [0,1]
            save_image(
                add_red_dividers(final_img),
                os.path.join(case_dir, "final_image.png"),
            )

            # 4) progression strip — selected steps as a single row with red dividers
            n_strip = 10
            step_idxs = np.linspace(0, args.steps, n_strip, dtype=int)
            strip_imgs = [add_red_dividers(decode(intermediates_sq[si])) for si in step_idxs]
            strip_grid = make_grid(torch.stack(strip_imgs), nrow=n_strip, padding=2, pad_value=0.3)
            save_image(strip_grid, os.path.join(case_dir, "progression_strip.png"))

            # 5) text report + channel-0 printout
            report_path = os.path.join(case_dir, "report.txt")
            with open(report_path, "w") as f:
                orig_stdout = sys.stdout
                sys.stdout = f

                print(f"Case {case_num+1}  |  sample_idx={global_idx}")
                print(f"Hallucination type : {analysis['hall_type']}")
                print(f"Hallucination score: {analysis['score']}")
                for name in COLUMN_NAMES:
                    nb = analysis['col_blobs'][name]
                    flag = " <-- 2+ shapes" if nb >= 2 else ""
                    print(f"  {name:10s}: {nb} shape(s){flag}")
                if analysis['hall_type'] == 'empty':
                    print(f"  *** EMPTY IMAGE — no shapes detected anywhere ***")
                print(f"\nFlow Matching is DETERMINISTIC.")
                print(f"  noise_init.pt  → re-run with this noise to reproduce exactly.")
                print(f"  intermediates  → {args.steps+1} steps × [3,16,16] float32")

                print_channel0_steps(intermediates_sq, n_print=6)

                sys.stdout = orig_stdout

            # Also print to terminal
            with open(report_path) as f:
                for line in f:
                    print("  " + line, end="")

        print(f"\nAll traces saved to: {trace_dir}")

    # ─────────────────────────────────────────────────────────────
    # Summary stats file
    # ─────────────────────────────────────────────────────────────
    stats_path = os.path.join(ANALYSIS_DIR, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Checkpoint : {os.path.basename(ckpt_path)}\n")
        f.write(f"Total      : {args.n_total}\n")
        f.write(f"Steps      : {args.steps}\n")
        f.write(f"Hallucin.  : {n_hall} ({100*n_hall/args.n_total:.2f}%)\n")
        f.write(f"  empty img: {n_empty} ({100*n_empty/args.n_total:.2f}%)\n")
        f.write(f"  double col: {n_double} ({100*n_double/args.n_total:.2f}%)\n")
        f.write(f"Normal     : {n_norm} ({100*n_norm/args.n_total:.2f}%)\n\n")
        f.write("Per-column breakdown:\n")
        for name in COLUMN_NAMES:
            c0 = sum(1 for a in all_analyses if a["col_blobs"][name] == 0)
            c1 = sum(1 for a in all_analyses if a["col_blobs"][name] == 1)
            c2 = sum(1 for a in all_analyses if a["col_blobs"][name] >= 2)
            f.write(f"  {name:10s}: 0-shape={c0}  1-shape={c1}  2+-shape={c2}\n")
        f.write(f"\nHallucination indices (first 1000):\n")
        f.write(" ".join(str(i) for i in hall_indices[:1000]) + "\n")

    print(f"\nSummary: {stats_path}")
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Hallucination grids: {hall_dir}/")
    print(f"  Normal grid        : {norm_dir}/grid_normal.png")
    print(f"  Trace cases        : {trace_dir}/  ({n_trace} cases)")
    print(f"  Stats              : {stats_path}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def _latest_ckpt() -> str:
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {CKPT_DIR}")
    return ckpts[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default=None,
                   help="Checkpoint path (default: latest)")
    p.add_argument("--n_total",    type=int, default=30000,
                   help="Total samples to generate (default: 30000)")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size per forward pass (default: 512)")
    p.add_argument("--steps",      type=int, default=100,
                   help="ODE Euler steps (default: 100)")
    p.add_argument("--n_trace",    type=int, default=50,
                   help="Max hallucination cases to trace with full intermediates (default: 50)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

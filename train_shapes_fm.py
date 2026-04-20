"""
Flow Matching training script for 16x16x3 simple-shapes dataset.
UNet: 64 model_channels, 3 levels (channel_mult=[1,2,2]), 3 res_blocks per level.

Usage:
    # Train từ đầu
    python train_shapes_fm.py --epochs 500 --batch_size 128 --lr 1e-4

    # Resume từ checkpoint cụ thể, train tiếp đến epoch 500
    python train_shapes_fm.py --resume shapes_fm_output/checkpoints/unet_epoch0200.pt --epochs 500

    # Resume từ checkpoint mới nhất
    python train_shapes_fm.py --resume latest --epochs 500

--epochs luôn là TỔNG số epoch mục tiêu (không phải số epoch thêm vào).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "image"))

from flow_matching.path import CondOTProbPath          # noqa: E402
from flow_matching.solver import ODESolver             # noqa: E402
from flow_matching.utils import ModelWrapper           # noqa: E402
from models.unet import UNetModel                      # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    REPO_ROOT,
    "..",
    "diffusion-model-hallucination/simple-datasets/simple-shapes-16x16",
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "shapes_fm_output")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")


# ── Dataset ───────────────────────────────────────────────────────────────────
class ShapesDataset(Dataset):
    """Loads all PNGs in a flat directory as 16x16 RGB tensors scaled to [-1, 1]."""

    def __init__(self, root: str):
        self.paths = sorted(
            [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith(".png")
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),                      # [0, 1]
                transforms.Normalize([0.5, 0.5, 0.5],      # → [-1, 1]
                                     [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Model wrapper for ODESolver ────────────────────────────────────────────────
class UNetVelocityWrapper(ModelWrapper):
    """
    Bridges ODESolver's call convention  (x=x, t=t)
    to UNetModel's forward signature     (x, timesteps, extra).
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        # t may be a scalar tensor during ODE integration — expand to batch.
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.model(x, t, extra=extras)


# ── Build UNet ────────────────────────────────────────────────────────────────
def build_unet() -> UNetModel:
    """
    64 model_channels, 3 levels via channel_mult=[1,2,2],
    3 residual blocks per level, attention at 2× downsampling.
    For 16×16 input: levels run at 16×16 → 8×8 → 4×4.
    """
    return UNetModel(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=(2,),   # attention at 8×8
        dropout=0.1,
        channel_mult=(1, 2, 2),       # 3 levels
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
        with_fourier_features=False,
    )


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    dataset = ShapesDataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Model ──
    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    # ── Flow matching path ──
    path = CondOTProbPath()

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # ── Resume từ checkpoint ──
    start_epoch = 1
    if args.resume:
        ckpt_path = _resolve_resume(args.resume)
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {ckpt['epoch']}  loss={ckpt.get('loss', float('nan')):.5f}")
        if start_epoch > args.epochs:
            print(f"Đã train đủ {args.epochs} epochs rồi. Tăng --epochs nếu muốn train thêm.")
            return

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x_1 = batch.to(device)                          # [B, 3, 16, 16], in [-1,1]
            B = x_1.shape[0]

            x_0 = torch.randn_like(x_1)                     # Gaussian noise source
            t = torch.rand(B, device=device)                 # uniform t ∈ [0,1]

            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t                           # target velocity

            # Predict velocity and compute MSE loss
            u_pred = model(x_t, t, extra={})
            loss = torch.pow(u_pred - u_t, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.5f}  lr={lr:.2e}")

        # ── Save checkpoint ──
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(CKPT_DIR, f"unet_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # ── Generate samples at milestones ──
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _sample_and_save(model, device, epoch, n_samples=64, steps=args.sample_steps)

    print("Training complete.")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"Samples     → {SAMPLE_DIR}")


# ── Sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def _sample_and_save(model, device, epoch, n_samples=64, steps=100):
    model.eval()

    wrapper = UNetVelocityWrapper(model)
    solver = ODESolver(velocity_model=wrapper)

    x_init = torch.randn(n_samples, 3, 16, 16, device=device)
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)

    x_gen = solver.sample(
        x_init=x_init,
        step_size=None,          # step_size=None → use time_grid spacing
        method="euler",
        time_grid=time_grid,
        return_intermediates=False,
    )

    # Rescale [-1, 1] → [0, 1] for saving
    x_gen = (x_gen.clamp(-1, 1) + 1) / 2

    out_path = os.path.join(SAMPLE_DIR, f"samples_epoch{epoch:04d}.png")
    save_image(x_gen, out_path, nrow=8, padding=1)
    print(f"  Samples saved: {out_path}")

    model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_resume(resume: str) -> str:
    """'latest' → path checkpoint mới nhất; path khác giữ nguyên."""
    if resume == "latest":
        import glob
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
        if not ckpts:
            raise FileNotFoundError(f"Không có checkpoint trong {CKPT_DIR}")
        return ckpts[-1]
    return resume


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Flow Matching on simple-shapes-16x16")
    p.add_argument("--epochs",       type=int,   default=300,
                   help="Tổng số epoch mục tiêu (default: 300)")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path checkpoint để resume, hoặc 'latest'")
    p.add_argument("--log_every",    type=int,   default=10)
    p.add_argument("--save_every",   type=int,   default=50)
    p.add_argument("--sample_every", type=int,   default=50)
    p.add_argument("--sample_steps", type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

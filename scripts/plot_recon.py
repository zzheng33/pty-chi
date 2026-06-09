#!/usr/bin/env python
"""Plot a saved pty-chi reconstruction as amplitude/phase figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def center_crop(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0 = max((arr.shape[-2] - shape[0]) // 2, 0)
    x0 = max((arr.shape[-1] - shape[1]) // 2, 0)
    return arr[..., y0 : y0 + shape[0], x0 : x0 + shape[1]]


def common_center_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1]))
    return center_crop(a, shape), center_crop(b, shape)


def add_complex_image(fig, ax_amp, ax_phase, data: np.ndarray, title: str) -> None:
    amp = np.abs(data)
    phase = np.angle(data)
    im_amp = ax_amp.imshow(amp, cmap="viridis")
    im_phase = ax_phase.imshow(phase, cmap="gray")
    ax_amp.set_title(f"{title} amplitude")
    ax_phase.set_title(f"{title} phase")
    ax_amp.axis("off")
    ax_phase.axis("off")
    fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
    fig.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)


def load_reference(result: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "para_file" not in result.files:
        return None
    para_file = Path(str(result["para_file"]))
    if not para_file.exists():
        return None
    with h5py.File(para_file, "r") as f:
        if "object/initial_guess" not in f:
            return None
        return f["object/initial_guess"][...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recon_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.recon_npz, allow_pickle=True) as result:
        recon = result["object"]
        reference = load_reference(result)

    stem = args.recon_npz.stem

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    add_complex_image(fig, axes[0], axes[1], recon, "reconstruction")
    recon_path = args.output_dir / f"{stem}_object.{args.format}"
    fig.savefig(recon_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {recon_path}")

    if reference is not None:
        reference, recon_cmp = common_center_crop(reference, recon)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        add_complex_image(fig, axes[0, 0], axes[0, 1], recon_cmp, "reconstruction")
        add_complex_image(fig, axes[1, 0], axes[1, 1], reference, "reference")
        cmp_path = args.output_dir / f"{stem}_comparison.{args.format}"
        fig.savefig(cmp_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {cmp_path}")


if __name__ == "__main__":
    main()

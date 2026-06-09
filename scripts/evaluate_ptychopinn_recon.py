#!/usr/bin/env python
"""Evaluate a pty-chi reconstruction against the PtychoPINN reference object."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

PTYCHOPINN_REPO = Path("/home/zhong.zheng/PtychoPINN")
if PTYCHOPINN_REPO.exists():
    sys.path.insert(0, str(PTYCHOPINN_REPO))

try:
    from ptychopinn_torch.eval.frc import frc_preprocess_images
    from ptychopinn_torch.eval.eval_metrics import FSC
except ImportError:
    frc_preprocess_images = None
    FSC = None


def center_crop(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0 = max((arr.shape[-2] - shape[0]) // 2, 0)
    x0 = max((arr.shape[-1] - shape[1]) // 2, 0)
    return arr[..., y0 : y0 + shape[0], x0 : x0 + shape[1]]


def common_center_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1]))
    return center_crop(a, shape), center_crop(b, shape)


def common_square_crop(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    side = min(a.shape[-2], a.shape[-1], b.shape[-2], b.shape[-1])
    return center_crop(a, (side, side)), center_crop(b, (side, side))


def align_complex_scale(reference: np.ndarray, recon: np.ndarray) -> np.ndarray:
    # Solves min_s ||reference - s * recon||_2 for one complex scalar s.
    denom = np.vdot(recon, recon)
    if np.abs(denom) < 1e-12:
        return recon
    scale = np.vdot(recon, reference) / denom
    return scale * recon


def phase_error(reference: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.angle(recon) - np.angle(reference))))


def frc_curve(reference: np.ndarray, recon: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    f_ref = np.fft.fftshift(np.fft.fft2(reference))
    f_rec = np.fft.fftshift(np.fft.fft2(recon))

    h, w = reference.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    radius = np.rint(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    max_r = min(h, w) // 2

    frc = np.empty(max_r, dtype=np.float64)
    frc[:] = np.nan
    for r in range(max_r):
        mask = radius == r
        if not np.any(mask):
            continue
        numerator = np.abs(np.sum(f_ref[mask] * np.conj(f_rec[mask])))
        denominator = np.sqrt(
            np.sum(np.abs(f_ref[mask]) ** 2) * np.sum(np.abs(f_rec[mask]) ** 2)
        )
        frc[r] = numerator / denominator if denominator > 0 else np.nan

    freq = np.arange(max_r) / max_r
    valid = np.isfinite(frc)
    auc = float(np.nanmean(frc[valid & (freq <= 0.5)]))
    return freq, frc, auc


def ptychopinn_frc_auc(
    reference: np.ndarray,
    recon: np.ndarray,
    cutoff: float = 0.5,
    align: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    if frc_preprocess_images is None or FSC is None:
        raise RuntimeError(
            "Could not import PtychoPINN FRC functions. Run with the ptychopinn_torch "
            "environment or make sure /home/zhong.zheng/PtychoPINN is available."
        )

    reference_sq, recon_sq = common_square_crop(reference, recon)
    aligned_ref, aligned_recon = frc_preprocess_images(
        reference_sq,
        recon_sq,
        image_prop="complex",
        verbose=verbose,
        align=align,
    )
    fr_curve, x_fr, _t_curve, _x_t = FSC(aligned_ref, aligned_recon)
    stop = np.where(x_fr - cutoff > 0)[0]
    stop_idx = int(stop[0]) if len(stop) else len(fr_curve)
    fr_auc = float(np.sum(fr_curve[:stop_idx]) / max(stop_idx, 1))
    return fr_curve, x_fr, fr_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recon_npz", type=Path)
    parser.add_argument(
        "--amp-mask-threshold",
        type=float,
        default=0.05,
        help="Phase metrics are computed where reference amplitude exceeds this fraction of max amplitude.",
    )
    parser.add_argument(
        "--frc-align",
        action="store_true",
        help="Use PtychoPINN's sub-pixel registration step before FRC. Manuscript helper often used align=False.",
    )
    parser.add_argument(
        "--frc-cutoff",
        type=float,
        default=0.5,
        help="Normalized spatial-frequency cutoff for PtychoPINN-style FRC AUC.",
    )
    args = parser.parse_args()

    result = np.load(args.recon_npz, allow_pickle=True)
    recon = result["object"]
    para_file = Path(str(result["para_file"]))

    with h5py.File(para_file, "r") as f:
        reference = f["object/initial_guess"][...]

    reference, recon = common_center_crop(reference, recon)
    recon = align_complex_scale(reference, recon)

    amp_ref = np.abs(reference)
    amp_rec = np.abs(recon)
    amp_range = max(float(amp_ref.max() - amp_ref.min()), 1e-12)
    amp_rmse = float(np.sqrt(np.mean((amp_rec - amp_ref) ** 2)))
    amp_nrmse = amp_rmse / amp_range
    amp_mae = float(np.mean(np.abs(amp_rec - amp_ref)))

    mask = amp_ref > args.amp_mask_threshold * amp_ref.max()
    ph_err = phase_error(reference, recon)
    phase_mae_rad = float(np.mean(np.abs(ph_err[mask])))
    phase_rmse_rad = float(np.sqrt(np.mean(ph_err[mask] ** 2)))

    complex_nrmse = float(
        np.linalg.norm((recon - reference).ravel()) / max(np.linalg.norm(reference.ravel()), 1e-12)
    )
    if frc_preprocess_images is not None and FSC is not None:
        _fr_curve, _x_fr, frc_auc = ptychopinn_frc_auc(
            reference,
            recon,
            cutoff=args.frc_cutoff,
            align=args.frc_align,
        )
        frc_name = f"ptychopinn_frc_auc_0_to_{args.frc_cutoff:g}"
    else:
        _, _, frc_auc = frc_curve(*common_square_crop(reference, recon))
        frc_name = f"fallback_frc_auc_0_to_{args.frc_cutoff:g}"

    print(f"reconstruction: {args.recon_npz}")
    print(f"reference:      {para_file}:object/initial_guess")
    print(f"comparison shape: {reference.shape}")
    print(f"complex_nrmse:      {complex_nrmse:.6g}  lower is better")
    print(f"amplitude_mae:      {amp_mae:.6g}  lower is better")
    print(f"amplitude_nrmse:    {amp_nrmse:.6g}  lower is better")
    print(f"phase_mae_rad:      {phase_mae_rad:.6g}  lower is better")
    print(f"phase_rmse_rad:     {phase_rmse_rad:.6g}  lower is better")
    print(f"{frc_name}: {frc_auc:.6g}  higher is better")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run pty-chi on PtychoPINN data converted to Ptychodus-style HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import ptychi.api as api
import ptychi.device
from ptychi.api.task import PtychographyTask
from ptychi.utils import (
    get_default_complex_dtype,
    get_suggested_object_size,
    rescale_probe,
    set_default_complex_dtype,
)


DATASETS = {
    "FLY1": "FLY1/supervised_pinn_fly001",
    "IC1": "IC1/supervised_pinn_ic_1",
    "IC2": "IC2/supervised_pinn_ic_2",
    "LCLS": "LCLS/xppl1026722_Run0396_64_train",
    "LFP": "LFP/supervised_pinn_micro_lfp",
    "NCM": "NCM/supervised_pinn_ncm",
    "TP1": "TP1/supervised_pinn_gold_tp_1",
    "TP2": "TP2/supervised_pinn_gold_tp_2",
    "W": "W/supervised_pinn_cnm",
}


def resolve_dataset(dataset: str, data_root: Path) -> tuple[Path, Path]:
    if dataset in DATASETS:
        stem = data_root / DATASETS[dataset]
    else:
        stem = Path(dataset)
        if not stem.is_absolute():
            stem = data_root / stem

    dp_file = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
    para_file = stem.with_name(stem.name + "_ptychodus_para.hdf5")
    if not dp_file.exists():
        raise FileNotFoundError(f"Missing diffraction file: {dp_file}")
    if not para_file.exists():
        raise FileNotFoundError(f"Missing parameter file: {para_file}")
    return dp_file, para_file


def load_converted_data(
    dp_file: Path,
    para_file: Path,
    center_positions: bool = True,
    scale_probe: bool = True,
) -> tuple[np.ndarray, torch.Tensor, float, np.ndarray]:
    with h5py.File(dp_file, "r") as f:
        patterns = f["dp"][...]

    with h5py.File(para_file, "r") as f:
        probe = f["probe"][...]
        pixel_size_m = float(f["object"].attrs["pixel_height_m"])
        positions = np.stack(
            [f["probe_position_y_m"][...], f["probe_position_x_m"][...]],
            axis=1,
        )

    if scale_probe:
        probe = rescale_probe(probe, patterns)
    if probe.ndim == 3:
        probe = probe[None, :, :, :]

    positions_px = positions / pixel_size_m
    if center_positions:
        positions_px = positions_px - positions_px.mean(axis=0, keepdims=True)

    probe_tensor = torch.as_tensor(probe, dtype=get_default_complex_dtype())
    return patterns, probe_tensor, pixel_size_m, positions_px


def make_options(
    algorithm: str,
    data: np.ndarray,
    probe: torch.Tensor,
    pixel_size_m: float,
    positions_px: np.ndarray,
    epochs: int,
    batch_size: int,
    extra_object_pixels: int,
    object_step_size: float,
    probe_step_size: float,
    optimize_probe: bool,
):
    algorithm = algorithm.lower()
    if algorithm == "pie":
        options = api.PIEOptions()
        options.object_options.alpha = 1.0
        options.probe_options.alpha = 1.0
    elif algorithm == "epie":
        options = api.EPIEOptions()
        options.object_options.alpha = 1.0
        options.probe_options.alpha = 1.0
    elif algorithm == "rpie":
        options = api.RPIEOptions()
        options.object_options.alpha = 1.0
        options.probe_options.alpha = 1.0
    elif algorithm == "mpie":
        options = api.RPIEOptions()
        options.object_options.alpha = 1.0
        options.probe_options.alpha = 1.0
        options.object_options.optimizer_params = {"momentum": 0.1, "nesterov": True}
        options.probe_options.optimizer_params = {"momentum": 0.1, "nesterov": True}
    elif algorithm == "lsqml":
        options = api.LSQMLOptions()
        options.object_options.build_preconditioner_with_all_modes = True
        options.reconstructor_options.noise_model = api.NoiseModels.GAUSSIAN
    elif algorithm == "dm":
        options = api.DMOptions()
        options.probe_options.power_constraint.probe_power = np.sum(
            np.max(data, axis=-3), axis=(-2, -1)
        )
        options.probe_options.power_constraint.enabled = True
    elif algorithm == "bh":
        options = api.BHOptions()
        options.probe_options.rho = 0.1
        options.probe_position_options.rho = 2
        options.reconstructor_options.method = "GD"
        options.reconstructor_options.use_double_precision_for_fft = False
        options.reconstructor_options.forward_model_options.pad_for_shift = 4
    elif algorithm == "ad_ptycho":
        options = api.AutodiffPtychographyOptions()
    else:
        raise ValueError(
            "algorithm must be one of: pie, epie, rpie, mpie, dm, lsqml, bh, ad_ptycho"
        )

    options.data_options.data = data
    object_shape = get_suggested_object_size(
        positions_px,
        probe.shape[-2:],
        extra=extra_object_pixels,
    )
    options.object_options.initial_guess = torch.ones(
        [1, *object_shape],
        dtype=get_default_complex_dtype(),
    )
    options.object_options.pixel_size_m = pixel_size_m
    options.object_options.optimizable = True
    options.object_options.optimizer = api.Optimizers.SGD
    options.object_options.step_size = object_step_size

    options.probe_options.initial_guess = probe
    options.probe_options.optimizable = optimize_probe
    options.probe_options.optimizer = api.Optimizers.SGD
    options.probe_options.step_size = probe_step_size

    options.probe_position_options.position_x_px = positions_px[:, 1]
    options.probe_position_options.position_y_px = positions_px[:, 0]
    options.probe_position_options.optimizable = False

    if algorithm == "pie":
        options.reconstructor_options.batch_size = 1
    else:
        options.reconstructor_options.batch_size = min(batch_size, data.shape[0])
    options.reconstructor_options.num_epochs = epochs
    options.reconstructor_options.allow_nondeterministic_algorithms = False
    if algorithm == "dm":
        options.reconstructor_options.chunk_length = min(batch_size, data.shape[0])
    return options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="TP1",
        help="Dataset key such as TP1/TP2/IC1, or a converted file stem.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--algorithm",
        choices=["pie", "epie", "rpie", "mpie", "dm", "lsqml", "bh", "ad_ptycho"],
        default="epie",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extra-object-pixels", type=int, default=64)
    parser.add_argument("--object-step-size", type=float, default=0.1)
    parser.add_argument("--probe-step-size", type=float, default=0.1)
    parser.add_argument("--fixed-probe", action="store_true")
    parser.add_argument("--no-center-positions", action="store_true")
    parser.add_argument("--no-probe-rescale", action="store_true")
    parser.add_argument("--device", choices=["cuda", "xpu", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Default: outputs/<dataset>_<algorithm>_recon.npz",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"
    if args.device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print("Intel XPU is not available; falling back to CPU.")
            args.device = "cpu"
        else:
            ptychi.device.set_torch_accelerator_module(torch.xpu)
    torch.set_default_device(args.device)
    torch.set_default_dtype(torch.float32)
    set_default_complex_dtype(torch.complex64)

    dp_file, para_file = resolve_dataset(args.dataset, args.data_root)
    data, probe, pixel_size_m, positions_px = load_converted_data(
        dp_file,
        para_file,
        center_positions=not args.no_center_positions,
        scale_probe=not args.no_probe_rescale,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Diffraction: {dp_file}")
    print(f"Parameters:  {para_file}")
    print(f"data shape: {data.shape}, dtype: {data.dtype}")
    print(f"probe shape: {tuple(probe.shape)}, dtype: {probe.dtype}")
    print(f"position y range px: {positions_px[:, 0].min():.3f} to {positions_px[:, 0].max():.3f}")
    print(f"position x range px: {positions_px[:, 1].min():.3f} to {positions_px[:, 1].max():.3f}")
    print(f"pixel_size_m: {pixel_size_m}")

    options = make_options(
        args.algorithm,
        data,
        probe,
        pixel_size_m,
        positions_px,
        args.epochs,
        args.batch_size,
        args.extra_object_pixels,
        args.object_step_size,
        args.probe_step_size,
        optimize_probe=not args.fixed_probe,
    )
    options.reconstructor_options.default_device = (
        api.Devices.CPU if args.device == "cpu" else api.Devices.GPU
    )
    print(f"object initial shape: {tuple(options.object_options.initial_guess.shape)}")

    if args.dry_run:
        print("Dry run complete; reconstruction was not started.")
        return

    task = PtychographyTask(options)
    task.run()

    recon = task.get_data_to_cpu("object", as_numpy=True)[0]
    probe_out = task.get_data_to_cpu("probe", as_numpy=True)

    output = args.output
    if output is None:
        output = Path("outputs") / f"{args.dataset}_{args.algorithm}_recon.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        object=recon,
        probe=probe_out,
        position_y_px=positions_px[:, 0],
        position_x_px=positions_px[:, 1],
        dp_file=str(dp_file),
        para_file=str(para_file),
        algorithm=args.algorithm,
    )
    print(f"Saved reconstruction: {output}")


if __name__ == "__main__":
    main()

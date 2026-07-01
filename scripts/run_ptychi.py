#!/usr/bin/env python
"""Run pty-chi on PtychoPINN data converted to Ptychodus-style HDF5."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
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
    # "FLY1": "FLY1/supervised_pinn_fly001",
    # "IC1": "IC1/supervised_pinn_ic_1",
    # "IC2": "IC2/supervised_pinn_ic_2",
    # "LCLS": "LCLS/xppl1026722_Run0396_64_train",
    # "LFP": "LFP/supervised_pinn_micro_lfp",
    # "NCM": "NCM/supervised_pinn_ncm",
    # "TP1": "TP1/supervised_pinn_gold_tp_1",
    # "TP2": "TP2/supervised_pinn_gold_tp_2",
    # "W": "W/supervised_pinn_cnm",
    "R1000": "R1000/synthetic_1000_N64",
    "R2000": "R2000/synthetic_2000_N64",
    "R4000": "R4000/synthetic_4000_N64",
    "R8000": "R8000/synthetic_8000_N64",
    "R12000": "R12000/synthetic_12000_N64",
    # "R16000": "R16000/synthetic_16000_N64",
    # "R20000": "R20000/synthetic_20000_N64",
    # "R26000": "R26000/synthetic_26000_N64",
}

SYNTHETIC_DATASETS = list(DATASETS.keys())
ALGORITHMS = ["pie", "epie", "rpie", "mpie", "dm", "lsqml", "bh", "ad_ptycho"]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONITOR_SCRIPT = (
    Path("/home/zhong.zheng/PtychoPINN") / "scripts" / "monitor_gpu_power.py"
)
TIMING_PATTERNS = {
    "io_load_time_s": re.compile(r"^io_load_time_s:\s*([0-9.eE+-]+)", re.MULTILINE),
    "setup_time_s": re.compile(r"^setup_time_s:\s*([0-9.eE+-]+)", re.MULTILINE),
    "task_setup_time_s": re.compile(r"^task_setup_time_s:\s*([0-9.eE+-]+)", re.MULTILINE),
    "reconstruction_run_time_s": re.compile(
        r"^reconstruction_run_time_s:\s*([0-9.eE+-]+)", re.MULTILINE
    ),
    "save_time_s": re.compile(r"^save_time_s:\s*([0-9.eE+-]+)", re.MULTILINE),
    "total_time_s": re.compile(r"^total_time_s:\s*([0-9.eE+-]+)", re.MULTILINE),
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


def start_power_monitor(
    power_csv: Path | None,
    monitor_script: Path,
    vendor: str,
    interval: float,
    label: str,
    devices: str | None,
    device: str,
) -> subprocess.Popen | None:
    if power_csv is None or device == "cpu":
        return None

    power_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(monitor_script),
        "--vendor",
        vendor,
        "--interval",
        str(interval),
        "--output",
        str(power_csv),
        "--label",
        label,
    ]
    if devices:
        cmd.extend(["--devices", devices])

    env = os.environ.copy()
    env.pop("ZE_AFFINITY_MASK", None)
    print(f"Starting power monitor at IO start: {power_csv}", flush=True)
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)


def stop_power_monitor(monitor: subprocess.Popen | None) -> None:
    if monitor is None:
        return
    monitor.terminate()
    try:
        monitor.wait(timeout=5)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait()


def run_single() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="IC1",
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1000)
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
    parser.add_argument(
        "--no-save-output",
        action="store_true",
        help="Skip copying reconstruction tensors to CPU and writing the output .npz.",
    )
    parser.add_argument("--power-csv", type=Path, default=None)
    parser.add_argument("--monitor-script", type=Path, default=DEFAULT_MONITOR_SCRIPT)
    parser.add_argument("--vendor", default="auto", choices=("auto", "nvidia", "amd", "intel"))
    parser.add_argument("--devices", default=None, help="Comma-separated GPU indices to monitor.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--power-label", default=None)
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

    power_label = args.power_label or f"{args.dataset}_{args.algorithm}"
    monitor = start_power_monitor(
        args.power_csv,
        args.monitor_script,
        args.vendor,
        args.interval,
        power_label,
        args.devices,
        args.device,
    )
    power_monitor_start_s = time.perf_counter()

    try:
        total_start = time.perf_counter()
        dp_file, para_file = resolve_dataset(args.dataset, args.data_root)
        io_start = time.perf_counter()
        data, probe, pixel_size_m, positions_px = load_converted_data(
            dp_file,
            para_file,
            center_positions=not args.no_center_positions,
            scale_probe=not args.no_probe_rescale,
        )
        io_load_time_s = time.perf_counter() - io_start

        print(f"Dataset: {args.dataset}")
        print(f"Diffraction: {dp_file}")
        print(f"Parameters:  {para_file}")
        print(f"data shape: {data.shape}, dtype: {data.dtype}")
        print(f"probe shape: {tuple(probe.shape)}, dtype: {probe.dtype}")
        print(f"position y range px: {positions_px[:, 0].min():.3f} to {positions_px[:, 0].max():.3f}")
        print(f"position x range px: {positions_px[:, 1].min():.3f} to {positions_px[:, 1].max():.3f}")
        print(f"pixel_size_m: {pixel_size_m}")
        print(f"power_monitor_to_io_start_s: {io_start - power_monitor_start_s:.6f}")
        print(f"io_load_time_s: {io_load_time_s:.6f}")

        setup_start = time.perf_counter()
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
        setup_time_s = time.perf_counter() - setup_start
        print(f"object initial shape: {tuple(options.object_options.initial_guess.shape)}")
        print(f"setup_time_s: {setup_time_s:.6f}")

        if args.dry_run:
            print("Dry run complete; reconstruction was not started.")
            return

        task_setup_start = time.perf_counter()
        task = PtychographyTask(options)
        task_setup_time_s = time.perf_counter() - task_setup_start

        run_start = time.perf_counter()
        task.run()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
        reconstruction_run_time_s = time.perf_counter() - run_start

        print(f"task_setup_time_s: {task_setup_time_s:.6f}")
        print(f"reconstruction_run_time_s: {reconstruction_run_time_s:.6f}")
        if args.no_save_output:
            print("save_time_s: 0.000000")
            print(f"total_time_s: {time.perf_counter() - total_start:.6f}")
            print("Skipped reconstruction save.")
            return

        save_start = time.perf_counter()
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
            io_load_time_s=io_load_time_s,
            setup_time_s=setup_time_s,
            task_setup_time_s=task_setup_time_s,
            reconstruction_run_time_s=reconstruction_run_time_s,
            total_time_before_save_s=save_start - total_start,
        )
        save_time_s = time.perf_counter() - save_start
        print(f"save_time_s: {save_time_s:.6f}")
        print(f"total_time_s: {time.perf_counter() - total_start:.6f}")
        print(f"Saved reconstruction: {output}")
    finally:
        stop_power_monitor(monitor)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def short_gpu_name(name: str) -> str:
    normalized = name.upper().replace("_", " ").replace("-", " ")
    for known_name in ("V100", "A100", "H100", "H200", "B200", "MI300X", "MI300A", "MAX"):
        if known_name in normalized:
            return "Max" if known_name == "MAX" else known_name
    return safe_name(name)


def expand_selection(values: list[str], all_values: list[str]) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return list(all_values)
    return values


def parse_timing_log(log_path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if not log_path.exists():
        return metrics
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for key, pattern in TIMING_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            metrics[key] = f"{float(matches[-1]):.6f}"
    return metrics


def detect_gpu_label(args) -> tuple[str, str]:
    if args.gpu_label:
        vendor = args.vendor if args.vendor != "auto" else "unknown_vendor"
        return vendor, safe_name(args.gpu_label)
    if args.test or args.device == "cpu" or not args.monitor_script.exists():
        return args.vendor, "cpu" if args.device == "cpu" else "unknown_gpu"

    cmd = [
        sys.executable,
        str(args.monitor_script),
        "--vendor",
        args.vendor,
        "--list-gpus",
    ]
    if args.devices:
        cmd.extend(["--devices", args.devices])

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return args.vendor, "unknown_gpu"

    rows = [line.split(",", 2) for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return args.vendor, "unknown_gpu"

    vendor = rows[0][0]
    names = [row[2] for row in rows if len(row) == 3]
    unique_names = sorted(set(short_gpu_name(name) for name in names))
    if len(unique_names) == 1:
        return vendor, unique_names[0]
    return vendor, safe_name("_".join(unique_names))


def run_modeling_one(args, dataset: str, algorithm: str, run_dir: Path) -> dict[str, object]:
    run_name = f"{algorithm}_e{args.epochs}_bs{args.batch_size}"
    dataset_dir = run_dir / safe_name(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    log_file = dataset_dir / f"{run_name}.log"
    power_csv = dataset_dir / f"{run_name}_power.csv"
    label = f"{dataset}_{run_name}"

    ptychi_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        dataset,
        "--data-root",
        str(args.data_root),
        "--algorithm",
        algorithm,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--extra-object-pixels",
        str(args.extra_object_pixels),
        "--object-step-size",
        str(args.object_step_size),
        "--probe-step-size",
        str(args.probe_step_size),
        "--no-save-output",
    ]
    if not args.test and args.device != "cpu":
        ptychi_cmd.extend(
            [
                "--power-csv",
                str(power_csv),
                "--monitor-script",
                str(args.monitor_script),
                "--vendor",
                args.vendor,
                "--interval",
                str(args.interval),
                "--power-label",
                label,
            ]
        )
        if args.devices:
            ptychi_cmd.extend(["--devices", args.devices])
    if args.fixed_probe:
        ptychi_cmd.append("--fixed-probe")
    if args.no_center_positions:
        ptychi_cmd.append("--no-center-positions")
    if args.no_probe_rescale:
        ptychi_cmd.append("--no-probe-rescale")
    if args.dry_run_ptychi:
        ptychi_cmd.append("--dry-run")

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
        env["HIP_VISIBLE_DEVICES"] = args.devices
        env["ROCR_VISIBLE_DEVICES"] = args.devices
        env["GPU_DEVICE_ORDINAL"] = args.devices
        env["ZE_AFFINITY_MASK"] = args.devices

    if args.test or args.device == "cpu":
        print(f"Skipping GPU power monitor for {label}.", flush=True)
    else:
        print(f"Power monitor will start inside pty-chi at IO start: {power_csv}", flush=True)

    start = time.time()
    returncode = 1
    print(
        "Running pty-chi: "
        f"dataset={dataset}, algorithm={algorithm}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, device={args.device}",
        flush=True,
    )
    with log_file.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            ptychi_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()

    row = {
        "dataset": dataset,
        "algorithm": algorithm,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "vendor": args.vendor,
        "detected_vendor": args.detected_vendor,
        "gpu_label": args.gpu_label,
        "devices": args.devices or "all",
        "returncode": returncode,
        "wall_duration_s": f"{time.time() - start:.6f}",
        "power_csv": "" if args.test or args.device == "cpu" else str(power_csv),
        "log_file": str(log_file),
    }
    row.update(parse_timing_log(log_file))
    return row


def run_modeling() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["R1000"])
    parser.add_argument("--algorithms", nargs="+", default=["epie"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "xpu"))
    parser.add_argument("--vendor", default="auto", choices=("auto", "nvidia", "amd", "intel"))
    parser.add_argument("--devices", default=None, help="Comma-separated GPU indices to run/monitor.")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "modeling_exp")
    parser.add_argument("--monitor-script", type=Path, default=DEFAULT_MONITOR_SCRIPT)
    parser.add_argument("--gpu-label", default=None, help="Manual output-folder GPU label, e.g. A100.")
    parser.add_argument("--extra-object-pixels", type=int, default=64)
    parser.add_argument("--object-step-size", type=float, default=0.1)
    parser.add_argument("--probe-step-size", type=float, default=0.1)
    parser.add_argument("--fixed-probe", action="store_true")
    parser.add_argument("--no-center-positions", action="store_true")
    parser.add_argument("--no-probe-rescale", action="store_true")
    parser.add_argument("--dry-run-ptychi", action="store_true")
    parser.add_argument("--test", action="store_true", help="Skip GPU power monitoring.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.datasets = expand_selection(args.datasets, SYNTHETIC_DATASETS)
    args.algorithms = expand_selection(args.algorithms, ALGORITHMS)

    detected_vendor, gpu_label = detect_gpu_label(args)
    args.detected_vendor = detected_vendor
    args.gpu_label = gpu_label
    run_dir = args.output_root / gpu_label
    run_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for algorithm in args.algorithms:
            row = run_modeling_one(args, dataset, algorithm, run_dir)
            if row["returncode"] != 0 and not args.continue_on_error:
                print(f"Stopping after failed run: {row}", file=sys.stderr)
                return int(row["returncode"])

    return 0


def main() -> int | None:
    if "--modeling" in sys.argv:
        return run_modeling()
    run_single()
    return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        raise SystemExit(result)

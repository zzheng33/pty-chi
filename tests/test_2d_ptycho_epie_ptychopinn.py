import argparse
import os
from pathlib import Path

import pytest
import torch

import ptychi.api as api
from ptychi.api.task import PtychographyTask
from ptychi.utils import get_default_complex_dtype, get_suggested_object_size

import test_utils as tutils


DATASETS = {
    "TP1": "TP1/supervised_pinn_gold_tp_1",
    "TP2": "TP2/supervised_pinn_gold_tp_2",
    "IC1": "IC1/supervised_pinn_ic_1",
    "IC2": "IC2/supervised_pinn_ic_2",
    "NCM": "NCM/supervised_pinn_ncm",
    "FLY1": "FLY1/supervised_pinn_fly001",
    "LFP": "LFP/supervised_pinn_micro_lfp",
    "W": "W/supervised_pinn_cnm",
    "LCLS": "LCLS/xppl1026722_Run0396_64_train",
}


class Test2DPtychoEPIEPtychoPINN(tutils.BaseTester):
    def load_ptychopinn_data(self, dataset="TP1"):
        data_root = Path(
            os.environ.get(
                "PTYCHOPINN_CONVERTED_DATA_DIR",
                Path(__file__).resolve().parents[1] / "data",
            )
        )
        stem = data_root / DATASETS[dataset]
        dp_file = stem.with_name(stem.name + "_ptychodus_dp.hdf5")
        para_file = stem.with_name(stem.name + "_ptychodus_para.hdf5")
        data, probe, pixel_size_m, positions_px = self.load_data_ptychodus(
            dp_file,
            para_file,
            subtract_position_mean=True,
            additional_opr_modes=0,
        )
        return data, probe[:, [0], :, :], pixel_size_m, positions_px

    @pytest.mark.local
    @tutils.BaseTester.wrap_recon_tester(
        name="test_2d_ptycho_epie_ptychopinn",
        run_comparison=False,
    )
    def test_2d_ptycho_epie_ptychopinn(self):
        device = os.environ.get("PTYCHOPINN_TEST_DEVICE", "cpu").lower()
        use_cpu = device == "cpu"
        self.setup_ptychi(cpu_only=use_cpu)

        data, probe, pixel_size_m, positions_px = self.load_ptychopinn_data(
            dataset=os.environ.get("PTYCHOPINN_TEST_DATASET", "TP1")
        )

        options = api.EPIEOptions()
        options.data_options.data = data

        options.object_options.initial_guess = torch.ones(
            [1, *get_suggested_object_size(positions_px, probe.shape[-2:], extra=64)],
            dtype=get_default_complex_dtype(),
        )
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.optimizable = True
        options.object_options.optimizer = api.Optimizers.SGD
        options.object_options.step_size = 0.1
        options.object_options.alpha = 1

        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True
        options.probe_options.optimizer = api.Optimizers.SGD
        options.probe_options.step_size = 0.1
        options.probe_options.alpha = 1

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]
        options.probe_position_options.optimizable = False

        options.reconstructor_options.default_device = api.Devices.CPU if use_cpu else api.Devices.GPU
        options.reconstructor_options.batch_size = 128
        options.reconstructor_options.num_epochs = int(os.environ.get("PTYCHOPINN_TEST_EPOCHS", "1"))
        options.reconstructor_options.allow_nondeterministic_algorithms = False

        task = PtychographyTask(options)
        task.run()

        return task.get_data_to_cpu("object", as_numpy=True)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TP1", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    os.environ["PTYCHOPINN_TEST_DATASET"] = args.dataset
    os.environ["PTYCHOPINN_TEST_EPOCHS"] = str(args.epochs)
    os.environ["PTYCHOPINN_TEST_DEVICE"] = args.device

    tester = Test2DPtychoEPIEPtychoPINN()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_2d_ptycho_epie_ptychopinn()

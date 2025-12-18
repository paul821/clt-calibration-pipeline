# calibration/setup_model.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch

import clt_toolkit as clt
import flu_core as flu


@dataclass
class ModelBundle:
    flu_demo_model: object
    base_state: object
    base_params: object
    base_schedules: object
    base_precomputed: dict
    estimation_config: dict


def build_model_bundle(
    T: int,
    timesteps_per_day: int,
    seed: int = 0,
) -> ModelBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # you can keep these if you want
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    texas_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"
    calibration_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "calibration_research_input_files"

    # load files
    calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)
    humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)

    schedules_info = flu.FluSubpopSchedules(
        absolute_humidity=humidity_df,
        flu_contact_matrix=calendar_df,
        daily_vaccines=vaccines_df,
    )

    # jsons
    subpopA_init_vals = clt.make_dataclass_from_json(calibration_files_path / "subpopA_init_vals.json", flu.FluSubpopState)
    subpopB_init_vals = clt.make_dataclass_from_json(calibration_files_path / "subpopB_init_vals.json", flu.FluSubpopState)
    subpopC_init_vals = clt.make_dataclass_from_json(calibration_files_path / "subpopC_init_vals.json", flu.FluSubpopState)

    common_subpop_params = clt.make_dataclass_from_json(texas_files_path / "common_subpop_params.json", flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(calibration_files_path / "ABC_mixing_params.json", flu.FluMixingParams)
    simulation_settings = clt.make_dataclass_from_json(texas_files_path / "simulation_settings.json", flu.SimulationSettings)
    simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day})

    # distinct RNGs (tiny hygiene improvement)
    rngA = np.random.default_rng(88888)
    rngB = np.random.default_rng(88889)
    rngC = np.random.default_rng(88890)

    subpopA_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 1.5})
    subpopB_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.5})
    subpopC_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.2})

    subpopA = flu.FluSubpopModel(subpopA_init_vals, subpopA_params, simulation_settings, rngA, schedules_info, name="subpopA")
    subpopB = flu.FluSubpopModel(subpopB_init_vals, subpopB_params, simulation_settings, rngB, schedules_info, name="subpopB")
    subpopC = flu.FluSubpopModel(subpopC_init_vals, subpopC_params, simulation_settings, rngC, schedules_info, name="subpopC")

    flu_demo_model = flu.FluMetapopModel([subpopA, subpopB, subpopC], mixing_params)

    base_torch_inputs = flu_demo_model.get_flu_torch_inputs()
    base_state = base_torch_inputs["state_tensors"]
    base_params = base_torch_inputs["params_tensors"]
    base_schedules = base_torch_inputs["schedule_tensors"]
    base_precomputed = base_torch_inputs["precomputed"]

    if hasattr(base_state, "HR"):
        base_state.HR[:] = 0.0
    if hasattr(base_state, "HD"):
        base_state.HD[:] = 0.0

    estimation_config = {
        "beta_param": "L",
        "estimate_initial": {"E": False, "IP": False, "ISR": False, "ISH": False, "IA": False},
        "time_zero_mode": "absolute",
        "t0_day": 0,
        "window_days": T,
        "ihr_mode": False,
        "ihr_param": "L",
        "ihr_aggregate_over_age": True,
        "restart_widths_ihr_mode": {"Wide Search": 0.25, "Medium Search": 0.15, "Narrow Search": 0.05},
        "hold_beta_fixed": False,
        "hold_ihr_fixed": False,
        "fixed_beta_tensor": None,
        "fixed_ihr_tensor": None,
    }

    return ModelBundle(
        flu_demo_model=flu_demo_model,
        base_state=base_state,
        base_params=base_params,
        base_schedules=base_schedules,
        base_precomputed=base_precomputed,
        estimation_config=estimation_config,
    )

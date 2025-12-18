import torch, time as global_time, pandas as pd, matplotlib.pyplot as plt, numpy as np, copy
from dataclasses import fields

import clt_toolkit as clt
import flu_core as flu

from scipy.optimize import least_squares, minimize
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue

from calibration.setup_model import build_model_bundle
from calibration.optimize import run_metapop_calibration_suite
from calibration.plotting import plot_calibration_results

STOP_R2 = 0.5
STABLE_R2_FOR_LOSS_CHECK = STOP_R2 * 0.8
T = 180
timesteps_per_day = 4
true_gamma = 0.2
    
def main():
  torch.autograd.set_detect_anomaly(True)
  
  bundle = build_model_bundle(T=T, timesteps_per_day=TIMESTEPS_PER_DAY, seed=0)
  
  L, A, R = base_params.beta_baseline.shape
  true_betas_3_param_L = torch.tensor([0.0412, 0.0422, 0.0422], dtype=torch.float64)
  true_beta_3_param_tensor = true_betas_3_param_L.view(L, 1, 1).expand(L, A, R)
  true_betas_3_param_flat = true_betas_3_param_L.tolist()
  
  params_3_param = copy.deepcopy(base_params)
  params_3_param.beta_baseline = true_beta_3_param_tensor
  
  with torch.no_grad():
      true_admits_history_3_param = flu.torch_simulate_hospital_admits(
          base_state, params_3_param, base_precomputed, base_schedules, T, timesteps_per_day
          )
  
  print(f"Running Experiment 1: {len(true_betas_3_param_flat)} Parameters")
  print(f"True Betas: {true_betas_3_param_flat}")
  
  config_overrides_exp1 = {
      "beta_param": "L",
      "estimate_initial": {"E": False, "IP": False, "ISR": False, "ISH": False, "IA": False},
      "time_zero_mode": "absolute",
  }
  
  scale_factors = {'beta': 1.0}
  
  exp1_results_df, exp1_initial_guesses, exp1_theta_structure, exp1_config = run_metapop_calibration_suite(
      true_admits_history=true_admits_history_3_param,
      true_betas_flat=true_betas_3_param_flat,
      scale_factors=scale_factors,
      lambda_l2=1e-2,
      t0_day=0,
      window_days=T,
      config_overrides=config_overrides_exp1,
      state_at_t0=None   # uses base_state at day 0
  )
  
  plot_calibration_results(
      true_admits_history=true_admits_history_3_param,
      results_df=exp1_results_df,
      scale_factors={'beta': 1.0},
      true_betas_flat=true_betas_3_param_flat,
      initial_guesses=exp1_initial_guesses) # ADDED



  plot_subpop_timeseries_from_results(
      results_df=exp1_results_df,
      theta_structure=exp1_theta_structure,
      scale_factors={"beta": 1.0},
      true_params=params_3_param,
      simulate_states_fn=simulate_states_fn,
      T=T,
      timesteps_per_day=timesteps_per_day,
      mode="per_location",         # or "per_age", "per_loc_age"
      theta_col="final_theta",
      max_runs=1,
  )

  
  state_truth_E = copy.deepcopy(base_state)
  if hasattr(state_truth_E, "E"):
      state_truth_E.E = state_truth_E.E * 2.0  # arbitrary, just to have nontrivial E(0)
  
  with torch.no_grad():
      true_admits_history_3_param_E = flu.torch_simulate_hospital_admits(
          state_truth_E, params_3_param, base_precomputed, base_schedules, T, timesteps_per_day
      )
  
  print(f"\nRunning Experiment 2: Beta + initial E(0) parameters")
  
  config_overrides_exp2 = {
      "beta_param": "L",
      "estimate_initial": {"E": True, "IP": False, "ISR": False, "ISH": False, "IA": False},
      "time_zero_mode": "absolute",
  }
  
  exp2_results_df, exp2_initial_guesses, exp2_theta_structure, exp2_config = run_metapop_calibration_suite(
      true_admits_history=true_admits_history_3_param_E,
      true_betas_flat=true_betas_3_param_flat,
      scale_factors=scale_factors,
      lambda_l2=1e-2,
      t0_day=0,
      window_days=T,
      config_overrides=config_overrides_exp2,
      state_at_t0=state_truth_E   # we know the true E(0) in this synthetic example
  )

  plot_calibration_results(
    true_admits_history=true_admits_history_3_param_E,
    results_df=exp2_results_df,
    scale_factors={'beta': 1.0, 'E': 1.0},
    true_betas_flat=true_betas_3_param_flat,
    initial_guesses=exp2_initial_guesses,
    theta_structure=exp2_theta_structure,
    true_initial_compartments=true_initial_compartments,
)

  plot_compartment_timeseries(
    results_df=exp2_results_df,
    theta_structure=exp2_theta_structure,
    scale_factors={"beta": 1.0, "E": 1.0},
    true_state_at_t0=state_truth_E,      # the one you doubled E on
    true_params=params_3_param,
    simulate_states_fn=simulate_states_fn,
    T=T,
    timesteps_per_day=timesteps_per_day,
    compartments=("E",),                # later: ("E", "IP", "IA", ...)
)




import torch, time as global_time, pandas as pd, matplotlib.pyplot as plt, numpy as np, copy
from dataclasses import fields

import clt_toolkit as clt
import flu_core as flu

from scipy.optimize import least_squares, minimize
import multiprocessing as mp
from multiprocessing import Process, Queue
import queue

STOP_R2 = 0.5
STABLE_R2_FOR_LOSS_CHECK = STOP_R2 * 0.8
T = 180
timesteps_per_day = 4
true_gamma = 0.2
    
def main():
  torch.autograd.set_detect_anomaly(True)
  
  
  
  
  dt = 1.0 / timesteps_per_day
  time_steps = int(T/dt)
  
  torch.manual_seed(0)
  np.random.seed(0)
  
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 200)
  
  #load initial files
  texas_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"
  calibration_files_path = clt.utils.PROJECT_ROOT / "flu_instances" / "calibration_research_input_files"
  
  #json files for initial states and parameters per sub population
  subpopA_init_vals_filepath = calibration_files_path / "subpopA_init_vals.json"
  subpopB_init_vals_filepath = calibration_files_path / "subpopB_init_vals.json"
  subpopC_init_vals_filepath = calibration_files_path / "subpopC_init_vals.json"   # (note: your pasted file used B here; should be C)
  
  common_subpop_params_filepath = texas_files_path / "common_subpop_params.json"
  mixing_params_filepath = calibration_files_path / "ABC_mixing_params.json"
  simulation_settings_filepath = texas_files_path / "simulation_settings.json"
  
  #schedules
  calendar_df = pd.read_csv(texas_files_path / "school_work_calendar.csv", index_col=0)    # contact patterns over time
  humidity_df = pd.read_csv(texas_files_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
  vaccines_df = pd.read_csv(texas_files_path / "daily_vaccines_constant.csv", index_col=0)
  
  #time varying inputs the model uses each day
  schedules_info = flu.FluSubpopSchedules(
      absolute_humidity=humidity_df,
      flu_contact_matrix=calendar_df,
      daily_vaccines=vaccines_df
  )
  
  #initial values for sub population for schedules
  subpopA_init_vals = clt.make_dataclass_from_json(subpopA_init_vals_filepath, flu.FluSubpopState)
  subpopB_init_vals = clt.make_dataclass_from_json(subpopB_init_vals_filepath, flu.FluSubpopState)
  subpopC_init_vals = clt.make_dataclass_from_json(subpopC_init_vals_filepath, flu.FluSubpopState)
  
  common_subpop_params = clt.make_dataclass_from_json(common_subpop_params_filepath, flu.FluSubpopParams)
  
  mixing_params = clt.make_dataclass_from_json(mixing_params_filepath, flu.FluMixingParams)
  simulation_settings = clt.make_dataclass_from_json(simulation_settings_filepath, flu.SimulationSettings)
  
  
  simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": 4})
  
  L = 3 #locations / number of sub populations
  A = 5 #age groups
  R = 1 #risk strata
  
  bit_generator = np.random.MT19937(88888)
  jumped_bit_generator = bit_generator.jumped(1)
  
  #timestep
  
  #different beta baseline per subpop (this is what I will later estimate)
  subpopA_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 1.5})
  subpopB_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.5})
  subpopC_params = clt.updated_dataclass(common_subpop_params, {"beta_baseline": 2.2})
  
  #build object subpop models
  subpopA = flu.FluSubpopModel(subpopA_init_vals, subpopA_params, simulation_settings,
                               np.random.Generator(bit_generator), schedules_info, name="subpopA")
  subpopB = flu.FluSubpopModel(subpopB_init_vals, subpopB_params, simulation_settings,
                               np.random.Generator(jumped_bit_generator), schedules_info, name="subpopB")
  subpopC = flu.FluSubpopModel(subpopC_init_vals, subpopC_params, simulation_settings,
                               np.random.Generator(jumped_bit_generator), schedules_info, name="subpopC")
  
  #wrapper to apply mixing and traveling
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
  
  # ------------------------------------------------------------------
  # CONFIG: what do we estimate, and what is "time zero"?
  # ------------------------------------------------------------------
  ESTIMATION_CONFIG = {
      # β parameterization
      #   'L'  : one β per location (L)
      #   'LA' : one β per (location, age) (L×A)
      "beta_param": "L",
  
      # Which initial compartments to estimate at time zero.
      # Keys must match attributes on base_state: e.g., base_state.E, base_state.IP, etc.
      # True  => include in θ and estimate
      # False => keep JSON default and do NOT estimate
      "estimate_initial": {
          "E":   False,
          "IP":  False,   # later: split into IP1/IP2 if CLT exposes those
          "ISR":  False,   # later: split into ISR/ISH
          "ISH":  False,
          "IA":  False,
          # add more later if you like, e.g. "ISR", "ISH", "IP1", "IP2"
      },
  
      # Time-zero semantics
      #   'absolute' : time zero is season start (what your script does now)
      #   'relative' : time zero is some later day (e.g. 150) – see comments in loss fn
      "time_zero_mode": "absolute",
  
      # If time_zero_mode == 'relative', these matter:
      # "t0_day": 150,           # day index we define as time zero (0-based)
      # "window_days": 30,       # how many days of data we fit over
  
      # For now, with absolute t0, just fit entire T-day horizon:
      "t0_day": 0,
      "window_days": T,
  
      # IHR mode (2-stage calibration)
      "ihr_mode": False,
      "ihr_param": "L",                 # analogous to beta_param
      "ihr_aggregate_over_age": True,   # truth aggregation behavior
      "restart_widths_ihr_mode": {"Wide Search": 0.25, "Medium Search": 0.15, "Narrow Search": 0.05},
  
      # stage control (internal; set by suite)
      "hold_beta_fixed": False,
      "hold_ihr_fixed": False,
      "fixed_beta_tensor": None,
      "fixed_ihr_tensor": None,
  
  }
  
  
  d = flu_demo_model.get_flu_torch_inputs()
  
  state = d["state_tensors"] #tensors of compartments for S, E, etc.
  params = d["params_tensors"] #all static model parameters
  schedules = d["schedule_tensors"] #time varying parameters
  precomputed = d["precomputed"]
  
  init_state = copy.deepcopy(state)
  
  #for optimization
  opt_state = copy.deepcopy(state)
  opt_params = copy.deepcopy(params)
  
  L, A, R = init_state.S.shape
  
  true_admits_history = flu.torch_simulate_hospital_admits(
      state, params, precomputed, schedules, T, timesteps_per_day
  ).clone().detach()
  
  THETA_STRUCTURE = build_theta_structure(ESTIMATION_CONFIG, base_state, base_params)
  base_true_admits_history_for_worker = None


  #Case
  
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




import copy
import numpy as np
import torch
import multiprocessing as mp

import flu_core as flu
import clt_toolkit as clt
STOP_R2 = 0.5
STABLE_R2_FOR_LOSS_CHECK = STOP_R2 * 0.8
T = 180
timesteps_per_day = 4
true_gamma = 0.2

# ------------------------------------------------------------------
# θ structure: slices for β and initial compartments
# ------------------------------------------------------------------

def build_theta_structure(config, base_state, base_params):
    """
    Decide how many parameters we have and where each block lives in θ.

    Returns a dict:
        {
          'beta_param': 'L' or 'LA',
          'ihr_param': 'L' or 'LAR' (if estimate_ihr),
          'slices': {
              'beta': slice(...),
              'ihr': slice(...),
              'init_E': slice(...),
              ...
          },
          'size': total_length
        }
    """
    L, A, R = base_params.beta_baseline.shape
    slices = {}
    idx = 0

    estimate_beta = config.get("estimate_beta", True)
    estimate_ihr = config.get("estimate_ihr", False)

    # --- β block ---
    if estimate_beta:
        if config["beta_param"] == "L":
            n_beta = L
        elif config["beta_param"] == "LA":
            n_beta = L * A
        else:
            raise ValueError("beta_param must be 'L' or 'LA'")

        slices["beta"] = slice(idx, idx + n_beta)
        idx += n_beta

    # --- IHR block (optional) ---
    if estimate_ihr:
        ihr_param = config.get("ihr_param", "L")

        if ihr_param == "L":
            n_ihr = L
        elif ihr_param == "LAR":
            n_ihr = L * A * R
        else:
            raise ValueError("ihr_param must be 'L' or 'LAR'")

        slices["ihr"] = slice(idx, idx + n_ihr)
        idx += n_ihr

    # --- initial compartment blocks ---
    for comp_name, do_est in config["estimate_initial"].items():
        if not do_est:
            continue
        comp_tensor = getattr(base_state, comp_name)
        n_comp = comp_tensor.numel()          # typically L×A×R
        slices[f"init_{comp_name}"] = slice(idx, idx + n_comp)
        idx += n_comp

    out = {
        "beta_param": config["beta_param"],
        "slices": slices,
        "size": idx,
    }
    if estimate_ihr:
        out["ihr_param"] = config.get("ihr_param", "L")

    return out




def theta_to_betas(theta_vec, scale_factors, beta_param, L, A, R):
    """
    theta_vec: 1D numpy or torch of *log* β-parameters only (not the whole θ).
    beta_param: 'L' or 'LA'.
    Returns β_baseline tensor of shape [L, A, R].
    """
    if isinstance(theta_vec, np.ndarray):
        theta_t = torch.tensor(theta_vec, dtype=torch.float64)
    else:
        theta_t = theta_vec

    betas = torch.exp(theta_t) / scale_factors['beta']  # invert log-scale

    if beta_param == "L":
        return betas.view(L, 1, 1).expand(L, A, R)
    elif beta_param == "LA":
        return betas.view(L, A, 1).expand(L, A, R)
    else:
        raise ValueError("beta_param must be 'L' or 'LA'")

def theta_to_ihr(theta_ihr, scale_factors, ihr_param, L, A, R):
    ihr = torch.exp(theta_ihr) / float(scale_factors["ihr"])
    if ihr_param == "L":
        return ihr.view(L, 1, 1).expand(L, A, R)
    elif ihr_param == "LAR":
        return ihr.view(L, A, R)
    else:
        raise ValueError(f"Unsupported ihr_param={ihr_param}")


def apply_theta_to_model(theta_input, config, structure, base_state, base_params, scale_factors):
    """
    Given θ, build:
      - init_state (with possibly modified initial compartments)
      - params     (with β_baseline set from θ)

    Returns (init_state, params, theta_beta)
    """
    # Normalize input to a torch tensor, but do NOT break gradient if it's already a tensor
    if isinstance(theta_input, np.ndarray):
        theta = torch.tensor(theta_input, dtype=torch.float64, requires_grad=True)
    else:
        theta = theta_input.to(dtype=torch.float64)

    L, A, R = base_params.beta_baseline.shape
    slices = structure["slices"]

    # --- β block (θ stores log(β * scale_beta)) ---
    s_beta = slices["beta"]
    theta_beta = theta[s_beta]
    beta_tensor = theta_to_betas(
        theta_beta,
        scale_factors=scale_factors,
        beta_param=structure["beta_param"],
        L=L, A=A, R=R,
    )

    params = copy.deepcopy(base_params)
    # Respect fixed-parameter modes (used by IHR 2-stage)
    if config.get("hold_beta_fixed", False) and (config.get("fixed_beta_tensor", None) is not None):
        params.beta_baseline = config["fixed_beta_tensor"]
    else:
        params.beta_baseline = beta_tensor
    # Optional IHR estimation (unless held fixed)
    if config.get("hold_ihr_fixed", False) and (config.get("fixed_ihr_tensor", None) is not None):
        params.ihr_baseline = config["fixed_ihr_tensor"]
    elif structure is not None and "ihr" in structure.get("slices", {}) and config.get("estimate_ihr", False):
        s_ihr = structure["slices"]["ihr"]
        theta_ihr = theta[s_ihr]
        ihr_tensor = theta_to_ihr(
            theta_ihr,
            scale_factors=scale_factors,
            ihr_param=structure.get("ihr_param", config.get("ihr_param", "L")),
            L=L, A=A, R=R,
        )
        params.ihr_baseline = ihr_tensor


    # --- initial state blocks (θ stores log(compartment * scale_comp)) ---
    init_state = copy.deepcopy(base_state)

    for comp_name, do_est in config["estimate_initial"].items():
        if not do_est:
            continue

        key = f"init_{comp_name}"
        if key not in slices:
            raise KeyError(f"{key} missing from theta_structure['slices'].")

        s_comp = slices[key]
        theta_comp = theta[s_comp]

        comp_scale = float(scale_factors.get(comp_name, 1.0))
        # invert log-scale and scaling
        new_vals = (torch.exp(theta_comp) / comp_scale).view_as(getattr(init_state, comp_name))
        setattr(init_state, comp_name, new_vals)

    # Ensure H(0) is zero if present
    if hasattr(init_state, "H"):
        init_state.H[:] = 0.0

    return init_state, params, theta_beta


def build_trajectory_importance_weights(observed_history):
    observed_1d = torch.sum(observed_history, dim=(1, 2, 3))
    dI_dt = torch.abs(observed_1d[1:] - observed_1d[:-1]) / dt
    dI_dt = torch.cat([dI_dt[:1], dI_dt])
    importance = torch.sqrt(dI_dt + 1e-6)
    weights = importance / importance.mean()
    weights = torch.clamp(weights, min=0.1, max=5.0)
    return weights.view(-1, 1, 1, 1)

def create_metapop_loss_function(
    observed_history,      # tensor [T_obs, L, A, R] – already sliced to the window you care about
    base_state_t0,
    base_params, precomputed, schedules,
    scale_factors,
    lambda_l2=0.0,
    config=None,
    structure=None,
    timesteps_per_day=4
):
    """
    observed_history should already reflect:
      - absolute t0 (day 0) OR
      - a relative window (e.g. days 150–180) if you decide to slice outside.
      base_state_t0   : state to start from at that t0.

    This keeps the function focused: given observed_history and θ, compute MSE.
    """
    if config is None:
        config = ESTIMATION_CONFIG
    if structure is None:
        structure = THETA_STRUCTURE

    observed_history = observed_history.double()
    y_mean = torch.mean(observed_history)
    ss_tot = torch.sum((observed_history - y_mean) ** 2).item()

    T_obs = observed_history.shape[0]

    # same importance weighting you had before
    def build_trajectory_importance_weights(obs):
        obs_1d = torch.sum(obs, dim=(1, 2, 3))
        dI_dt = torch.abs(obs_1d[1:] - obs_1d[:-1]) * timesteps_per_day  # per day
        dI_dt = torch.cat([dI_dt[:1], dI_dt])
        importance = torch.sqrt(dI_dt + 1e-6)
        weights = importance / importance.mean()
        return torch.clamp(weights, min=0.1, max=5.0).view(-1, 1, 1, 1)

    weights = build_trajectory_importance_weights(observed_history)

    current_stats = {'loss': None, 'r_squared': None, 'should_stop': False}

    def loss_and_grad(x_np, tracer):
        """
        x_np: θ as numpy array (log-βs and optionally log-initials).
        tracer: None or any truthy value; when truthy we print progress.
        """
        nonlocal current_stats

        if current_stats.get('should_stop', False):
            # short-circuit if we already decided to stop
            return current_stats['loss'], np.zeros_like(x_np)

        try:
            theta_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)

            # Map θ to (state_t0, params_t0)
            init_state_grad, params_grad, theta_beta = apply_theta_to_model(
                theta_t, config, structure, base_state_t0, base_params, scale_factors
            )

            pred_history = flu.torch_simulate_hospital_admits(
                init_state_grad, params_grad, precomputed, schedules,
                T_obs, timesteps_per_day
            )

            weighted_mse = torch.sum(((pred_history - observed_history) ** 2) * weights) / torch.sum(weights)
            l2_penalty = lambda_l2 * torch.sum(theta_beta**2)
            loss = weighted_mse + l2_penalty

            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite")

            ss_res = torch.sum((observed_history - pred_history) ** 2).item()
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            current_stats.update({'loss': loss.item(), 'r_squared': r_squared})

            # Early stop if R² high enough
            if r_squared >= STOP_R2 and tracer is not None:
                print(f"Stopping optimization: R-squared = {r_squared:.6f} >= {STOP_R2}")
                current_stats['should_stop'] = True
                return loss.item(), np.zeros_like(x_np)

            loss.backward()
            grad = theta_t.grad.detach().numpy() if theta_t.grad is not None else np.zeros_like(x_np)

            if tracer is not None:
                print(f"Loss: {loss.item():.4f}, Grad norm: {np.linalg.norm(grad):.2e}, R²: {r_squared:.4f}")

            return loss.item(), grad

        except Exception as e:
            print(f"[loss_and_grad] Exception: {e}")
            return 1e12, np.zeros_like(x_np)

    return loss_and_grad, ss_tot, current_stats, weights

class StopOptimization(Exception):
    pass

def get_parameter_bounds(num_params, beta_min=1e-5, beta_max=5.0, scale_factors=None):
    log_min = np.log(beta_min * scale_factors['beta']) if scale_factors else np.log(beta_min)
    log_max = np.log(beta_max * scale_factors['beta']) if scale_factors else np.log(beta_max)
    return [(log_min, log_max)] * num_params

def generate_restart_guess(deviation_factor, true_betas, scale_factors):
    log_true_betas = np.log10(np.clip(true_betas, 1e-9, 1e9))
    log_random_betas = np.random.uniform(log_true_betas - deviation_factor, log_true_betas + deviation_factor)
    return np.log((10 ** log_random_betas) * scale_factors['beta'])

def optimizer_worker(result_queue, progress_queue,
                     optimizer_name, x0, loss_and_grad_fn,
                     scale_factors, rsquared_threshold,
                     lambda_l2, current_stats, weights):
    try:
        iteration_counter = 0

        if 'least_squares' in optimizer_name:

            def residual_fn(x_np):
                with torch.no_grad():

                    betas = np.exp(x_np) / scale_factors['beta']
                    opt_params_copy = copy.deepcopy(base_params)
                    Lp, Ap, Rp = opt_params_copy.beta_baseline.shape
                    if len(betas) == Lp * Ap:
                        opt_params_copy.beta_baseline = torch.tensor(betas, dtype=torch.float64).view(Lp, Ap, Rp)
                    elif len(betas) == Lp:
                        opt_params_copy.beta_baseline = torch.tensor(betas, dtype=torch.float64).view(Lp, 1, 1).expand(Lp, Ap, Rp)
                    else:
                        raise ValueError("β shape mismatch in least_squares residual_fn")

                                        # Number of time steps in the observed window
                    T_steps = base_true_admits_history_for_worker.shape[0]
                    steps_per_day = timesteps_per_day

                    # Convert to an integer number of days
                    T_days = int(T_steps // steps_per_day)

                    pred_history = flu.torch_simulate_hospital_admits(
                        base_state, opt_params_copy, base_precomputed, base_schedules,
                        T_days,
                        timesteps_per_day
                    )

                    residuals = (pred_history - base_true_admits_history_for_worker)
                    return residuals.cpu().numpy().flatten()

            res = least_squares(fun=residual_fn, x0=x0, method='trf', jac='2-point')

            result_queue.put({
                'x': res.x, 'loss': res.cost * 2,
                'success': res.success, 'status': 'Completed',
                'nit': res.nfev
                })

        elif 'Adam' in optimizer_name:
            theta = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            adam_optimizer = torch.optim.Adam([theta], lr=0.01)
            final_loss = 1e12

            for i in range(1000):
                adam_optimizer.zero_grad()
                loss, grad_np = loss_and_grad_fn(theta.detach().numpy(), None)
                final_loss = loss
                if not np.isfinite(loss):
                  break
                theta.grad = torch.tensor(grad_np)
                adam_optimizer.step()

                r_squared = current_stats['r_squared']
                progress_queue.put_nowait({
                    'iter': i + 1, 'loss': loss,
                    'r_squared': r_squared,
                    'params': theta.detach().numpy()
                    })
                if r_squared >= rsquared_threshold:
                  raise StopOptimization("R-squared threshold met")

            res = {
                'x': theta.detach().numpy(),
                'loss': final_loss,
                'success': True,
                'status': 'Completed',
                'nit': 1000
                }
            result_queue.put(res)

        else: #CG or L-BFGS-B
            _prev = {'x': None, 'f': None, 'g': None}

            def loss_and_grad_for_scipy(x_np):
                if _prev['x'] is not None and np.allclose(x_np, _prev['x']):
                  return _prev['f'], _prev['g']
                f, g = loss_and_grad_fn(x_np, "print")
                _prev.update({'x': x_np.copy(), 'f': f, 'g': g.copy()})
                return f, g

            def callback(xk):
                nonlocal iteration_counter

                iteration_counter += 1

                progress_queue.put_nowait({
                    'iter': iteration_counter,
                    'loss': _prev['f'],
                    'r_squared': current_stats['r_squared'],
                    'params': xk
                    })
                if current_stats['r_squared'] >= rsquared_threshold:
                  raise StopOptimization("R-squared threshold met")

            res = minimize(
                fun=loss_and_grad_for_scipy,
                x0=x0,
                jac=True,
                method=optimizer_name,
                bounds=get_parameter_bounds(len(x0), scale_factors=scale_factors)
                  if 'L-BFGS-B' in optimizer_name else None,
                callback=callback,
                options={'maxiter': 1000}
                )
            result_queue.put({
                'x': res.x,
                'loss': res.fun,
                'success': res.success,
                'status': 'Completed',
                'nit': res.nit
                })

    except StopOptimization as e:
        final_params = x0
        while not progress_queue.empty():
          final_params = progress_queue.get()['params']
        result_queue.put({
            'x': final_params,
            'loss': None,
            'success': True,
            'status': f'Success: {e}',
            'nit': iteration_counter
            })

    except Exception as e:
        result_queue.put({
            'x': x0,
            'loss': 1e7,
            'success': False,
            'status': f'Failed: {e}',
            'nit': 0
            })


def optimizer_worker_process(
    optimizer_name,
    x0,
    loss_and_grad_fn,
    lambda_l2,
    weights,
    time_limit,
    result_queue,
):
    """
    Worker process that runs a single optimization attempt for one optimizer.

    It:
      - starts from x0 (scaled parameter vector),
      - calls loss_and_grad_fn(theta_np, lambda_l2, weights),
      - runs the chosen optimizer (CG, L-BFGS-B, Adam, least_squares_fd),
      - pushes a dict into result_queue with theta_opt, loss, r_squared, nit.
    """
    start_time = global_time.time()
    x0 = np.asarray(x0, dtype=float)

    # Default payload if something goes wrong
    payload = {
        "theta_opt": x0.copy(),
        "loss": float("inf"),
        "r_squared": float("-inf"),
        "nit": 0,
    }

    try:
        # -----------------------------
        # Common wrappers
        # -----------------------------
        def fun(theta_vec):
            loss, grad, r2 = loss_and_grad_fn(
                np.asarray(theta_vec, dtype=float),
                lambda_l2,
                weights
            )
            return float(loss)

        def jac(theta_vec):
            loss, grad, r2 = loss_and_grad_fn(
                np.asarray(theta_vec, dtype=float),
                lambda_l2,
                weights
            )
            return np.asarray(grad, dtype=float)

        # -----------------------------------
        # 1) CG / L-BFGS-B via scipy.minimize
        # -----------------------------------
        if optimizer_name in ("CG", "L-BFGS-B"):
            res = minimize(
                fun,
                x0,
                method=optimizer_name,
                jac=jac,
                options={"maxiter": 1000}
            )
            theta_opt = np.asarray(res.x, dtype=float)
            nit = int(getattr(res, "nit", 0))
            loss, grad, r2 = loss_and_grad_fn(theta_opt, lambda_l2, weights)

        # -----------------------------------
        # 2) Adam: manual implementation
        # -----------------------------------
        elif optimizer_name == "Adam":
            theta = x0.copy()
            m = np.zeros_like(theta)
            v = np.zeros_like(theta)
            beta1, beta2 = 0.9, 0.999
            alpha = 0.05
            eps = 1e-8
            max_iters = 2000

            nit = 0
            last_loss = None
            last_r2 = None

            while nit < max_iters and (global_time.time() - start_time) < time_limit:
                loss, grad, r2 = loss_and_grad_fn(theta, lambda_l2, weights)
                nit += 1

                g = np.asarray(grad, dtype=float)

                # Adam updates
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g * g)

                m_hat = m / (1.0 - beta1 ** nit)
                v_hat = v / (1.0 - beta2 ** nit)

                theta = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)

                last_loss = float(loss)
                last_r2 = float(r2)

            # Final eval at last theta
            if last_loss is None or last_r2 is None:
                loss, grad, r2 = loss_and_grad_fn(theta, lambda_l2, weights)
            else:
                loss, r2 = last_loss, last_r2

            theta_opt = np.asarray(theta, dtype=float)

        # -----------------------------------
        # 3) least_squares_fd via scipy.least_squares
        # -----------------------------------
        elif optimizer_name == "least_squares_fd":
            # Use sqrt(loss) as a single residual. This is enough for LS to
            # mimic minimizing your scalar loss, with finite-difference Jacobian.
            def residuals(theta_vec):
                loss, grad, r2 = loss_and_grad_fn(
                    np.asarray(theta_vec, dtype=float),
                    lambda_l2,
                    weights
                )
                # Make sure it's non-negative before sqrt
                return np.array([np.sqrt(max(float(loss), 0.0))], dtype=float)

            res = least_squares(
                residuals,
                x0,
                jac='2-point',
                max_nfev=1000
            )
            theta_opt = np.asarray(res.x, dtype=float)
            nit = int(getattr(res, "nfev", 0))
            loss, grad, r2 = loss_and_grad_fn(theta_opt, lambda_l2, weights)

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        payload["theta_opt"] = theta_opt
        payload["loss"] = float(loss)
        payload["r_squared"] = float(r2)
        payload["nit"] = int(nit)

    except Exception as e:
        # Leave payload as default; parent will see inf / -inf and can log
        # We still push something, so the parent knows the worker completed.
        print(f"[optimizer_worker_process] Exception in {optimizer_name}: {e}")

    result_queue.put(payload)

def manage_single_optimization_attempt(
    optimizer_name,
    x0,
    loss_and_grad_fn,
    scale_factors,
    lambda_l2,
    rsquared_threshold,
    current_stats,
    weights,
    time_limit,
):
    """
    Run a single optimization attempt in a separate process (with timeout).

    Returns a dict with attempt metadata + theta/loss/r2.
    """
    start_time = global_time.time()
    x0 = np.asarray(x0, dtype=float)

    result = {
        "optimizer": optimizer_name,
        "initial_guess_id": current_stats.get("initial_guess_id", None),
        "phase": current_stats.get("phase", None),
        "restart_num": current_stats.get("restart_num", 0),
        "loss": float("inf"),
        "r_squared": float("-inf"),
        "duration": 0.0,
        "nit": 0,
        "theta_opt": x0.copy(),
        "final_theta": x0.tolist(),      # <-- NEW: always present for plotting
        "timed_out": False,              # <-- NEW
        "success": False,                # <-- NEW (based on r2 threshold)
    }

    result_queue = mp.Queue()

    print(f"Running {optimizer_name} (PID: TBD) with a {time_limit}s time limit")

    p = mp.Process(
        target=optimizer_worker_process,
        args=(
            optimizer_name,
            x0,
            loss_and_grad_fn,
            lambda_l2,
            weights,
            time_limit,
            result_queue,
        )
    )
    p.start()
    print(f"{optimizer_name} worker started with PID: {p.pid}")

    # Give the worker a small grace window to flush payload into the queue.
    grace_s = 5.0
    p.join(time_limit + grace_s)

    duration = global_time.time() - start_time
    result["duration"] = float(duration)

    if p.is_alive():
        print(f"{optimizer_name} (PID: {p.pid}) timed out after {time_limit}s (+{grace_s}s grace). Terminating.")
        result["timed_out"] = True
        p.terminate()
        p.join()

    # Even if we timed out, try to read whatever the worker managed to enqueue.
    payload = None
    try:
        # Prefer non-blocking read
        if not result_queue.empty():
            payload = result_queue.get_nowait()
    except Exception:
        payload = None

    if payload is not None:
        theta_opt = np.asarray(payload.get("theta_opt", x0), dtype=float)
        loss_val = float(payload.get("loss", float("inf")))
        r2_val = float(payload.get("r_squared", float("-inf")))
        nit_val = int(payload.get("nit", 0))

        result["theta_opt"] = theta_opt
        result["final_theta"] = theta_opt.tolist()
        result["loss"] = loss_val
        result["r_squared"] = r2_val
        result["nit"] = nit_val

        # Define "success" in the one place that matters: the reported r2.
        if np.isfinite(result["r_squared"]) and (result["r_squared"] >= float(rsquared_threshold)):
            result["success"] = True
    else:
        print(f"{optimizer_name} (PID: {p.pid}) finished but returned no payload.")

    print(
        f"{optimizer_name} finished attempt: "
        f"loss={result['loss']:.4f}, R²={result['r_squared']:.4f}, "
        f"nit={result['nit']}, duration={duration:.2f}s, timed_out={result['timed_out']}"
    )

    return result


def build_loss_function_and_stats(
    observed_window,
    state_at_t0,
    config,
    structure,
    scale_factors,
    true_betas_flat,   # still accepted for interface, not used here
    lambda_l2,
):
    """
    Build a standardized loss_and_grad_fn that ALWAYS has signature:
        loss_and_grad_fn(theta_np, lambda_l2, weights)
    and ALWAYS returns a 3-tuple:
        (loss_value, grad_flat, aux_dict)

    Returns
    -------
    loss_and_grad_fn : callable
    ss_tot : any
    current_stats : dict
    weights : any
    """

    # --- Resolve base_params / precomputed / schedules from config or globals ---
    g = globals()

    if "base_params" in config:
        base_params_local = config["base_params"]
    else:
        if "base_params" not in g:
            raise RuntimeError("base_params not found in config or globals.")
        base_params_local = g["base_params"]

    if "base_precomputed" in config:
        precomputed_local = config["base_precomputed"]
    else:
        if "base_precomputed" not in g:
            raise RuntimeError("base_precomputed not found in config or globals.")
        precomputed_local = g["base_precomputed"]

    if "base_schedules" in config:
        schedules_local = config["base_schedules"]
    else:
        if "base_schedules" not in g:
            raise RuntimeError("base_schedules not found in config or globals.")
        schedules_local = g["base_schedules"]

    # --- Call your existing helper exactly with the args it expects ---
    base_loss_fn, ss_tot, current_stats, weights = create_metapop_loss_function(
        observed_history = observed_window,
        base_state_t0    = state_at_t0,
        base_params      = base_params_local,
        precomputed      = precomputed_local,
        schedules        = schedules_local,
        structure        = structure,
        scale_factors    = scale_factors,
        lambda_l2        = lambda_l2,
    )

    # IHR mode: compare admits aggregated over age/risk (per-location series)
    if config.get("ihr_mode", False) and config.get("ihr_aggregate_over_age", True):
        # predicted: (time, L, A, R) -> (time, L)
        pred_window = pred_window.sum(dim=(-2, -1))
        # observed stored in global base_true_admits_history_for_worker
        # ensure it is also (time, L)
        obs = base_true_admits_history_for_worker
        if torch.is_tensor(obs) and obs.dim() > 2:
            obs = obs.sum(dim=(-2, -1))
    else:
        obs = base_true_admits_history_for_worker


    # --- Adapter: normalize arguments AND outputs ---
    def loss_and_grad_adapter(theta_np, lambda_l2_unused, weights_local):
        """
        Called by optimizer_worker_process as:
            loss_and_grad_fn(theta_np, lambda_l2, weights)

        We must ALWAYS return (loss_value, grad_flat, aux_dict).
        """
        # First handle argument arity
        try:
            out = base_loss_fn(theta_np, weights_local)
        except TypeError:
            # If base_loss_fn only takes theta_np
            out = base_loss_fn(theta_np)

        r2 = current_stats.get("r_squared", 0.0)

        # Now normalize the output shape
        if isinstance(out, tuple):
          if len(out) == 3:
              # (loss, grad, r2_like) – if you ever return that directly
              loss_val, grad_flat, r2_local = out
              # trust r2_local if it’s numeric
              try:
                  r2_val = float(r2_local)
              except (TypeError, ValueError):
                  r2_val = float(r2)
              return loss_val, grad_flat, r2_val

          elif len(out) == 2:
              # (loss, grad) – add r2
              loss_val, grad_flat = out
              return loss_val, grad_flat, float(r2)

          elif len(out) == 1:
              # (loss,) – no gradient
              loss_val = out[0]
              grad_flat = None
              return loss_val, grad_flat, float(r2)


        # Non-tuple: treat as scalar loss only
        loss_val = out
        grad_flat = None
        return loss_val, grad_flat, float(r2)

    return loss_and_grad_adapter, ss_tot, current_stats, weights

def draw_restart_point_around_true(
    base_x0,
    phase,
    structure,
    scale_factors,
    true_betas_flat,
    rng=None,
):
    """
    Generate a new starting point where the *beta* coordinates are sampled
    uniformly from a box around the TRUE betas.

    - Wide:   beta in [0.25 * true, 1.75 * true]   (±75%)
    - Medium: beta in [0.50 * true, 1.50 * true]   (±50%)
    - Narrow: beta in [0.75 * true, 1.25 * true]   (±25%)

    Other parameters (e.g., gamma) are copied from base_x0.
    """

    if rng is None:
        rng = np.random.default_rng()

    phase_lower = str(phase).lower()
    # IHR mode: narrower restart widths
    default_widths = {"Wide Search": 0.75, "Medium Search": 0.50, "Narrow Search": 0.25}
    ihr_widths = {"Wide Search": 0.25, "Medium Search": 0.15, "Narrow Search": 0.05}

    use_ihr_widths = False
    if "config" in structure and isinstance(structure["config"], dict):
        use_ihr_widths = structure["config"].get("ihr_mode", False)

    widths = ihr_widths if use_ihr_widths else default_widths
    restart_width = widths.get(phase, 0.25)


    x0 = np.asarray(base_x0, dtype=float).copy()

    # Where betas live inside theta:
    beta_slice = structure["slices"]["beta"]
    true_betas = np.asarray(true_betas_flat, dtype=float)

    # Sanity: match dimensions
    n_beta_theta = beta_slice.stop - beta_slice.start
    if true_betas.shape[0] != n_beta_theta:
        raise ValueError(
            f"true_betas_flat has length {true_betas.shape[0]}, "
            f"but beta_slice indicates {n_beta_theta} betas in theta."
        )

    # Build the box in beta-space
    low  = true_betas * (1.0 - restart_width)
    high = true_betas * (1.0 + restart_width)

    # Ensure strictly positive betas (important before log)
    low = np.maximum(low, 1e-8)

    # Sample uniformly within the box for each beta
    sampled_betas = rng.uniform(low=low, high=high)

    # Map beta -> theta (log-scaled using your scale_factors)
    beta_scale = scale_factors["beta"]
    theta_beta = np.log(sampled_betas * beta_scale)

    # Overwrite the beta part of theta; leave others (e.g. gamma) as in base_x0
    x0[beta_slice] = theta_beta

    return x0









def run_optimizer_with_restarts(
    optimizer_name,
    x0,
    observed_window,
    state_at_t0,
    config,
    structure,
    scale_factors,
    lambda_l2,
    true_betas_flat,
    stable_loss_tolerance=0.05,
    time_limit=300,
):
    """
    Runs a given optimizer multiple times (initial + restarts) with different
    search widths (wide/medium/narrow).

    IMPORTANT FIXES:
      - phase / restart_num are set meaningfully for every attempt
      - returns attempts with final_theta populated (via manage_single_optimization_attempt)
    """
    loss_fn, _, current_stats, weights = build_loss_function_and_stats(
        observed_window=observed_window,
        state_at_t0=state_at_t0,
        config=config,
        structure=structure,
        scale_factors=scale_factors,
        true_betas_flat=true_betas_flat,
        lambda_l2=lambda_l2
    )

    STOP_R2 = config.get("rsquared_threshold", 0.9)

    all_attempts_for_this_run = []
    best_result_so_far = None

    # -------------------------
    # Initial attempt
    # -------------------------
    current_stats["phase"] = "Initial"
    current_stats["restart_num"] = 0

    initial_result = manage_single_optimization_attempt(
        optimizer_name=optimizer_name,
        x0=x0,
        loss_and_grad_fn=loss_fn,
        scale_factors=scale_factors,
        lambda_l2=lambda_l2,
        rsquared_threshold=STOP_R2,
        current_stats=current_stats,
        weights=weights,
        time_limit=time_limit,
    )

    beta_slice = structure["slices"]["beta"]
    thetalog = np.asarray(initial_result["theta_opt"], dtype=float)[beta_slice]
    betas = np.exp(thetalog) / scale_factors["beta"]
    initial_result["final_betas"] = betas.tolist()

    all_attempts_for_this_run.append(initial_result)
    best_result_so_far = initial_result

    if initial_result.get("success", False) and initial_result.get("r_squared", -1.0) >= STOP_R2:
        return best_result_so_far, all_attempts_for_this_run

    # -------------------------
    # Restarts
    # -------------------------
    restart_settings = [
        ("Wide Search",   config.get("num_wide_restarts",   2)),
        ("Medium Search", config.get("num_medium_restarts", 2)),
        ("Narrow Search", config.get("num_narrow_restarts", 2)),
    ]

    # IHR mode uses narrower restart widths (handled inside draw_restart_point_around_true)
    if config.get("ihr_mode", False):
        if "config" not in structure:
            structure["config"] = {}
        structure["config"]["ihr_mode"] = True


    for phase_name, num_restarts in restart_settings:
        for restart_idx in range(num_restarts):
            print(f"\n{optimizer_name}: Restart {restart_idx+1}/{num_restarts} ({phase_name})")

            current_stats["phase"] = phase_name
            current_stats["restart_num"] = int(restart_idx + 1)

            restart_x0 = draw_restart_point_around_true(
                base_x0=x0,
                phase=phase_name,
                structure=structure,
                scale_factors=scale_factors,
                true_betas_flat=true_betas_flat,
            )

            restart_result = manage_single_optimization_attempt(
                optimizer_name=optimizer_name,
                x0=restart_x0,
                loss_and_grad_fn=loss_fn,
                scale_factors=scale_factors,
                lambda_l2=lambda_l2,
                rsquared_threshold=STOP_R2,
                current_stats=current_stats,
                weights=weights,
                time_limit=time_limit,
            )

            thetalog = np.asarray(restart_result["theta_opt"], dtype=float)[beta_slice]
            betas = np.exp(thetalog) / scale_factors["beta"]
            restart_result["final_betas"] = betas.tolist()

            all_attempts_for_this_run.append(restart_result)

            if (best_result_so_far is None) or (restart_result["loss"] < best_result_so_far["loss"]):
                best_result_so_far = restart_result

            if restart_result.get("success", False) and restart_result.get("r_squared", -1.0) >= STOP_R2:
                print("Stopping: R-squared threshold met; no further restarts for this optimizer.")
                return best_result_so_far, all_attempts_for_this_run

            loss_improvement = (
                (best_result_so_far["loss"] - restart_result["loss"])
                / max(abs(best_result_so_far["loss"]), 1e-8)
            )
            if abs(loss_improvement) <= stable_loss_tolerance:
                print("Stopping: Stable local optimum found (loss no longer improving).")
                return best_result_so_far, all_attempts_for_this_run

    return best_result_so_far, all_attempts_for_this_run


def _run_single_stage(
    observed_window,
    state_at_t0,
    config,
    structure,
    scale_factors,
    lambda_l2,
    true_betas_flat,
    optimizers_to_run=None,
    time_limit=300,
):
    """
    Runs the optimization suite for ONE stage (e.g., beta-only, or IHR-only).

    Returns:
        results_df, initial_guesses
    """
    if optimizers_to_run is None:
        optimizers_to_run = ['CG', 'L-BFGS-B', 'Adam', 'least_squares_fd']

    theta_size = int(structure["size"])
    slices = structure["slices"]

    # ---- initial guess builder (stage-aware) ----
    def _make_stage_theta_guess(randomize=False):
        theta0 = np.zeros(theta_size, dtype=float)

        # --- β (only if present in structure) ---
        if "beta" in slices:
            beta_slice = slices["beta"]
            n_beta = beta_slice.stop - beta_slice.start
            if not randomize:
                beta0 = np.full(n_beta, 0.005, dtype=float)
            else:
                beta0 = np.random.uniform(0.003, 0.006, size=n_beta)
            theta0[beta_slice] = np.log(beta0 * float(scale_factors["beta"]))

        # --- IHR (only if present in structure) ---
        if "ihr" in slices:
            ihr_slice = slices["ihr"]
            n_ihr = ihr_slice.stop - ihr_slice.start

            # default guess for IHR
            if not randomize:
                ihr0 = np.full(n_ihr, 0.02, dtype=float)
            else:
                ihr0 = np.random.uniform(0.01, 0.03, size=n_ihr)

            theta0[ihr_slice] = np.log(ihr0 * float(scale_factors.get("ihr", 1.0)))

        # --- initial compartments (if enabled) ---
        for comp_name, do_est in config["estimate_initial"].items():
            if not do_est:
                continue
            key = f"init_{comp_name}"
            if key not in slices:
                raise KeyError(f"{key} missing from theta_structure['slices'] in this stage.")
            s = slices[key]

            base_comp = getattr(state_at_t0, comp_name)
            comp_scale = float(scale_factors.get(comp_name, 1.0))
            comp0 = base_comp.detach().cpu().numpy().reshape(-1).astype(float)
            comp0 = np.clip(comp0, 1e-12, None)

            if not randomize:
                theta0[s] = np.log(comp0 * comp_scale)
            else:
                jitter = np.random.uniform(0.8, 1.25, size=comp0.shape[0])
                theta0[s] = np.log((comp0 * jitter) * comp_scale)

        return theta0

    initial_guesses = [
        _make_stage_theta_guess(randomize=False),
        _make_stage_theta_guess(randomize=True),
    ]

    # ---- attempt log ----
    all_attempts_log = []

    # helper to attach init/final compartment blocks
    def _attach_init_final_compartments(attempt_dict, x0_theta, theta_opt):
        for comp_name, do_est in config["estimate_initial"].items():
            if not do_est:
                continue
            key = f"init_{comp_name}"
            if key not in slices:
                continue

            s = slices[key]
            comp_scale = float(scale_factors.get(comp_name, 1.0))

            init_vals = np.exp(np.asarray(x0_theta[s], dtype=float)) / comp_scale
            final_vals = np.exp(np.asarray(theta_opt[s], dtype=float)) / comp_scale

            attempt_dict[f"initial_{comp_name}"] = init_vals.tolist()
            attempt_dict[f"final_{comp_name}"] = final_vals.tolist()

    # ---- run optimizers ----
    for optimizer_name in optimizers_to_run:
        print("\n" + "#" * 60 + f"\nTESTING OPTIMIZER: {optimizer_name}\n" + "#" * 60)

        for i, x0 in enumerate(initial_guesses):
            print("\n" + "=" * 50 + f"\nOPTIMIZATION RUN {i+1}/{len(initial_guesses)} for {optimizer_name}\n" + "=" * 50)

            best_result_so_far, attempts = run_optimizer_with_restarts(
                optimizer_name=optimizer_name,
                x0=x0,
                observed_window=observed_window,
                state_at_t0=state_at_t0,
                config=config,
                structure=structure,
                scale_factors=scale_factors,
                lambda_l2=lambda_l2,
                true_betas_flat=true_betas_flat,
                time_limit=time_limit,
            )

            # Initial betas (only if this stage includes beta)
            if "beta" in slices:
                beta_slice = slices["beta"]
                initial_betas_for_this_run = (np.exp(x0[beta_slice]) / float(scale_factors["beta"])).tolist()
            else:
                initial_betas_for_this_run = None

            for attempt in attempts:
                attempt["initial_guess_id"] = i + 1

                if initial_betas_for_this_run is not None:
                    attempt["initial_betas"] = initial_betas_for_this_run

                # Ensure final_theta exists for plotting
                if "final_theta" not in attempt:
                    attempt["final_theta"] = np.asarray(attempt.get("theta_opt", x0), dtype=float).tolist()

                theta_opt = np.asarray(attempt.get("theta_opt", x0), dtype=float)

                # Derive final_betas if this stage includes beta
                if "beta" in slices:
                    beta_log = theta_opt[slices["beta"]]
                    attempt["final_betas"] = (np.exp(beta_log) / float(scale_factors["beta"])).tolist()

                # Derive final_ihr if this stage includes ihr
                if "ihr" in slices:
                    ihr_log = theta_opt[slices["ihr"]]
                    attempt["final_ihr"] = (np.exp(ihr_log) / float(scale_factors.get("ihr", 1.0))).tolist()

                _attach_init_final_compartments(attempt, x0_theta=x0, theta_opt=theta_opt)

                all_attempts_log.append(attempt)

    results_df = pd.DataFrame(all_attempts_log)
    return results_df, initial_guesses


def run_metapop_calibration_suite(
    true_admits_history,
    true_betas_flat,
    scale_factors,
    lambda_l2,
    t0_day=0,
    window_days=None,
    config_overrides=None,
    state_at_t0=None
):
    """
    IMPORTANT FIXES:
      - initial_guesses are ALWAYS full-theta length (theta_size), not sometimes num_params
      - attempts log includes final_theta (from manage_single_optimization_attempt)
      - best-runs summary now includes initial/final compartment tensors (flattened lists)
        for any estimated initial compartments (e.g., E) via init_* slices.
    """
    global base_true_admits_history_for_worker

    if window_days is None:
        window_days = T

    local_config = copy.deepcopy(ESTIMATION_CONFIG)
    local_config["t0_day"] = t0_day
    local_config["window_days"] = window_days

    if config_overrides is not None:
        for k, v in config_overrides.items():
            local_config[k] = v

    local_structure = build_theta_structure(local_config, base_state, base_params)
    observed_window = true_admits_history[start_step:start_step + n_steps]
    # IHR mode uses a 2-stage calibration:
    #   Stage 1: estimate beta only (hold IHR fixed)
    #   Stage 2: estimate IHR only (hold beta fixed at stage-1 optimum)
    if local_config.get("ihr_mode", False):
        # ---------- Stage 1 ----------
        cfg_stage1 = copy.deepcopy(local_config)
        cfg_stage1["estimate_ihr"] = False
        cfg_stage1["hold_ihr_fixed"] = True
        cfg_stage1["fixed_ihr_tensor"] = getattr(base_params, "ihr_baseline", None)

        structure_stage1 = build_theta_structure(cfg_stage1, base_state, base_params)

        # Run stage 1 by calling a small internal helper that performs your current “optimizer loops”
        results_stage1, init_guesses_stage1 = _run_single_stage(
            observed_window=observed_window,
            state_at_t0=state_at_t0,
            config=cfg_stage1,
            structure=structure_stage1,
            scale_factors=scale_factors,
            lambda_l2=lambda_l2,
            true_betas_flat=true_betas_flat,
        )

        # Pick best overall beta tensor from stage 1
        best1 = results_stage1.loc[results_stage1["loss"].idxmin()]
        theta_hat1 = np.asarray(best1["theta_opt"], dtype=float)
        beta_log1 = theta_hat1[structure_stage1["slices"]["beta"]]
        beta_vec1 = np.exp(beta_log1) / float(scale_factors["beta"])
        L, A, R = base_params.beta_baseline.shape
        beta_tensor_fixed = theta_to_betas(
            torch.tensor(beta_vec1, dtype=torch.float64),
            scale_factors=scale_factors,
            beta_param=structure_stage1.get("beta_param", cfg_stage1.get("beta_param", "L")),
            L=L, A=A, R=R,
        ).detach()

        # ---------- Stage 2 ----------
        cfg_stage2 = copy.deepcopy(local_config)
        cfg_stage2["estimate_ihr"] = True
        cfg_stage2["hold_beta_fixed"] = True
        cfg_stage2["fixed_beta_tensor"] = beta_tensor_fixed
        # crucial: DO NOT estimate beta in stage 2
        cfg_stage2["beta_param"] = cfg_stage1.get("beta_param", "L")
        cfg_stage2["estimate_beta"] = False  # used by your structure builder if you add it

        structure_stage2 = build_theta_structure(cfg_stage2, base_state, base_params)

        results_stage2, init_guesses_stage2 = _run_single_stage(
            observed_window=observed_window,
            state_at_t0=state_at_t0,
            config=cfg_stage2,
            structure=structure_stage2,
            scale_factors=scale_factors,
            lambda_l2=lambda_l2,
            true_betas_flat=true_betas_flat,
        )

        # Combine logs (optional) and return stage-2 as primary
        results_df = pd.concat([results_stage1.assign(stage="beta"),
                                results_stage2.assign(stage="ihr")],
                              ignore_index=True)

        initial_guesses = init_guesses_stage1 + init_guesses_stage2
        local_structure = {"stage1": structure_stage1, "stage2": structure_stage2}
        return results_df, initial_guesses, local_structure, local_config

    theta_size = int(local_structure["size"])
    slices = local_structure["slices"]

    steps_per_day = timesteps_per_day
    start_step = int(t0_day * steps_per_day)
    n_steps = int(window_days * steps_per_day)

    
    # IHR mode: aggregate observed truth over age (and risk) => per-location (T x L)
    if local_config.get("ihr_mode", False) and local_config.get("ihr_aggregate_over_age", True):
        # expected dims: (time, L, A, R) or similar; sum over A and R
        if torch.is_tensor(observed_window):
            observed_window = observed_window.sum(dim=(-2, -1))
        else:
            # if numpy
            observed_window = observed_window.sum(axis=(-2, -1))

    base_true_admits_history_for_worker = observed_window

    if state_at_t0 is None:
        state_at_t0 = base_state

    num_params = len(true_betas_flat)
    optimizers_to_run = ['CG', 'L-BFGS-B', 'Adam', 'least_squares_fd']

    def _make_full_theta_guess(beta_guess_level=0.005, randomize=False):
        """
        Build a full θ (log-scaled) of length theta_size:
          - β part: log(beta * scale_beta)
          - init compartments: log(comp * scale_comp) for any enabled init_* blocks
        """
        theta0 = np.zeros(theta_size, dtype=float)

        # --- β ---
        beta_slice = slices["beta"]
        if not randomize:
            beta0 = np.full(beta_slice.stop - beta_slice.start, beta_guess_level, dtype=float)
        else:
            beta0 = np.random.uniform(0.003, 0.006, size=(beta_slice.stop - beta_slice.start))
        theta0[beta_slice] = np.log(beta0 * float(scale_factors["beta"]))

        # --- IHR ---
        if local_config.get("estimate_ihr", False) and ("ihr" in slices):
            s_ihr = slices["ihr"]
            # choose a sane default; use base_params.ihr_baseline aggregated to L if needed
            base_ihr = getattr(base_params, "ihr_baseline", None)
            if base_ihr is None:
                ihr0 = np.full(s_ihr.stop - s_ihr.start, 0.02, dtype=float)
            else:
                # derive per-L baseline
                base_ihr_L = base_ihr.detach().cpu().numpy().mean(axis=(1,2))
                ihr0 = np.clip(base_ihr_L, 1e-12, None)

            theta0[s_ihr] = np.log(ihr0 * float(scale_factors["ihr"]))
            if randomize:
                jitter = np.random.uniform(0.8, 1.25, size=theta0[s_ihr].shape[0])
                theta0[s_ihr] = np.log((ihr0 * jitter) * float(scale_factors["ihr"]))


        # --- initial compartments (if enabled) ---
        for comp_name, do_est in local_config["estimate_initial"].items():
            if not do_est:
                continue
            key = f"init_{comp_name}"
            s = slices[key]
            base_comp = getattr(state_at_t0, comp_name)
            comp_scale = float(scale_factors.get(comp_name, 1.0))

            comp0 = base_comp.detach().cpu().numpy().reshape(-1).astype(float)
            comp0 = np.clip(comp0, 1e-12, None)  # avoid log(0)
            theta0[s] = np.log(comp0 * comp_scale)

            if randomize:
                # small multiplicative jitter in comp-space
                jitter = np.random.uniform(0.8, 1.25, size=theta0[s].shape[0])
                theta0[s] = np.log((comp0 * jitter) * comp_scale)

        return theta0

    initial_guesses = [
        _make_full_theta_guess(beta_guess_level=0.005, randomize=False),
        _make_full_theta_guess(beta_guess_level=0.005, randomize=True),
    ]

    all_attempts_log = []

    # Helper to attach init/final compartment blocks for any enabled initial compartments
    def _attach_init_final_compartments(attempt_dict, x0_theta, theta_opt):
        for comp_name, do_est in local_config["estimate_initial"].items():
            if not do_est:
                continue
            key = f"init_{comp_name}"
            s = slices[key]
            comp_scale = float(scale_factors.get(comp_name, 1.0))

            init_vals = np.exp(np.asarray(x0_theta[s], dtype=float)) / comp_scale
            final_vals = np.exp(np.asarray(theta_opt[s], dtype=float)) / comp_scale

            # Store as flattened lists (full LxAxR in one vector)
            attempt_dict[f"initial_{comp_name}"] = init_vals.tolist()
            attempt_dict[f"final_{comp_name}"] = final_vals.tolist()

    results_df, initial_guesses = _run_single_stage(
          observed_window=observed_window,
          state_at_t0=state_at_t0,
          config=local_config,
          structure=local_structure,
          scale_factors=scale_factors,
          lambda_l2=lambda_l2,
          true_betas_flat=true_betas_flat,
      )


    results_df = pd.DataFrame(all_attempts_log)

    print("\n\n" + "=" * 70 + "\n=== DETAILED OPTIMIZATION ATTEMPTS ===\n" + "=" * 70)
    if not results_df.empty:
        print(results_df[['optimizer', 'initial_guess_id', 'phase', 'restart_num',
                          'loss', 'r_squared', 'duration', 'nit', 'timed_out']].to_string(
            index=False, float_format="%.4f"
        ))
    else:
        print("No attempts recorded.")

    if results_df.empty:
        return results_df, initial_guesses, local_structure, local_config

    best_runs = results_df.loc[
        results_df.groupby(['optimizer', 'initial_guess_id'])['loss'].idxmin()
    ].copy()

    # β summary columns
    for i in range(num_params):
        best_runs[f'true_beta_{i+1}'] = true_betas_flat[i]
        best_runs[f'initial_beta_{i+1}'] = best_runs['initial_betas'].apply(
            lambda betas: betas[i] if isinstance(betas, list) and len(betas) > i else np.nan
        )
        best_runs[f'final_beta_{i+1}'] = best_runs.apply(
            lambda row: row['final_betas'][i] if isinstance(row.get('final_betas', None), list) and len(row['final_betas']) > i else np.nan,
            axis=1
        )

    print("\n\n" + "=" * 70 + "\n=== SUMMARY OF BEST RUNS ===\n" + "=" * 70)

    summary_cols = ['optimizer', 'initial_guess_id', 'loss', 'r_squared', 'timed_out']
    param_cols = [
        c for c in best_runs.columns
        if c.startswith(('true_', 'initial_', 'final_'))
        and c not in ('true_betas_flat', 'final_betas')
    ]

    print(best_runs[summary_cols + param_cols].sort_values('loss'))

    r2_success_threshold = local_config.get("r2_success_threshold", 0.9)
    success_mask = (
        np.isfinite(results_df["loss"]) &
        np.isfinite(results_df["r_squared"]) &
        (results_df["r_squared"] >= r2_success_threshold)
    )
    successful_runs = results_df[success_mask]

    if successful_runs.empty:
        print("\nNo successful optimization runs found that meet the R squared threshold.")
        return results_df, initial_guesses, local_structure, local_config

    best_overall = successful_runs.loc[successful_runs['loss'].idxmin()]
    estimated_betas = np.array(best_overall['final_betas'][:num_params], dtype=float)
    mape = 100 * np.mean(np.abs((np.array(true_betas_flat) - estimated_betas) / np.array(true_betas_flat)))

    print("\n\n" + "=" * 70 +
          f"\n=== OVERALL BEST OPTIMIZER: {best_overall['optimizer']} (from Guess {best_overall['initial_guess_id']}) ===\n" +
          "=" * 70)
    print(f"Final Loss: {best_overall['loss']:.6f}, R squared: {best_overall['r_squared']:.6f}")
    print(f"Mean Absolute Pct Error (MAPE) across all {len(true_betas_flat)} betas: {mape:.2f}%\n")

    return results_df, initial_guesses, local_structure, local_config


def _make_true_series(true_admits_history, dt, mode='per_location'):
    """Create canonical truth series for plotting."""
    # Sum over risk (dim=3)
    true_LA = torch.sum(true_admits_history, dim=3)  # [T, L, A]
    if mode == 'per_location':
        return true_LA.sum(dim=2).cpu().numpy()      # [T, L]
    elif mode == 'per_LA':
        return true_LA.cpu().numpy()                 # [T, L, A]
    else:
        raise ValueError("mode must be 'per_location' or 'per_LA'")


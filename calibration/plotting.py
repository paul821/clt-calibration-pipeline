import matplotlib.pyplot as plt
import pandas as pd


def plot_calibration_results(
    true_admits_history,
    results_df,
    scale_factors,
    true_betas_flat,
    initial_guesses,
    theta_structure=None,
    true_initial_compartments=None,
    theta_col="final_theta",
):
    """
    FIXES:
      - initial_guess_id is 1-based; we index initial_guesses with (id-1)
      - when using theta_structure + final_theta, betas are log-scaled => exp()/scale
      - same for initial guess betas (they are log-scaled)
    """
    if results_df.empty:
        print("Result DataFrame is empty, skipping plot.")
        return

    if true_initial_compartments is None:
        true_initial_compartments = {}

    true_plot_data = torch.sum(true_admits_history, dim=(1, 2, 3)).cpu().numpy()
    series_len = true_plot_data.shape[0]

    try:
        _T = T
    except NameError:
        _T = series_len

    try:
        _dt = dt
    except NameError:
        _dt = None

    if (_dt is not None) and (series_len == int(round(_T / _dt))):
        xs_days = np.arange(series_len) * _dt
    elif series_len == _T:
        xs_days = np.arange(_T)
    else:
        xs_days = np.linspace(0.0, float(_T), num=series_len, endpoint=False)

    plt.rcParams.update({'font.size': 14})

    # ============================================================
    # Figure 1: True vs fitted admissions, separate figures per initial guess
    # ============================================================
    best_per_optimizer = results_df.loc[
        results_df.groupby(['optimizer', 'initial_guess_id'])['loss'].idxmin()
    ].copy()

    for guess_id, df_guess in best_per_optimizer.groupby('initial_guess_id'):
        plt.figure(figsize=(14, 7))
        plt.plot(xs_days, true_plot_data, label="True Hospital Admits",
                 color='black', linestyle='--', marker='o', markersize=3)

        for _, row in df_guess.iterrows():
            final_betas = None

            # Preferred: derive from final_theta (log) + theta_structure
            if theta_structure is not None and theta_col in row and isinstance(row[theta_col], (list, np.ndarray)):
                slices = theta_structure.get("slices", {})
                if "beta" in slices:
                    theta_hat = np.asarray(row[theta_col], dtype=float)
                    beta_slice = slices["beta"]
                    beta_log = theta_hat[beta_slice]
                    final_betas = np.exp(beta_log) / float(scale_factors["beta"])
            else:
                # Fallback: legacy already-in-beta-space
                if isinstance(row.get('final_betas', None), list):
                    final_betas = np.array(row['final_betas'], dtype=float)

            if final_betas is None or np.any(~np.isfinite(final_betas)):
                print(f"Skipping plot for {row['optimizer']} due to invalid final beta values.")
                continue

            opt_params_copy = copy.deepcopy(base_params)
            Lp, Ap, Rp = opt_params_copy.beta_baseline.shape
            run_betas = torch.tensor(final_betas, dtype=torch.float64)

            if run_betas.numel() == Lp * Ap:
                opt_params_copy.beta_baseline = run_betas.view(Lp, Ap, Rp)
            elif run_betas.numel() == Lp:
                opt_params_copy.beta_baseline = run_betas.view(Lp, 1, 1).expand(Lp, Ap, Rp)
            else:
                print(f"Skipping {row['optimizer']}: beta shape mismatch.")
                continue

            with torch.no_grad():
                fitted_history = flu.torch_simulate_hospital_admits(
                    base_state, opt_params_copy, base_precomputed, base_schedules, _T, timesteps_per_day
                )

            fitted_plot_data = torch.sum(fitted_history, dim=(1, 2, 3)).cpu().numpy()
            if fitted_plot_data.shape[0] != series_len:
                x_fit = np.linspace(xs_days[0], xs_days[-1], num=fitted_plot_data.shape[0], endpoint=True)
            else:
                x_fit = xs_days

            plt.plot(
                x_fit, fitted_plot_data,
                label=f"Fitted - {row['optimizer']} (Loss: {row['loss']:.2f})",
                alpha=0.9, linewidth=2
            )

        plt.title(f"Comparison of True vs. Fitted Hospital Admissions (Initial Guess {guess_id})")
        plt.xlabel("Time (days)")
        plt.ylabel("Total Daily Hospital Admissions")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    # ============================================================
    # Figure 2: Bar charts comparing true/initial/final betas
    # ============================================================
    unique_opts = results_df['optimizer'].unique()
    n_opts = len(unique_opts)
    fig, axes = plt.subplots(1, n_opts, figsize=(6 * n_opts, 6), sharey=True)

    if n_opts == 1:
        axes = [axes]

    true_betas_flat = np.asarray(true_betas_flat, dtype=float)

    for ax, opt_name in zip(axes, unique_opts):
        best_row = results_df.loc[
            results_df[results_df['optimizer'] == opt_name]['loss'].idxmin()
        ]

        final_betas = None
        if theta_structure is not None and theta_col in best_row:
            slices = theta_structure.get("slices", {})
            if "beta" in slices and isinstance(best_row[theta_col], (list, np.ndarray)):
                theta_hat = np.asarray(best_row[theta_col], dtype=float)
                beta_slice = slices["beta"]
                beta_log = theta_hat[beta_slice]
                final_betas = np.exp(beta_log) / float(scale_factors["beta"])

        if final_betas is None and 'final_betas' in best_row:
            final_betas = np.array(best_row['final_betas'], dtype=float)

        if final_betas is None:
            print(f"Skipping β bar chart for {opt_name}: no beta information found.")
            continue

        # 1-based id -> 0-based index
        init_guess_id = int(best_row['initial_guess_id'])
        init_guess_idx = init_guess_id - 1
        if init_guess_idx < 0 or init_guess_idx >= len(initial_guesses):
            print(f"Skipping {opt_name}: initial_guess_id={init_guess_id} out of range for initial_guesses.")
            continue

        init_theta = np.asarray(initial_guesses[init_guess_idx], dtype=float)

        if theta_structure is not None and "beta" in theta_structure.get("slices", {}):
            beta_slice = theta_structure["slices"]["beta"]
            init_beta_log = init_theta[beta_slice]
            init_betas = np.exp(init_beta_log) / float(scale_factors["beta"])
        else:
            # legacy: initial_guesses already in beta-space
            init_betas = init_theta

        if true_betas_flat.shape[0] != final_betas.shape[0]:
            print(f"Skipping {opt_name}: true_betas vs final_betas length mismatch.")
            continue
        if init_betas.shape[0] != true_betas_flat.shape[0]:
            print(f"Skipping {opt_name}: true_betas vs initial_betas length mismatch.")
            continue

        width = 0.25
        x = np.arange(len(true_betas_flat))
        ax.bar(x - width, true_betas_flat, width, label="True")
        ax.bar(x, init_betas, width, label=f"Initial Guess {init_guess_id}")
        ax.bar(x + width, final_betas, width, label="Final (Fitted)")

        ax.set_title(f"β Comparison - {opt_name}")
        ax.set_xlabel("β Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_compartment_timeseries(
    results_df,
    theta_structure,
    scale_factors,
    true_state_at_t0,
    true_params,
    simulate_states_fn,
    T,
    timesteps_per_day,
    compartments=("E",),
    theta_col="final_theta",
):
    """
    Plot true vs estimated compartment trajectories over time for the best run.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results table from run_metapop_calibration_suite.
    theta_structure : dict
        Contains 'slices' mapping family -> slice in theta (e.g., 'beta', 'E', 'IP', ...).
    scale_factors : dict
        Scale factors for theta families, e.g. {'beta': 1.0, 'E': 1.0, ...}.
    true_state_at_t0 : state object
        State at t0 for the *true* trajectory (e.g. state_truth_E).
    true_params : params object
        Parameters used for the true trajectory (e.g. params_3_param).
    simulate_states_fn : callable
        Function that runs the full state simulation.
        Expected signature:
            simulate_states_fn(state, params, precomputed, schedules, T, timesteps_per_day)
        and returns a dict mapping compartment name -> torch.Tensor of shape (time, ...).
        You can adapt this to exactly match your own API.
    T, timesteps_per_day : int
        Time horizon and substeps per day.
    compartments : iterable of str
        Which compartments to plot (e.g. ('E', 'IP')).
    theta_col : str
        Column name in results_df that contains the flattened theta vector.
    """

    if results_df.empty:
        print("No results to plot.")
        return

    if 'loss' not in results_df.columns:
        raise ValueError("results_df must contain a 'loss' column.")

    slices = theta_structure.get("slices", {})
    scale_factors = scale_factors or {}

    # -------- Pick best run (minimum finite loss) --------
    mask_finite = np.isfinite(results_df['loss'])
    if not mask_finite.any():
        print("No finite-loss runs to plot compartments.")
        return

    best_idx = results_df.loc[mask_finite, 'loss'].idxmin()
    best_run = results_df.loc[best_idx]

    if theta_col not in best_run:
        raise ValueError(f"{theta_col} not found in best_run.")

    theta_hat = np.asarray(best_run[theta_col], dtype=float)

    # -------- Build fitted params & initial state from theta_hat --------
    # 1) betas
    fitted_params = copy.deepcopy(true_params)
    if "beta" in slices:
        beta_slice = slices["beta"]
        beta_flat = theta_hat[beta_slice] * scale_factors.get("beta", 1.0)

        L, A, R = fitted_params.beta_baseline.shape
        beta_tensor = torch.tensor(beta_flat, dtype=fitted_params.beta_baseline.dtype)
        if beta_tensor.numel() == L * A:
            fitted_params.beta_baseline = beta_tensor.view(L, A, R)
        elif beta_tensor.numel() == L:
            fitted_params.beta_baseline = beta_tensor.view(L, 1, 1).expand(L, A, R)
        else:
            raise ValueError(
                f"beta size {beta_tensor.numel()} doesn't match expected L*A={L*A} or L={L}."
            )

    # 2) initial compartments (E, IP, etc.)
    fitted_state = copy.deepcopy(true_state_at_t0)
    for family in compartments:
        if family not in slices:
            print(f"Warning: '{family}' not in theta_structure['slices']; skipping.")
            continue
        if not hasattr(fitted_state, family):
            print(f"Warning: state has no compartment '{family}'; skipping.")
            continue

        sl = slices[family]
        comp_flat = theta_hat[sl] * scale_factors.get(family, 1.0)
        comp_tensor = getattr(fitted_state, family)

        shape = comp_tensor.shape
        if comp_flat.size != int(np.prod(shape)):
            raise ValueError(
                f"Shape mismatch for {family}: theta slice has size {comp_flat.size}, "
                f"state tensor has shape {shape}."
            )

        new_vals = torch.tensor(
            comp_flat.reshape(shape),
            dtype=comp_tensor.dtype,
            device=comp_tensor.device,
        )
        setattr(fitted_state, family, new_vals)

    # -------- Run true and fitted simulations --------
    with torch.no_grad():
        true_traj = simulate_states_fn(
            true_state_at_t0, true_params, base_precomputed, base_schedules, T, timesteps_per_day
        )
        fitted_traj = simulate_states_fn(
            fitted_state, fitted_params, base_precomputed, base_schedules, T, timesteps_per_day
        )

    # true_traj / fitted_traj should be dicts: family -> tensor (time, ...)
    # Aggregate over non-time dimensions for plotting (sum over loc/age/etc.)
    xs = np.arange(true_traj[compartments[0]].shape[0]) / timesteps_per_day

    n_comp = len(compartments)
    fig, axes = plt.subplots(n_comp, 1, figsize=(10, 4 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]

    for ax, family in zip(axes, compartments):
        if family not in true_traj or family not in fitted_traj:
            print(f"Compartment '{family}' missing from trajectories; skipping.")
            continue

        true_series = true_traj[family].sum(dim=tuple(range(1, true_traj[family].ndim))).cpu().numpy()
        fitted_series = fitted_traj[family].sum(dim=tuple(range(1, fitted_traj[family].ndim))).cpu().numpy()

        ax.plot(xs, true_series, label=f"True {family}(t)", color="black", linestyle="--")
        ax.plot(xs, fitted_series, label=f"Estimated {family}(t)", alpha=0.8)
        ax.set_ylabel(f"{family} (aggregated)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(
        f"Compartment trajectories (best run, loss={best_run['loss']:.3g}, "
        f"R²={best_run.get('r_squared', np.nan):.4f})"
    )
    plt.tight_layout()
    plt.show()


def plot_subpop_timeseries(result,
                           mode='per_location',   # 'per_location' or 'per_LA'
                           locations=None,        # list of L indices to plot (None = all)
                           ages=None,             # list of A indices to plot (used when mode='per_LA')
                           max_cols=3,
                           title_prefix=None):
    """
    Plot true vs fitted hospital admissions over time for subpopulations.

    Parameters
    ----------
    result : dict
        One row from your results (e.g., best attempt) with keys:
        'true_L_series','fitted_L_series','true_LA_series','fitted_LA_series','dt'
    mode : str
        'per_location': plots [T, L] series (sum over A)
        'per_LA'      : plots [T] per (L,A) small multiples
    locations : list[int] or None
        Which L indices to include; None -> all
    ages : list[int] or None
        Which A indices to include (only used in mode='per_LA'); None -> all
    max_cols : int
        Max subplot columns for small multiples
    title_prefix : str or None
        Prefix text for figure title
    """

    dt = float(result.get('dt', 1.0))
    if mode == 'per_location':
        true_L = result['true_L_series']    # [T, L]
        fit_L  = result['fitted_L_series']  # [T, L]
        T, L = true_L.shape
        xs = np.arange(T) * dt

        loc_idx = locations if locations is not None else list(range(L))
        n = len(loc_idx)
        ncols = min(max_cols, n if n > 0 else 1)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)
        ax_iter = axes.flatten()

        for k, li in enumerate(loc_idx):
            ax = ax_iter[k]
            ax.plot(xs, true_L[:, li], label='True', linewidth=2)
            ax.plot(xs, fit_L[:, li],  label='Fitted', linewidth=2, alpha=0.85)
            ax.set_title(f'Location L={li}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Admissions')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()

        # Hide any unused axes
        for m in range(len(loc_idx), len(ax_iter)):
            ax_iter[m].axis('off')

        fig.suptitle((title_prefix + ' – ' if title_prefix else '') + 'True vs Fitted Admissions per Location (sum over ages)', y=1.02)
        fig.tight_layout()
        plt.show()

    elif mode == 'per_LA':
        true_LA = result['true_LA_series']    # [T, L, A]
        fit_LA  = result['fitted_LA_series']  # [T, L, A]
        T, L, A = true_LA.shape
        xs = np.arange(T) * dt

        loc_idx = locations if locations is not None else list(range(L))
        age_idx = ages if ages is not None else list(range(A))

        pairs = [(l, a) for l in loc_idx for a in age_idx]
        n = len(pairs)
        ncols = min(max_cols, n if n > 0 else 1)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)
        ax_iter = axes.flatten()

        for k, (li, ai) in enumerate(pairs):
            ax = ax_iter[k]
            ax.plot(xs, true_LA[:, li, ai], label='True', linewidth=2)
            ax.plot(xs, fit_LA[:, li, ai],  label='Fitted', linewidth=2, alpha=0.85)
            ax.set_title(f'L={li}, A={ai}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Admissions')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()

        for m in range(len(pairs), len(ax_iter)):
            ax_iter[m].axis('off')

        fig.suptitle((title_prefix + ' – ' if title_prefix else '') + 'True vs Fitted Admissions per (Location, Age)', y=1.02)
        fig.tight_layout()
        plt.show()

    else:
        raise ValueError("mode must be 'per_location' or 'per_LA'")


def plot_subpop_timeseries_from_results(
    results_df,
    theta_structure,
    scale_factors,
    true_state_at_t0=None,      # <-- FIX: optional now (prevents your missing-arg TypeError)
    true_params=None,
    simulate_states_fn=None,
    T=None,
    timesteps_per_day=None,
    compartment="L",            # e.g. "L", "E", "IP", etc.
    mode="per_location",        # or "per_age", "per_loc_age"
    theta_col="final_theta",
    max_runs=1,
):
    """
    FIX:
      - true_state_at_t0 is now optional. If omitted, we fall back to global base_state.
    """
    if results_df.empty:
        print("No results to plot.")
        return

    if true_state_at_t0 is None:
        if "base_state" in globals():
            true_state_at_t0 = globals()["base_state"]
        else:
            raise ValueError("true_state_at_t0 was not provided and base_state is not available in globals().")

    if true_params is None:
        if "base_params" in globals():
            true_params = globals()["base_params"]
        else:
            raise ValueError("true_params was not provided and base_params is not available in globals().")

    if simulate_states_fn is None:
        raise ValueError("simulate_states_fn must be provided (a full-state simulator).")

    if T is None or timesteps_per_day is None:
        raise ValueError("T and timesteps_per_day must be provided.")

    # Assume standard shape: (time, L, A, ...)
    # infer L and A for labeling
    time_len = true_tensor.shape[0]
    L_dim = true_tensor.shape[1]
    A_dim = true_tensor.shape[2] if ndim >= 3 else 1

    xs = np.arange(time_len) / timesteps_per_day

    # ---- Select best finite-loss runs per optimizer ----
    mask_finite = np.isfinite(results_df["loss"])
    if not mask_finite.any():
        print("No finite-loss runs to plot.")
        return

    finite_df = results_df[mask_finite].copy()
    best_indices = (
        finite_df.sort_values("loss")
        .groupby("optimizer")
        .head(max_runs)
        .index
    )

    # ---- For each selected run: reconstruct, simulate, and plot ----
    plt.figure(figsize=(14, 6))

    # Plot true subpop series first (as reference)
    if mode == "per_location":
        # Sum over all non-time, non-location dims
        # dimensions: (time, L, A, rest...)
        other_dims = tuple(range(2, ndim))
        true_series = true_tensor.sum(dim=other_dims).cpu().numpy()  # shape (time, L)
        for loc in range(L_dim):
            plt.plot(xs, true_series[:, loc], color="black", alpha=0.3,
                     linestyle="--", label=f"True loc {loc}" if loc == 0 else None)

    elif mode == "per_age":
        # sum over all non-time, non-age dims
        # here treat dim=2 as age
        other_dims = (1,) + tuple(range(3, ndim))
        true_series = true_tensor.sum(dim=other_dims).cpu().numpy()  # shape (time, A)
        for age in range(A_dim):
            plt.plot(xs, true_series[:, age], color="black", alpha=0.3,
                     linestyle="--", label=f"True age {age}" if age == 0 else None)

    elif mode == "per_loc_age":
        # each (loc, age) separately, aggregated over remaining dims
        other_dims = tuple(range(3, ndim))
        true_series = true_tensor.sum(dim=other_dims).cpu().numpy()  # (time, L, A)
        for loc in range(L_dim):
            for age in range(A_dim):
                plt.plot(xs, true_series[:, loc, age], color="black", alpha=0.15,
                         linestyle="--",
                         label=f"True loc{loc},age{age}" if (loc == 0 and age == 0) else None)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # ---- Overlay fitted trajectories from best runs ----
    for idx in best_indices:
        row = finite_df.loc[idx]
        opt_name = row.get("optimizer", "Unknown")
        if theta_col not in row:
            print(f"Skipping row {idx}: missing '{theta_col}'.")
            continue

        theta_hat = np.asarray(row[theta_col], dtype=float)

        # 1) reconstruct params from θ
        fitted_params = copy.deepcopy(true_params)
        if "beta" in slices:
            beta_slice = slices["beta"]
            beta_flat = theta_hat[beta_slice] * scale_factors.get("beta", 1.0)

            Lp, Ap, Rp = fitted_params.beta_baseline.shape
            beta_tensor = torch.tensor(beta_flat, dtype=fitted_params.beta_baseline.dtype)
            if beta_tensor.numel() == Lp * Ap:
                fitted_params.beta_baseline = beta_tensor.view(Lp, Ap, Rp)
            elif beta_tensor.numel() == Lp:
                fitted_params.beta_baseline = beta_tensor.view(Lp, 1, 1).expand(Lp, Ap, Rp)
            else:
                print(f"Skipping row {idx}: beta size mismatch.")
                continue

        # 2) reconstruct initial compartments from θ (if present)
        fitted_state = copy.deepcopy(true_state_at_t0)
        for fam, sl in slices.items():
            if fam == "beta":
                continue
            if not hasattr(fitted_state, fam):
                continue

            comp_flat = theta_hat[sl] * scale_factors.get(fam, 1.0)
            comp_tensor = getattr(fitted_state, fam)
            shape = comp_tensor.shape
            if comp_flat.size != int(np.prod(shape)):
                print(f"Skipping row {idx}: shape mismatch for family '{fam}'.")
                continue

            new_vals = torch.tensor(
                comp_flat.reshape(shape),
                dtype=comp_tensor.dtype,
                device=comp_tensor.device,
            )
            setattr(fitted_state, fam, new_vals)

        # 3) simulate fitted trajectory
        with torch.no_grad():
            fitted_traj = simulate_states_fn(
                fitted_state, fitted_params, base_precomputed, base_schedules, T, timesteps_per_day
            )

        if compartment not in fitted_traj:
            print(f"Skipping row {idx}: compartment '{compartment}' not in fitted_traj.")
            continue

        fitted_tensor = fitted_traj[compartment]

        if fitted_tensor.shape[0] != time_len:
            print(f"Skipping row {idx}: time length mismatch ({fitted_tensor.shape[0]} vs {time_len}).")
            continue

        # Aggregate the same way as the true series
        if mode == "per_location":
            other_dims = tuple(range(2, fitted_tensor.ndim))
            series = fitted_tensor.sum(dim=other_dims).cpu().numpy()  # (time, L)
            for loc in range(L_dim):
                plt.plot(xs, series[:, loc],
                         label=f"{opt_name} loc {loc}",
                         alpha=0.7)
        elif mode == "per_age":
            other_dims = (1,) + tuple(range(3, fitted_tensor.ndim))
            series = fitted_tensor.sum(dim=other_dims).cpu().numpy()  # (time, A)
            for age in range(A_dim):
                plt.plot(xs, series[:, age],
                         label=f"{opt_name} age {age}",
                         alpha=0.7)
        elif mode == "per_loc_age":
            other_dims = tuple(range(3, fitted_tensor.ndim))
            series = fitted_tensor.sum(dim=other_dims).cpu().numpy()  # (time, L, A)
            for loc in range(L_dim):
                for age in range(A_dim):
                    plt.plot(xs, series[:, loc, age],
                             label=f"{opt_name} loc{loc},age{age}",
                             alpha=0.4)

    plt.xlabel("Time (days)")
    plt.ylabel(f"{compartment} (aggregated)")
    plt.title(f"{compartment}(t) subpopulation trajectories (mode={mode})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

def simulate_states_fn(state, params, precomputed, schedules, T, timesteps_per_day):
    # adapt to your actual API
    return flu.torch_simulate_full_state(
        state, params, precomputed, schedules, T, timesteps_per_day
    )
    

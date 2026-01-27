import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional

def save_convergence_plot(grid_results, filename="SSE_Stage1_Convergence.png", prefix=""):
    """
    SSE vs Offset Visualization with Global R2
    """
    if not grid_results:
        return
    
    if prefix:
        filename = prefix + filename
    
    sorted_results = sorted(grid_results, key=lambda x: x[0])
    offsets = [x[0] for x in sorted_results]
    sse_vals = [x[1]['loss'] for x in sorted_results]
    r2_vals = [x[1]['global_r2'] for x in sorted_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_sse = '#2c3e50'
    ax1.set_xlabel('Offset (Days)')
    ax1.set_ylabel('SSE SUM (Objective)', color=color_sse, fontsize=12, fontweight='bold')
    ax1.plot(offsets, sse_vals, marker='o', color=color_sse, linewidth=2, label='Objective Loss')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_sse)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    ax2 = ax1.twinx()
    color_r2 = '#e74c3c'
    ax2.set_ylabel('Global R²', color=color_r2, fontsize=12, fontweight='bold')
    ax2.plot(offsets, r2_vals, marker='s', color=color_r2, linestyle='--', alpha=0.6, label='Global R²')
    ax2.tick_params(axis='y', labelcolor=color_r2)

    plt.title('Stage 1 Calibration: Objective & Global R² vs Offset', fontsize=14)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def save_regional_aggregate_plot(truth_noisy, truth_clean, opt_pred, current_T, filename="calibration_regional_aggregate.png", prefix=""):
    """
    Aggregate plot with uniform styling
    """
    if prefix:
        filename = prefix + filename
    t_days = np.arange(current_T)
    obs_agg = truth_noisy.sum(dim=(2, 3)).detach().cpu().numpy()
    tru_agg = truth_clean.sum(dim=(2, 3)).detach().cpu().numpy()
    est_agg = opt_pred.sum(dim=(2, 3)).detach().cpu().numpy()
    
    num_locs = obs_agg.shape[1]
    # Calculate grid size (e.g., ceilings)
    n_plots = num_locs + 1
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()
    loc_names = ["Loc A", "Loc B", "Loc C", "Loc D", "Loc E"][:num_locs] + ["Global"]
    
    for i in range(n_plots):
        ax = axes[i]
        if i < num_locs:
            o, t, e = obs_agg[:, i], tru_agg[:, i], est_agg[:, i]
        else:
            o, t, e = obs_agg.sum(axis=1), tru_agg.sum(axis=1), est_agg.sum(axis=1)
        
        ax.scatter(t_days, o, color='black', alpha=0.3, s=10, label='Noisy Obs')
        ax.plot(t_days, t, color='green', label='Ground Truth', linewidth=2)
        ax.plot(t_days, e, color='red', linestyle='--', label='Estimated', linewidth=2)
        
        ss_tot = np.sum((o - np.mean(o))**2)
        r2_tru = 1 - (np.sum((o - t)**2) / ss_tot) if ss_tot > 0 else 0
        r2_est = 1 - (np.sum((o - e)**2) / ss_tot) if ss_tot > 0 else 0
        
        ax.set_title(loc_names[i])
        ax.text(0.95, 0.94, f"Ground Truth R²: {r2_tru:.3f}", transform=ax.transAxes, ha='right', color='green', fontsize=9, fontweight='bold')
        ax.text(0.95, 0.89, f"Estimated R²: {r2_est:.3f}", transform=ax.transAxes, ha='right', color='red', fontsize=9, fontweight='bold')
        
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85))
            
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    if "fit_offset" not in filename and "fit_L-BFGS-B" not in filename and "fit_CG" not in filename and "fit_Adam" not in filename: 
        print(f"Saved: {filename}")

def save_diagnostic_plots(truth_noisy, truth_clean, opt_pred, current_T, age_labels=None, prefix=""):
    """
    15-panel diagnostic plot
    """
    if age_labels is None:
        age_labels = ["0-4", "5-17", "18-49", "50-64", "65+"]
    
    filename = f"{prefix}calibration_15_panel.png" if prefix else "calibration_15_panel.png"
    
    t_days = np.arange(current_T)
    obs_np, tru_np, est_np = [x.detach().cpu().numpy().squeeze(-1) for x in [truth_noisy, truth_clean, opt_pred]]
    num_locs = obs_np.shape[1]
    loc_names = ["Loc A", "Loc B", "Loc C", "Loc D", "Loc E"][:num_locs]
    
    fig1, axes1 = plt.subplots(5, num_locs, figsize=(5 * num_locs, 20), sharex=True, squeeze=False)
    for r_idx, r_name in enumerate(loc_names):
        for a_idx in range(5):
            ax = axes1[a_idx, r_idx]
            obs, tru, est = obs_np[:, r_idx, a_idx], tru_np[:, r_idx, a_idx], est_np[:, r_idx, a_idx]
            ax.scatter(t_days, obs, color='black', alpha=0.3, s=8, label='Noisy Obs')
            ax.plot(t_days, tru, color='green', linewidth=1.5, label='Ground Truth')
            ax.plot(t_days, est, color='red', linestyle='--', linewidth=1.5, label='Estimated')
            
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_tru = (1-(np.sum((obs-tru)**2)/ss_tot)) if ss_tot > 0 else 0
            r2_est = (1-(np.sum((obs-est)**2)/ss_tot)) if ss_tot > 0 else 0
            
            ax.set_title(f"{r_name} - Age {age_labels[a_idx]}")
            ax.text(0.95, 0.94, f"Ground Truth R²: {r2_tru:.3f}", transform=ax.transAxes, ha='right', color='green', fontsize=9)
            ax.text(0.95, 0.89, f"Estimated R²: {r2_est:.3f}", transform=ax.transAxes, ha='right', color='red', fontsize=9)
            
            if r_idx == 0 and a_idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_multi_optimizer_comparison(results_dict: Dict, stage_name: str = "Stage 1", save_path: str = None, prefix: str = ""):
    """
    Plot comparison of multiple optimizers
    
    Args:
        results_dict: dict mapping optimizer name to result dict with 'loss', 'r_squared', 'duration'
        stage_name: "Stage 1" or "Stage 2"
        save_path: optional save path
    """
    if not results_dict:
        print("No results to plot")
        return
    
    optimizers = list(results_dict.keys())
    losses = [results_dict[opt].get('loss', float('inf')) for opt in optimizers]
    r2s = [results_dict[opt].get('global_r2', results_dict[opt].get('r_squared', 0.0)) for opt in optimizers]
    durations = [results_dict[opt].get('duration', 0.0) for opt in optimizers]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss comparison
    axes[0].bar(optimizers, losses, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{stage_name} - Loss Comparison', fontsize=14)
    axes[0].set_yscale('log')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1].bar(optimizers, r2s, color='seagreen', alpha=0.7)
    axes[1].set_ylabel('R²', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{stage_name} - R² Comparison', fontsize=14)
    axes[1].set_ylim([0, 1])
    axes[1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target R²=0.9')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    
    # Duration comparison
    axes[2].bar(optimizers, durations, color='coral', alpha=0.7)
    axes[2].set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_title(f'{stage_name} - Runtime Comparison', fontsize=14)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        if prefix and not save_path.startswith(prefix):
            save_path = prefix + save_path
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        default_name = f"{stage_name.replace(' ', '_')}_optimizer_comparison.png"
        filename = prefix + default_name if prefix else default_name
        plt.savefig(filename)
        print(f"Saved: {filename}")
        
    plt.close()

def plot_parameter_recovery_bars(
    true_params: Dict[str, np.ndarray],
    opt_params_dict: Dict[str, Dict[str, np.ndarray]],
    param_names: List[str],
    save_path: str = "parameter_recovery_comparison.png",
    prefix: str = ""
):
    """
    Bar chart comparing true vs optimized parameters across multiple optimizers
    
    Args:
        true_params: dict mapping param name to true values (e.g., {"beta": [0.22, 0.28, 0.25]})
        opt_params_dict: dict mapping optimizer name to dict of param name then to optimized values
        param_names: list of parameter names to plot (["beta", "E"])
        save_path: path to save figure
    """
    if prefix and not save_path.startswith(prefix):
        save_path = prefix + save_path
    
    n_params = len(param_names)
    n_optimizers = len(opt_params_dict)
    
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 6), squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_optimizers + 1))
    
    for param_idx, param_name in enumerate(param_names):
        ax = axes[param_idx]
        
        true_vals = true_params.get(param_name, np.array([]))
        if true_vals.ndim > 1:
            # For multi-dimensional params (like E with shape (L, A, R)), aggregate
            true_vals = true_vals.sum(axis=tuple(range(1, true_vals.ndim)))  # Sum over non-location dims
        
        n_values = len(true_vals)
        x = np.arange(n_values)
        width = 0.8 / (n_optimizers + 1)
        
        # True values
        ax.bar(x - width * n_optimizers / 2, true_vals, width, label='True', color='black', alpha=0.6)
        
        # Optimized values per optimizer
        for opt_idx, (opt_name, opt_params) in enumerate(opt_params_dict.items()):
            opt_vals = opt_params.get(param_name, np.array([]))
            if opt_vals.ndim > 1:
                opt_vals = opt_vals.sum(axis=tuple(range(1, opt_vals.ndim)))
            
            if len(opt_vals) == n_values:
                ax.bar(x - width * n_optimizers / 2 + width * (opt_idx + 1), opt_vals, width, 
                       label=opt_name, color=colors[opt_idx], alpha=0.8)
        
        ax.set_xlabel('Index' if param_name == "beta" else 'Location', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{param_name.upper()} Recovery', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        if param_name == "beta":
            ax.set_xticklabels([f'β{i}' for i in range(n_values)])
        else:
            ax.set_xticklabels([f'Loc {i}' for i in range(n_values)])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_compartment_timeseries(
    truth_clean: torch.Tensor,
    predictions_dict: Dict[str, torch.Tensor],
    compartment_name: str,
    current_T: int,
    timesteps_per_day: int = 4,
    save_path: str = None,
    prefix: str = ""
):
    """
    Plot compartment trajectories for multiple optimizers
    
    Args:
        truth_clean: clean truth data (T, L, A, R)
        predictions_dict: dict mapping optimizer name to predicted data (T, L, A, R)
        compartment_name: which compartment to plot (used in title)
        current_T: time horizon
        timesteps_per_day: temporal resolution
        save_path: optional save path
    """
    t_days = np.arange(current_T)
    
    # Aggregate over all dims except time
    truth_agg = truth_clean.sum(dim=(1, 2, 3)).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_days, truth_agg, color='black', linewidth=2, label='Ground Truth', linestyle='--')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for i, (opt_name, pred) in enumerate(predictions_dict.items()):
        pred_agg = pred.sum(dim=(1, 2, 3)).cpu().numpy()
        plt.plot(t_days, pred_agg, color=colors[i], linewidth=1.5, alpha=0.8, label=f'{opt_name}')
    
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel(f'{compartment_name} (aggregated)', fontsize=12)
    plt.title(f'{compartment_name} Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        if prefix and not save_path.startswith(prefix):
            save_path = prefix + save_path
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        default_name = f"{compartment_name.replace(' ', '_')}_optimizer_comparison.png"
        filename = prefix + default_name if prefix else default_name
        plt.savefig(filename)
        print(f"Saved: {filename}")
    
    plt.close()

def plot_loss_component_breakdown(results_dict: Dict, stage_name: str = "Stage 1", save_path: str = None, prefix: str = ""):
    """
    Stacked bar chart showing loss component breakdown
    
    Args:
        results_dict: dict mapping optimizer name to result dict with 'pure_fit_sse' and 'reg_breakdown'
        stage_name: "Stage 1" or "Stage 2"
        save_path: optional save path
    """
    optimizers = list(results_dict.keys())
    
    # Extract components
    sse_vals = []
    reg_components = {}
    
    for opt_name in optimizers:
        result = results_dict[opt_name]
        sse_vals.append(result.get('pure_fit_sse', result.get('sse', 0.0)))
        
        if 'reg_breakdown' in result:
            for reg_name, reg_val in result['reg_breakdown'].items():
                if reg_name not in reg_components:
                    reg_components[reg_name] = []
                reg_components[reg_name].append(reg_val)
        else:
            for reg_name in reg_components.keys():
                reg_components[reg_name].append(0.0)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(optimizers))
    width = 0.6
    
    # SSE base
    ax.bar(x, sse_vals, width, label='Data Fit (SSE)', color='steelblue', alpha=0.8)
    
    # Stack regularization components
    bottom = np.array(sse_vals)
    colors_reg = plt.cm.Pastel1(np.linspace(0, 1, len(reg_components)))
    
    for i, (reg_name, reg_vals) in enumerate(reg_components.items()):
        # Pad with zeros if needed
        while len(reg_vals) < len(optimizers):
            reg_vals.append(0.0)
        
        ax.bar(x, reg_vals, width, bottom=bottom, label=reg_name, color=colors_reg[i], alpha=0.8)
        bottom += np.array(reg_vals)
    
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{stage_name} - Loss Component Breakdown', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        if prefix and not save_path.startswith(prefix):
            save_path = prefix + save_path
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        default_name = f"{stage_name.replace(' ', '_')}_loss_breakdown.png"
        filename = prefix + default_name if prefix else default_name
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()

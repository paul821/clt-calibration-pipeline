import matplotlib
import numpy as np

def save_convergence_plot(grid_results):
    """Surgical Edit 92/93: SSE vs Offset Visualization with Global R2"""
    if not grid_results:
        return
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
    plt.savefig('SSE_Stage1_Convergence.png')
    plt.close()
    print("Saved: SSE_Stage1_Convergence.png")

def save_regional_aggregate_plot(truth_noisy, truth_clean, opt_pred, current_T, filename="calibration_regional_aggregate.png"):
    """Surgical Edit 86/90: Aggregate plot with uniform styling"""
    t_days = np.arange(current_T)
    obs_agg = truth_noisy.sum(dim=(2, 3)).detach().cpu().numpy()
    tru_agg = truth_clean.sum(dim=(2, 3)).detach().cpu().numpy()
    est_agg = opt_pred.sum(dim=(2, 3)).detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    loc_names = ["Loc A", "Loc B", "Loc C", "Global"]
    
    for i in range(4):
        ax = axes[i]
        if i < 3:
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
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    if "fit_offset" not in filename: print(f"Saved: {filename}")


def save_diagnostic_plots(truth_noisy, truth_clean, opt_pred, current_T):
    t_days = np.arange(current_T)
    obs_np, tru_np, est_np = [x.detach().cpu().numpy().squeeze(-1) for x in [truth_noisy, truth_clean, opt_pred]]
    fig1, axes1 = plt.subplots(5, 3, figsize=(15, 20), sharex=True)
    for r_idx, r_name in enumerate(["Loc A", "Loc B", "Loc C"]):
        for a_idx in range(5):
            ax = axes1[a_idx, r_idx]; obs, tru, est = obs_np[:, r_idx, a_idx], tru_np[:, r_idx, a_idx], est_np[:, r_idx, a_idx]
            ax.scatter(t_days, obs, color='black', alpha=0.3, s=8, label='Noisy Obs')
            ax.plot(t_days, tru, color='green', linewidth=1.5, label='Ground Truth')
            ax.plot(t_days, est, color='red', linestyle='--', linewidth=1.5, label='Estimated')
            
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_tru = (1-(np.sum((obs-tru)**2)/ss_tot)) if ss_tot > 0 else 0
            r2_est = (1-(np.sum((obs-est)**2)/ss_tot)) if ss_tot > 0 else 0
            
            ax.set_title(f"{r_name} - Age {AGE_LABELS[a_idx]}")
            ax.text(0.95, 0.94, f"Ground Truth R²: {r2_tru:.3f}", transform=ax.transAxes, ha='right', color='green', fontsize=9)
            ax.text(0.95, 0.89, f"Estimated R²: {r2_est:.3f}", transform=ax.transAxes, ha='right', color='red', fontsize=9)
            
            if r_idx == 0 and a_idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig("calibration_15_panel.png"); print("Saved: calibration_15_panel.png")

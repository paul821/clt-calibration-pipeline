

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

def print_beta_e0_table(true_betas, opt_betas, true_e0, opt_e0, best_probe_details=None):
    print(f"\n>>> STAGE 1: BETA & E0 PARAMETER RECOVERY <<<")
    print("="*105)
    print(f"{'LOCATION':<12} | {'AGE':<6} | {'TRUE E0':<12} | {'OPT E0':<12} | {'BETA (TRUE)':<12} | {'BETA (OPT)':<12} | {'% DEV':<8}")
    print("-" * 105)
    loc_names = ["Loc A", "Loc B", "Loc C"]
    for r_idx, name in enumerate(loc_names):
        t_b, o_b = true_betas[r_idx], opt_betas[r_idx]
        b_dev = ((o_b - t_b) / t_b) * 100
        for a_idx in range(5):
            t_e, o_e = true_e0[r_idx, a_idx].item(), opt_e0[r_idx, a_idx].item()
            row_name = name if a_idx == 0 else ""
            beta_t_str = f"{t_b:<12.6f}" if a_idx == 0 else ""
            beta_o_str = f"{o_b:<12.6f}" if a_idx == 0 else ""
            dev_str = f"{b_dev:>+7.2f}%" if a_idx == 0 else ""
            print(f"{row_name:<12} | Age {a_idx} | {t_e:<12.6f} | {o_e:<12.6f} | {beta_t_str} | {beta_o_str} | {dev_str}")
        print(f"{'':<12} | TOTAL  | {true_e0[r_idx].sum().item():<12.6f} | {opt_e0[r_idx].sum().item():<12.6f} | {'':<12} | {'':<12} |")
        print("-" * 105)
    if best_probe_details:
        print(f"FIT QUALITY AT OPTIMAL OFFSET ({best_probe_details['offset']} days):")
        for r_idx, name in enumerate(["A", "B", "C"]):
            print(f"  Subpop {name}: SSE = {best_probe_details['reg_sse'][r_idx]:.3f}, R2 = {best_probe_details['reg_r2'][r_idx]:.5f}")
        # Surgical Edit: Objective Breakdown
        print(f"  GLOBAL RESULTS | SSE SUM (Objective): {best_probe_details['loss']:.3f} | Global R2: {best_probe_details['global_r2']:.6f}")
        print(f"  SSE SUM = {best_probe_details['pure_fit_sse']:.3f} (SSE) + {best_probe_details['reg_penalty']:.3f} (regularization)")
    print("="*105)

def print_results_table(label, true_ihrs, opt_ihrs, truth_data, pred_data):
    print(f"\n>>> {label} <<<")
    print("="*80)
    print(f"{'LOC-AGE':<10} | {'TRUE IHR':<10} | {'OPT IHR':<10} | {'% DEV':<10} | {'SSE':<12} | {'R2':<8}")
    print("-" * 80)
    objective_sse_sum = 0.0
    for r_idx, r_name in enumerate(["A", "B", "C"]):
        sub_sse = 0
        for a_idx in range(5):
            t_val = true_ihrs[r_idx, a_idx, 0]
            o_str, d_str = "", ""
            if opt_ihrs is not None:
                o_val = opt_ihrs[r_idx * 5 + a_idx]
                diff = o_val - t_val
                o_str, d_str = f"{o_val:.6f}", f"{'+' if diff >= 0 else ''}{(diff/t_val)*100:.2f}%"
            age_sse = torch.sum((pred_data[:, r_idx, a_idx] - truth_data[:, r_idx, a_idx])**2).item()
            age_ss_tot = torch.sum((truth_data[:, r_idx, a_idx] - torch.mean(truth_data[:, r_idx, a_idx]))**2).item()
            age_r2 = 1.0 - (age_sse / age_ss_tot) if age_ss_tot > 0 else 0.0
            print(f"{r_name}-Age {a_idx:<2} | {t_val:<10.6f} | {o_str:<10} | {d_str:<10} | {age_sse:<12.3f} | {age_r2:<8.4f}")
            sub_sse += age_sse; objective_sse_sum += age_sse
        reg_true, reg_pred = truth_data[:, r_idx].sum(dim=(1, 2)), pred_data[:, r_idx].sum(dim=(1, 2))
        reg_sse = torch.sum((reg_pred - reg_true)**2).item()
        reg_sstot = torch.sum((reg_true - torch.mean(reg_true))**2).item()
        print(f"--- Subpop {r_name} R2: {1-(reg_sse/reg_sstot):.5f} | Total SSE: {sub_sse:.3f}")
        print("-" * 80)
    g_true, g_pred = truth_data.sum(dim=(1,2,3)), pred_data.sum(dim=(1,2,3))
    g_sse_total = torch.sum((g_pred - g_true)**2).item()
    g_sstot = torch.sum((g_true - torch.mean(g_true))**2).item()
    print(f"GLOBAL RESULTS | SSE: {g_sse_total:.3f} | R2: {1-(g_sse_total/g_sstot):.6f}")
    print(f"SSE SUM (Objective): {objective_sse_sum:.3f}")
    if opt_ihrs is not None:
        print(f"PARAMETER SSE: {np.sum((opt_ihrs - true_ihrs.flatten())**2):.8f}")
    print("="*80)



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

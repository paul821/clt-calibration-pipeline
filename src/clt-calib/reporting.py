import torch

def format_iter_report(pred, truth, subpop_truths, iteration_count, g_norm=None, sse_obj=None):
    # Surgical Edit: Logic to only print if flag is True
    if not VERBOSE_LBFGS: return 0.0, 0.0
    sse_global = torch.sum((pred.sum(dim=(1,2,3)) - truth.sum(dim=(1,2,3))) ** 2).item()
    ss_tot_g = torch.sum((truth.sum(dim=(1,2,3)) - torch.mean(truth.sum(dim=(1,2,3)))) ** 2).item()
    r2_global = 1.0 - (sse_global / ss_tot_g) if ss_tot_g > 0 else 0.0
    p_err_g = abs(pred.sum(dim=(1,2,3)).max().item() - truth.sum(dim=(1,2,3)).max().item()) / truth.sum(dim=(1,2,3)).max().item()
    out = f"Iter {iteration_count:02d} | Global: SSE={sse_global:.3f}, R2={r2_global:.4f}, P.Err={p_err_g*100:.1f}%"
    if g_norm is not None: out += f", Grad={g_norm:.5f}"
    print(out)
    if sse_obj is not None: print(f"         SSE SUM (Objective): {sse_obj:.3f}")
    for i, name in enumerate(['A', 'B', 'C']):
        p_sub, t_sub = pred[:, i].sum(dim=(1, 2)), subpop_truths[i]
        s_sse = torch.sum((p_sub - t_sub) ** 2).item()
        s_ss_tot = torch.sum((t_sub - torch.mean(t_sub)) ** 2).item()
        s_r2 = 1.0 - (s_sse / s_ss_tot) if s_ss_tot > 0 else 0.0
        print(f"         Subpop {name}: SSE={s_sse:.3f}, R2={s_r2:.4f}, P.Err={(abs(p_sub.max().item()-t_sub.max().item())/t_sub.max().item())*100:.1f}%")
    return sse_global, r2_global


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


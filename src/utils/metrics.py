# Libraries to import:
import torch
import numpy as np

def format_iter_report(pred, truth, subpop_truths, iteration_count, g_norm=None, sse_obj=None, verbose=True):
    """
    Format and print iteration report (Professor's style)
    
    Args:
        pred: predicted admissions (T, L, A, R)
        truth: truth admissions (T, L, A, R)
        subpop_truths: list of per-location truth tensors
        iteration_count: current iteration number
        g_norm: gradient norm (optional)
        sse_obj: SSE objective value (optional)
        verbose: whether to print
    
    Returns:
        sse_global, r2_global
    """
    if not verbose:
        return 0.0, 0.0
    
    sse_global = torch.sum((pred.sum(dim=(1,2,3)) - truth.sum(dim=(1,2,3))) ** 2).item()
    ss_tot_g = torch.sum((truth.sum(dim=(1,2,3)) - torch.mean(truth.sum(dim=(1,2,3)))) ** 2).item()
    r2_global = 1.0 - (sse_global / ss_tot_g) if ss_tot_g > 0 else 0.0
    p_err_g = abs(pred.sum(dim=(1,2,3)).max().item() - truth.sum(dim=(1,2,3)).max().item()) / truth.sum(dim=(1,2,3)).max().item()
    
    out = f"Iter {iteration_count:02d} | Global: SSE={sse_global:.3f}, R2={r2_global:.4f}, P.Err={p_err_g*100:.1f}%"
    if g_norm is not None: 
        out += f", Grad={g_norm:.5f}"
    print(out)
    
    if sse_obj is not None: 
        print(f"         SSE SUM (Objective): {sse_obj:.3f}")
    
    for i, name in enumerate(['A', 'B', 'C']):
        p_sub, t_sub = pred[:, i].sum(dim=(1, 2)), subpop_truths[i]
        s_sse = torch.sum((p_sub - t_sub) ** 2).item()
        s_ss_tot = torch.sum((t_sub - torch.mean(t_sub)) ** 2).item()
        s_r2 = 1.0 - (s_sse / s_ss_tot) if s_ss_tot > 0 else 0.0
        print(f"         Subpop {name}: SSE={s_sse:.3f}, R2={s_r2:.4f}, P.Err={(abs(p_sub.max().item()-t_sub.max().item())/t_sub.max().item())*100:.1f}%")
    
    return sse_global, r2_global

def print_beta_e0_table(true_betas, opt_betas, true_e0, opt_e0, best_probe_details=None):
    """
    Print Stage 1 Beta & E0 parameter recovery table (Professor's style)
    
    FULL PASTE from professor's code (lines 111-139)
    """
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
        print(f"  GLOBAL RESULTS | SSE SUM (Objective): {best_probe_details['loss']:.3f} | Global R2: {best_probe_details['global_r2']:.6f}")
        print(f"  SSE SUM = {best_probe_details['pure_fit_sse']:.3f} (SSE) + {best_probe_details.get('total_reg', 0.0):.3f} (regularization)")
    print("="*105)

def print_multi_compartment_table(true_params, opt_params, compartments, best_details=None):
    """
    Print multi-compartment parameter recovery table (NEW - enhanced version)
    
    Args:
        true_params: dict mapping compartment name → true values (L, A, R) or (L,)
        opt_params: dict mapping compartment name → optimized values
        compartments: list of compartment names to display
        best_details: optional dict with fit quality metrics
    """
    print(f"\n>>> MULTI-COMPARTMENT PARAMETER RECOVERY <<<")
    print("="*150)
    
    # Beta section
    if "beta" in compartments:
        print(f"\n{'BETA PARAMETERS':^150}")
        print("-" * 150)
        print(f"{'LOCATION':<15} | {'TRUE BETA':<15} | {'OPT BETA':<15} | {'% DEV':<10}")
        print("-" * 150)
        
        true_beta = true_params.get("beta", np.zeros(3))
        opt_beta = opt_params.get("beta", np.zeros(3))
        
        for i, loc_name in enumerate(["Loc A", "Loc B", "Loc C"]):
            t_val = true_beta[i] if i < len(true_beta) else 0.0
            o_val = opt_beta[i] if i < len(opt_beta) else 0.0
            dev = ((o_val - t_val) / t_val * 100) if t_val != 0 else 0.0
            print(f"{loc_name:<15} | {t_val:<15.6f} | {o_val:<15.6f} | {dev:>+9.2f}%")
    
    # Compartment sections (E, IP, ISR, ISH, IA)
    comp_names_display = [c for c in compartments if c != "beta"]
    for comp_name in comp_names_display:
        print(f"\n{comp_name.upper() + ' INITIAL VALUES':^150}")
        print("-" * 150)
        print(f"{'LOCATION':<12} | {'AGE':<6} | {'TRUE {}':<15} | {'OPT {}':<15} | {'ABS DEV':<12}".format(comp_name.upper(), comp_name.upper()))
        print("-" * 150)
        
        true_comp = true_params.get(comp_name, np.zeros((3, 5, 1)))
        opt_comp = opt_params.get(comp_name, np.zeros((3, 5, 1)))
        
        # Ensure 3D
        if true_comp.ndim == 2:
            true_comp = true_comp[..., np.newaxis]
        if opt_comp.ndim == 2:
            opt_comp = opt_comp[..., np.newaxis]
        
        for r_idx, loc_name in enumerate(["Loc A", "Loc B", "Loc C"]):
            for a_idx in range(5):
                t_val = true_comp[r_idx, a_idx, 0] if r_idx < true_comp.shape[0] and a_idx < true_comp.shape[1] else 0.0
                o_val = opt_comp[r_idx, a_idx, 0] if r_idx < opt_comp.shape[0] and a_idx < opt_comp.shape[1] else 0.0
                abs_dev = abs(o_val - t_val)
                
                row_name = loc_name if a_idx == 0 else ""
                print(f"{row_name:<12} | Age {a_idx} | {t_val:<15.6f} | {o_val:<15.6f} | {abs_dev:<12.6f}")
            
            # Location total
            true_total = true_comp[r_idx].sum() if r_idx < true_comp.shape[0] else 0.0
            opt_total = opt_comp[r_idx].sum() if r_idx < opt_comp.shape[0] else 0.0
            print(f"{'':<12} | TOTAL  | {true_total:<15.6f} | {opt_total:<15.6f} |")
            print("-" * 150)
    
    # Fit quality summary
    if best_details:
        print(f"\nFIT QUALITY SUMMARY:")
        print(f"  Global R²: {best_details.get('global_r2', 0.0):.6f}")
        print(f"  Total Loss: {best_details.get('loss', 0.0):.3f}")
        if 'reg_breakdown' in best_details:
            print(f"  Regularization Breakdown:")
            for reg_name, reg_val in best_details['reg_breakdown'].items():
                print(f"    {reg_name}: {reg_val:.3f}")
    
    print("="*150)

def print_results_table(label, true_ihrs, opt_ihrs, truth_data, pred_data):
    """
    Print IHR calibration results table (Professor's style)
    
    FIXED: Handle both per-location (L) and per-location-age (L*A) IHR
    """
    print(f"\n>>> {label} <<<")
    print("="*80)
    print(f"{'LOC-AGE':<10} | {'TRUE IHR':<10} | {'OPT IHR':<10} | {'% DEV':<10} | {'SSE':<12} | {'R2':<8}")
    print("-" * 80)
    
    objective_sse_sum = 0.0
    
    # Determine IHR dimensionality
    if opt_ihrs is not None:
        opt_ihrs = np.array(opt_ihrs)
        if opt_ihrs.ndim == 0:
            opt_ihrs = opt_ihrs.reshape(1)
        
        # Check if per-location (3 values) or per-location-age (15 values)
        is_per_location = (len(opt_ihrs) == 3)
    else:
        is_per_location = False
    
    for r_idx, r_name in enumerate(["A", "B", "C"]):
        sub_sse = 0
        for a_idx in range(5):
            t_val = true_ihrs[r_idx, a_idx, 0]
            o_str, d_str = "", ""
            
            if opt_ihrs is not None:
                if is_per_location:
                    # Same IHR for all ages in this location
                    o_val = opt_ihrs[r_idx]
                else:
                    # Per-location-age IHR
                    o_val = opt_ihrs[r_idx * 5 + a_idx]
                
                diff = o_val - t_val
                o_str, d_str = f"{o_val:.6f}", f"{'+' if diff >= 0 else ''}{(diff/t_val)*100:.2f}%"
            
            age_sse = torch.sum((pred_data[:, r_idx, a_idx] - truth_data[:, r_idx, a_idx])**2).item()
            age_ss_tot = torch.sum((truth_data[:, r_idx, a_idx] - torch.mean(truth_data[:, r_idx, a_idx]))**2).item()
            age_r2 = 1.0 - (age_sse / age_ss_tot) if age_ss_tot > 0 else 0.0
            print(f"{r_name}-Age {a_idx:<2} | {t_val:<10.6f} | {o_str:<10} | {d_str:<10} | {age_sse:<12.3f} | {age_r2:<8.4f}")
            sub_sse += age_sse
            objective_sse_sum += age_sse
        
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
        if is_per_location:
            # Compare per-location averages
            true_avg = true_ihrs.reshape(3, 5, -1).mean(axis=(1, 2))
            param_sse = np.sum((opt_ihrs - true_avg)**2)
        else:
            # Compare all values
            param_sse = np.sum((opt_ihrs - true_ihrs.flatten())**2)
        
        print(f"PARAMETER SSE: {param_sse:.8f}")
    
    print("="*80)
    
def print_optimizer_comparison_table(results_dict, stage_name="Stage 1"):
    """
    Print comparison table across multiple optimizers (NEW)
    
    Args:
        results_dict: dict mapping optimizer name → result dict
        stage_name: "Stage 1" or "Stage 2"
    """
    print(f"\n>>> {stage_name.upper()} - OPTIMIZER COMPARISON <<<")
    print("="*100)
    print(f"{'OPTIMIZER':<20} | {'LOSS':<12} | {'R²':<10} | {'DURATION (s)':<15} | {'ITERATIONS':<12}")
    print("-" * 100)
    
    for opt_name, result in sorted(results_dict.items(), key=lambda x: x[1].get('loss', float('inf'))):
        loss = result.get('loss', float('inf'))
        r2 = result.get('global_r2', result.get('r_squared', 0.0))
        duration = result.get('duration', 0.0)
        nit = result.get('nit', 0)
        
        print(f"{opt_name:<20} | {loss:<12.4f} | {r2:<10.6f} | {duration:<15.2f} | {nit:<12}")
    
    print("="*100)
    
    # Best optimizer
    best_opt = min(results_dict.items(), key=lambda x: x[1].get('loss', float('inf')))
    print(f"\nBEST OPTIMIZER: {best_opt[0]} (Loss: {best_opt[1].get('loss', 0.0):.4f})")
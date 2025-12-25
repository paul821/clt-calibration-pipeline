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

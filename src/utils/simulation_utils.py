import torch
import clt_toolkit as clt

def apply_time_stretching(params, time_stretch_factor):
    """
    Apply time stretching to transition rates
    
    Divides all transition rates by time_stretch_factor to potentially elongate epidemic dynamics
    """
    if isinstance(time_stretch_factor, (float, int)):
        if time_stretch_factor == 1.0:
            return params
    
    stretched_params = clt.updated_dataclass(params, {
        attr: getattr(params, attr) / time_stretch_factor
        for attr in ['E_to_I_rate', 'IP_to_IS_rate', 'ISR_to_R_rate', 'IA_to_R_rate', 
                     'ISH_to_H_rate', 'HR_to_R_rate', 'HD_to_D_rate', 'R_to_S_rate']
    })
    return stretched_params

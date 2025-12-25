
# --- 2. GLOBAL CONSTANTS & CONFIG ---
T = 180
timesteps_per_day = 4
torch.manual_seed(0)
# #dpm
np.random.seed(0)

#MODE = "BETA_ONLY"
#MODE = "IHR_ONLY"
MODE = "SEQUENTIAL"

CALIBRATION_METHOD = "L-BFGS-B"
GSS_TOLERANCE = 1.0
VERBOSE_LBFGS = False # Surgical Edit: Gating flag for all iteration outputs

AGE_LABELS = ["0-4", "5-17", "18-49", "50-64", "65+"]

ESTIMATION_CONFIG = {
    "beta_param": "L",
    "estimate_initial": {"E": True, "IP": False, "ISR": False, "ISH": False, "IA": False},
}

REG_CONFIG = {
    "lambda_e0_zero": 10.0,      
    "lambda_e0_target": 10.0,    
    "target_e0_values": [0.0, 1.0, 0.0],    
    "target_age_idx": 2         
}

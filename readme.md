# CLT Calibration Pipeline

A modular, multi-optimizer calibration framework for the CLT_BaseModel with support for:
- **Multiple calibration modes**: Beta-only, IHR mode, sequential 2-stage
- **Multi-optimizer support**: L-BFGS-B, CG, Adam, least_squares_fd
- **Regional loss decomposition** for better identifiability
- **Structured regularization** to prevent shadow solutions
- **Golden Section Search** for temporal offset discovery (IHR mode)
- **Noise robustness testing** (Poisson, Gaussian)
- **Time stretching** for epidemic dynamics tuning

## Features

### IHR Mode
- **Stage 1**: Transmission rate (Beta) and initial compartments (E0, IP0, etc.) with optional GSS offset discovery
- **Stage 2**: Age-stratified infection-hospitalization rates (IHR)
- Regional loss decomposition preserves location-specific signals
- Structured regularization prevents misassigned epidemic seeding

### Multi-Optimizer Mode
- Run multiple optimizers simultaneously for comparison
- Restart strategy: Wide, Medium, Narrow search phases
- Warm-starting between restart phases for efficiency
- Comprehensive reporting and visualization

### Key Mechanisms
1. **Log-space parameterization** for all parameters (guarantees positivity)
2. **Regional SSE decomposition** (not global MSE) for better gradients
3. **Structural regularization** for initial compartments (not just magnitude penalties)
4. **Loss component breakdown** (data fit + regularization terms)
5. **Multi-compartment support** (E, IP, ISR, ISH, IA)

## Installation
```bash
git clone https://github.com/paul821/clt-calibration-pipeline.git
cd clt-calibration-pipeline
pip install -e .
```

## Quick Start

### Example 1: Basic Beta Calibration
```python
from config.calibration_config import CalibrationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration

config = CalibrationConfig(
    mode="BETA_ONLY",
    optimizers=["L-BFGS-B"],
    noise_type="poisson"
)

model_config = ModelConfig()
stage1, stage2 = run_calibration(config, model_config)
```

### Example 2: IHR Mode with Multiple Optimizers
```python
config = CalibrationConfig(
    mode="IHR_MODE",
    optimizers=["L-BFGS-B", "CG"],
    enable_gss=True,
    gss_offset_range=(-30, 15)
)

config.estimation_config = {
    "beta_param": "L",
    "estimate_initial": {"E": True, ...},
    "ihr_param": "L"
}

stage1, stage2 = run_calibration(config, model_config)
```

### Example 3: Multi-Compartment Calibration
```python
config = CalibrationConfig(
    mode="BETA_ONLY",
    optimizers=["L-BFGS-B", "CG", "Adam"]
)

config.estimation_config = {
    "beta_param": "L",
    "estimate_initial": {
        "E": True,   # Estimate E0
        "IP": True,  # Estimate IP0
        ...
    }
}

# Structured regularization
from config.calibration_config import RegularizationConfig
config.regularization = RegularizationConfig(
    compartment_configs={
        "E": {
            "type": "structural",
            "location_targets": [0.0, 1.0, 0.0],
            "age_targets": [0, 0, 1, 0, 0],
            "lambda_on_target": 10.0,
            "lambda_off_target": 10.0
        }
    }
)

stage1, _ = run_calibration(config, model_config)
```

## Running Examples
```bash
# Basic calibration
python examples/basic_calibration.py

# IHR mode with GSS
python examples/ihr_mode_example.py

# Multi-compartment
python examples/multi_compartment_example.py

# Two-Region Austin (Time Stretch Estimation)
python examples/austin_2region_tester.py
```

## Configuration Guide

### Calibration Modes

- **`BETA_ONLY`**: Estimate transmission rates (and optionally initial compartments)
- **`IHR_MODE`**: Two-stage calibration with GSS offset discovery
- **`SEQUENTIAL`**: Two-stage without offset discovery (beta → IHR)

### Optimizer Selection
```python
config.optimizers = ["L-BFGS-B", "CG", "Adam", "least_squares_fd"]
```

- **L-BFGS-B**: Best for smooth objectives, supports bounds
- **CG**: Fast, no bounds, good for large-scale
- **Adam**: Robust to noise, slower convergence
- **least_squares_fd**: Finite-difference Jacobian, good baseline

### Loss Aggregation
```python
config.loss_aggregation = "regional"  # or "location_age" or "global"
```

- **`regional`**: Sum of per-location SSE (recommended for Stage 1)
- **`location_age`**: Sum of per-(location,age) SSE (recommended for Stage 2/IHR)
- **`global`**: Single global SSE (legacy, not recommended)

### Regularization
```python
from config.calibration_config import RegularizationConfig

config.regularization = RegularizationConfig(
    beta_type="l2_magnitude",
    beta_lambda=1e-2,
    compartment_configs={
        "E": {
            "type": "structural",  # or "l2_magnitude"
            ...
        }
    }
)
```

## Output Files

### Plots
- `stage1_optimizer_comparison.png`: Bar charts comparing loss/R2/runtime
- `stage1_loss_breakdown.png`: Stacked bars showing data fit vs regularization
- `stage1_parameter_recovery.png`: True vs estimated parameters
- `calibration_15_panel.png`: 15-panel (3 locations × 5 ages) fit quality
- `final_calibration_regional_aggregate.png`: 4-panel regional summary
- `SSE_Stage1_Convergence_*.png`: GSS convergence curves (IHR mode only)

### Console Output
- Optimizer comparison tables
- Parameter recovery tables (beta, E0, IP0, etc.)
- Fit quality metrics (SSE, R2, per location and global)
- Loss component breakdown (data fit + regularization)

## Project Structure
```
clt-calibration-pipeline/
├── config/              # Configuration dataclasses
├── src/
│   ├── loss/           # Loss functions (regional, regularization)
│   ├── optimization/   # Optimizer modules (GSS, multi-optimizer)
│   ├── utils/          # Utilities (theta transforms, metrics, noise)
│   └── visualization/  # Plotting and reporting
├── scripts/            # Main calibration script
└── examples/           # Usage examples
```

## Advanced Usage

### Time Stretching
```python
config.apply_time_stretch = True
config.time_stretch_factor = 1.1 
config.estimate_time_stretch = True  # Enable optimization
config.time_stretch_bounds = (0.5, 2.0)
```

Elongates epidemic dynamics by dividing transition rates (1.0 = no stretch, >1.0 = slower dynamics). Can be estimated as a free parameter during calibration.

### Custom Regularization
```python
# Enforce specific seeding pattern
config.regularization.compartment_configs["E"] = {
    "type": "structural",
    "location_targets": [0.0, 1.0, 0.0],  # Seed in location 1
    "age_targets": [0, 0, 1, 0, 0],       # Seed in age group 2
    "lambda_on_target": 10.0,
    "lambda_off_target": 10.0
}
```

### Verbosity Control
```python
config.verbosity = 0  # Silent
config.verbosity = 1  # Summary only
config.verbosity = 2  # Detailed (iteration-level)
config.verbosity = 3  # Debug (everything)
```

## Troubleshooting

**Optimizer fails to converge:**
- Try different optimizer (CG often more robust than L-BFGS-B)
- Increase regularization lambda
- Check initial guess quality

**Shadow solutions (wrong seeding location):**
- Use structured regularization (not just L2)
- Increase `lambda_on_target` and `lambda_off_target`

**GSS takes too long:**
- Reduce offset range: `config.gss_offset_range = (-10, 10)`
- Increase tolerance: `config.gss_tolerance = 5.0`
- Or disable: `config.enable_gss = False`

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate #for mac/linux
venv\Scripts\activate #for windows
pip install -e .
python3 examples/basic_calibration.py
python3 scripts/run_calibration.py
```
or
```bash
# Create a custom run script
python3 -c "
from scripts.run_calibration import run_calibration
from config.calibration_config import CalibrationConfig
from config.model_config import ModelConfig

config = CalibrationConfig(
    mode='SEQUENTIAL',
    verbose_lbfgs=True,
    gss_tolerance=0.5
)
model = ModelConfig()

run_calibration(config, model)
"
```

## How to Commit
```bash
git add .
git status 
git commit -m "Comment"
git push
```

# CLT Calibration Pipeline

A modular calibration framework for the CLT_BaseModel using multi-stage parameter estimation with Golden Section Search and L-BFGS-B optimization.

## Features

- **Stage 1**: Transmission rate (β) and initial seeding (E₀) calibration with temporal offset discovery
- **Stage 2**: Infection-hospitalization ratio (IHR) calibration across age-stratified populations
- **Golden Section Search**: Automatic epidemic start date discovery
- **Stochastic Truth Anchoring**: Realistic Poisson-noised synthetic data
- **Regularization**: Structural penalties to prevent shadow solutions

## Installation
```bash
git clone https://github.com/yourusername/clt-calibration-pipeline.git
cd clt-calibration-pipeline
pip install -e .
```

## Quick Start
```python
from config.calibration_config import CalibrationConfig
from config.model_config import ModelConfig
from scripts.run_calibration import run_calibration

# Configure calibration
calib_config = CalibrationConfig(mode="SEQUENTIAL")
model_config = ModelConfig()

# Run
final_state, final_params, predictions = run_calibration(calib_config, model_config)
```

## Project Structure
```
clt-calibration-pipeline/
├── config/              # Configuration dataclasses
├── src/
│   ├── optimization/    # Stage 1 (GSS) and Stage 2 (IHR) optimizers
│   ├── utils/          # Parameter transforms and metrics
│   └── visualization/   # Plotting utilities
├── scripts/            # Main execution scripts
└── examples/           # Usage examples
```

## Configuration

### Calibration Modes

- `BETA_ONLY`: Stage 1 only (transmission + seeding)
- `IHR_ONLY`: Stage 2 only (hospitalization rates)
- `SEQUENTIAL`: Full two-stage pipeline

### Key Parameters
```python
CalibrationConfig(
    T=180,                    # Simulation days
    timesteps_per_day=4,      # Temporal resolution
    mode="SEQUENTIAL",        # Calibration mode
    gss_tolerance=1.0,        # GSS convergence threshold
    verbose_lbfgs=False       # Iteration-level logging
)
```


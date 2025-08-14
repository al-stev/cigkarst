# W&B Enterprise Workshop: Advanced Geological AI

A hands-on demonstration of Weights & Biases for conditional diffusion workflows in subsurface modeling.

## Overview

This workshop demonstrates Weights & Biases features for geological AI workflows. Using a simulated conditional diffusion model trained on the CigKarst geological dataset, participants can explore ML experiment tracking, visualization, and reporting capabilities.

**Demonstration Areas:**
- Experiment tracking with metrics and system monitoring
- Interactive visualizations including 3D geological volumes and well logs
- Artifact lineage from dataset to trained models
- Hyperparameter optimization with early termination
- Programmatic reporting

## Workshop Content

**Live Training Monitoring** - Stream metrics and system telemetry  
**3D Volume Inspection** - View prediction, condition, and residual volumes  
**Validation Tables** - Log per-epoch comparisons with images and interactive plots  
**Hyperparameter Sweeps** - Launch Bayesian optimization with early termination  
**Model Registry** - Version checkpoints with aliases  
**Executive Reports** - Generate programmatic summaries  

## Quick Start

### Prerequisites

- Python 3.8+
- Weights & Biases account ([sign up here](https://wandb.ai))

*Note: Dependencies are installed automatically by the notebook*

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Launch the workshop
jupyter notebook workshop.ipynb
```

### Configuration

1. **Set your W&B credentials** in the notebook:
   ```python
   ENTITY = "your-team-entity"    # Your W&B team/organization
   PROJECT = "your-project-name"  # Project workspace name
   ```

2. **Optional environment variables** (skip prompts):
   ```bash
   export WANDB_API_KEY="your-api-key"
   export WANDB_ENTITY="your-team-entity"
   export WANDB_PROJECT="your-project-name"
   ```

## Workshop Structure

### 1. **Setup & Configuration** (~5 min)
- W&B authentication and project setup
- Configuration management for conditional diffusion parameters

### 2. **Dataset & Lineage** (~10 min)
- Load versioned CigKarst geological dataset from W&B Registry
- Establish data-to-model lineage for full auditability

### 3. **Training & Monitoring** (~20 min)
- Simulate conditional diffusion training with live metrics
- Log rich media: 2D slice grids, 3D volume renders, validation tables
- Track system resources alongside model performance

### 4. **Hyperparameter Optimization** (~15 min)
- Launch Bayesian sweep with early termination (Hyperband)
- Compare multiple runs and identify optimal configurations

### 5. **Model Registry & Governance** (~5 min)
- Version best checkpoint as artifact with metadata
- Link to Model Registry with staging/production aliases

### 6. **Executive Reporting** (~5 min)
- Generate programmatic reports for stakeholder communication
- Combine metrics, visualizations, and governance information

## Technical Details

### Geological Context

The workshop simulates a **conditional diffusion model** for geological structure generation:

- **Input (Y)**: Seismic amplitude volumes (64³ float32)
- **Output (X)**: Karst structure predictions (64³ float32)  
- **Conditioning**: Y = f(X) relationship for physical consistency
- **Validation**: Compare predictions against well logs and forward models

### Simulation Approach

Rather than training a computationally expensive real model, the workshop uses **high-fidelity simulation** that:

- Generates realistic loss curves and convergence patterns
- Produces geological volumes with proper spatial structure
- Simulates forward modeling for condition consistency
- Maintains deterministic behavior for reproducible demos

### 3D Visualization Options

- **PyVista (default)**: High-quality HTML renders for stable inline viewing
- **ipyvolume (optional)**: Interactive widgets for detailed exploration
- **Plotly (tier 1/2)**: Downsampled comparison views

Configure via training parameters:
```python
config = {
    "enable_3d": True,
    "enable_high_fidelity_3d": True,  # PyVista renders
    "enable_ipyvolume": False,        # Optional interactive widgets
}
```

## Files Structure

```
├── workshop.ipynb              # Main workshop notebook
├── prepare_dataset_shell.py    # Dataset preparation utilities
├── Wandb_Diffusion_Demo.ipynb  # Standalone demo version
└── README.md                   # This file
```

## Key Features Demonstrated

### MLOps Features
- **Experiment Tracking**: Experiments, configs, and results tracking
- **Team Collaboration**: Share runs, artifacts, and insights
- **Audit Trails**: Complete lineage tracking

### Visualizations
- **3D Geological Volumes**: Inline HTML renders of predictions and residuals
- **Interactive Well Logs**: Plotly charts comparing ground truth vs predictions
- **Validation Tables**: Per-epoch media with sortable metrics and images

### Workflow Features
- **Hyperparameter Sweeps**: Bayesian optimization with early termination
- **Model Registry**: Versioning with staging/production aliases
- **Reports**: Programmatic generation of summaries

## Target Audience

**ML Practitioners & Geoscientists**: Experiment tracking, visualizations, and model development workflows.

**Technical Leaders & Stakeholders**: Governance, metrics tracking, and reporting capabilities.

## Workshop Focus

This workshop addresses common challenges in ML workflows:

- **Tool Fragmentation**: Multiple disconnected tools for different workflow stages
- **Monitoring Gaps**: Difficulty tracking long-running model training
- **Audit Requirements**: Need for transparent experiment and model lineage
- **Communication**: Translating technical results for non-technical stakeholders

The workshop demonstrates W&B features that may help address these challenges in geological AI workflows.

## Support

- **W&B Documentation**: [docs.wandb.ai](https://docs.wandb.ai)
- **Enterprise Solutions**: Contact your W&B representative

## License

This workshop is provided for educational and evaluation purposes. Please refer to your W&B licensing agreement for terms of use.

---

*This workshop provides a hands-on exploration of W&B features for geological AI workflows.*
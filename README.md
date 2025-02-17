# OocyteForestBoost

A machine learning framework for predicting cell-cell interactions during oocyte maturation using ensemble methods (Random Forest and XGBoost).

## Architecture

```mermaid
flowchart TB
    subgraph Input["1 Input Data"]
        direction LR
        A1["Cell Type 1\nGene Expression"] --> FP
        A2["Cell Type 2\nGene Expression"] --> FP
        style A1 fill:#f9e6e6
        style A2 fill:#f9e6e6
    end

    subgraph Feature["2 Feature Engineering"]
        direction TB
        FP["Feature Preparation"] --> F1["Basic Features"]
        FP --> F2["Interaction Features"]
        FP --> F3["Correlation Features"]
        F1 & F2 & F3 --> CM["Combined Matrix"]
        style FP fill:#e6f3ff
        style F1 fill:#e6f3ff
        style F2 fill:#e6f3ff
        style F3 fill:#e6f3ff
        style CM fill:#e6f3ff
    end

    subgraph Model["3 Model Training"]
        direction TB
        CM --> RF["Random Forest Model"]
        CM --> XGB["XGBoost Model"]
        RF --> EP["Ensemble\nPredictions"]
        XGB --> EP
        style RF fill:#e6ffe6
        style XGB fill:#e6ffe6
        style EP fill:#e6ffe6
    end

    subgraph Evaluation["4 Model Evaluation"]
        direction TB
        EP --> M1["ROC Curve\nAUC Score"]
        EP --> M2["PR Curve\nPrecision/Recall"]
        EP --> M3["Feature\nImportance Analysis"]
        EP --> M4["FDR Control\n(p < 0.05)"]
        style M1 fill:#fff5e6
        style M2 fill:#fff5e6
        style M3 fill:#fff5e6
        style M4 fill:#fff5e6
    end

    subgraph Output["5 Final Results"]
        direction TB
        M1 & M2 & M3 & M4 --> R1["Model Performance\nMetrics"]
        M1 & M2 & M3 & M4 --> R2["Significant\nInteractions"]
        style R1 fill:#e6e6ff
        style R2 fill:#e6e6ff
    end

    classDef section fill:#fff,stroke:#333,stroke-width:2px
    class Input,Feature,Model,Evaluation,Output section
    linkStyle default stroke:#666,stroke-width:2px
```

## Features

- Ensemble learning combining Random Forest and XGBoost
- Advanced feature engineering for gene expression data
- Statistical validation with FDR control (< 0.05)
- Comprehensive evaluation metrics (ROC, PR curves)
- Interactive visualization of results
- 35% improvement in prediction accuracy over baseline methods

## Installation

You can install OocyteForestBoost directly from PyPI:

```bash
pip install oocyteforestboost
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/qianxu05172019/OocyteForestBoost.git

# Navigate to the directory
cd OocyteForestBoost

# Install from source
pip install .
```

## Quick Start

```python
from oocyteforestboost.predictor import CellInteractionPredictor
import numpy as np

# Generate example data (replace with your own data)
n_samples = 1000
n_features = 100
cell_type1_expression = np.random.normal(0, 1, (n_samples, n_features))
cell_type2_expression = np.random.normal(0, 1, (n_samples, n_features))

# Initialize predictor
predictor = CellInteractionPredictor()

# Prepare features
X = predictor.prepare_features(cell_type1_expression, cell_type2_expression)

# If you have interaction labels (for training)
y = np.random.binomial(1, 0.5, n_samples)  # Replace with your actual labels

# Train models
predictor.train_models(X, y)

# Make predictions
predictions = predictor.predict_proba(X)

# Identify significant interactions
significant_interactions = predictor.apply_fdr_control(predictions)

# Evaluate model performance
eval_results = predictor.evaluate_models(X, y)

# Visualize results
predictor.plot_evaluation_metrics(eval_results)
```

## Input Data Format

The predictor expects gene expression data in the following format:
- Gene expression matrices for two cell types (numpy arrays)
- Each row represents a sample
- Each column represents a gene
- Labels should be binary (0: no interaction, 1: interaction)

## Citation

If you use this tool in your research, please cite:

```
@article{OocyteForestBoost2024,
  title={OocyteForestBoost: A machine learning framework for predicting cell-cell interactions during oocyte maturation},
  author={Xu, Qian},
  year={2024}
}
```

## License

MIT License

## Contact

Qian Xu - qianxu0517@gmail.com

Project Link: [https://github.com/qianxu05172019/OocyteForestBoost](https://github.com/qianxu05172019/OocyteForestBoost)
# Data-Centric AI for Time Series Classification

This repository provides a flexible experimentation framework to evaluate **data-centric adaptation strategies** (like label flipping, instance reduction, and length reduction) on time series classification tasks.

---
## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Experiments [optional]

Edit the `configs/experiment.yaml` file to define:

- Datasets and classifiers
- Data-centric strategies (random, systematic, baseline, instance/length reduction)
- Parameter grids (e.g., flip ratios, reduction ratios, seeds)

Example configuration:

```yaml
experiment:
  - dataset: ["Beef", "GunPoint"]
    classifier: ["mini-rocket", "catch22"]
    random_seed: [0]
    strategy:
      - type: "length_reduction"
        mode: "random"
        params:
          reduction_fraction: [0.1, 0.3]
          take_from_end: [false, true]
      - type: "number_instances"
        mode: "random"
        params:
          reduction_ratio: [0.1, 0.3]
      - type: "label_flipping"
        mode: "random"
        params:
          flip_ratio: [0.0, 0.1]
      - type: "baseline"
        mode: "none"
        params: {}
```

### 3. Run Experiments

```bash
python main.py
```

The script will:
- Download datasets if needed
- Run all configured experiments with progress tracking
- Skip already completed experiments automatically
- Log all activities to console and `logs/data_centric.log`

Results and metrics will be saved in the `results/` folder with timestamps and tracked in `results/summary.csv`.

---

## 📊 Outputs

Each experiment creates a timestamped folder in `results/` containing:
- `config.json`: Complete experiment configuration
- `preds.npy`: Model predictions (class labels)
- `y_test.npy`: Ground truth test labels
- `metrics.json`: Performance metrics (accuracy and F1 score)
- The `results/summary.csv` file provides a cumulative overview of all experiments.

---

## 📈 Result Analysis

### Using the Jupyter Notebook
- Open `notebooks/visualize_results.ipynb` to explore results

```python
from src.utils.resultHandler import ResultHandler
from src.utils.configHandler import ConfigHandler

config_handler = ConfigHandler()
result_handler = ResultHandler(config_handler.SUMMARY_FILE)

# Iterate through all results
for folder, result in result_handler.iter_results():
    config = result['config']
    metrics = result['metrics'] 
    predictions = result['preds']
    ground_truth = result['y_test']
    # Analyze results...
```

---

## 🧪 Supported Strategies

### Data-Centric Adaptation Strategies

- ✅ **RandomLabelFlipping**: Randomly flip a percentage of training labels
  - Parameters: `flip_ratio` (0.0-1.0)
  
- ✅ **SystematicLabelFlipping**: Flip labels based on a confusion matrix
  - Parameters: `confusion_matrix` (nested dict of transition probabilities)
  
- ✅ **NumberInstanceStrategy**: Randomly reduce training instances
  - Parameters: `reduction_ratio` (0.0-1.0)
  
- ✅ **LengthReductionStrategy**: Truncate time series length
  - Parameters: `reduction_fraction` (0.0-1.0), `take_from_end` (bool)
  
- ✅ **BaselineStrategy**: No adaptation (control condition)
  - Parameters: none

### Classifiers
The framework supports all classifiers from the time series classification bakeoff:
- `mini-rocket`: MiniRocket transformer with Ridge classifier
- `catch22`: Catch22 features with Random Forest
- `hydra`: Hydra transformer with Ridge classifier
- And many more from the `tsml-eval` package

### Datasets
All UCR Time Series Classification Archive datasets are supported automatically through the `aeon` package.

---

## 📁 Project Structure


```plaintext
src/
├── data_centric/
│   └── dataCentricStrategy.py      # Data-centric strategy implementations
├── data_handling/
│   └── datasetHandler.py           # UCR dataset loading utilities
├── experiments/
│   └── experiment.py               # Experiment orchestration
├── models/
│   └── classifierHandler.py        # Classifier abstraction (bakeoff models)
└── utils/
    ├── configHandler.py            # Configuration and path management
    ├── logger.py                   # Logging utilities
    └── resultHandler.py            # Results loading and analysis
configs/
├── experiment.yaml                 # Experiment configuration
├── paths.yaml                      # Path configurations
└── visualisations.mplstyle         # Matplotlib style settings
main.py                             # Main experiment runner
notebooks/
└── visualize_results.ipynb         # Jupyter notebook for result analysis
results/                            # Output folders and summary.csv
logs/                               # Application logs
Univariate_ts/                      # Downloaded UCR datasets (auto-created)
```

---

## 🔧 Extending the Framework

### Adding New Strategies
1. Create a new class inheriting from `DataCentricStrategy`
2. Implement the `apply(X, y)` method
3. Add it to the strategy registry in `DataCentricStrategy.from_config()`

Example:
```python
class CustomStrategy(DataCentricStrategy):
    def __init__(self, custom_param: float):
        super().__init__()
        self.custom_param = custom_param
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Your custom logic here
        return X_modified, y_modified
```

## 📬 Questions?

Open an issue or reach out to the authors for questions and contributions!
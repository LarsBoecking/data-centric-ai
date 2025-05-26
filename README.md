# Data-Centric AI for Time Series Classification

This repository provides a flexible experimentation framework to evaluate **data-centric adaptation strategies** (like label flipping, instance reduction, and length reduction) on time series classification tasks.

---

## 📁 Project Structure

```plaintext
src/
├── classifierHandler.py        # Classifier abstraction
├── dataCentricStrategy.py      # Data-centric strategy implementations
├── datasetHandler.py           # Dataset loading utilities
├── experiment.py               # Experiment orchestration
├── resultHandler.py            # Results loading and analysis
├── utils.py                    # Shared utilities (config, logging, download)
main.py                         # Main experiment runner
experiment.yaml                 # Experiment configuration
notebooks/
    visualize_results.ipynb     # Jupyter notebook for result analysis
results/                        # Output folders and summary.csv
Univariate_ts/                  # Downloaded UCR datasets
```

---

## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### 2. Prepare Data

The dataset will be automatically downloaded and extracted to `Univariate_ts/` on first run if not present. No manual download is required.

### 3. Configure Experiments

Edit the `experiment.yaml` file to define:

- Datasets and classifiers
- Data-centric strategies (random, systematic, baseline, instance/length reduction)
- Parameter grids (e.g., flip ratios, reduction ratios, seeds)

Example:

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

### 4. Run Experiments

    python main.py

Results and metrics will be saved in the `results/` folder with a timestamp and tracked in `results/summary.csv`.

---

## 📊 Outputs

Each experiment saves:
- `config.json`: Full configuration
- `preds.npy`, `y_test.npy`: Predictions and ground truth (class labels)
- `metrics.json`: Accuracy and F1 score
- `summary.csv`: Cumulative overview of all runs

---

## 📈 Result Analysis

- Use the notebook in `notebooks/visualize_results.ipynb` to load and analyze results.
- The `ResultHandler` class loads results from `summary.csv` and provides access to all experiment outputs.
- The notebook demonstrates how to visualize confusion matrices and class distributions for predictions vs. ground truth.

---

## 🧪 Strategy Support

- ✅ `RandomLabelFlipping`: Randomly flip a percentage of labels
- ✅ `SystematicLabelFlipping`: Flip labels based on a confusion matrix
- ✅ `NumberInstanceStrategy`: Randomly reduce the number of training instances
- ✅ `LengthReductionStrategy`: Reduce the length of time series
- ✅ `BaselineStrategy`: No adaptation (control)

To add your own strategy, implement a subclass of `DataCentricStrategy` and register it in the factory method.

---

## 📬 Questions?

Open an issue or reach out to the authors for questions and contributions!

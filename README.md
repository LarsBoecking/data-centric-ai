# Data-Centric AI for Time Series Classification

This repository provides a flexible experimentation framework to evaluate **data-centric adaptation strategies** (like label flipping) on time series classification tasks.

---

## 📁 Project Structure

```plaintext
src/
├── classifierHandler.py        
├── dataCentricStrategy.py      
├── datasetHandler.py          
├── experiment.py               
├── utils.py                    
main.py                         
experiment.yaml                 
results/                       
```

---

## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt

### 2. Prepare Data

Download the UCR Time Series Classification Archive:
http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip

Extract it into a folder named `112UCRFolds/` in the project root.

### 3. Configure Experiments

Edit the `experiment.yaml` file to define:

- Datasets and classifiers
- Data-centric strategies (random, systematic, baseline)
- Parameter grids (e.g., flip ratios, seeds)

Example:

    experiment:
      - dataset: ["Beef", "GunPoint"]
        classifier: ["mini-rocket", "InceptionTime"]
        random_seed: [0, 42]
        strategy:
          - type: "label_flipping"
            mode: "random"
            params:
              flip_ratio: [0.0, 0.1]
          - type: "baseline"

### 4. Run Experiments

    python main.py

Results and metrics will be saved in the `results/` folder with a timestamp and tracked in `results/summary.csv`.

---

## 📊 Outputs

Each experiment saves:
- `config.json`: Full configuration
- `preds.npy`, `y_test.npy`: Predictions and ground truth
- `metrics.json`: Accuracy and F1 score
- `summary.csv`: Cumulative overview of all runs

---

## 🧪 Strategy Support

- ✅ `RandomLabelFlipping`: Randomly flip a percentage of labels
- ✅ `SystematicLabelFlipping`: Flip labels based on a confusion matrix
- ✅ `BaselineStrategy`: No adaptation (control)

To add your own strategy, implement a subclass of `DataCentricStrategy` and register it in the factory method.

---

## 📬 Questions?

Open an issue or reach out to the authors for questions and contributions!

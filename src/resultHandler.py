import os
import json
import numpy as np
import pandas as pd

class ResultHandler:
    def __init__(self, summary_csv_path: str):
        self.summary_csv_path = summary_csv_path
        self.summary_df = pd.read_csv(summary_csv_path)

    def get_result_folders(self):
        return self.summary_df['folder'].tolist()

    def load_result(self, folder: str):
        result = {}
        config_path = os.path.join(folder, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                result['config'] = json.load(f)
        else:
            result['config'] = None
        metrics_path = os.path.join(folder, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                result['metrics'] = json.load(f)
        else:
            result['metrics'] = None
        preds_path = os.path.join(folder, 'preds.npy')
        if os.path.exists(preds_path):
            result['preds'] = np.load(preds_path)
        else:
            result['preds'] = None
        y_test_path = os.path.join(folder, 'y_test.npy')
        if os.path.exists(y_test_path):
            result['y_test'] = np.load(y_test_path)
        else:
            result['y_test'] = None
        return result

    def iter_results(self):
        for folder in self.get_result_folders():
            yield folder, self.load_result(folder)

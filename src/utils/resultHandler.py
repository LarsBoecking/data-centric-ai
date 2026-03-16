import os
import json
import numpy as np
import pandas as pd
from ..utils.logger import get_logger

class ResultHandler:
    def __init__(self, summary_csv_path: str):
        self.summary_csv_path = summary_csv_path
        self.summary_df = pd.read_csv(summary_csv_path)
        self.logger = get_logger(__name__)
        
        # Normalize column names: map old schema to new schema
        column_mapping = {
            'strategy': 'type',
            'strategy_mode': 'mode',
        }
        for old_col, new_col in column_mapping.items():
            if old_col in self.summary_df.columns and new_col not in self.summary_df.columns:
                self.summary_df[new_col] = self.summary_df[old_col]
        
        # Parse strategy_params JSON string into individual columns
        if 'strategy_params' in self.summary_df.columns:
            self._parse_strategy_params()
        
        self.logger.info(f"Initialized ResultHandler with summary_csv_path: {summary_csv_path}")
    
    def _parse_strategy_params(self):
        """Parse strategy_params JSON column into individual parameter columns."""
        import json
        
        # Extract all unique parameters from strategy_params
        all_params = set()
        for params_str in self.summary_df['strategy_params']:
            try:
                params = json.loads(params_str.replace("'", '"'))
                all_params.update(params.keys())
            except:
                pass
        
        # Create columns for each parameter
        for param in all_params:
            self.summary_df[param] = None
            for idx, params_str in enumerate(self.summary_df['strategy_params']):
                try:
                    params = json.loads(params_str.replace("'", '"'))
                    self.summary_df.at[idx, param] = params.get(param)
                except:
                    pass
        
        self.logger.info(f"Parsed strategy_params into columns: {list(all_params)}")


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


import pandas as pd
import numpy as np
from src.data_handling.datasetHandler import UCRDataset


def prepare_model_data(analyse_df: pd.DataFrame, dataset_names: list, datasets_path: str) -> pd.DataFrame:
    """
    Prepare and merge results dataframe with dataset metadata.
    """
    print(f"Loading metadata for {len(dataset_names)} datasets...")
    
    dataset_metadata = {}
    for dataset_name in dataset_names:
        try:
            dataset = UCRDataset(dataset_name, path=datasets_path)
            X_train, y_train, X_test, y_test, meta = dataset.load()
            dataset_metadata[dataset_name] = {
                'N_train': X_train.shape[0],
                'T': X_train.shape[2],
                'K': len(np.unique(y_train))
            }
        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
    
    metadata_df = pd.DataFrame(dataset_metadata).T
    metadata_df.index.name = 'dataset'
    metadata_df = metadata_df.reset_index()
    
    model_df = analyse_df.merge(metadata_df, on='dataset', how='left')
    
    model_df['p'] = pd.to_numeric(
        model_df['strategy_params'].str.extract(r'"p":\s*([0-9.]+)')[0]
    )
    
    def classify_type(clf_name):
        if 'cnn' in clf_name.lower() or 'conv' in clf_name.lower():
            return 'CNN'
        elif 'rocket' in clf_name.lower():
            return 'Rocket-based'
        elif 'forest' in clf_name.lower() or 'ensemble' in clf_name.lower():
            return 'Ensemble'
        else:
            return 'Other'
    
    model_df['classifier_type'] = model_df['classifier'].apply(classify_type)
    model_df = model_df.dropna(subset=['accuracy', 'N_train', 'T', 'K', 'p'])
    return model_df


def preprocess_variables(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create and standardize derived variables. Skip columns with zero variance.
    """
    df = model_df.copy()
    zero_var_cols = []
    
    df['log_N'] = np.log(df['N_train'])
    df['log_T'] = np.log(df['T'])
    
    for col in ['p', 'log_N', 'log_T', 'K']:
        std_col = f'{col}_std'
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 1e-10:
            df[std_col] = (df[col] - mean_val) / std_val
        else:
            zero_var_cols.append(col)
            print(f"Skipping {col} - zero variance")
    
    if zero_var_cols:
        print(f"Excluded from analysis: {zero_var_cols}")
    
    return df


def select_strategy_subset_results(selected_type: str, selected_mode: str, 
                                   result_handler, config_handler) -> pd.DataFrame:
    """
    Load and prepare data for a specific strategy type and mode.
    """
    summary_df = result_handler.summary_df
    dataset_names = summary_df['dataset'].unique()
    datasets_path = config_handler.DATASETS_PATH
    
    analyse_df = summary_df[
        (summary_df["type"] == selected_type) & 
        (summary_df["mode"] == selected_mode)
    ]
    
    model_df = prepare_model_data(analyse_df, dataset_names, datasets_path)
    model_df = preprocess_variables(model_df)
    
    return model_df
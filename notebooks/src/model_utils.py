"""
Utility functions for statistical modeling of degradation effects.
"""

import pandas as pd
import numpy as np
import traceback
import warnings
import statsmodels.formula.api as smf
from typing import Dict, Any

def fit_ols_model(model_df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    """
    Fit OLS regression model.
    
    Parameters
    ----------
    model_df : pd.DataFrame
        Input dataframe with all required variables
    formula : str
        Model formula (e.g., "accuracy ~ p + log_N_std + K")
    
    Returns
    -------
    Dict containing model results, or None if fitting fails
    """
    model_df = model_df.dropna()
    
    try:
        import patsy
        y, X = patsy.dmatrices(formula, data=model_df, return_type='dataframe')
        
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            print(f"Warning: Design matrix rank deficiency (rank={rank}, cols={X.shape[1]})")
        
        model = smf.ols(formula, data=model_df).fit()
    except Exception as e:
        print(f"Model fitting failed for formula: {formula}")
        print(f"Data shape: {model_df.shape}, Missing: {model_df.isna().sum().sum()}")
        print(f"Exception: {e}")
        traceback.print_exc()
        return None
    
    results = {
        'model': model,
        'formula': formula,
        'n_obs': len(model_df),
        'method': 'OLS',
        'data': model_df
    }
    
    return results


def fit_mixed_effects_model(model_df: pd.DataFrame, formula: str, 
                            groups_col: str = "classifier_type",
                            re_formula: str = "~1") -> Dict[str, Any]:
    """
    Fit mixed-effects model with specified grouping variable.
    
    Parameters
    ----------
    model_df : pd.DataFrame
        Input dataframe
    formula : str
        Model formula
    groups_col : str
        Column to use for grouping (e.g., "classifier_type")
    re_formula : str
        Random effects formula
    
    Returns
    -------
    Dict containing model results, or None if fitting fails
    """
    model_df = model_df.dropna()
    
    try:
        if groups_col not in model_df.columns:
            raise ValueError(f"Grouping column '{groups_col}' not in dataframe")
        
        n_groups = model_df[groups_col].nunique()
        if n_groups < 2:
            raise ValueError(f"Only {n_groups} groups in '{groups_col}' - need at least 2")
        
        import patsy
        y, X = patsy.dmatrices(formula, data=model_df, return_type='dataframe')
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            print(f"Warning: Design matrix rank deficiency (rank={rank}, cols={X.shape[1]})")
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df[groups_col],
                re_formula=re_formula
            ).fit(method='powell')
    except Exception as e:
        print(f"Mixed effects model fitting failed")
        print(f"Formula: {formula}")
        print(f"Groups: {groups_col} (RE: {re_formula})")
        print(f"Data shape: {model_df.shape}, Missing: {model_df.isna().sum().sum()}")
        if groups_col in model_df.columns:
            print(f"Group distribution: {model_df[groups_col].nunique()} unique")
            print(f"{model_df[groups_col].value_counts().to_dict()}")
        print(f"Exception: {e}")
        traceback.print_exc()
        return None
    
    re_dict = model.random_effects
    re_df = pd.DataFrame(re_dict).T
    if len(re_df.columns) > 0:
        re_df.index.name = groups_col
    
    results = {
        'model': model,
        'formula': formula,
        'n_obs': len(model_df),
        'groups_col': groups_col,
        're_summary': re_df,
        'method': f'Mixed Effects (grouped by {groups_col})',
        'data': model_df
    }
    
    return results
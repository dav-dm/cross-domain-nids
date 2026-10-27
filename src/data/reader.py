import json
import numpy as np
import pandas as pd
from pathlib import Path
from filelock import FileLock
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data.dataset_config import dataset_config
from util.config import load_config


def get_data_labels(dataset, num_pkts, fields, is_flat, seed):
    """
    Preprocesses a dataset and returns the input features and labels.

    Args:
        dataset (str): The name of the dataset to be used.
        num_pkts (int): The number of packets to consider.
        fields (list): List of fields to include in the input features.
        is_flat (bool): If True, returns flattened input features (for ML models).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple (x, y) where x is the input features and y are the labels.
    """
    dataset_dict = dict()
    dc = dataset_config[dataset]
    full_path = dc['path']
    label_column = dc.get('label_column', 'LABEL')
    cf = load_config()
    
    scaler_fn = f'{cf["scaler"]}_' if 'symlog' in cf['scaler'] else ''
    p = Path(full_path)
    prep_df_path = p.parent / f'{p.stem}_{scaler_fn}{label_column.lower()}_prep{seed}{p.suffix}'
    
    # Lock
    lock = FileLock(str(prep_df_path) + '.lock', timeout=-1)
    with lock:
        if not prep_df_path.exists():
            # First time reading the dataset 
            print(f'Processing {dataset} dataframe...')
    
            df = pd.read_parquet(full_path)
            df = _preprocess_dataframe(df, label_column, p.parent, cf)
            df.to_parquet(prep_df_path)
        else:
            # Already pre-processed
            print(f'WARNING: using pre-processed dataframe for {dataset}.')
            df = pd.read_parquet(prep_df_path)
            
    # Compute PSQ input N_p x F
    data_series = df[[f'SCALED_{f}' for f in fields]].apply(
        lambda row: _process_row(row, num_pkts), axis=1
    )
    data = np.concatenate(data_series.tolist(), axis=0).astype(np.float32) 
    data = np.expand_dims(data, axis=1)
    
    dataset_dict['labels'] = np.array([label for label in df['ENC_LABEL']], dtype=np.int64)
    dataset_dict['quintuple'] = df.index.to_list()
    if is_flat:
        dataset_dict['data'] = np.array([np.ravel(a.T) for a in data])
        return dataset_dict
    dataset_dict['data'] = data
    return dataset_dict


def _preprocess_dataframe(df, label_column, parent_dir, config):
    """
    Preprocess the dataframe by performing label encoding, field padding, and scaling.
    """
    all_fields = config['all_fields']
    pad_value = config['pad_value']
    pad_value_dir = config['pad_value_dir']

    # Label encoding
    le = LabelEncoder()
    le.fit(df[label_column])
    df['ENC_LABEL'] = le.transform(df[label_column])
    
    # Save encoding informations
    label_conv = {
        str(k): int(v)
        for k, v in zip(le.classes_, le.transform(le.classes_))
    }
    with open(parent_dir / f'{label_column.lower()}_conv.json', 'w') as f:
        json.dump(label_conv, f)
        
    processed_fields = []
    for f in all_fields:
        
        if f not in df.columns:
            continue
        
        # Field padding
        pv = pad_value_dir if f == 'DIR' else pad_value
        df[f] = df[[f, 'FEAT_PAD']].apply(
            lambda x: np.concatenate((x[f], [pv] * x['FEAT_PAD'])), axis=1)

        # Field scaling
        if config['scaler'] == 'minmax_symlog':
            df[f'SCALED_{f}'] = df[f].apply(
                lambda x: _symlog_normalization(
                    x, config['symlog_params'][f], linear_fraction=config['linear_fraction']))
        else:
            scaler = MinMaxScaler((0, 1))
            scaler.fit(np.concatenate(df[f].values, axis=0).reshape(-1, 1))
            df[f'SCALED_{f}'] = df[f].apply(
                lambda x: scaler.transform(x.reshape(-1, 1)).reshape(-1))
        processed_fields.append(f)
        
    # Pick only preprocessed fields and encoded labels
    return df[[f'SCALED_{f}' for f in processed_fields] + ['ENC_LABEL']]


def _process_row(row, num_pkts):
    """
    Process a single row by slicing each field to num_pkts, stacking them,
    transposing the result, and adding an extra dimension.
    """
    # Convert each field's data to a numpy array and slice to num_pkts
    field_arrays = [np.array(f)[:num_pkts] for f in row]
    # Stack along a new axis to get shape (F, num_pkts)
    stacked = np.stack(field_arrays, axis=0)
    # Transpose to (num_pkts, F) and add a new axis at the beginning => (1, num_pkts, F)
    return np.expand_dims(stacked.T, axis=0)


def _symlog_normalization(x, p, linear_fraction=0.9):
    """
    Applies a symmetric logarithmic scaling to the input array based on the provided parameters.

    Args:
        x (np.ndarray): Input array to be scaled.
        p (tuple): Scaling parameters. For two elements (v_min, v_clip), linear scaling is applied.
                   For three elements (v_min, v_lin, v_clip), a combination of linear and logarithmic 
                   caling is applied.
        linear_fraction (float): Fraction of the range to be scaled linearly. Default is 0.9.
    Returns:
        np.ndarray: Scaled array with values normalized between 0 and 1.
    """
    x = np.asarray(x, dtype=np.float64)

    if len(p) == 2:
        v_min, v_clip = p
        x = np.clip(x, v_min, v_clip)
        return (x - v_min) / (v_clip - v_min)

    # len == 3
    v_min, v_lin, v_clip = p
    x = np.clip(x, v_min, v_clip)

    m_lin = x <= v_lin
    m_soft = ~m_lin

    out = np.empty_like(x, dtype=np.float64)
    out[m_lin] = (x[m_lin] - v_min) / (v_lin - v_min) * linear_fraction

    if m_soft.any():
        soft = np.log1p(x[m_soft] - v_lin)
        soft_max = np.log1p(v_clip - v_lin)
        out[m_soft] = linear_fraction + (1.0 - linear_fraction) * (soft / soft_max)

    return out

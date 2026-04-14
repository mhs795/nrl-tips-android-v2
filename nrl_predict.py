#!/usr/bin/env python3
"""
NRL Predict — lightweight inference for Android.
Loads .npz models and runs predictions without scikit-learn.
"""

import numpy as np
import os

def load_model(path):
    """Load an exported .npz model."""
    if not os.path.exists(path):
        return None
    # Load into memory buffer to bypass potential archive access issues
    # and to prevent "seek of closed file" errors when file is closed by 'with' block.
    import io
    with open(path, 'rb') as f:
        data = io.BytesIO(f.read())
    return np.load(data, allow_pickle=True)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _predict_tree(node_idx, features, children_left, children_right, feature, threshold, value):
    """Recursive tree traversal for HistGradientBoosting."""
    f_idx = feature[node_idx]
    if f_idx == -1:  # Leaf node
        return value[node_idx]
    
    if features[f_idx] <= threshold[node_idx]:
        return _predict_tree(children_left[node_idx], features, children_left, children_right, feature, threshold, value)
    else:
        return _predict_tree(children_right[node_idx], features, children_left, children_right, feature, threshold, value)

def predict_proba_home(model, features):
    """
    Run inference on feature data.
    features: numpy array, either 1D (single game) or 2D (multiple games).
    Returns probability of home win [0..1] as float or array.
    """
    is_1d = (features.ndim == 1)
    if is_1d:
        features = features.reshape(1, -1)
        
    n_samples = features.shape[0]
    n_folds = int(model['n_folds'][0])
    
    all_sample_probs = []
    
    for i in range(n_samples):
        row = features[i]
        fold_probs = []
        
        for fi in range(n_folds):
            n_trees = int(model[f'f{fi}_n_trees'][0])
            lr = float(model[f'f{fi}_lr'][0])
            raw_score = float(model[f'f{fi}_init'][0])
            
            for ti in range(n_trees):
                cl = model[f'f{fi}_t{ti}_cl']
                cr = model[f'f{fi}_t{ti}_cr']
                feat = model[f'f{fi}_feat']
                thr = model[f'f{fi}_thr']
                val = model[f'f{fi}_val']
                
                raw_score += lr * _predict_tree(0, row, cl, cr, feat, thr, val)
            
            # Apply isotonic calibration
            iso_x = model[f'f{fi}_iso_x']
            iso_y = model[f'f{fi}_iso_y']
            prob = np.interp(raw_score, iso_x, iso_y)
            fold_probs.append(prob)
            
        all_sample_probs.append(np.mean(fold_probs))
        
    res = np.array(all_sample_probs)
    return res[0] if is_1d else res

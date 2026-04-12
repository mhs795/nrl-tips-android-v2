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
    return np.load(path, allow_pickle=True)

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

def predict_proba_home(model, features_row):
    """
    Run inference on a single feature row (numpy array).
    Returns probability of home win [0..1].
    """
    n_folds = int(model['n_folds'][0])
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
            
            raw_score += lr * _predict_tree(0, features_row, cl, cr, feat, thr, val)
        
        # Apply isotonic calibration
        iso_x = model[f'f{fi}_iso_x']
        iso_y = model[f'f{fi}_iso_y']
        prob = np.interp(raw_score, iso_x, iso_y)
        fold_probs.append(prob)
        
    return np.mean(fold_probs)

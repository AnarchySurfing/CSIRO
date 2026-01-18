import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CFG


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]; yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores

def weighted_r2_score_global(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2

def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        print(f'{"Index":<6} | {"Image Path":<40} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}')
        print('-' * 90)
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        header = f'{"Index":<6} | {"Image Path":<40} | {"State":<6} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}'
        print(header)
        print('-' * len(header))
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            state = val_df.iloc[idx]['State'] if 'State' in val_df.columns else 'NA'
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {str(state):<6} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def compare_train_val(tr_df, val_df, targets, show_plots=True):
    """Quick comparison of target distributions and metadata between train and val splits."""
    print("\n--- Train / Val Comparison ---")

    for t in targets:
        tr = tr_df.get(t, pd.Series(dtype=float)).dropna()
        val = val_df.get(t, pd.Series(dtype=float)).dropna()
        print(f"\nTarget: {t}")
        print(f"  Train: n={len(tr)} mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
        print(f"  Val  : n={len(val)} mean={val.mean():.3f} std={val.std():.3f} min={val.min():.3f} max={val.max():.3f}")
        if show_plots:
            try:
                plt.figure(figsize=(6, 3))
                sns.kdeplot(tr, label='train', fill=True)
                sns.kdeplot(val, label='val', fill=True)
                plt.legend()
                plt.title(f'Distribution: {t}')
                plt.show()
            except Exception as e:
                print('  Could not plot distributions for', t, '-', e)

    # Compare Sampling_Date and State if present
    if 'Sampling_Date' in tr_df.columns:
        try:
            tr_dates = pd.to_datetime(tr_df['Sampling_Date'], errors='coerce')
            val_dates = pd.to_datetime(val_df['Sampling_Date'], errors='coerce')
            print("\nSampling_Date range:")
            print(f"  Train: {tr_dates.min()} -> {tr_dates.max()} (missing {tr_dates.isna().sum()})")
            print(f"  Val  : {val_dates.min()} -> {val_dates.max()} (missing {val_dates.isna().sum()})")
        except Exception as e:
            print('  Could not parse Sampling_Date:', e)
    if 'State' in tr_df.columns:
        print("\nState distribution (train vs val):")
        tr_state = tr_df['State'].value_counts(normalize=True)
        val_state = val_df['State'].value_counts(normalize=True)
        state_df = pd.concat([tr_state, val_state], axis=1, keys=['train', 'val']).fillna(0)

        print(state_df)
import gc
import numpy as np
import os
import pandas as pd

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from config import CFG
from dataset import BiomassDataset, get_train_transforms, get_tta_transforms
from model import BiomassModel
from train import train_epoch, valid_epoch_tta, build_optimizer, build_scheduler, set_backbone_requires_grad
from metric import weighted_r2_score_global,compare_train_val


print('Loading data...')
df_long = pd.read_csv(CFG.TRAIN_CSV)
df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
assert df_wide['image_path'].is_unique, 'Leakage risk: duplicate image_path rows'

# Merge metadata (Sampling_Date, State) for stratification
if 'Sampling_Date' in df_long.columns and 'State' in df_long.columns:
    print('Merging metadata for stratification...')
    meta_df = df_long[['image_path', 'Sampling_Date', 'State']].drop_duplicates()
    df_wide = df_wide.merge(meta_df, on='image_path', how='left')

# Keep necessary columns
df_wide = df_wide[['image_path', 'Sampling_Date', 'State'] + CFG.ALL_TARGET_COLS]
print(f'{len(df_wide)} training images')

# Use StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_true, oof_pred, fold_summary = [], [], []

# Split based on groups (Sampling_Date) and stratification target (State)
groups = df_wide['Sampling_Date']
y_stratify = df_wide['State']

# models_list = [] # Removed to save memory

for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_wide, y_stratify, groups=groups)):
    if fold not in CFG.FOLDS_TO_TRAIN:
        print(f'Skipping fold {fold} as per configuration.')
        continue
    print('\n' + '='*70)
    print(f'FOLD {fold+1}/{CFG.N_FOLDS} | {len(tr_idx)} train / {len(val_idx)} val')
    print('='*70)
    torch.cuda.empty_cache(); gc.collect()

    tr_df  = df_wide.iloc[tr_idx].reset_index(drop=True)
    val_df = df_wide.iloc[val_idx].reset_index(drop=True)

    # Quick train/val comparison for this fold
    try:
        compare_train_val(tr_df, val_df, CFG.ALL_TARGET_COLS, show_plots=True)
    except Exception as e:
        print('Warning: compare_train_val failed:', e)

    tr_set = BiomassDataset(tr_df,  get_train_transforms(), CFG.TRAIN_IMAGE_DIR)

    # Create TTA loaders
    val_loaders = []
    for mode in range(CFG.VAL_TTA_TIMES): # 0: orig, 1: hflip, 2: vflip, 3: rot90
        val_set_tta = BiomassDataset(val_df, get_tta_transforms(mode), CFG.TRAIN_IMAGE_DIR)
        val_loader_tta = DataLoader(val_set_tta, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
        val_loaders.append(val_loader_tta)

    tr_loader  = DataLoader(tr_set, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    print('Building model...')
    backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
    base_model = BiomassModel(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, backbone_path=backbone_path).to(CFG.DEVICE)

    # Load pretrained fold weights if available (for resuming or fine-tuning)
    if getattr(CFG, 'PRETRAINED_DIR', None) and os.path.isdir(CFG.PRETRAINED_DIR):
        pretrained_path = os.path.join(CFG.PRETRAINED_DIR, f'best_model_fold{fold}.pth')
        if os.path.exists(pretrained_path):
            try:
                state = torch.load(pretrained_path, map_location='cpu')
                # support raw state_dict or dict-with-keys
                if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
                    key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
                    sd = state[key]
                else:
                    sd = state
                base_model.load_state_dict(sd, strict=False)
                base_model.to(CFG.DEVICE)
                print(f'  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}')
            except Exception as e:
                print(f'  ✗ Failed to load pretrained fold {fold}: {e}')
        else:
            print(f'  (No pretrained file for fold {fold} at {pretrained_path})')
    else:
        print('  (No PRETRAINED_DIR configured or directory missing)')

    model = nn.DataParallel(base_model)
    set_backbone_requires_grad(base_model, False)
    optimizer = build_optimizer(base_model)
    scheduler = build_scheduler(optimizer)
    ema = ModelEmaV2(base_model, decay=CFG.EMA_DECAY)

    best_global_r2 = -np.inf
    patience = 0
    best_fold_preds = None; best_fold_true = None
    best_avg_r2 = -np.inf

    # Define save path
    save_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{fold}.pth')

    for epoch in range(1, CFG.EPOCHS + 1):
        if epoch == CFG.FREEZE_EPOCHS + 1:
            patience = 0
            set_backbone_requires_grad(base_model, True)
            print(f'Epoch {epoch}: backbone unfrozen')

        tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, CFG.DEVICE, ema)
        eval_model = ema.module if ema is not None else (model.module if hasattr(model, 'module') else model)

        # Use TTA validation
        val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(eval_model, val_loaders, CFG.DEVICE)

        per_r2_str = ' | '.join([f'{CFG.ALL_TARGET_COLS[i][:5]}: {r2:.3f}' for i, r2 in enumerate(per_r2)])
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f'Fold {fold} | Epoch {epoch:02d} | TLoss {tr_loss:.5f} | VLoss {val_loss:.5f} |avgR2 {avg_r2:.4f}| GlobalR² {global_r2:.4f} {"[BEST]" if global_r2 > best_global_r2 else ""}')
        print(f'  → {per_r2_str}')

        if global_r2 > best_global_r2:
            best_global_r2 = global_r2
            best_avg_r2 = avg_r2

            # Save the EMA weights (best state) to disk immediately
            # Clone to CPU to avoid memory issues
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f'  → SAVED EMA weights to {save_path} (GlobalR²: {best_global_r2:.4f})')
            del best_state # Free memory

            patience = 0
            best_fold_preds = preds_fold; best_fold_true = true_fold
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                    print(f'  → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')
                    break

        del preds_fold, true_fold
        torch.cuda.empty_cache()
        gc.collect()

    if best_fold_preds is not None:
        oof_true.append(best_fold_true); oof_pred.append(best_fold_preds)
        fold_summary.append({'fold': fold, 'global_r2': best_global_r2,'avg_r2':avg_r2})

    # Cleanup for this fold
    del model, base_model, tr_loader, val_loaders, optimizer, scheduler, ema
    if 'eval_model' in locals(): del eval_model
    torch.cuda.empty_cache(); gc.collect()

if oof_true:
    oof_true_arr = np.concatenate(oof_true, axis=0)
    oof_pred_arr = np.concatenate(oof_pred, axis=0)
    oof_global_r2, oof_avg_r2, oof_per_r2 = weighted_r2_score_global(oof_true_arr, oof_pred_arr)

    print('\nTraining complete! Models saved in:', CFG.MODEL_DIR)
    print('Fold summary:')
    for fs in fold_summary:
        print(f"  Fold {fs['fold']}: Global R² = {fs['global_r2']:.4f}, Avg R² = {fs.get('avg_r2', float('nan')):.4f}")
    print(f'OOF Global Weighted R²: {oof_global_r2:.4f} | OOF Avg Target R²: {oof_avg_r2:.4f}')
    print('OOF Per-target:', dict(zip(CFG.ALL_TARGET_COLS, [f"{r:.4f}" for r in oof_per_r2])))
else:
    print('No OOF predictions collected.')
# Config
import numpy as np
import os

import torch


class CFG:
    CREATE_SUBMISSION = True
    USE_TQDM        = False
    PRETRAINED_DIR  = None
    PRETRAINED      = True
    BASE_PATH       = '/kaggle/input/csiro-biomass'
    SEED            = 82947501
    FOLDS_TO_TRAIN   = [0,1,2,3,4]
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_IMAGE_DIR = '/kaggle/input/csiro-biomass/test'
    TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
    SUBMISSION_DIR  = '/kaggle/working/'
    MODEL_DIR_012       = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/1'
    MODEL_DIR_34       = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/2'
    N_FOLDS         = 5


    # MODEL_NAME      = 'vit_large_patch16_dinov3.lvd1689m'  
    # BACKBONE_PATH   = '/kaggle/input/vit-large-patch16-dinov3-lvd1689m-backbone-pth/vit_large_patch16_dinov3.lvd1689m_backbone.pth'
    MODEL_NAME      = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH   = '/kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth'
    
    IMG_SIZE        = 512

    VAL_TTA_TIMES   = 1
    TTA_STEPS       = 1
    
    
    BATCH_SIZE      = 1
    GRAD_ACC        = 4
    NUM_WORKERS     = 4
    EPOCHS          = 1
    FREEZE_EPOCHS   = 0
    WARMUP_EPOCHS   = 3
    LR_REST         = 1e-3
    LR_BACKBONE     = 5e-4
    WD              = 1e-2
    EMA_DECAY       = 0.9
    PATIENCE        = 5
    TARGET_COLS     = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS    = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    LOSS_WEIGHTS    = np.array([0.1, 0.1, 0.1, 0.0, 0.0])
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')
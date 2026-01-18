import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import CFG
from model import BiomassModel, biomass_loss
from metric import weighted_r2_score_global
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

@torch.no_grad()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()
    running = 0.0
    preds_total, preds_gdm, preds_green, preds_clover, preds_dead, all_labels = [], [], [], [], [], []
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    
    for l, r, lab in loader:
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)
        running += loss.item() * l.size(0)
        preds_total.extend(p_total.cpu().numpy().ravel())
        preds_gdm.extend(p_gdm.cpu().numpy().ravel())
        preds_green.extend(p_green.cpu().numpy().ravel())
        preds_clover.extend(p_clover.cpu().numpy().ravel())
        preds_dead.extend(p_dead.cpu().numpy().ravel())
        all_labels.extend(lab.cpu().numpy())
    
    pred_total  = np.array(preds_total)
    pred_gdm    = np.array(preds_gdm)
    pred_green  = np.array(preds_green)
    pred_clover = np.array(preds_clover)
    pred_dead   = np.array(preds_dead)
    true_labels = np.stack(all_labels)
    
    pred_all = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)
    return running / len(loader.dataset), global_r2, avg_r2, per_r2, pred_all, true_labels

@torch.no_grad()
def valid_epoch_tta(eval_model, loaders, device):
    eval_model.eval()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    
    # We need to aggregate predictions from all loaders
    # Assuming all loaders have same order and size (which they should if shuffle=False)
    
    all_preds_accum = None
    all_labels = None
    total_loss = 0.0
    
    for loader_idx, loader in enumerate(loaders):
        preds_total, preds_gdm, preds_green, preds_clover, preds_dead = [], [], [], [], []
        current_labels = []
        running_loss = 0.0
        
        for l, r, lab in loader:
            l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            with amp_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)
            
            running_loss += loss.item() * l.size(0)
            
            preds_total.extend(p_total.cpu().numpy().ravel())
            preds_gdm.extend(p_gdm.cpu().numpy().ravel())
            preds_green.extend(p_green.cpu().numpy().ravel())
            preds_clover.extend(p_clover.cpu().numpy().ravel())
            preds_dead.extend(p_dead.cpu().numpy().ravel())
            
            if loader_idx == 0:
                current_labels.extend(lab.cpu().numpy())
        
        total_loss += (running_loss / len(loader.dataset))
        
        # Stack predictions for this loader: (N, 5)
        # Order: Green, Dead, Clover, GDM, Total (matching CFG.ALL_TARGET_COLS order roughly, but let's be precise)
        # CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
        # preds lists are just raw outputs.
        # Let's stack them in the order expected by weighted_r2_score_global which expects:
        # y_true, y_pred where columns match.
        # The model returns: total, gdm, green, clover, dead
        # We need to stack them to match true_labels which comes from CFG.ALL_TARGET_COLS
        # CFG.ALL_TARGET_COLS is ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
        
        pred_stack = np.stack([
            np.array(preds_green),
            np.array(preds_dead),
            np.array(preds_clover),
            np.array(preds_gdm),
            np.array(preds_total)
        ], axis=1)
        
        if all_preds_accum is None:
            all_preds_accum = pred_stack
            all_labels = np.stack(current_labels)
        else:
            all_preds_accum += pred_stack
            
    # Average predictions
    avg_preds = all_preds_accum / len(loaders)
    avg_loss = total_loss / len(loaders)
    
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(all_labels, avg_preds)
    return avg_loss, global_r2, avg_r2, per_r2, avg_preds, all_labels

def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad

def build_optimizer(model: BiomassModel):
    head_params = (list(model.head_green_raw.parameters()) +
                   list(model.head_clover_raw.parameters()) +
                   list(model.head_dead_raw.parameters()))
    backbone_params = list(model.backbone.parameters())
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': head_params,     'lr': CFG.LR_HEAD,     'weight_decay': CFG.WD},
    ])
def build_optimizer(model: BiomassModel):
    # 1. Get backbone parameter IDs for exclusion
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    
    # 2. Separate params into backbone vs. everything else (heads, fusion, etc.)
    backbone_params = []
    rest_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                rest_params.append(p)
    
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': rest_params,     'lr': CFG.LR_REST,     'weight_decay': CFG.WD},
])
def build_scheduler(optimizer):
    def lr_lambda(epoch):
        e = max(0, epoch - 1)
        if e < CFG.WARMUP_EPOCHS:
            return float(e + 1) / float(max(1, CFG.WARMUP_EPOCHS))
        progress = (e - CFG.WARMUP_EPOCHS) / float(max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0
    opt.zero_grad()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if torch.cuda.is_available() else (lambda: nullcontext())
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader
    for i, (l, r, lab) in enumerate(itera):
        l, r, lab = l.to(device, non_blocking=True), r.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            total, gdm, green, clover, dead = model(l, r)
            loss = biomass_loss((total, gdm, green, clover, dead), lab, w=CFG.LOSS_WEIGHTS) / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        running += loss.item() * l.size(0) * CFG.GRAD_ACC
        
        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)
            opt.zero_grad()
    scheduler.step()
    return running / len(loader.dataset)
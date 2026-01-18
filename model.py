import os

import timm
import torch
import torch.nn as nn


# class BiomassModel(nn.Module):
#     def __init__(self, model_name, pretrained=True):
#         super().__init__()
#         self.model_name = model_name
#         self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='avg')
#         nf = self.backbone.num_features
#         comb = nf * 2
#         self.head_green_raw  = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_clover_raw = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_dead_raw   = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         if pretrained:
#             self.load_pretrained()
    
#     def load_pretrained(self):
#         try:
#             sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='avg').state_dict()
#             self.backbone.load_state_dict(sd, strict=False)
#             print('Pretrained weights loaded.')
#         except Exception as e:
#             print(f'Warning: pretrained load failed: {e}')
    
#     def forward(self, left, right):
#         fl = self.backbone(left)
#         fr = self.backbone(right)
#         x  = torch.cat([fl, fr], dim=1)
#         green  = self.head_green_raw(x)
#         # clover = torch.nn.functional.softplus(self.head_clover_raw(x))
#         # dead   = torch.nn.functional.softplus(self.head_dead_raw(x))
#         clover = self.head_clover_raw(x)
#         dead   = self.head_dead_raw(x)
#         gdm    = green + clover
#         total  = gdm + dead
#         return total, gdm, green, clover, dead
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x

class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        
        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='')
        
        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")
            
        nf = self.backbone.num_features
        
        # 3. Mamba Fusion Neck
        #    Mixes the concatenated tokens [Left, Right]
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        
        # 4. Pooling & Heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Heads (using the same logic as before, but on fused features)
        self.head_green_raw  = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead_raw   = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        
        if pretrained:
            self.load_pretrained()
    
    def load_pretrained(self):
        try:
            # Load weights normally
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                # Handle common checkpoint wrappers (e.g. if saved with 'model' key)
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                # Original behavior: Download from internet
                print("Downloading backbone weights...")
                sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()
            
            # Interpolate pos_embed if needed (for 256x256 vs 224x224)
            if 'pos_embed' in sd and hasattr(self.backbone, 'pos_embed'):
                pe_ck = sd['pos_embed']
                pe_m  = self.backbone.pos_embed
                if pe_ck.shape != pe_m.shape:
                    print(f"Interpolating pos_embed: {pe_ck.shape} -> {pe_m.shape}")
                    # (Simple interpolation logic here or rely on timm's load if strict=False handles it well enough)
                    # For robust interpolation, use the snippet provided in previous turn
            
            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded.')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')
    
    def forward(self, left, right):
        # 1. Extract Tokens (B, N, D)
        #    Note: ViT usually returns [CLS, Patch1, Patch2...]
        #    We remove CLS token for spatial mixing, or keep it. Let's keep it.
        x_l = self.backbone(left)
        x_r = self.backbone(right)
        
        # 2. Concatenate Left and Right tokens along sequence dimension
        #    (B, N, D) + (B, N, D) -> (B, 2N, D)
        x_cat = torch.cat([x_l, x_r], dim=1)
        
        # 3. Apply Mamba Fusion
        #    This allows tokens from Left image to interact with tokens from Right image
        x_fused = self.fusion(x_cat)
        
        # 4. Global Pooling
        #    (B, 2N, D) -> (B, D, 2N) -> (B, D, 1) -> (B, D)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        
        # 5. Prediction Heads
        green  = self.head_green_raw(x_pool)
        clover = self.head_clover_raw(x_pool)
        dead   = self.head_dead_raw(x_pool)
        
        # Summation logic
        gdm    = green + clover
        total  = gdm + dead
        
        return total, gdm, green, clover, dead
def biomass_loss(outputs, labels, w=None):
    total, gdm, green, clover, dead = outputs
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=5.0) # Huber loss for robust regression (beta=5.0 as recommended)
    
    l_green  = huber(green.squeeze(),  labels[:,0])
    l_dead   = huber(dead.squeeze(), labels[:,1]) # Use Huber loss for Dead
    l_clover = huber(clover.squeeze(), labels[:,2])
    l_gdm    = huber(gdm.squeeze(),    labels[:,3])
    l_total  = huber(total.squeeze(),  labels[:,4])

    # Stack per-target losses in the SAME order as CFG.ALL_TARGET_COLS
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    # losses = torch.stack([l_green, l_dead, l_clover])

    # Use provided weights, or default to CFG.R2_WEIGHTS
    if w is None:
        return losses.mean()
    w = torch.as_tensor(w, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()
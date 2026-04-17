#!/usr/bin/env python3
# 1. 这个必须排在第一行（除了上面的环境声明）
from __future__ import annotations

# 2. 然后设置环境变量，解决国内服务器连接 Hugging Face 的问题
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from timm import create_model 
from tqdm import tqdm

# 导入你项目中的数据集定义
from datasets.copper_xray_dataset import CopperXRayDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swin-T + brightness CE + thickness consistency (Corrected)")
    
    # 路径配置
    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    p.add_argument("--split-column", type=str, default="split")
    
    # 训练超参数
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224) 
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=True)
    
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--best-metric", type=str, default="macro_f1")

    # 物理启发扰动参数 [cite: 49, 136]
    p.add_argument("--brightness-delta-min", type=float, default=-0.15)
    p.add_argument("--brightness-delta-max", type=float, default=0.15)
    p.add_argument("--brightness-apply-p", type=float, default=0.5)
    p.add_argument("--thickness-delta-min", type=float, default=0.0)
    p.add_argument("--thickness-delta-max", type=float, default=0.15)
    p.add_argument("--consistency-weight", type=float, default=0.5)

    return p.parse_args()

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_swin_t_2ch(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """适配 Swin-T 以处理双能 X 射线双通道输入 [cite: 75]"""
    model = create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
    old_proj = model.patch_embed.proj
    model.patch_embed.proj = nn.Conv2d(
        2, old_proj.out_channels, 
        kernel_size=old_proj.kernel_size, 
        stride=old_proj.stride, 
        padding=old_proj.padding
    )
    with torch.no_grad():
        if pretrained:
            w_mean = old_proj.weight.mean(dim=1, keepdim=True)
            model.patch_embed.proj.weight.copy_(w_mean.repeat(1, 2, 1, 1))
    return model

def apply_brightness_batch(x, delta_min, delta_max, apply_p):
    x_aug = x.clone()
    B = x.shape[0]
    probs = torch.rand(B, device=x.device)
    deltas = torch.empty(B, device=x.device).uniform_(delta_min, delta_max)
    for i in range(B):
        if probs[i] < apply_p:
            x_aug[i] = torch.clamp(x_aug[i] * (1.0 + deltas[i]), 0.0, 1.0)
    return x_aug

def apply_thickness_batch(x, delta_min, delta_max, eps=1e-6):
    """在对数域构造厚度扰动以模拟输入分布偏移 [cite: 49, 136]"""
    deltas = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(delta_min, delta_max)
    x_safe = torch.clamp(x, eps, 1.0)
    x_shift = torch.exp(torch.log(x_safe) + deltas)
    return torch.clamp(x_shift, 0.0, 1.0)

@torch.no_grad()
def eval_metrics(model, loader, device, loss_fn):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        all_p.extend(logits.argmax(1).cpu().tolist())
        all_y.extend(y.cpu().tolist())
    return {
        "acc": float(accuracy_score(all_y, all_p)),
        "bal_acc": float(balanced_accuracy_score(all_y, all_p)),
        "macro_f1": float(f1_score(all_y, all_p, average="macro", zero_division=0)),
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds_train = CopperXRayDataset(args.csv_path, split_value="train", data_root=args.data_root, out_size=args.img_size, hflip_p=0.5)
    ds_val = CopperXRayDataset(args.csv_path, split_value="val", data_root=args.data_root, out_size=args.img_size)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.eval_batch_size, shuffle=False)

    print(f"[INFO] 正在加载 Swin-T (通过镜像站 hf-mirror.com)...")
    model = build_swin_t_2ch(num_classes=2, pretrained=True).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(str(args.log_dir))

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device)
            
            x_b = apply_brightness_batch(x, args.brightness_delta_min, args.brightness_delta_max, args.brightness_apply_p)
            x_t = apply_thickness_batch(x, args.thickness_delta_min, args.thickness_delta_max)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                z = model(x)
                z_b = model(x_b)
                z_t = model(x_t)
                
                # 联合分类损失与厚度一致性损失 [cite: 68]
                ce_loss = loss_fn(z, y) + loss_fn(z_b, y)
                cons_loss = F.mse_loss(z_t, z.detach()) # 干净分支 stop-grad [cite: 63, 64]
                loss = ce_loss + args.consistency_weight * cons_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val = eval_metrics(model, val_loader, device, loss_fn)
        score = val[args.best_metric]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.ckpt_path)
        
        print(f"\n[FINISH] Epoch {epoch}: Val F1={val['macro_f1']:.4f} | Best={best_score:.4f}")

    summary = {"best_val_f1": best_score, "model": "swin_tiny", "seed": args.seed}
    with open(args.log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    writer.close()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from datasets.copper_xray_dataset import CopperXRayDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            print("[WARN] 未安装 tensorboard，使用空 SummaryWriter。")
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet18 + brightness CE + thickness consistency")

    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    p.add_argument("--split-column", type=str, default="split")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=192)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--best-metric", type=str, default="macro_f1", choices=["macro_f1", "bal_acc", "acc"])

    p.add_argument("--hflip-p", type=float, default=0.5)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)

    # brightness branch
    p.add_argument("--brightness-delta-min", type=float, default=-0.15)
    p.add_argument("--brightness-delta-max", type=float, default=0.15)
    p.add_argument("--brightness-apply-p", type=float, default=0.5)

    # thickness consistency branch
    p.add_argument("--thickness-delta-min", type=float, default=0.0)
    p.add_argument("--thickness-delta-max", type=float, default=0.15)
    p.add_argument("--consistency-weight", type=float, default=0.5)

    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet18_2ch(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)

    old = m.conv1
    m.conv1 = nn.Conv2d(
        2, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if pretrained and old.weight.shape[1] == 3:
            w_mean = old.weight.mean(dim=1, keepdim=True)
            m.conv1.weight.copy_(w_mean.repeat(1, 2, 1, 1))
        else:
            nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        if m.conv1.bias is not None:
            nn.init.zeros_(m.conv1.bias)

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def make_ds(args: argparse.Namespace, split_value: str) -> CopperXRayDataset:
    # 这里 dataset 本身不开 brightness / thickness aug
    return CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value=split_value,
        data_root=args.data_root,
        out_size=args.img_size,
        hflip_p=(args.hflip_p if split_value == "train" else 0.0),
        thickness_aug=False,
        use_brightness_aug=False,
    )


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin = (args.device == "cuda" and torch.cuda.is_available())

    train_ds = make_ds(args, "train")
    val_ds = make_ds(args, "val")
    test_ds = make_ds(args, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def apply_brightness_batch(x: torch.Tensor, delta_min: float, delta_max: float, apply_p: float) -> torch.Tensor:
    # x: (B,C,H,W), [0,1]
    x_aug = x.clone()
    B = x.shape[0]
    probs = torch.rand(B, device=x.device)
    deltas = torch.empty(B, device=x.device).uniform_(delta_min, delta_max)
    for i in range(B):
        if probs[i] < apply_p:
            x_aug[i] = torch.clamp(x_aug[i] * (1.0 + deltas[i]), 0.0, 1.0)
    return x_aug


def apply_thickness_batch(x: torch.Tensor, delta_min: float, delta_max: float, eps: float = 1e-6) -> torch.Tensor:
    # log-domain attenuation perturbation
    deltas = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(delta_min, delta_max)
    x_safe = torch.clamp(x, eps, 1.0)
    x_shift = torch.exp(torch.log(x_safe) + deltas)
    x_shift = torch.clamp(x_shift, 0.0, 1.0)
    return x_shift


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    all_p: List[int] = []
    all_y: List[int] = []
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        p = logits.argmax(1)

        total_loss += float(loss.item()) * y.numel()
        n += y.numel()
        all_p.extend(p.cpu().tolist())
        all_y.extend(y.cpu().tolist())

    return {
        "loss": total_loss / max(n, 1),
        "acc": float(accuracy_score(all_y, all_p)),
        "bal_acc": float(balanced_accuracy_score(all_y, all_p)),
        "macro_f1": float(f1_score(all_y, all_p, average="macro", zero_division=0)),
        "precision": float(precision_score(all_y, all_p, average="macro", zero_division=0)),
        "recall": float(recall_score(all_y, all_p, average="macro", zero_division=0)),
        "pred0": int(Counter(all_p).get(0, 0)),
        "pred1": int(Counter(all_p).get(1, 0)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.log_dir))

    train_loader, val_loader, test_loader = make_loaders(args)

    model = build_resnet18_2ch(pretrained=args.pretrained).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    best_name = args.best_metric

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_ce_b = 0.0
        total_cons = 0.0
        corr = 0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device)

            x_b = apply_brightness_batch(
                x,
                delta_min=args.brightness_delta_min,
                delta_max=args.brightness_delta_max,
                apply_p=args.brightness_apply_p,
            )
            x_t = apply_thickness_batch(
                x,
                delta_min=args.thickness_delta_min,
                delta_max=args.thickness_delta_max,
            )

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(x)
                logits_b = model(x_b)
                logits_t = model(x_t)

                ce = loss_fn(logits, y)
                ce_b = loss_fn(logits_b, y)
                cons = F.mse_loss(logits_t, logits.detach())

                loss = ce + ce_b + args.consistency_weight * cons

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item()) * y.numel()
            total_ce += float(ce.item()) * y.numel()
            total_ce_b += float(ce_b.item()) * y.numel()
            total_cons += float(cons.item()) * y.numel()
            n += y.numel()
            corr += (logits.argmax(1) == y).sum().item()

        train_loss = total_loss / max(n, 1)
        train_ce = total_ce / max(n, 1)
        train_ce_b = total_ce_b / max(n, 1)
        train_cons = total_cons / max(n, 1)
        train_acc = corr / max(n, 1)

        val = eval_metrics(model, val_loader, device, loss_fn)
        score = float(val[best_name])
        if score > best:
            best = score
            torch.save(model.state_dict(), args.ckpt_path)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/ce_clean", train_ce, epoch)
        writer.add_scalar("train/ce_brightness", train_ce_b, epoch)
        writer.add_scalar("train/consistency", train_cons, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        for k in ["loss", "acc", "bal_acc", "macro_f1", "precision", "recall"]:
            writer.add_scalar(f"val/{k}", val[k], epoch)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} ce={train_ce:.4f} ce_b={train_ce_b:.4f} cons={train_cons:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val['acc']:.4f} val_bal_acc={val['bal_acc']:.4f} val_macro_f1={val['macro_f1']:.4f} "
            f"best_{best_name}={best:.4f}",
            flush=True,
        )

    print(f"[INFO] Loading best checkpoint: {args.ckpt_path}", flush=True)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    test = eval_metrics(model, test_loader, device, loss_fn)
    print("[TEST]", json.dumps(test, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "overall_test": test,
        "best_metric": best_name,
        "best_metric_value": float(best),
        "brightness_delta_min": float(args.brightness_delta_min),
        "brightness_delta_max": float(args.brightness_delta_max),
        "brightness_apply_p": float(args.brightness_apply_p),
        "thickness_delta_min": float(args.thickness_delta_min),
        "thickness_delta_max": float(args.thickness_delta_max),
        "consistency_weight": float(args.consistency_weight),
        "seed": int(args.seed),
    }

    out_json = args.log_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] saved to {out_json}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()
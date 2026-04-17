#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18

from datasets.copper_xray_dataset import CopperXRayDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate brightness-control model under thickness shift on test set")

    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    p.add_argument("--split-column", type=str, default="split")

    p.add_argument("--img-size", type=int, default=192)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--method-name", type=str, default="brightness_control")

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)

    p.add_argument("--delta-list", type=str, default="-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25")
    p.add_argument("--apply-p", type=float, default=1.0)

    p.add_argument("--out-dir", type=Path, required=True)

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


def make_test_ds(args: argparse.Namespace, fixed_delta: float) -> CopperXRayDataset:
    return CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value="test",
        data_root=args.data_root,
        out_size=args.img_size,
        hflip_p=0.0,
        thickness_aug=True,
        thickness_delta_min=fixed_delta,
        thickness_delta_max=fixed_delta,
        thickness_apply_p=args.apply_p,
        force_thickness_on_eval=True,
        use_brightness_aug=False,
    )


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

    acc = accuracy_score(all_y, all_p)
    bal = balanced_accuracy_score(all_y, all_p)
    mf1 = f1_score(all_y, all_p, average="macro", zero_division=0)
    pre = precision_score(all_y, all_p, average="macro", zero_division=0)
    rec = recall_score(all_y, all_p, average="macro", zero_division=0)
    dist = Counter(all_p)

    return {
        "loss": total_loss / max(n, 1),
        "acc": float(acc),
        "bal_acc": float(bal),
        "macro_f1": float(mf1),
        "precision": float(pre),
        "recall": float(rec),
        "pred0": int(dist.get(0, 0)),
        "pred1": int(dist.get(1, 0)),
    }


def _safe_group_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))

    uniq = set(y_true)
    if len(uniq) < 2:
        out["bal_acc"] = None
        out["macro_f1"] = None
        out["precision"] = None
        out["recall"] = None
        return out

    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return out


@torch.no_grad()
def group_report_test(args: argparse.Namespace, model: nn.Module, device: torch.device, ds: CopperXRayDataset) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_csv(args.csv_path)
    required = {"thickness_group", "sample_id", "label", args.split_column}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(), {}

    test_df = df[df[args.split_column] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame(), {}

    groups = sorted([str(g) for g in test_df["thickness_group"].dropna().unique().tolist()])
    if not groups:
        return pd.DataFrame(), {}

    id_to_idx = {str(ds.samples.iloc[i]["sample_id"]).strip(): i for i in range(len(ds.samples))}

    rows = []
    worst_acc = 1.0
    worst_group = None

    for g in groups:
        gdf = test_df[test_df["thickness_group"].astype(str) == g]
        label_counts = gdf["label"].value_counts().to_dict()
        label0_n = int(label_counts.get(0, 0))
        label1_n = int(label_counts.get(1, 0))

        sids = [str(x).strip() for x in gdf["sample_id"].tolist()]
        idxs = [id_to_idx[sid] for sid in sids if sid in id_to_idx]
        n = len(idxs)

        if n == 0:
            rows.append({
                "thickness_group": g, "n": 0, "label0_n": label0_n, "label1_n": label1_n,
                "acc": None, "bal_acc": None, "macro_f1": None, "precision": None, "recall": None,
                "pred0": 0, "pred1": 0, "note": "empty"
            })
            continue

        subset = Subset(ds, idxs)
        loader = DataLoader(subset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

        y_true: List[int] = []
        y_pred: List[int] = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = logits.argmax(1).cpu().tolist()
            y_pred.extend(p)
            y_true.extend(y.cpu().tolist())

        dist = Counter(y_pred)
        m = _safe_group_metrics(y_true, y_pred)
        note = "single-class" if (label0_n == 0 or label1_n == 0) else ""

        acc_v = m["acc"] if m["acc"] is not None else None
        if acc_v is not None and acc_v < worst_acc:
            worst_acc = acc_v
            worst_group = g

        rows.append({
            "thickness_group": g,
            "n": n,
            "label0_n": label0_n,
            "label1_n": label1_n,
            "acc": m["acc"],
            "bal_acc": m["bal_acc"],
            "macro_f1": m["macro_f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pred0": int(dist.get(0, 0)),
            "pred1": int(dist.get(1, 0)),
            "note": note,
        })

    out = pd.DataFrame(rows).sort_values("thickness_group").reset_index(drop=True)

    summary = {}
    if worst_group is not None:
        summary["worst_group_acc"] = float(worst_acc)
        summary["worst_group_name"] = str(worst_group)

    return out, summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18_2ch(pretrained=args.pretrained).to(device)
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}", flush=True)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    delta_list = [float(x.strip()) for x in args.delta_list.split(",") if x.strip()]
    summary_rows = []

    for delta in delta_list:
        print(f"\n[INFO] Evaluating delta={delta:.4f}", flush=True)

        ds = make_test_ds(args, fixed_delta=delta)
        loader = DataLoader(
            ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
        )

        test = eval_metrics(model, loader, device, loss_fn)
        gr, gr_summary = group_report_test(args, model, device, ds)

        delta_tag = f"{delta:+.2f}".replace("+", "p").replace("-", "m")
        gr_csv = args.out_dir / f"group_report_{args.method_name}_{delta_tag}.csv"
        js_path = args.out_dir / f"summary_{args.method_name}_{delta_tag}.json"

        if not gr.empty:
            gr.to_csv(gr_csv, index=False)

        payload = {
            "method": args.method_name,
            "delta": delta,
            "apply_p": args.apply_p,
            "overall_test": test,
            "worst_group_acc": gr_summary.get("worst_group_acc", None),
            "worst_group_name": gr_summary.get("worst_group_name", None),
        }
        js_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "method": args.method_name,
            "delta": delta,
            "apply_p": args.apply_p,
            "loss": test["loss"],
            "acc": test["acc"],
            "bal_acc": test["bal_acc"],
            "macro_f1": test["macro_f1"],
            "precision": test["precision"],
            "recall": test["recall"],
            "pred0": test["pred0"],
            "pred1": test["pred1"],
            "worst_group_acc": gr_summary.get("worst_group_acc", None),
            "worst_group_name": gr_summary.get("worst_group_name", None),
        }

        if not gr.empty:
            for _, r in gr.iterrows():
                g = str(r["thickness_group"])
                row[f"{g}_acc"] = r["acc"]
                row[f"{g}_bal_acc"] = r["bal_acc"]
                row[f"{g}_macro_f1"] = r["macro_f1"]

        summary_rows.append(row)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    out_csv = args.out_dir / f"shift_summary_{args.method_name}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from timm import create_model
from eval_thickness_shift import eval_metrics, group_report_test, make_test_ds
from torch.utils.data import DataLoader
import pandas as pd

def build_swin_t_2ch(num_classes=2):
    model = create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    old_proj = model.patch_embed.proj
    model.patch_embed.proj = nn.Conv2d(
        2, old_proj.out_channels, 
        kernel_size=old_proj.kernel_size, 
        stride=old_proj.stride, 
        padding=old_proj.padding
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    # 基础路径与模型参数
    parser.add_argument("--csv-path", type=Path, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--split-column", type=str, default="split") 
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    
    # 评估专用核心参数（补齐了 apply_p 等）
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--apply-p", type=float, default=1.0) # 扫描时必须 100% 应用偏移
    parser.add_argument("--delta-list", type=str, default="-0.25,-0.15,-0.05,0.0,0.05,0.15,0.25")
    
    # 运行环境参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_swin_t_2ch().to(device)
    print(f"[INFO] Loading Swin-T checkpoint: {args.ckpt_path}")
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    delta_list = [float(x) for x in args.delta_list.split(",")]
    summary_rows = []

    for delta in delta_list:
        print(f"Scanning delta={delta:+.2f}...")
        # 此时 args 已经包含了 make_test_ds 所需的所有属性
        ds = make_test_ds(args, fixed_delta=delta)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        
        test_res = eval_metrics(model, loader, device, loss_fn)
        gr_df, gr_sum = group_report_test(args, model, device, ds)

        row = {
            "method": args.method_name,
            "delta": delta,
            "acc": test_res["acc"],
            "macro_f1": test_res["macro_f1"],
            "worst_group_acc": gr_sum.get("worst_group_acc")
        }
        summary_rows.append(row)

    out_csv = args.out_dir / f"shift_summary_{args.method_name}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] 鲁棒性扫描完成！结果路径: {out_csv}")

if __name__ == "__main__":
    main()

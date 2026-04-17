# DE-XRT Thickness Consistency Regularization for Copper Ore Sorting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and data for the paper:

**"Enhancing Robustness of Copper Ore Sorting in DE-XRT Imaging Through Physically-Informed Thickness-Consistency Regularization"**  
*Zhu Zhi-yong, He Jian-feng, Wang Xue-yuan, Nie Feng-jun, Wang Wen, Zou Yang-hui, Li Wei-dong, Zhong Guo-yun, Ye Zhi-Xiang, Diao Fan*

## 📝 Overview

This work proposes a robust classification method to handle thickness variations in Dual-Energy X-ray Transmission (DE-XRT) copper ore sorting. The method:

- Models thickness variations as physically‑informed log‑domain additive shifts based on the Beer–Lambert law.
- Introduces a consistency regularization framework that enforces stable predictions between original and thickness‑perturbed samples.
- Achieves **92.18%** accuracy on the clean test set (comparable to the strong Brightness baseline at 92.16%).
- Under a large positive thickness shift ($\delta = +0.25$), improves Macro‑F1 by **7.76** percentage points and Worst‑group accuracy by **11.09** percentage points over Brightness.
- Generalizes across both CNN (ResNet18) and Transformer (Swin‑T) backbones.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NVIDIA GPU (tested on RTX 4090)

### Installation

Clone the repository:
```bash
git clone https://github.com/AI-Zhu001/DE-XRT-thickness-regularization.git
cd DE-XRT-thickness-regularization

Install dependencies:
pip install -r requirements.txt

📊 Dataset
The original dataset contains 7,245 dual‑energy X‑ray images (2,806 copper, 4,439 waste). Due to proprietary restrictions, the full dataset cannot be publicly released. A synthetic/miniature dataset is provided in data/ to demonstrate the data structure and allow code execution for testing.


Data Format
Organize your dataset as:
dataset_root/
├── mine/
│   ├── high/     # high‑energy images for copper ore
│   └── low/      # low‑energy images for copper ore
└── waste/
    ├── high/     # high‑energy images for waste rock
    └── low/      # low‑energy images for waste rock

Generate CSV splits and thickness group annotations:
python copper_dataset_split.py --root /path/to/dataset --output-dir split_outputs
This creates copper_xray_all_splits.csv with columns: sample_id, label, high_path, low_path, mean_gray, thickness_group, split (train/val/test), and cross‑split columns.

Data Format
Each sample is a 2‑channel tensor (high + low) resized to 192×192 (ResNet) or 224×224 (Swin).
Split ratio: 70% train / 15% val / 15% test.
Thickness groups (thin/medium/thick) are defined by sorting samples by mean grayscale.

🏃 How to Run
All ResNet experiments are run for 20 epochs with batch size 32, using AdamW optimizer (lr = 1e‑4, weight decay = 1e‑4).

Baseline (no augmentation)
python train_resnet_brightness_control.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --log-dir logs/baseline \
    --ckpt-path checkpoints/baseline.pth \
    --pretrained \
    --seed 0

Brightness augmentation (empirical baseline)
python train_resnet_brightness_control.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --brightness-aug \
    --brightness-delta-min -0.15 \
    --brightness-delta-max 0.15 \
    --brightness-apply-p 0.5 \
    --log-dir logs/brightness \
    --ckpt-path checkpoints/brightness.pth \
    --pretrained \
    --seed 0

Proposed thickness consistency regularization
python train_resnet_consistency.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --brightness-delta-min -0.15 \
    --brightness-delta-max 0.15 \
    --brightness-apply-p 0.5 \
    --thickness-delta-min 0.0 \
    --thickness-delta-max 0.15 \
    --consistency-weight 0.5 \
    --log-dir logs/consistency \
    --ckpt-path checkpoints/consistency.pth \
    --pretrained \
    --seed 0

Evaluation
Clean test set
python eval_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/consistency.pth \
    --method-name consistency \
    --out-dir eval_results \
    --img-size 192 \
    --delta-list 0.0 \
    --apply-p 1.0
Thickness shift robustness evaluation
Fixed shift intensities: δ ∈ {-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25}
python eval_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/consistency.pth \
    --method-name consistency \
    --out-dir eval_results \
    --img-size 192 \
    --delta-list -0.25,-0.15,-0.05,0.0,0.05,0.15,0.25 \
    --apply-p 1.0
Swin Transformer (Architectural Generalizability)
Train Swin‑T with the proposed consistency regularization (learning rate 1e-4, consistent with the paper):
python train_swin_consistency.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 224 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --consistency-weight 0.5 \
    --log-dir logs/swin_consistency \
    --ckpt-path checkpoints/swin_consistency.pth \
    --seed 0
Evaluate Swin‑T under thickness shifts:
python eval_swin_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/swin_consistency.pth \
    --method-name swin_consistency \
    --out-dir eval_results \
    --img-size 224 \
    --delta-list -0.25,-0.15,-0.05,0.0,0.05,0.15,0.25
For multi‑seed experiments (5 seeds), you can use the provided batch script:
bash run_swin_5seeds.sh
Batch evaluation across all shifts:
python evaluate.py --model_path checkpoints/consistency_best.pth --shift_list -0.25 -0.15 -0.05 0 0.05 0.15 0.25
Evaluation metrics reported: Accuracy, Balanced Accuracy, macro-averaged Precision, macro-averaged Recall, Macro-F1, and Worst-group accuracy (lowest accuracy among thin/medium/thick groups).

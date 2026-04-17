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
```

Install dependencies:
```bash
pip install -r requirements.txt
```

##  📊 Dataset
The original dataset contains 7,245 dual‑energy X‑ray images (2,806 copper, 4,439 waste). Due to proprietary restrictions, the full dataset cannot be publicly released. A synthetic/miniature dataset is provided in data/ to demonstrate the data structure and allow code execution for testing.

### Data Preparation
Organize your dataset as:

```text
dataset_root/
├── mine/
│   ├── high/
│   └── low/
└── waste/
    ├── high/
    └── low/
```
Generate CSV splits and thickness group annotations:

```bash
python copper_dataset_split.py --root /path/to/dataset --output-dir split_outputs
```
This creates copper_xray_all_splits.csv with columns: sample_id, label, high_path, low_path, mean_gray, thickness_group, split (train/val/test), and cross‑split columns.

### Data Format
Each sample is a 2‑channel tensor (high + low) resized to 192×192 (ResNet) or 224×224 (Swin).
Split ratio: 70% train / 15% val / 15% test.
Thickness groups (thin/medium/thick) are defined by sorting samples by mean grayscale.

##  🏃 How to Run
All ResNet experiments are run for 20 epochs with batch size 32, using AdamW optimizer (lr = 1e‑4, weight decay = 1e‑4).

1. Baseline (no augmentation)
```bash
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
 ```      
2. Brightness Augmentation (empirical baseline)
```bash
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
```       
3. Proposed Thickness Consistency Regularization
```bash
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
```     
Multi‑seed experiments: In our paper, each method was trained with 5 different random seeds. The command above shows an example with --seed 0. To reproduce the full results, run the same command with multiple distinct seeds (you may choose any set of 5 seeds) and average the results. For Swin‑Transformer, a batch script run_swin_5seeds.sh is provided.

4. Evaluation
Clean test set
```bash
python eval_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/consistency.pth \
    --method-name consistency \
    --out-dir eval_results \
    --img-size 192 \
    --delta-list 0.0 \
    --apply-p 1.0
``` 
Thickness shift robustness scan
Fixed shift intensities: δ ∈ {‑0.25, ‑0.15, ‑0.05, 0, 0.05, 0.15, 0.25}
```bash
python eval_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/consistency.pth \
    --method-name consistency \
    --out-dir eval_results \
    --img-size 192 \
    --delta-list -0.25,-0.15,-0.05,0.0,0.05,0.15,0.25 \
    --apply-p 1.0
```   
5. Swin Transformer (Architectural Generalizability)
Train Swin‑T with the proposed consistency regularization (learning rate 1e-4, consistent with the paper):
```bash
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
 ```     
Evaluate Swin‑T under thickness shifts:
```bash
python eval_swin_thickness_shift.py \
    --csv-path split_outputs/copper_xray_all_splits.csv \
    --data-root /path/to/dataset \
    --ckpt-path checkpoints/swin_consistency.pth \
    --method-name swin_consistency \
    --out-dir eval_results \
    --img-size 224 \
    --delta-list -0.25,-0.15,-0.05,0.0,0.05,0.15,0.25
 ```

For multi‑seed experiments (5 seeds), you can use the provided batch script:
```bash
bash run_swin_5seeds.sh
```

## 📁 Repository Structure
```text
├── copper_dataset_split.py # CSV generation with thickness groups
├── copper_xray_dataset.py # PyTorch Dataset with thickness/brightness augmentations
├── train_resnet_brightness_control.py # Baseline + brightness augmentation
├── train_resnet_consistency.py # Proposed thickness consistency regularization (ResNet18)
├── train_swin_consistency.py # Swin‑Transformer + consistency (generalizability)
├── eval_thickness_shift.py # Evaluate ResNet18 under thickness shifts
├── eval_swin_thickness_shift.py # Evaluate Swin‑T under thickness shifts
├── eval_brightness_shift.py # Evaluate brightness‑control models (optional)
├── final_collect_plots.py # Generate summary tables and plots
├── plot_shift_summary_auto.py # Plot Macro‑F1 and worst‑group accuracy curves
├── run_swin_5seeds.sh # Batch training script for Swin (5 seeds)
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # This file
```
> **Note**: Files such as `train_resnet_thickness.py` and `train_resnet_hybrid_curriculum.py` were used in preliminary experiments and are not required for reproducing the final results reported in the paper.

## 📜 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{Zhu2026Thickness,
  title={Enhancing Robustness of Copper Ore Sorting in DE-XRT Imaging Through Physically-Informed Thickness-Consistency Regularization},
  author={Zhu, Zhi-yong and He, Jian-feng and Wang, Xue-yuan and Nie, Feng-jun and Wang, Wen and Zou, Yang-hui and Li, Wei-dong and Zhong, Guo-yun and Ye, Zhi-Xiang and Diao, Fan},
  journal={Computers \& Geosciences},
  year={2026}
}
```

## Contact

For questions or issues, please contact the corresponding author:  
**Jianfeng He** – hjf_10@yeah.net

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


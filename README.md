# DE-XRT Thickness Consistency Regularization for Copper Ore Sorting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official source code and data for the paper:

**"Enhancing Robustness of Copper Ore Sorting in DE-XRT Imaging Through Physically-Informed Thickness-Consistency Regularization"**  
*Zhu Zhi-yong, He Jian-feng, Wang Xue-yuan, Nie Feng-jun, Wang Wen, Zou Yang-hui, Li Wei-dong, Zhong Guo-yun, Ye Zhi-Xiang, Diao Fan*

## 📝 Overview

This work proposes a robust classification method to handle thickness variations in Dual-Energy X-ray Transmission (DE-XRT) copper ore sorting. The method:

- Models thickness variations as physically-informed log-domain additive shifts based on the Beer–Lambert law.
- Introduces a consistency regularization framework that enforces stable predictions between original and thickness-perturbed samples.
- Achieves **92.18%** accuracy on the clean test set (comparable to the strong Brightness baseline at 92.16%).
- Under a large positive thickness shift ($\delta = +0.25$), improves Macro-F1 by **7.76** percentage points and Worst-group accuracy by **11.09** percentage points over Brightness.
- Generalizes across both CNN (ResNet18) and Transformer (Swin-T) backbones.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NVIDIA GPU (recommended; tested on RTX 4090)

### Installation

Clone the repository:
```bash
git clone https://github.com/AI-Zhu001/DE-XRT-thickness-regularization.git
cd DE-XRT-thickness-regularization

Install dependencies:
pip install -r requirements.txt

📊 Dataset
The original dataset contains 7,245 dual-energy X-ray images of copper ore and waste rock (2,806 copper, 4,439 waste). Due to proprietary restrictions, the full dataset cannot be publicly released.

A synthetic/miniature dataset is provided in data/ to demonstrate the data structure and allow code execution for testing.

Data Format
Each sample consists of two channels (high-energy and low-energy) resized to 192×192 pixels. The dataset is split into training/validation/test sets at a ratio of 7:1.5:1.5 (5,072 / 1,087 / 1,086 samples). Class proportions are preserved via stratified sampling.

Thickness grouping (thin/medium/thick) is based on the average grayscale value of the dual-channel images, serving as a thickness proxy.

🏃 How to Run
All experiments are run for 20 epochs with a batch size of 32, using the AdamW optimizer (learning rate = 1e-4, weight decay = 1e-4). The input size is 192×192.

Baseline (no augmentation)
python train.py --method baseline --epochs 20 --batch_size 32

Brightness augmentation (empirical baseline)
Random brightness scaling factor sampled from [-0.15, 0.15] with 0.5 probability.
python train.py --method brightness --epochs 20 --batch_size 32

Proposed thickness consistency regularization
Applies log-domain thickness perturbations (δ sampled from [0.0, 0.15]) and uses consistency loss (MSE with stop-gradient, weight λ=0.5).
python train.py --method consistency --epochs 20 --batch_size 32
Model selection is based on Macro-F1 on the validation set. Early stopping if no improvement for 5 consecutive epochs.

Evaluation
Clean test set
python evaluate.py --model_path checkpoints/consistency_best.pth --shift 0

Thickness shift robustness evaluation
Fixed shift intensities: δ ∈ {-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25}
python evaluate.py --model_path checkpoints/consistency_best.pth --shift 0.25

Batch evaluation across all shifts:
python evaluate.py --model_path checkpoints/consistency_best.pth --shift_list -0.25 -0.15 -0.05 0 0.05 0.15 0.25
Evaluation metrics reported: Accuracy, Balanced Accuracy, macro-averaged Precision, macro-averaged Recall, Macro-F1, and Worst-group accuracy (lowest accuracy among thin/medium/thick groups).

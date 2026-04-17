#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def _resolve_path(path_str: str, data_root: Optional[str] = None) -> Path:
    p = Path(path_str)
    if p.exists():
        return p

    if data_root is not None:
        root = Path(data_root)
        # 兼容 CSV 里可能是 Windows 风格路径
        path_norm = str(path_str).replace("\\", "/")
        filename = Path(path_norm).name

        # 常见情况 1：CSV 中是绝对 Windows 路径，直接取文件名拼到 root 下
        cand = root / filename
        if cand.exists():
            return cand

        # 常见情况 2：保留最后几级目录
        parts = [x for x in Path(path_norm).parts if x not in ("/", "\\")]
        for k in range(min(5, len(parts)), 0, -1):
            cand = root.joinpath(*parts[-k:])
            if cand.exists():
                return cand

    raise FileNotFoundError(f"图像不存在，且无法通过 data_root 解析：{path_str}")


def _load_single_channel_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 2:
        ch = arr
    else:
        # 保持和你原始 dataset 一致：取第 0 通道
        ch = arr[:, :, 0]

    ch = ch / 255.0
    ch = np.clip(ch, 0.0, 1.0).astype(np.float32)
    return ch


def _resize_tensor_img(img: torch.Tensor, out_size: int) -> torch.Tensor:
    # img: (C,H,W)
    if img.shape[-2] == out_size and img.shape[-1] == out_size:
        return img
    img = img.unsqueeze(0)  # (1,C,H,W)
    img = F.interpolate(img, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return img.squeeze(0)


def apply_thickness_aug(
    x: np.ndarray,
    delta: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    log 域衰减模拟：
    输入 x ∈ [0,1]，值越大表示越亮/透过率越高。
    用 I' = exp(log(I) + delta) = I * exp(delta)
    这里保持和你的论文主线一致：在 log 域做厚度相关扰动。
    """
    x = np.clip(x, eps, 1.0)
    log_x = np.log(x)
    log_x = log_x + delta
    x_aug = np.exp(log_x)
    x_aug = np.clip(x_aug, 0.0, 1.0).astype(np.float32)
    return x_aug


def apply_brightness_aug(x: np.ndarray, delta: float) -> np.ndarray:
    """
    普通增强对照：图像域 brightness scaling
    x: (C,H,W), [0,1]
    delta in [-a, a]
    """
    scale = 1.0 + delta
    x_aug = x * scale
    x_aug = np.clip(x_aug, 0.0, 1.0).astype(np.float32)
    return x_aug


class CopperXRayDataset(Dataset):
    """
    兼容 train_resnet_thickness.py / eval_thickness_shift.py 的版本

    输出:
        img_tensor: (2, out_size, out_size), float32, [0,1]
        label_tensor: long
    """

    def __init__(
        self,
        csv_path: str | Path,
        split_column: str = "split",
        split_value: str = "train",
        data_root: Optional[str] = None,
        out_size: int = 192,
        hflip_p: float = 0.0,
        thickness_aug: bool = False,
        thickness_delta_min: float = -0.25,
        thickness_delta_max: float = 0.25,
        thickness_apply_p: float = 0.8,
        force_thickness_on_eval: bool = False,
        thickness_enabled: bool = True,
        use_brightness_aug: bool = False,
        brightness_delta_min: float = -0.15,
        brightness_delta_max: float = 0.15,
        brightness_apply_p: float = 0.5,
        transform=None,
        target_transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 不存在：{self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if split_column not in df.columns:
            raise ValueError(f"CSV 中缺少列：{split_column}")

        mask = df[split_column] == split_value
        self.samples = df.loc[mask].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"列 {split_column} 下没有值为 {split_value} 的样本。")

        self.split_column = split_column
        self.split_value = split_value
        self.is_train = (split_value == "train")

        self.data_root = data_root
        self.out_size = out_size
        self.hflip_p = hflip_p

        self.thickness_aug = thickness_aug
        self.thickness_delta_min = thickness_delta_min
        self.thickness_delta_max = thickness_delta_max
        self.thickness_apply_p = thickness_apply_p
        self.force_thickness_on_eval = force_thickness_on_eval
        self.thickness_enabled = thickness_enabled

        self.use_brightness_aug = use_brightness_aug
        self.brightness_delta_min = brightness_delta_min
        self.brightness_delta_max = brightness_delta_max
        self.brightness_apply_p = brightness_apply_p

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.samples.iloc[idx]

        high_path = _resolve_path(row["high_path"], self.data_root)
        low_path = _resolve_path(row["low_path"], self.data_root)

        high = _load_single_channel_image(high_path)
        low = _load_single_channel_image(low_path)

        x = np.stack([high, low], axis=0).astype(np.float32)  # (2,H,W)

        # hflip
        if self.hflip_p > 0 and np.random.rand() < self.hflip_p:
            x = np.flip(x, axis=2).copy()

        # thickness aug
        use_thickness_now = (
            self.thickness_enabled and
            self.thickness_aug and (
                self.is_train or self.force_thickness_on_eval
            )
        )
        if use_thickness_now and np.random.rand() < self.thickness_apply_p:
            delta = np.random.uniform(self.thickness_delta_min, self.thickness_delta_max)
            x = apply_thickness_aug(x, delta)

        # ordinary augmentation control: brightness
        if self.use_brightness_aug and self.is_train and np.random.rand() < self.brightness_apply_p:
            delta = np.random.uniform(self.brightness_delta_min, self.brightness_delta_max)
            x = apply_brightness_aug(x, delta)

        img_tensor = torch.from_numpy(x).float()
        img_tensor = _resize_tensor_img(img_tensor, self.out_size)

        label = int(row["label"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)

        return img_tensor, label_tensor


def create_dataloader(
    csv_path: str | Path,
    split_column: str,
    split_value: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    data_root: Optional[str] = None,
    out_size: int = 192,
) -> DataLoader:
    dataset = CopperXRayDataset(
        csv_path=csv_path,
        split_column=split_column,
        split_value=split_value,
        data_root=data_root,
        out_size=out_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
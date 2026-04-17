#!/bin/bash

# 统一参数：20轮训练，1e-4学习率
LR=1e-4
EPOCHS=20
WEIGHT=0.5
IMG_SIZE=224

for SEED in 0 1 2 3 4
do
    echo "============================================"
    echo "正在开始训练 Swin-T (20 Epochs): Seed $SEED"
    echo "============================================"
    
    # 1. 训练
    python train_swin_consistency.py \
        --log-dir /root/autodl-tmp/paper_runs_swin/consistency_seed$SEED \
        --ckpt-path /root/autodl-tmp/paper_runs_swin/consistency_seed$SEED/best_swin_consistency.pth \
        --lr $LR \
        --epochs $EPOCHS \
        --consistency-weight $WEIGHT \
        --img-size $IMG_SIZE \
        --seed $SEED

    # 2. 评估（厚度偏移扫描）
    echo "正在评估 Swin-T: Seed $SEED"
    python eval_swin_thickness_shift.py \
        --ckpt-path /root/autodl-tmp/paper_runs_swin/consistency_seed$SEED/best_swin_consistency.pth \
        --method-name swin_consistency_seed$SEED \
        --out-dir /root/autodl-tmp/paper_runs_swin/eval_results \
        --img-size $IMG_SIZE
done

echo "所有 5-seed 实验（20 Epochs/轮）已完成！"

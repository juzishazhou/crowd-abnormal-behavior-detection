#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""YOLOv8 model evaluation — PR curves, F1 curves, confusion matrix, comparison table.

Usage:
    python tools/run_eval.py --weights <model.pt> --data <dataset.yaml>

Note: This is an advanced tool that requires a YOLO-format dataset and a trained model.
      The WiderPerson dataset is NOT included in this repository.
      See README for dataset preparation instructions.
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def pct(new, base):
    """Percent change, e.g. +12.3% or -5.1%."""
    if base == 0:
        return "+inf" if new > 0 else "0.0%"
    change = (new - base) / base * 100
    return f"{change:+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 model evaluation")
    parser.add_argument("--weights", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="数据集 YAML 配置文件路径")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (默认: 0 即 CUDA GPU 0; 用 'cpu' 切换 CPU)")
    parser.add_argument("--project", type=str, default="runs/eval",
                        help="评估输出目录 (默认: runs/eval)")
    parser.add_argument("--name", type=str, default="eval",
                        help="评估运行名称 (默认: eval)")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if not os.path.exists(args.weights):
        print(f"[X] 权重文件不存在: {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[X] 数据集配置不存在: {args.data}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {args.weights}")
    print(f"  Data:       {args.data}")
    print(f"  Device:     {args.device}")
    print(f"{'=' * 60}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        exist_ok=True,
    )

    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print(f"\n  Overall Results:")
    print(f"    mAP50:     {map50:.4f}")
    print(f"    mAP50-95:  {map50_95:.4f}")
    print(f"    Precision: {mp:.4f}")
    print(f"    Recall:    {mr:.4f}")

    # Per-class breakdown
    if len(metrics.box.ap_class_index) >= 1:
        print(f"\n  Per-class results:")
        for idx, cls_id in enumerate(metrics.box.ap_class_index):
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            print(f"    {cls_name:12s}  P={metrics.box.p[idx]:.4f}  R={metrics.box.r[idx]:.4f}  "
                  f"ap50={metrics.box.ap50[idx]:.4f}  ap={metrics.box.ap[idx]:.4f}")

    print(f"\n  Output: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()

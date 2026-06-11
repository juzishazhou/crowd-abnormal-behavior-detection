# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
图4.2 摔倒过程中宽高比变化曲线
论文：密集人群背景下人体异常行为检测算法设计
对应章节：4.5.3 跌倒检测 - 条件二（宽高比突变 η_r > 2.0）
输出：assets/fig4_2_aspect_ratio.png  (300 DPI, 可直接插入 Word)

Usage:
    python scripts/plot_fig4_2.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# ===== 输出路径 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "assets" / "fig4_2_aspect_ratio.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ===== 中文显示设置（Windows 用 SimHei；Mac 可改为 'Arial Unicode MS'）=====
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False   # 正常显示负号

# ===== 构造宽高比序列（按论文设定的三阶段）=====
# 直立行走 r≈0.3-0.5；失衡跌倒骤升越过 2.0；触地静止 r≈1.8-2.2
frames = np.arange(0, 61)

def aspect_ratio(f):
    if f <= 22:                       # 直立行走
        return 0.40 + 0.04 * np.sin(f * 0.6)
    elif f <= 38:                     # 失衡跌倒（平滑骤升）
        t = (f - 22) / 16
        e = t * t * (3 - 2 * t)       # smoothstep
        return 0.42 + (2.05 - 0.42) * e
    else:                             # 触地静止
        return 2.05 + 0.12 * np.sin((f - 38) * 0.5) - 0.04 * (f - 38) / 22

r = np.array([aspect_ratio(f) for f in frames])
conf_idx = 38                          # 阈值突破 / 跌倒确认帧

# ===== 绘图 =====
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)

# 正常行走基线带 0.3-0.5
ax.axhspan(0.3, 0.5, color='#40B43E', alpha=0.12, zorder=0)

# 突变阈值线 η_r > 2.0
ax.axhline(2.0, color='#E65C53', ls='--', lw=1.4, zorder=2)

# 宽高比主曲线
ax.plot(frames, r, color='#F3A04C', lw=2.4, zorder=3)
ax.fill_between(frames, r, color='#F3A04C', alpha=0.10, zorder=1)

# 跌倒确认点
ax.scatter([conf_idx], [r[conf_idx]], s=70, color='#E65C53',
           zorder=5, edgecolors='white', linewidths=1.2)

# ===== 标注文字 =====
ax.text(11, 0.78, '直立行走', ha='center', fontsize=11, color='0.35')
ax.text(30, 1.35, '失衡跌倒', ha='center', fontsize=11, color='0.35')
ax.text(50, 2.42, '触地静止', ha='center', fontsize=11, color='0.35')
ax.text(40, 1.78, r'$\eta_r>2.0$ 触发', fontsize=11, color='#E65C53', weight='bold')
ax.text(60, 2.06, '阈值 2.0', ha='right', va='bottom', fontsize=10, color='#E65C53')
ax.text(0.5, 0.31, '正常行走基线 0.3–0.5', fontsize=9.5, color='#2f8a2e')

# ===== 坐标轴 =====
ax.set_xlabel('帧序号 (frame)', fontsize=12)
ax.set_ylabel('宽高比 r = w / h', fontsize=12)
ax.set_xlim(0, 60)
ax.set_ylim(0, 2.7)
ax.set_yticks(np.arange(0, 2.8, 0.5))
ax.grid(True, color='0.85', lw=0.6)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(str(OUTPUT_PATH), dpi=300, bbox_inches='tight')
print(f'已保存 {OUTPUT_PATH}（300 DPI）')
plt.show()

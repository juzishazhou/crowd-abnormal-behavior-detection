# 密集人群背景下人体异常行为检测算法

> Crowd Abnormal Behavior Detection Based on Ultralytics YOLO

基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的毕业设计项目，支持三类人体异常行为检测：

1. **跌倒检测** — 姿态估计 + ByteTrack 跟踪 + 轨迹分析 + 时序平滑
2. **奔跑检测** — 目标检测 + ByteTrack 跟踪 + 像素速度阈值
3. **禁入区域入侵检测** — 多边形区域配置 + 脚底中心点/轨迹判定

---

## 效果演示

### 跌倒检测
![fall detection demo](assets/demo/fall_demo.gif)

### 奔跑检测
![running detection demo](assets/demo/running_demo.gif)

### 禁入区域检测
![intrusion detection demo](assets/demo/intrusion_demo.gif)

---

## 项目结构

```
crowd-abnormal-behavior-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
├── scripts/                         # 主检测脚本
│   ├── fall_detection.py            # 跌倒检测
│   ├── running_detection.py         # 奔跑检测
│   └── intrusion_detection.py       # 禁入区域入侵检测
├── configs/                         # 配置文件
│   ├── bytetrack_fall.yaml          # ByteTrack 跌倒检测配置
│   ├── bytetrack.yaml               # ByteTrack 奔跑检测配置
│   └── forbidden_zone.example.json  # 禁入区域示例配置
├── tools/                           # 辅助工具
│   ├── check_env.py                 # 环境检查
│   ├── zone_selector.py             # 禁入区域交互式标定
│   ├── export_alerts.py             # 告警导出
│   ├── frame_viewer.py              # 视频帧查看器
│   ├── measure_fps.py               # FPS 测量
│   ├── measure_pipeline.py          # 流水线延迟测量
│   ├── measure_tracking.py          # 跟踪性能测量
│   └── run_eval.py                  # 模型评估
├── docs/                            # 文档
│   └── USAGE.md                     # 使用说明
├── assets/                          # 资源文件
│   ├── fall.mp4                     # 跌倒检测测试视频
│   ├── running.mp4                  # 奔跑检测测试视频
│   ├── intrusion.mp4                # 禁入区域测试视频
│   └── demo/                        # 效果演示 GIF
│       ├── fall_demo.gif
│       ├── running_demo.gif
│       └── intrusion_demo.gif
├── outputs/                         # 本地运行结果（不入库）
└── tests/                           # 测试
    └── test_imports.py
```

---

## 环境安装

**Python 版本**: 建议 Python 3.8+

```bash
# 克隆仓库
git clone https://github.com/juzishazhou/crowd-abnormal-behavior-detection.git
cd crowd-abnormal-behavior-detection

# 安装依赖
pip install -r requirements.txt
```

### 依赖清单

| 包 | 用途 |
|---|---|
| `ultralytics>=8.4.0` | YOLO 推理 + 跟踪 |
| `opencv-python>=4.6.0` | 视频读写 + 可视化 |
| `numpy>=1.23.0` | 数组运算 |
| `torch>=1.8.0` | PyTorch 后端 |
| `shapely>=2.0.0` | 多边形精确判断（可选，禁入区域检测有 fallback） |
| `psutil>=5.8.0` | 系统信息检查 |
| `matplotlib>=3.3.0` | 论文配图绘制 |

---

## 权重准备

**本仓库不包含任何 `.pt` 权重文件。**

### 官方预训练权重

`pip install ultralytics` 后，首次运行脚本时会自动从 Ultralytics 服务器下载所需权重（如 `yolov8n.pt`、`yolov8n-pose.pt`、`yolov8s.pt` 等）。

你也可以手动下载放入 `weights/` 目录（此目录已加入 `.gitignore`）：

```bash
mkdir weights
# 下载官方权重
# https://github.com/ultralytics/assets/releases
```

### 微调权重

WiderPerson 数据集微调后的权重文件较大（~225 MB），暂不在仓库中提供。

> **TODO**: 后续通过 GitHub Release 提供下载链接。

---

## 使用方法

所有脚本从项目根目录运行。

### 跌倒检测

```bash
python scripts/fall_detection.py --source <video_path> --output outputs/fall_demo.mp4
```

- 模型: 自动使用 `weights/yolov8n-pose.pt`（姿态估计）
- 追踪器: `configs/bytetrack_fall.yaml`
- 可选参数: `--extract-keyframes` 抽取跌倒关键帧

### 奔跑检测

```bash
python scripts/running_detection.py --source <video_path> --output outputs/running_demo.mp4
```

- 模型: 自动使用 `weights/yolov8s.pt`（目标检测）
- 追踪器: `configs/bytetrack.yaml`
- 日志: `outputs/running_events.json`
- 灵敏度调节: `--conf-trigger 0.40`（降低阈值更多检测，提高阈值减少误检）

### 禁入区域入侵检测

两步流程：

```bash
# 步骤 1：交互式标定禁入区域
python tools/zone_selector.py --source <video_path> --output configs/forbidden_zone.json

# 步骤 2：运行检测
python scripts/intrusion_detection.py --source <video_path> --zone configs/forbidden_zone.json --output outputs/intrusion_demo.mp4
```

如果不指定 `--zone`，脚本使用内置示例多边形（适用于 1280×720 视频中央区域）。

详细参数说明见 [docs/USAGE.md](docs/USAGE.md)。

---

## 工具脚本说明

| 工具 | 命令 | 用途 |
|------|------|------|
| 环境检查 | `python tools/check_env.py` | 检查 Python/PyTorch/Ultralytics 版本 |
| 区域标定 | `python tools/zone_selector.py --source <video> --output configs/zone.json` | 交互式标定禁入区域 |
| 告警导出 | `python tools/export_alerts.py --video <video>` | 批量导出异常告警 JSON |
| 帧查看器 | `python tools/frame_viewer.py <video_path>` | 逐帧查看视频 |
| FPS 测量 | `python tools/measure_fps.py --video <video> --weights <weights.pt>` | 测量纯检测 FPS |
| 流水线延迟 | `python tools/measure_pipeline.py --video <video> --weights <weights.pt>` | 测量各阶段延迟 |
| 跟踪性能 | `python tools/measure_tracking.py --video <video> --weights <weights.pt>` | 测量跟踪指标 |
| 模型评估 | `python tools/run_eval.py --weights <model.pt> --data <dataset.yaml>` | 评估 mAP/PR 曲线 |

> **注意**: `run_eval.py`、`export_alerts.py`、`measure_*.py` 是可选的高级工具，需要训练权重或数据集。

---

## 实验结果

简要实验数据记录见 [docs/experiment_results.md](docs/experiment_results.md)。

---

## 注意事项

1. **本仓库不包含**：数据集、权重文件（.pt）、测试视频、训练结果
2. 用户需要自行准备输入视频，通过 `--source` 参数传入
3. 禁入区域的坐标需根据实际视频分辨率通过 `tools/zone_selector.py` 标定
4. 权重文件放入 `weights/` 目录后即可使用（该目录已被 `.gitignore` 忽略）

---

## License

本项目基于 [AGPL-3.0](LICENSE) 开源。

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) 开发，沿用其许可证。

```
Copyright (C) 2026 juzishazhou
```

---

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 优秀的开源目标检测框架
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — 多目标跟踪算法
- [WiderPerson](https://github.com/ShiqiYu/WiderPerson) — 密集人群检测数据集

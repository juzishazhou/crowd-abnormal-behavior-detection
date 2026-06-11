# 测试报告：环境验证与脚本完整性测试

> 生成日期：2026-06-11  
> 环境：`test_clone` (conda)  
> 仓库：`crowd-abnormal-behavior-detection`

---

## 1. 测试环境

| 项目 | 值 |
|---|---|
| Python | 3.12.13 |
| 操作系统 | Windows 11 |
| CPU | Intel i7 (14C/20T) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4GB) |
| CUDA | ✅ 12.6 cuDNN 91002 |
| PyTorch | 2.12.0+cu126 |
| Ultralytics | 8.4.65 |

---

## 2. 核心检测脚本测试结果

### 2.1 跌倒检测 — `scripts/fall_detection.py`

| 项目 | 结果 |
|---|---|
| 状态 | **✅ 通过** |
| 输入 | `assets/fall.mp4` (720×1280, 30fps, 306 帧) |
| 模型 | `yolov8n-pose.pt` (6.5 MB) — 自动下载 ✅ |
| 推理设备 | CUDA (FP16) |
| 检测结果 | 3 帧确认跌倒 (#251~#253) |
| 输出视频 | `outputs/fall_demo.mp4` (18 MB) |
| 关键帧抽取 | 11 张 → `outputs/highlights/` |
| 自动补装 | `lap>=0.5.12` (ultralytics AutoUpdate) |
| 问题 | 无 |

### 2.2 奔跑检测 — `scripts/running_detection.py`

| 项目 | 结果 |
|---|---|
| 状态 | **⚠️ 部分通过（见已知问题）** |
| 输入 | `assets/running.mp4` (632×480, 30fps, 212 帧) |
| 模型 | `yolov8s.pt` (22.5 MB) — 需手动下载（国内网络慢） |
| 推理设备 | CUDA (FP16) |
| 检测结果 | 9 个奔跑告警 |
| 输出视频 | `outputs/running_demo.mp4` (2.4 MB) ✅ |
| 问题 | 见下方 |

**已知问题**：
- `TypeError: Object of type float32 is not JSON serializable` — 第 605 行 `json.dump(events_log)` 时因 `numpy.float32` 类型无法被标准 JSON 序列化而崩溃。输出视频已正常生成。未修改源代码。

### 2.3 禁入区域检测 — `scripts/intrusion_detection.py`

| 项目 | 结果 |
|---|---|
| 状态 | **✅ 通过** |
| 输入 | `assets/intrusion.mp4` (1920×1080, 263 帧) |
| 模型 | `yolov8n.pt` (6.2 MB) — 自动下载 ✅ |
| 推理设备 | CUDA |
| 检测结果 | 12 个入侵事件 |
| 输出视频 | `outputs/intrusion_demo.mp4` ✅ |
| shapely | 未安装时用 `cv2.pointPolygonTest` 降级 ✅（推荐安装 shapely） |
| 问题 | 无 |

---

## 3. 工具脚本启动测试

| 工具 | 状态 | 输出要点 |
|---|---|---|
| `tools/check_env.py` | ✅ | CUDA 12.6 / RTX 3050 / 4GB / Win11 |
| `tools/frame_viewer.py` | ✅ | 成功打开窗口，播放至结尾 |
| `tools/measure_fps.py` | ✅ | 26.6 FPS (yolov8n-pose, 720×1280) |
| `tools/measure_pipeline.py` | ✅ | 71.5 FPS (yolov8s, 632×480) |
| `tools/measure_tracking.py` | ✅ | 32 IDs, 33 ID switches, 平均轨迹 110.6 帧 |
| `tools/export_alerts.py` | ⏭️ 跳过 | 需训练权重和数据集 |
| `tools/run_eval.py` | ⏭️ 跳过 | 需数据集 |

---

## 4. 补装包清单（按时间顺序）

| 序号 | 包名 | 版本 | 触发脚本 | 原因 |
|---|---|---|---|---|
| 1 | `ultralytics` | 8.4.65 | 初始安装 | 核心推理框架 |
| 2 | `opencv-python` | 4.13.0.92 | 初始安装 | 视频 I/O + 可视化 |
| 3 | `matplotlib` | 3.10.9 | ultralytics 间接依赖 | 配图绘制 |
| 4 | `psutil` | 7.2.2 | ultralytics 间接依赖 | 系统信息 |
| 5 | `scipy` | 1.17.1 | ultralytics 间接依赖 | 科学计算 |
| 6 | `lap` | 0.5.13 | `fall_detection.py` | ultralytics 自动补装，ByteTrack 线性分配 |
| 7 | `wget` | 3.2 | `running_detection.py` | 辅助下载权重（临时工具，非必需） |
| 8 | `shapely` | 2.1.2 | 安装流程中补装 | 禁入区域几何计算（推荐） |

### 非必需 / 临时包说明

- `wget==3.2`：仅用于手动下载权重，非运行必需
- `modelscope==1.37.1`：尝试国内镜像下载时安装，未成功，非必需

---

## 5. requirements.txt 新旧对比

### 新增结构化分组（新）
新版本按用途分组并添加注释：
```
# ===== Core Inference =====
# ===== Video & Image Processing =====
# ===== Numerical Computing =====
# ===== Deep Learning Backend =====
# ===== Polygon Geometry (Intrusion Detection) =====
# ===== Utility Script Dependencies =====
```

### 版本变化

| 包 | 旧版本 | 新版本 | 变化 |
|---|---|---|---|
| ultralytics | >=8.4.0 | >=8.4.65 | ↑ |
| opencv-python | >=4.6.0 | >=4.13.0.92 | ↑ |
| numpy | >=1.23.0 | >=2.4.4 | ↑ |
| torch | >=1.8.0 | >=2.12.0 | ↑ 大幅 |
| torchvision | >=0.9.0 | >=0.27.0 | ↑ 大幅 |
| shapely | >=2.0.0 | >=2.1.2 | ↑ 微调 |
| psutil | >=5.8.0 | >=7.2.2 | ↑ |
| matplotlib | >=3.3.0 | >=3.10.9 | ↑ |

所有版本使用 `>=` 而非 `==`，保留升级空间。

### 未纳入 requirements.txt 的间接依赖（共 13 个）

`certifi`, `charset-normalizer`, `contourpy`, `cycler`, `filelock`, `fonttools`, `fsspec`, `idna`, `Jinja2`, `kiwisolver`, `MarkupSafe`, `mpmath`, `networkx`, `packaging`, `pillow`, `polars`, `polars-runtime-32`, `pyparsing`, `python-dateutil`, `PyYAML`, `requests`, `scipy`, `setuptools`, `six`, `sympy`, `tqdm`, `typing_extensions`, `ultralytics-thop`, `urllib3`, `wheel` 等均为间接依赖，由 pip 自动安装。

---

## 6. 已知问题与建议

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | `running_detection.py` 第 605 行 JSON 序列化 `numpy.float32` 失败 | 日志文件 `running_events.json` 无法保存 | 将 `conf` 值用 `float()` 包装后再存 JSON |
| 2 | `yolov8s.pt`（22MB）从 GitHub 下载超时 | 国内用户首次运行需手动下载 | 建议在 README 中注明国内镜像或 GitHub Release 方式 |
| 3 | `shapely` 未安装时轨迹交叉检测使用 cv2 降级 | 精度略低于 shapely | 默认纳入 requirements.txt（已处理） |

---

## 7. 推荐安装顺序

```bash
# 1. 创建环境（Python 3.8+ 均可）
conda create -n crowd-detection python=3.12
conda activate crowd-detection

# 2. 安装 PyTorch（按需选择 CPU / CUDA 版本）
# CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 首次运行任意脚本，权重自动下载：
python scripts/fall_detection.py --source assets/fall.mp4 --output outputs/test.mp4
```

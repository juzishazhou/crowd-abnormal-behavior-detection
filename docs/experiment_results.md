# 第五章实验数据记录

> 项目：密集人群异常行为检测系统
> 数据集：WiderPerson（检测）+ 公开测试视频（异常判别）

## 0. 环境与数据准备

### 0.1 关键文件清单

#### 预训练权重

| 文件 | 用途 |
|------|------|
| yolov8s.pt | 官方预训练权重（检测） |
| yolov8n-pose.pt | 姿态估计（跌倒检测） |

#### 配置文件

| 文件 | 内容 |
|------|------|
| configs/bytetrack.yaml | ByteTrack 奔跑检测配置（high=0.5, buffer=60） |
| configs/bytetrack_fall.yaml | ByteTrack 跌倒检测配置（high=0.35, buffer=60） |

#### 异常检测脚本

| 脚本 | 状态 |
|------|------|
| scripts/fall_detection.py | ✓ 可用 |
| scripts/running_detection.py | ✓ 可用 |
| scripts/intrusion_detection.py | ✓ 可用 |

### 0.2 WiderPerson 数据集准备

#### 原始数据

| 项目 | 数量 |
|------|------|
| 原始图片 | 13,382 张 |
| 原始标注 | 9,000 个 .txt 文件 |

原始标注格式（每行一个框）：
```
<图片目标数>
<class_id> <xmin> <ymin> <xmax> <ymax>
```
其中 class_id: 1=pedestrian, 2=rider, 3=partially-visible, 4=ignore-region, 5=crowd

#### YOLO 格式转换结果

| 数据集 | 图片数 | 标签数 | 类别 |
|--------|--------|--------|------|
| train | 8,000 | 8,000 | 1 class: person (id=0) |
| val | 1,000 | 1,000 | 1 class: person (id=0) |

**转换规则**：
- class_label 1/2/3 合并为 person (id=0)
- class_label 4 (ignore-region) 和 5 (crowd) 丢弃
- 坐标为归一化 YOLO 格式：`<class> <x_center> <y_center> <width> <height>`

#### 数据集配置示例

创建 `configs/data/widerperson.yaml`：
```yaml
path: /path/to/your/WiderPerson_yolo
train: images/train
val: images/val

names:
  0: person
```

> 注意：WiderPerson 数据集不包含在本仓库中，请从 [官方仓库](https://github.com/ShiqiYu/WiderPerson) 下载。

### 0.3 训练基线

| 项目 | 详情 |
|------|------|
| 模型 | yolov8s.pt 预训练权重 |
| Epochs | 5 |
| Batch Size | 4 |
| Image Size | 640 |

---

> 注：本文件已脱敏，删除了本地绝对路径和个人硬件信息。

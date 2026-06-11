#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 juzishazhou — AGPL-3.0 License
"""
跌倒检测脚本 v3 — 姿态估计 + ByteTrack + 轨迹分析 + 丢框补偿
核心改进（按用户建议）：
- 追踪检测框中心点高度，检测"骤降"特征判定跌倒
- 追踪丢失时，基于最后已知位置（近地、微动）触发"疑似跌倒"
- 姿态估计 + 轨迹特征双通道融合判定，不依赖单帧
- 输出 跌倒_效果.mp4

Usage:
    python scripts/fall_detection.py --source <video_path> [--output <output_path>]
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        extract_track_ids,
        show_frame,
        extract_event_keyframes,
        cleanup_resources,
    )
except ImportError:
    # Fallback when _common is not on sys.path (e.g. running script directly)
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from _common import (
        PROJECT_ROOT,
        open_video,
        get_video_props,
        create_video_writer,
        extract_track_ids,
        show_frame,
        extract_event_keyframes,
        cleanup_resources,
    )

# ============================================================
# 配置参数
# ============================================================
DEFAULT_SOURCE = None  # 用户必须通过 --source 指定输入视频
DEFAULT_OUTPUT = str(PROJECT_ROOT / "outputs" / "fall_demo.mp4")
MODEL_NAME = "weights/yolov8n-pose.pt"
TRACKER_CONFIG = PROJECT_ROOT / "configs" / "bytetrack_fall.yaml"

# ---- 推理 ----
DET_CONF = 0.25
IOU_THRESHOLD = 0.5

# ---- 姿态跌倒阈值 ----
FALL_ASPECT_RATIO_THRESH = 1.0
FALL_HEAD_HIP_RATIO_THRESH = 0.35

# ---- 时序平滑 ----
FALL_CONFIRM_WINDOW = 5
FALL_CONFIRM_MIN_RATIO = 0.5
FALL_HANGOVER_FRAMES = 30

# ---- 轨迹跌倒阈值 (核心新增) ----
TRAJ_HISTORY_LEN = 10
TRAJ_CENTER_DROP_RATIO = 0.15
TRAJ_AR_CHANGE_RATIO = 1.5
TRAJ_FALL_FRAMES = 5
LOST_PERSON_GROUND_ZONE = 0.82
LOST_PERSON_FRAMES = 12
LOST_PERSON_MAX_MOVE = 0.015
LOST_PERSON_MIN_TRAJ = 15

# ---- 绘制 ----
SKELETON_COLOR = (0, 255, 128)
KPT_COLOR = (0, 255, 0)
FALL_BOX_COLOR = (0, 0, 255)
SUSPECT_BOX_COLOR = (0, 140, 255)
NORMAL_BOX_COLOR = (255, 255, 0)
FALL_TEXT = "WARNING FALL"
FALL_FONT_SCALE = 2.0
FALL_FONT_THICKNESS = 4
FALL_TEXT_COLOR = (0, 0, 255)
KPT_CONF_THRESHOLD = 0.5

# ---- 轨迹线绘制 ----
TRAJ_LINE_MAX_POINTS = 60       # 最多保留的轨迹点
TRAJ_LINE_NORMAL_COLOR = (180, 180, 180)  # 普通人：灰色细线
TRAJ_LINE_FALL_COLOR = (0, 0, 255)        # 跌倒者：红色粗线
TRAJ_LINE_NORMAL_THICK = 1
TRAJ_LINE_FALL_THICK = 3

# ---- 关键帧自动抽取 ----
KEYFRAME_EXTRACT_ENABLED = True
KEYFRAME_RANGE = 10
KEYFRAME_STEP = 2
KEYFRAME_OUTPUT_DIR = "outputs/highlights"

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# ============================================================
# 全局状态
# ============================================================
fall_history = defaultdict(list)
last_fall_frame = {}
last_global_fall_frame = -999

traj_history = defaultdict(lambda: deque(maxlen=TRAJ_LINE_MAX_POINTS))
traj_fall_counter = defaultdict(int)
last_active_frame = {}
last_known_bbox = {}
lost_fall_triggered = set()
lost_fall_frame = {}


# ============================================================
# 姿态几何判定
# ============================================================
def is_fall_by_pose(keypoints_xy, bbox_width, bbox_height):
    if keypoints_xy is None or len(keypoints_xy) < 17:
        return False, 0.0

    kp = {name: kpt for name, kpt in zip(COCO_KEYPOINT_NAMES, keypoints_xy)}
    nose = kp["nose"]
    left_shoulder = kp["left_shoulder"]
    right_shoulder = kp["right_shoulder"]
    left_hip = kp["left_hip"]
    right_hip = kp["right_hip"]
    left_ankle = kp["left_ankle"]
    right_ankle = kp["right_ankle"]

    fall_score = 0.0

    if bbox_height > 0:
        ar = bbox_width / bbox_height
        if ar > FALL_ASPECT_RATIO_THRESH:
            fall_score += 0.5
        elif ar > 0.8:
            fall_score += 0.2

    head_y = nose[1] if nose[1] > 0 else None
    hip_y = (left_hip[1] + right_hip[1]) / 2.0 if (left_hip[1] > 0 and right_hip[1] > 0) else None
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2.0 if (left_ankle[1] > 0 and right_ankle[1] > 0) else None

    if head_y is not None and hip_y is not None and ankle_y is not None:
        body_height = abs(ankle_y - head_y) + 1e-5
        head_hip_dist = abs(head_y - hip_y)
        ratio = head_hip_dist / body_height
        if ratio < FALL_HEAD_HIP_RATIO_THRESH:
            fall_score += 0.4
        elif ratio < 0.5:
            fall_score += 0.15

    shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0]) / 2.0,
                              (left_shoulder[1] + right_shoulder[1]) / 2.0])
    hip_mid = np.array([(left_hip[0] + right_hip[0]) / 2.0,
                         (left_hip[1] + right_hip[1]) / 2.0])
    torso_vec = hip_mid - shoulder_mid
    torso_len = np.linalg.norm(torso_vec) + 1e-5
    torso_verticality = abs(torso_vec[1]) / torso_len
    if torso_verticality < 0.45:
        fall_score += 0.3
    elif torso_verticality < 0.6:
        fall_score += 0.1

    return fall_score >= 0.45, fall_score


# ============================================================
# 轨迹跌倒判定：中心点高度骤降 + 宽高比剧变
# ============================================================
def is_fall_by_trajectory(track_id, frame_idx, frame_h):
    traj = list(traj_history[track_id])
    if len(traj) < TRAJ_FALL_FRAMES:
        return False

    recent = traj[-TRAJ_FALL_FRAMES:]
    older = traj[:-TRAJ_FALL_FRAMES]
    if len(older) < 2:
        return False

    older_cy = np.mean([t[1] for t in older])
    recent_cy = np.mean([t[1] for t in recent])
    cy_drop = (recent_cy - older_cy) / frame_h

    older_ar = np.mean([t[2] / max(t[3], 1) for t in older])
    recent_ar = np.mean([t[2] / max(t[3], 1) for t in recent])
    ar_change = recent_ar / max(older_ar, 1e-5)

    is_traj_fall = (cy_drop > TRAJ_CENTER_DROP_RATIO) and (ar_change > TRAJ_AR_CHANGE_RATIO)

    if is_traj_fall:
        traj_fall_counter[track_id] += 1
    else:
        traj_fall_counter[track_id] = max(0, traj_fall_counter[track_id] - 1)

    return traj_fall_counter[track_id] >= 3


# ============================================================
# 丢框补偿：丢失追踪 + 最后位置近地 + 微动 = 疑似跌倒
# ============================================================
def check_lost_person_fall(track_id, frame_idx, frame_h):
    """
    丢框补偿：丢失追踪 + 最后位置近地 + 微动 + 足够轨迹 = 疑似跌倒。
    仅在 track_id 最近活跃过且积累了足够轨迹时才检查。
    """
    # 超时清理已触发标记
    if track_id in lost_fall_triggered:
        if track_id in lost_fall_frame:
            if frame_idx - lost_fall_frame[track_id] > FALL_HANGOVER_FRAMES + 10:
                lost_fall_triggered.discard(track_id)
                lost_fall_frame.pop(track_id, None)
        return True

    if track_id not in last_active_frame:
        return False

    lost_duration = frame_idx - last_active_frame[track_id]
    if lost_duration < LOST_PERSON_FRAMES:
        return False

    # 只在近期丢失的才检查（丢失不超过 N*2 帧）
    if lost_duration > LOST_PERSON_FRAMES * 3:
        return False

    # 必须有足够轨迹积累
    if len(traj_history[track_id]) < LOST_PERSON_MIN_TRAJ:
        return False

    if track_id not in last_known_bbox:
        return False

    lx1, ly1, lx2, ly2 = last_known_bbox[track_id]
    last_cy = (ly1 + ly2) / 2.0

    if last_cy / frame_h < LOST_PERSON_GROUND_ZONE:
        return False

    traj = list(traj_history[track_id])
    if len(traj) >= 3:
        positions = np.array([(t[0], t[1]) for t in traj[-3:]])
        if len(positions) >= 2:
            max_move = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            if max_move / frame_h > LOST_PERSON_MAX_MOVE:
                return False

    lost_fall_triggered.add(track_id)
    lost_fall_frame[track_id] = frame_idx
    global last_global_fall_frame
    last_global_fall_frame = frame_idx
    return True


# ============================================================
# 时序平滑 + 全局状态
# ============================================================
def should_display_fall(track_id, frame_idx, is_fall_now):
    fall_history[track_id].append((frame_idx, is_fall_now))
    if len(fall_history[track_id]) > FALL_CONFIRM_WINDOW:
        fall_history[track_id].pop(0)

    recent = fall_history[track_id]
    if len(recent) >= max(2, FALL_CONFIRM_WINDOW // 2):
        fall_ratio = sum(1 for _, f in recent if f) / len(recent)
    else:
        fall_ratio = 0.0

    if fall_ratio >= FALL_CONFIRM_MIN_RATIO:
        last_fall_frame[track_id] = frame_idx
        global last_global_fall_frame
        last_global_fall_frame = frame_idx

    if track_id in last_fall_frame:
        if frame_idx - last_fall_frame[track_id] <= FALL_HANGOVER_FRAMES:
            return True
        else:
            del last_fall_frame[track_id]

    return False


# ============================================================
# 绘制
# ============================================================
def draw_skeleton(frame, keypoints_xy, confidence, is_falling, bbox,
                  track_id=None, is_suspect=False):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    kpt_draw_thresh = KPT_CONF_THRESHOLD * 0.5 if (is_falling or is_suspect) else KPT_CONF_THRESHOLD

    for idx_a, idx_b in SKELETON_EDGES:
        if (idx_a >= len(keypoints_xy) or idx_b >= len(keypoints_xy)
                or idx_a >= len(confidence) or idx_b >= len(confidence)):
            continue
        if confidence[idx_a] < kpt_draw_thresh or confidence[idx_b] < kpt_draw_thresh:
            continue
        pt_a = (int(keypoints_xy[idx_a][0]), int(keypoints_xy[idx_a][1]))
        pt_b = (int(keypoints_xy[idx_b][0]), int(keypoints_xy[idx_b][1]))
        if pt_a[0] <= 0 or pt_a[1] <= 0 or pt_b[0] <= 0 or pt_b[1] <= 0:
            continue
        cv2.line(frame, pt_a, pt_b, SKELETON_COLOR, thickness=2, lineType=cv2.LINE_AA)

    for i, (kpt, conf) in enumerate(zip(keypoints_xy, confidence)):
        if conf < kpt_draw_thresh:
            continue
        x, y = int(kpt[0]), int(kpt[1])
        if x <= 0 or y <= 0:
            continue
        cv2.circle(frame, (x, y), radius=4, color=KPT_COLOR, thickness=-1, lineType=cv2.LINE_AA)

    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    if is_falling:
        box_color = FALL_BOX_COLOR
    elif is_suspect:
        box_color = SUSPECT_BOX_COLOR
    else:
        box_color = NORMAL_BOX_COLOR
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), box_color, thickness=2)

    id_label = f"ID:{track_id}" if track_id is not None else ""
    if id_label:
        (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (ix1, iy1 - th - 6), (ix1 + tw + 4, iy1), box_color, -1)
        cv2.putText(frame, id_label, (ix1 + 2, iy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_fall_banner(frame, is_suspect=False):
    h, w = frame.shape[:2]
    txt = "WARNING FALL?" if is_suspect else FALL_TEXT
    color = SUSPECT_BOX_COLOR if is_suspect else FALL_TEXT_COLOR
    text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                FALL_FONT_SCALE, FALL_FONT_THICKNESS)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 80
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (text_x - 20, text_y - text_size[1] - 20),
                  (text_x + text_size[0] + 20, text_y + 10),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)
    cv2.putText(frame, txt, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, FALL_FONT_SCALE, color,
                FALL_FONT_THICKNESS, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w - 1, 8), color, -1)
    cv2.rectangle(frame, (0, h - 8), (w - 1, h - 1), color, -1)


def draw_trajectory_lines(frame, is_falling=False):
    """在当前帧上绘制所有人的运动轨迹线。跌倒者红色粗线，普通人灰色细线。"""
    for tid, traj in traj_history.items():
        if len(traj) < 2:
            continue
        points = [(int(t[0]), int(t[1])) for t in traj]
        # 判断此人当前是否跌倒
        falling = (tid in last_fall_frame and
                   last_fall_frame.get(tid, -999) + FALL_HANGOVER_FRAMES >= traj[-1][4])
        color = TRAJ_LINE_FALL_COLOR if falling else TRAJ_LINE_NORMAL_COLOR
        thick = TRAJ_LINE_FALL_THICK if falling else TRAJ_LINE_NORMAL_THICK
        # 用小圆点标记最近位置
        cv2.circle(frame, points[-1], radius=4, color=color, thickness=-1)
        # 画折线
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, thickness=thick,
                     lineType=cv2.LINE_AA)


# 关键帧抽取通过 extract_event_keyframes 实现（已从 scripts._common 导入）


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="跌倒检测脚本 — 姿态估计 + ByteTrack + 轨迹分析 + 丢框补偿")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help="输入视频路径 (必须指定)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"输出视频路径 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--extract-keyframes", action="store_true", default=KEYFRAME_EXTRACT_ENABLED,
                        help="检测到跌倒后自动抽取关键帧 (默认: 开启)")
    parser.add_argument("--no-extract-keyframes", action="store_true",
                        help="禁用关键帧自动抽取")
    parser.add_argument("--keyframe-range", type=int, default=KEYFRAME_RANGE,
                        help=f"关键帧抽取范围 ±N 帧 (默认: {KEYFRAME_RANGE})")
    parser.add_argument("--keyframe-step", type=int, default=KEYFRAME_STEP,
                        help=f"关键帧抽取步长 (默认: {KEYFRAME_STEP})")
    parser.add_argument("--keyframe-dir", type=str, default=KEYFRAME_OUTPUT_DIR,
                        help=f"关键帧输出目录 (默认: {KEYFRAME_OUTPUT_DIR})")
    args = parser.parse_args()

    if args.source is None:
        print("[X] 错误: 必须通过 --source 指定输入视频路径")
        print("    用法: python scripts/fall_detection.py --source <video_path>")
        sys.exit(1)

    VIDEO_INPUT = Path(args.source)
    VIDEO_OUTPUT = Path(args.output)
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda")
    print(f"[i] 推理设备: {device.upper()}" + (" (FP16)" if half else ""))

    print(f"[i] 加载姿态估计模型: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"[X] 模型加载失败: {e}")
        sys.exit(1)

    cap = open_video(str(VIDEO_INPUT))
    props = get_video_props(cap)
    fps, frame_w, frame_h, total_frames = (props["fps"], props["width"],
                                            props["height"], props["total_frames"])
    print(f"[i] 视频信息: {frame_w}x{frame_h}, {fps:.1f} FPS, {total_frames} 帧")

    tracker_yaml = str(TRACKER_CONFIG) if TRACKER_CONFIG.exists() else "bytetrack.yaml"
    print(f"[i] 追踪器: {tracker_yaml}")
    print(f"[i] 轨迹分析: 中心骤降>{TRAJ_CENTER_DROP_RATIO*100:.0f}%H + AR变化>{TRAJ_AR_CHANGE_RATIO}x")
    print(f"[i] 丢框补偿: 丢失{LOST_PERSON_FRAMES}帧+近地>{LOST_PERSON_GROUND_ZONE*100:.0f}%H+需{LOST_PERSON_MIN_TRAJ}帧轨迹 -> 疑似跌倒")

    out = create_video_writer(str(VIDEO_OUTPUT), fps, (frame_w, frame_h))

    frame_idx = 0
    fall_frames = []
    suspect_frames = []

    print("\n[>] 开始逐帧推理 (追踪+轨迹+丢框补偿)...")
    print("    (按 Q 可提前终止)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, conf=DET_CONF, iou=IOU_THRESHOLD,
            device=device, half=half, tracker=tracker_yaml,
            persist=True, verbose=False,
        )

        frame_has_fall = False
        active_ids_this_frame = set()

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            keypoints_data = result.keypoints.data
            boxes_data = result.boxes.data

            if result.boxes.id is not None:
                track_ids = extract_track_ids(result.boxes)
            else:
                track_ids = list(range(len(boxes_data)))

            for pi in range(len(keypoints_data)):
                kp_tensor = keypoints_data[pi]
                box_tensor = boxes_data[pi]
                tid = int(track_ids[pi])

                active_ids_this_frame.add(tid)
                last_active_frame[tid] = frame_idx

                keypoints_xy = kp_tensor[:, :2].cpu().numpy()
                kpt_conf = kp_tensor[:, 2].cpu().numpy()

                x1, y1, x2, y2 = box_tensor[:4].cpu().numpy()
                bbox_w, bbox_h = x2 - x1, y2 - y1

                # 记录轨迹
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                traj_history[tid].append((cx, cy, bbox_w, bbox_h, frame_idx))
                last_known_bbox[tid] = (x1, y1, x2, y2)

                # 双通道判定
                pose_fall, _ = is_fall_by_pose(keypoints_xy, bbox_w, bbox_h)
                traj_fall = is_fall_by_trajectory(tid, frame_idx, frame_h)
                is_fall_now = pose_fall or traj_fall
                confirmed_fall = should_display_fall(tid, frame_idx, is_fall_now)

                if confirmed_fall:
                    frame_has_fall = True

                draw_skeleton(frame, keypoints_xy, kpt_conf, confirmed_fall,
                              (x1, y1, x2, y2), track_id=tid)

        # ---- 绘制轨迹线 (所有人) ----
        draw_trajectory_lines(frame)

        # ---- 丢框补偿 ----
        global last_global_fall_frame
        lost_fall_active = False
        all_known_ids = set(last_active_frame.keys())
        for tid in all_known_ids - active_ids_this_frame:
            if check_lost_person_fall(tid, frame_idx, frame_h):
                lost_fall_active = True
                suspect_frames.append(frame_idx)
                if tid in last_known_bbox:
                    lx1, ly1, lx2, ly2 = [int(v) for v in last_known_bbox[tid]]
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2),
                                  SUSPECT_BOX_COLOR, thickness=2)
                    cv2.putText(frame, f"ID:{tid} SUSPECT", (lx1, ly1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SUSPECT_BOX_COLOR, 2)

        # ---- 横幅 ----
        global_fall_active = (frame_idx - last_global_fall_frame <= FALL_HANGOVER_FRAMES)
        if frame_has_fall:
            draw_fall_banner(frame)
            fall_frames.append(frame_idx)
        elif lost_fall_active:
            draw_fall_banner(frame, is_suspect=True)
        elif global_fall_active:
            draw_fall_banner(frame)

        out.write(frame)
        frame_idx += 1

        if frame_idx % max(1, int(fps)) == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"    进度: {frame_idx}/{total_frames} ({pct:.1f}%)  "
                  f"| 跌: {len(fall_frames)} | 疑: {len(suspect_frames)}", end="\r")

        if not show_frame(frame, "Fall Detection (press Q to quit)", "q"):
            print("\n[!] 用户提前终止。")
            break

    cleanup_resources(cap, out)

    print("\n\n[OK] 处理完成！")
    print(f"    总帧数: {frame_idx}")
    print(f"    确认跌倒帧: {len(fall_frames)}")
    print(f"    丢框补偿疑似帧: {len(suspect_frames)}")
    print(f"    输出视频: {VIDEO_OUTPUT}")
    if fall_frames:
        print(f"    跌倒帧号: {fall_frames}")
    if suspect_frames:
        print(f"    疑似帧号(前30): {suspect_frames[:30]}{'...' if len(suspect_frames)>30 else ''}")
    if fall_frames or suspect_frames:
        all_fall = sorted(set(fall_frames + suspect_frames))
        print(f"\n[Tip] 请从 '{VIDEO_OUTPUT.name}' 中挑选最明显的跌倒帧。")
        print(f"      推荐区间: #{min(all_fall)} ~ #{max(all_fall)}")

        # ---- 自动抽取关键帧 ----
        do_extract = args.extract_keyframes and not args.no_extract_keyframes
        if do_extract:
            target_frames = fall_frames if fall_frames else suspect_frames
            extract_event_keyframes(
                video_path=VIDEO_OUTPUT,
                event_frames=target_frames,
                output_dir=args.keyframe_dir,
                range_frames=args.keyframe_range,
                step=args.keyframe_step,
                label="fall",
            )
    else:
        print("\n[!] 未检测到跌倒帧。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

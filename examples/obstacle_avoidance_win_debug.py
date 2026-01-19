# -*- coding: utf-8 -*-
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


def main():
    pipeline = Pipeline()
    config = Config()

    # 1. 开启硬件对齐 (D2C)，确保彩色和深度图坐标一致
    config.set_align_mode(OBAlignMode.HW_MODE)

    WIDTH, HEIGHT, FPS = 640, 480, 30
    try:
        # 配置双流
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_p = color_profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.MJPG, FPS)
        config.enable_stream(color_p)

        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_p = depth_profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.Y16, FPS)
        config.enable_stream(depth_p)
    except Exception as e:
        print(f"流开启失败: {e}")
        return

    pipeline.start(config)

    # --- 核心参数调整 ---
    SAFE_DISTANCE_MM = 400  # 安全边界设为 40cm
    MIN_NOISE_DIST = 50  # 过滤掉 5cm 以内的镜头噪点
    # 触发阈值：如果区域内超过 3% 的像素点低于 40cm，则判定为碰撞
    DANGER_PIXEL_RATIO = 0.03

    print(f"\n--- 避障参数已更新 ---")
    print(f"当前安全距离阈值: {SAFE_DISTANCE_MM} mm (40cm)")
    print("低于 40cm: 红色警告 | 高于 40cm: 绿色安全")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames: continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None: continue

            # 处理彩色图
            raw_color = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_img = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)

            # 处理深度图
            w, h = depth_frame.get_width(), depth_frame.get_height()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_raw.reshape((h, w))

            # 定义中心探测区域 (ROI)
            roi_w, roi_h = int(w * 0.4), int(h * 0.4)
            x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
            roi_depth = depth_data[y1:y1 + roi_h, x1:x1 + roi_w]

            # 碰撞判定逻辑
            # 统计在 [50mm, 400mm] 范围内的危险像素点
            danger_mask = (roi_depth > MIN_NOISE_DIST) & (roi_depth < SAFE_DISTANCE_MM)
            danger_count = np.sum(danger_mask)
            danger_ratio = danger_count / (roi_w * roi_h)

            # 低于 40cm 判定为 Collision
            is_collision = danger_ratio > DANGER_PIXEL_RATIO

            # 渲染深度伪彩色图
            depth_visual = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)

            # UI 反馈
            box_color = (0, 0, 255) if is_collision else (0, 255, 0)
            status_text = "COLLISION!" if is_collision else "SAFE"

            # 在彩色图上绘制
            cv2.rectangle(color_img, (x1, y1), (x1 + roi_w, y1 + roi_h), box_color, 3)
            cv2.putText(color_img, f"STATUS: {status_text}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 3)
            cv2.putText(color_img, f"Dist < 40cm: {danger_ratio * 100:.1f}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # 左右拼接并显示
            combined_view = cv2.hconcat([color_img, depth_visual])
            cv2.imshow("40cm Collision Detection Debug", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
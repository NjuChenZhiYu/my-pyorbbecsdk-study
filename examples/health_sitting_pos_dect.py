import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


def main():
    pipeline = Pipeline()
    config = Config()

    # 1. 开启 D2C 硬件对齐
    config.set_align_mode(OBAlignMode.HW_MODE)

    # 设定统一的分辨率
    WIDTH, HEIGHT, FPS = 640, 480, 30

    try:
        # 配置彩色流：建议明确指定 MJPG 格式以保证兼容性
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_p = color_profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.MJPG, FPS)
        config.enable_stream(color_p)

        # 配置深度流
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_p = depth_profiles.get_video_stream_profile(WIDTH, HEIGHT, OBFormat.Y16, FPS)
        config.enable_stream(depth_p)
    except Exception as e:
        print(f"流开启失败: {e}")
        return

    pipeline.start(config)

    # 坐姿提醒参数
    DIST_TOO_CLOSE = 450
    DIST_GOOD_MIN = 550
    DIST_GOOD_MAX = 750

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames: continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # --- 核心修正：处理 MJPG 压缩格式 ---
            raw_color = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            # 使用 imdecode 自动处理 168212 这种压缩字节流
            color_data = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)

            if color_data is None:
                print("解码彩色图像失败")
                continue

            # --- 深度数据处理 ---
            w, h = depth_frame.get_width(), depth_frame.get_height()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

            if depth_raw.size != w * h:
                continue
            depth_data = depth_raw.reshape((h, w))

            # --- 坐姿检测与 UI 渲染 ---
            roi_w, roi_h = 240, 240
            x1, y1 = (w - roi_w) // 2, (h - roi_h) // 2
            roi_depth = depth_data[y1:y1 + roi_h, x1:x1 + roi_w]

            valid_depths = roi_depth[roi_depth > 0]
            avg_dist = np.mean(valid_depths) if len(valid_depths) > 0 else 0

            # 状态判断
            if avg_dist == 0:
                msg, color = "Searching for user...", (255, 255, 255)
            elif avg_dist < DIST_TOO_CLOSE:
                msg, color = "TOO CLOSE! Lean Back", (0, 0, 255)
            elif DIST_GOOD_MIN <= avg_dist <= DIST_GOOD_MAX:
                msg, color = "Perfect Posture", (0, 255, 0)
            else:
                msg, color = f"Dist: {int(avg_dist)}mm", (255, 191, 0)

            # 绘制 UI
            cv2.rectangle(color_data, (x1, y1), (x1 + roi_w, y1 + roi_h), color, 2)
            cv2.rectangle(color_data, (0, 0), (w, 60), (30, 30, 30), -1)
            cv2.putText(color_data, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Orbbec AI Health Assistant", color_data)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
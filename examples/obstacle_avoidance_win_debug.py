import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def main():
    pipeline = Pipeline()
    config = Config()

    # 1. 启用深度流（避障的核心是深度数据）
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # 获取默认深度流配置，通常是 640x480 或 1280x720，30fps
        profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as e:
        print(f"ERROR: 无法开启深度流: {e}")
        return

    pipeline.start(config)

    # 定义避障参数（这些参数是你可以调整来测试不同场景的关键）
    MIN_DISTANCE_MM = 0  # 最小安全距离 (10cm)
    MAX_DISTANCE_MM = 350  # 最大检测距离 (80cm)

    # 避障检测区域 (ROI): 画面中心区域的宽度和高度
    ROI_WIDTH_PERCENT = 0.5  # 区域宽度占画面宽度的 50%
    ROI_HEIGHT_PERCENT = 0.4  # 区域高度占画面高度的 40%

    # 触发避障的危险像素点占比阈值
    DANGER_PIXEL_THRESHOLD_PERCENT = 0.03  # 如果危险像素超过 3% 就报警

    print("\n--- Orbbec 3D 避障预警服务已启动 ---")
    print(f"检测距离范围: {MIN_DISTANCE_MM}mm - {MAX_DISTANCE_MM}mm")
    print(f"危险像素占比阈值: {DANGER_PIXEL_THRESHOLD_PERCENT * 100:.1f}%")
    print("请将手或物体放到相机前，观察画面变化。按 'q' 退出。")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)  # 等待 100ms
            if not frames:
                continue

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            width, height = depth_frame.get_width(), depth_frame.get_height()
            # 1. 获取原始数据
            raw_data = depth_frame.get_data()
            # 2. 将数据转换为一维 numpy 数组
            depth_data_1d = np.frombuffer(raw_data, dtype=np.uint16)
            # 3. 核心修正：根据相机的宽高，将 1D 数组重塑为 2D 矩阵
            # 注意：Gemini 335Lg 的深度通常是 uint16 类型
            depth_data = depth_data_1d.reshape((height, width))

            # 计算动态 ROI 区域的实际像素坐标
            roi_pixel_w = int(width * ROI_WIDTH_PERCENT)
            roi_pixel_h = int(height * ROI_HEIGHT_PERCENT)

            x1 = (width - roi_pixel_w) // 2
            y1 = (height - roi_pixel_h) // 2
            x2 = x1 + roi_pixel_w
            y2 = y1 + roi_pixel_h

            # 提取 ROI 区域的深度数据
            # 4. 现在你可以安全地进行二维切片了
            roi_depth_data = depth_data[y1:y2, x1:x2]

            # 统计 ROI 区域内，处于危险距离范围的像素点
            # 排除 0 值（无效深度）和太远的背景
            danger_mask = (roi_depth_data > MIN_DISTANCE_MM) & (roi_depth_data < MAX_DISTANCE_MM)
            danger_pixel_count = np.sum(danger_mask)

            # 计算危险像素点占整个 ROI 区域的比例
            total_roi_pixels = roi_pixel_w * roi_pixel_h
            if total_roi_pixels == 0:  # 避免除以零
                collision_risk_ratio = 0
            else:
                collision_risk_ratio = danger_pixel_count / total_roi_pixels

            # 决策逻辑：根据危险像素占比判断是否触发报警
            is_collision = collision_risk_ratio >= DANGER_PIXEL_THRESHOLD_PERCENT

            # 5. 可视化 UI 反馈
            # 归一化深度图以便显示（0-255，伪彩色）
            # 注意：这里对深度数据进行了 clip 和 normalize，以增强显示效果，不影响原始避障计算
            display_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_JET)

            # 绘制避障检测框，根据状态改变颜色
            box_color = (0, 0, 255) if is_collision else (0, 255, 0)  # 红色警告，绿色安全
            cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, 3)

            # 绘制预警文字
            msg = "!!! COLLISION AHEAD !!!" if is_collision else "Path Clear"
            cv2.putText(display_img, msg, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)
            cv2.putText(display_img, f"Danger Pixels: {collision_risk_ratio * 100:.1f}%", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

            cv2.imshow("Orbbec 3D Obstacle Avoidance Service (Windows Debug)", display_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()  # 确保停止相机，释放资源


if __name__ == "__main__":
    main()
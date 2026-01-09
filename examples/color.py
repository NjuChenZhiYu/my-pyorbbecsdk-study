# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import cv2

from pyorbbecsdk import *
# 必须导入属性枚举
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBPropertyID, OBError
from pyorbbecsdk import OBPropertyID
from utils import frame_to_bgr_image
import time

ESC_KEY = 27

def auto_configure_camera():
    pipeline = Pipeline()
    device = pipeline.get_device()

    # 1. 获取并校验固件版本
    info = device.get_device_info()
    fw_version = info.get_firmware_version()
    print(f"检测到设备: {info.get_name()}, 固件版本: {fw_version}")

    # 针对你截图中的 1.6.21 版本进行逻辑判断
    target_fw = "1.6.21"
    if fw_version == target_fw:
        print(f"--- 固件版本匹配 ({target_fw})，开始执行自动配置 ---")

        try:
            # 2. 确保 LDP (激光保护) 处于开启状态以保证安全
            # 虽然你之前查到是 False，但在正式实验前建议设为 True
            print(device.get_bool_property(OBPropertyID.OB_PROP_LDP_BOOL))
            device.set_bool_property(OBPropertyID.OB_PROP_LDP_BOOL, True)
            print("已激活 LDP 激光保护机制")
            # 3. 开启激光投射器 (Laser Control)
            # 这对 335Lg 获取高质量深度图至关重要
            device.set_bool_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, True)
            print("激光投射器已开启，正在准备 3D 数据流...")

            # 4. 这里的延时是为了让激光功率稳定
            time.sleep(1)

        except OBError as e:
            print(f"配置属性时发生错误: {e}")
    else:
        print(f"警告: 固件版本为 {fw_version}，非预期的 {target_fw}，跳过自动配置。")


def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    # 在 pipeline.start() 之前添加
    device = pipeline.get_device()
    device_info = device.get_device_info()
    print(f"Device Name: {device_info.get_name()}")
    print(f"Serial Number: {device_info.get_serial_number()}")
    print(f"Firmware Version: {device_info.get_firmware_version()}")
    auto_configure_camera()
    pipeline.start(config)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":

    main()

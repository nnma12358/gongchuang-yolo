#!/bin/bash
# Astra 相机节点启动脚本（只用相机，不启动机械臂驱动）
# ------------------------------------------------------------------
# 用法（在 ros2_arm_container 内）：
#     bash /scripts/start_astra.sh          # 前台
#     nohup bash /scripts/start_astra.sh > /tmp/astra.log 2>&1 &     # 后台
# 关键点：
#   · 必须显式导出 ROS_DOMAIN_ID=95 / ROS_LOCALHOST_ONLY=1 / RMW=cyclonedds，
#     因为镜像 env 里没有这些，默认 domain 0 会与网关桥接（95）对不上；
#   · 话题按现场约定 remap 成 /camera/color/image_raw 与 /camera/depth/image_raw
#     （桥接容器的 COLOR_TOPIC/DEPTH_TOPIC 就是这两个）；
#   · 深度配准 depth_registration=true，保证深度与彩色像素对齐（2.5D 定位要用）。
set +u    # ROS 的 setup.bash 里有未定义变量（COLCON_TRACE），不能用 set -u
PREFIX=/ros2_ws/wheeltec_arm_ros2_foxy_ready/install/astra_camera
RT=$PREFIX/lib/astra_camera/openni2
# ROS 基础环境（两种布局都试一下）
set +u
[ -f /opt/ros/foxy/install/setup.bash ] && source /opt/ros/foxy/install/setup.bash
[ -f /opt/ros/foxy/setup.bash ] && source /opt/ros/foxy/setup.bash
source /ros2_ws/wheeltec_arm_ros2_foxy_ready/install/setup.bash
export OPENNI2_REDIST=$RT
export OPENNI2_DRIVERS_PATH=$RT/OpenNI2/Drivers
export LD_LIBRARY_PATH=$RT:${LD_LIBRARY_PATH:-}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-95}
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
echo "[astra] domain=$ROS_DOMAIN_ID rmw=$RMW_IMPLEMENTATION prefix=$PREFIX"
exec $PREFIX/lib/astra_camera/astra_camera_node --ros-args \
  -r /camera/rgb/image_raw:=/camera/color/image_raw \
  -r /camera/rgb/camera_info:=/camera/color/camera_info \
  -p device_uri:=ANY_DEVICE \
  -p enable_color:=true -p enable_depth:=true -p enable_ir:=false \
  -p depth_registration:=true \
  -p color_width:=640 -p color_height:=480 -p color_fps:=30 \
  -p depth_width:=640 -p depth_height:=480 -p depth_fps:=30

#!/usr/bin/env bash
# launch_openpilot.sh — cpv9 单入口(对齐 sp aunch_pc.sh 模式)
# 自带: 环境准备 / panda 自检 / tw_camera_cfg 相机服务生命周期 / 模拟 CAN(fake panda)
# 退出时自动关闭相机服务 + 模拟 CAN。用法:
#   实车:      bash launch_openpilot.sh
#   桌面模拟:  ENABLE_FAKE_PANDA=1 bash launch_openpilot.sh

set -uo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR" || exit 1

echo "========== cpv9-cuda-main — AGX Orin 启动 =========="

# ---- API 切换: EnableConnect=2 -> 胡萝卜 API; 默认 konik.ai ----
if [[ "$(cat /data/params/d/EnableConnect 2>/dev/null)" == "2" ]]; then
  export API_HOST="https://api.carrotpilot.app"
  export ATHENA_HOST="wss://athena.carrotpilot.app"
else
  export API_HOST='https://api.konik.ai'
  export ATHENA_HOST='wss://athena.konik.ai'
fi

# ---- venv + PYTHONPATH ----
if [ -f "$DIR/.venv/bin/activate" ]; then
  source "$DIR/.venv/bin/activate"
fi
export PYTHONPATH="$DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- panda 基准自检(快 <0.1s): 固件字节对齐 + 协议检查 ----
# 不一致只告警, 不阻塞启动; 修法: bash /data/openpilot/panda_版本核对/panda_维护.sh
if [ -f /data/openpilot/panda_版本核对/panda_boot_check.sh ]; then
  bash /data/openpilot/panda_版本核对/panda_boot_check.sh || true
elif [ -f /data/openpilot/deps/panda_版本核对/panda_boot_check.sh ]; then
  bash /data/openpilot/deps/panda_版本核对/panda_boot_check.sh || true
fi

# ---- AGX Orin 环境变量 (对齐 sp aunch_pc.sh) ----
export USE_WEBCAM=1                 # 走 Python webcamerad (V4L2/VIC 链路)
export GMSL_WEBCAM=1
export ROAD_CAM="${ROAD_CAM:-0}"    # /dev/video0 = road(前视)
export WIDE_CAM="${WIDE_CAM:-1}"    # /dev/video1 = wide(广角)
export ROAD_CAM_FRAMERATE="${ROAD_CAM_FRAMERATE:-20}"
export WIDE_CAM_FRAMERATE="${WIDE_CAM_FRAMERATE:-20}"
export NO_DM=1                      # 禁用驾驶员监控(需要额外模型)
export FULLSCREEN="${FULLSCREEN:-1}"

# nvgpu libcuda 必须最先出现 (缺失则 TRT 静默回退 tinygrad CPU 慢路径)
case "$(uname -m)" in
  aarch64) ACADOS_ARCH=aarch64 ;;
  x86_64)  ACADOS_ARCH=x86_64 ;;
  *)       ACADOS_ARCH=aarch64 ;;
esac
export LD_LIBRARY_PATH="/data/opt/nvidia/l4t-gpu-libs/nvgpu:$DIR/third_party/acados/$ACADOS_ARCH/lib:$DIR/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code:$DIR/selfdrive/controls/lib/lateral_mpc_lib/c_generated_code:$DIR/opendbc_repo/opendbc/can:${LD_LIBRARY_PATH:-}"

# VIC 时钟锁定 (闲时降频导致首帧转换延迟大)
if sudo -n true 2>/dev/null; then
  echo 358400000 | sudo tee /sys/class/devfreq/15340000.vic/min_freq > /dev/null 2>&1 || true
fi

# ---- 森云相机驱动初始化 (tw_camera_cfg: 配置 MAX9295A/IMX390 serdes, 必须先于 camerad) ----
# 不先跑它 VI 设备无图像输出, UI 黑屏。绑定脚本生命周期: 启动时拉起, 退出时 trap 清理。
if pgrep -x tw_camera_cfg > /dev/null 2>&1; then
  echo "[camera] tw_camera_cfg 已在运行, 跳过启动"
else
  echo "[camera] 启动森云相机驱动初始化 (tw_camera_cfg)..."
  sudo tw_camera_cfg &
  TW_CAM_PID=$!
  # 等待 serdes/i2c 初始化完成(探测到 /dev/i2c-9..12 被打开即视为就绪, 兜底最多 8 秒)
  for i in $(seq 1 16); do
    if sudo lsof -p "$(pgrep -xn tw_camera_cfg)" 2>/dev/null | grep -q "/dev/i2c-9"; then
      echo "[camera] 相机驱动就绪 (i2c 已初始化)"
      break
    fi
    sleep 0.5
  done
fi

# ---- 生命周期绑定: 脚本退出时连同 tw_camera_cfg / fake_panda 一起清理 ----
cleanup() {
  echo ""
  echo "[camera] 停止森云相机驱动 (tw_camera_cfg)..."
  pkill -f "tw_camera_cfg" 2>/dev/null || true
  echo "[fake_panda] 停止模拟 CAN (fake_panda_state)..."
  pkill -f "fake_panda_state" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- 模拟 CAN（fake panda），适配无 Panda 硬件的桌面/开发环境 ----
# 默认关闭（实车模式，走真实 panda）；桌面模拟: ENABLE_FAKE_PANDA=1 ./launch_openpilot.sh
if [ "${ENABLE_FAKE_PANDA:-1}" = "1" ]; then
  echo "启用 fake panda 模式（NOBOARD=1, SKIP_FW_QUERY=1）"
  export NOBOARD=1
  export SKIP_FW_QUERY=1
  export FINGERPRINT="${FINGERPRINT:-LEXUS_ES_TSS2}"  # 强制 fingerprint：雷克萨斯 ES 2019-24（ES200 2023）
  export BLOCK="${BLOCK},pandad"
  export FAKE_CAN_PROFILE="${FAKE_CAN_PROFILE:-toyota}"
  export FAKE_CAN_HZ="${FAKE_CAN_HZ:-100}"  # ES TSS2 parser 要求 KINEMATICS/WHEEL_SPEEDS 80Hz，50Hz 不够
  export FAKE_PANDA_SAFETY_MODEL="${FAKE_PANDA_SAFETY_MODEL:-toyota}"
  python3 "$DIR/tools/sim/fake_panda_state.py" &
  FAKE_PANDA_PID=$!
  sleep 1
fi

echo "启动配置: ROAD_CAM=/dev/video${ROAD_CAM} WIDE_CAM=/dev/video${WIDE_CAM} FAKE_CAN=${FAKE_CAN_PROFILE:-off}"
echo "================================================"

# ---- 启动 manager(完整进程树) ----
cd system/manager
if [ ! -f "$DIR/prebuilt" ]; then
  ./build.py
fi
./manager.py
rc=$?

# manager 退出(无论正常/崩溃)就结束脚本, 让 trap cleanup 收尾。
echo "[launch] manager.py 已退出 (rc=$rc)"
exit $rc

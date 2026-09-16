#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

function agnos_init {
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta

  # set success flag for current boot slot
  sudo abctl --set_success

  # TODO: do this without udev in AGNOS
  # udev does this, but sometimes we startup faster
  sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

  # Check if AGNOS update is required
  if [ $(< /VERSION) != "$AGNOS_VERSION" ]; then
    AGNOS_PY="$DIR/system/hardware/tici/agnos.py"
    MANIFEST="$DIR/system/hardware/tici/agnos.json"
    if $AGNOS_PY --verify $MANIFEST; then
      sudo reboot
    fi
    $DIR/system/hardware/tici/updater $AGNOS_PY $MANIFEST
  fi
}

function launch {
  # Remove orphaned git lock if it exists on boot
  [ -f "$DIR/.git/index.lock" ] && rm -f $DIR/.git/index.lock

  # use project virtualenv when running off-device
  if [ ! -f /AGNOS ] && [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
  fi

  # ---- AGX Orin GMSL 相机链路 (对齐 sp aunch_pc.sh) ----
  # USE_WEBCAM=1 → 走 Python webcamerad (tools/webcam/camerad.py):
  #   V4L2 采集 UYVY → VIC 硬件转 NV12 → FrameSync 20fps → VisionIPC → modeld(CUDA/TRT)
  # 注意: 不能设 DISABLE_CUDA_TRANSFORM=1 (那是 sp 的 USB 摄像头 PC 模式; 本机走 CUDA 变换)
  if [ ! -f /AGNOS ]; then
    export USE_WEBCAM=1
    export GMSL_WEBCAM=1
    export ROAD_CAM="${ROAD_CAM:-0}"
    export WIDE_CAM="${WIDE_CAM:-1}"
    export ROAD_CAM_FRAMERATE="${ROAD_CAM_FRAMERATE:-20}"
    export WIDE_CAM_FRAMERATE="${WIDE_CAM_FRAMERATE:-20}"
    export NO_DM=1            # 禁用驾驶员监控(需要额外模型)
    export FULLSCREEN="${FULLSCREEN:-1}"

    # nvgpu libcuda 必须最先出现 (缺失则 TRT 静默回退 tinygrad CPU 慢路径)
    case "$(uname -m)" in
      aarch64) ACADOS_ARCH=aarch64 ;;
      x86_64)  ACADOS_ARCH=x86_64 ;;
      *)       ACADOS_ARCH=aarch64 ;;
    esac
    export LD_LIBRARY_PATH="/data/opt/nvidia/l4t-gpu-libs/nvgpu:$DIR/third_party/acados/$ACADOS_ARCH/lib:$DIR/selfdrive/controls/lib/longitudinal_mpc_lib/c_generated_code:$DIR/selfdrive/controls/lib/lateral_mpc_lib/c_generated_code:$DIR/opendbc_repo/opendbc/can:${LD_LIBRARY_PATH:-}"
    # VIC 时钟锁定 (tegra_wmark governor 闲时降频, 首帧转换延迟大)
    if sudo -n true 2>/dev/null; then
      echo 358400000 | sudo tee /sys/class/devfreq/15340000.vic/min_freq > /dev/null 2>&1 || true
    fi
  fi

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The DIR init file has to exist, with a newer modtime than anything in
  #    the DIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${DIR}/.overlay_init" ]; then
    find ${DIR}/.git -newer ${DIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${DIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $DIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $DIR
          cd $DIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  export PYTHONPATH="$PWD"

  # hardware specific init
  if [ -f /AGNOS ]; then
    agnos_init
  fi

  # write tmux scrollback to a file (无 tmux 时静默跳过)
  if command -v tmux > /dev/null 2>&1 && tmux info > /dev/null 2>&1; then
    tmux capture-pane -pq -S-1500 > /tmp/launch_log 2>/dev/null
  fi
  # 依赖检查: 已装则完全静默, 只提示缺失的
  for mod in flask shapely kaitaistruct sounddevice pyaudio inputs; do
    if ! python -c "import $mod" > /dev/null 2>&1; then
      echo "$mod installing."
      pip install $mod
    fi
  done

  # events language init
  # 语言读取: AGNOS 真机读 /data/params, PC 模式读 $HOME/.comma/params
  # (cpv9 适配: 无 git 仓库, 不依赖 git status 判断, 直接按 LANG 强制覆盖)
  if [ -f /data/params/d/LanguageSetting ]; then
    LANG=$(cat /data/params/d/LanguageSetting)
  elif [ -f "${PARAMS_ROOT:-$HOME/.comma/params}/d/LanguageSetting" ]; then
    LANG=$(cat "${PARAMS_ROOT:-$HOME/.comma/params}/d/LanguageSetting")
  fi

  # events.py 语言切换 (events_en.py 保留英文原版备份)
  if [ "${LANG}" = "main_zh-CHS" ]; then
    # Backup current events.py (assumed English) and install Simplified Chinese events
    if [ ! -f $DIR/scripts/add/events_en.py ]; then
      cp -f $DIR/selfdrive/selfdrived/events.py $DIR/scripts/add/events_en.py
    fi
    cp -f $DIR/scripts/add/events_zh.py $DIR/selfdrive/selfdrived/events.py
  elif [ "${LANG}" = "main_ko" ]; then
    if [ ! -f $DIR/scripts/add/events_en.py ]; then
      cp -f $DIR/selfdrive/selfdrived/events.py $DIR/scripts/add/events_en.py
    fi
    cp -f $DIR/scripts/add/events_ko.py $DIR/selfdrive/selfdrived/events.py
  elif [ "${LANG}" = "main_en" ] && [ -f $DIR/scripts/add/events_en.py ]; then
    cp -f $DIR/scripts/add/events_en.py $DIR/selfdrive/selfdrived/events.py
  fi

  # c3xl amplifier file change
  C3XL=$(cat /data/params/d/HardwareC3xLite 2>/dev/null || echo 0)
  if [ "${C3XL}" = "1" ]; then
    cp -f $DIR/system/hardware/tici/amplifier.py $DIR/scripts/add/amplifier_org.py
    cp -f $DIR/scripts/add/amplifier_c3xl.py $DIR/system/hardware/tici/amplifier.py
  elif [ "${C3XL}" = "0" ] && [ -f $DIR/scripts/add/amplifier_org.py ]; then
    cp -f $DIR/scripts/add/amplifier_org.py $DIR/system/hardware/tici/amplifier.py
  fi

  # ---- panda 基准自检(快 <0.1s): 固件字节对齐 + 协议常量检查 ----
  # 让 cpv9 也统一到 cuda 那份 panda 固件(字节一致 -> 签名一致 -> 不再回刷)
  # 若提示基准不一致 -> bash /data/openpilot/deps/panda_版本核对/panda_维护.sh
  if [ -f /data/openpilot/deps/panda_版本核对/panda_boot_check.sh ]; then
    bash /data/openpilot/deps/panda_版本核对/panda_boot_check.sh || true
  elif [ -f /data/openpilot/panda_版本核对/panda_boot_check.sh ]; then
    bash /data/openpilot/panda_版本核对/panda_boot_check.sh || true
  fi

  # ---- 森云相机驱动初始化 (tw_camera_cfg: 配置 MAX9295A/IMX390 serdes, 必须先于 camerad 运行) ----
  # 移植自 sunnypilot-cuda/aunch_pc.sh: 不先跑它 VI 设备无图像输出, UI 黑屏。
  # 绑定脚本生命周期: 启动时拉起, 脚本退出时由 trap 清理。
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

  # 生命周期绑定: 脚本退出时连同 tw_camera_cfg / fake_panda 一起清理
  cleanup() {
    echo ""
    echo "[camera] 停止森云相机驱动 (tw_camera_cfg)..."
    pkill -f "tw_camera_cfg" 2>/dev/null || true
    echo "[fake_panda] 停止模拟 CAN (fake_panda_state)..."
    pkill -f "fake_panda_state" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  # ---- 模拟 CAN（fake panda），适配无 Panda 硬件的 PC 开发环境 ----
  # 默认关闭（实车模式，走真实 panda）；桌面模拟需显式打开: ENABLE_FAKE_PANDA=1 ./launch_openpilot.sh
  if [ "${ENABLE_FAKE_PANDA:-0}" = "1" ]; then
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

  # start manager
  cd system/manager
  if [ ! -f $DIR/prebuilt ]; then
    ./build.py
  fi
  ./manager.py
  rc=$?

  # manager 退出(无论正常/崩溃)就结束脚本, 让 trap cleanup 收尾。
  # 原来这里 while true 死循环导致 Ctrl+C 停不下来。
  echo "[launch] manager.py 已退出 (rc=$rc)"
  exit $rc
}

launch

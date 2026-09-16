#!/usr/bin/env python3
import os
import signal
import time

import cereal.messaging as messaging
from opendbc.can.packer import CANPacker
from opendbc.car.honda.values import HondaSafetyFlags
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp

RUNNING = True


def handle_signal(signum, frame):
  global RUNNING
  RUNNING = False


def env_bool(name: str, default: bool) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.lower() in ("1", "true", "yes", "on")


def build_honda_can_frames(packer: CANPacker, speed_kph: float, left_blinker: bool, right_blinker: bool):
  speed = max(0.0, speed_kph)
  is_moving = 1 if speed >= 1.0 else 0

  frames = []
  frames.append(packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": speed}))
  frames.append(packer.make_can_msg("WHEEL_SPEEDS", 0, {
    "WHEEL_SPEED_FL": speed,
    "WHEEL_SPEED_FR": speed,
    "WHEEL_SPEED_RL": speed,
    "WHEEL_SPEED_RR": speed,
  }))
  frames.append(packer.make_can_msg("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": 0}))
  # GEARBOX exists in honda_civic_ex_2022_can_generated; GEARBOX_CVT does NOT
  frames.append(packer.make_can_msg("GEARBOX", 0, {"SELECTED_D": 1, "GEAR_SHIFTER": 8}))
  frames.append(packer.make_can_msg("GAS_PEDAL_2", 0, {}))
  frames.append(packer.make_can_msg("SEATBELT_STATUS", 0, {"SEATBELT_DRIVER_LATCHED": 1}))
  frames.append(packer.make_can_msg("STEER_STATUS", 0, {"STEER_TORQUE_SENSOR": 0}))
  frames.append(packer.make_can_msg("STEERING_SENSORS", 0, {"STEER_ANGLE": 0}))
  frames.append(packer.make_can_msg("VSA_STATUS", 0, {}))
  frames.append(packer.make_can_msg("STANDSTILL", 0, {"WHEELS_MOVING": is_moving}))
  frames.append(packer.make_can_msg("STEER_MOTOR_TORQUE", 0, {}))
  frames.append(packer.make_can_msg("EPB_STATUS", 0, {}))
  frames.append(packer.make_can_msg("DOORS_STATUS", 0, {}))
  frames.append(packer.make_can_msg("CRUISE", 0, {}))
  frames.append(packer.make_can_msg("CRUISE_FAULT_STATUS", 0, {}))
  frames.append(packer.make_can_msg("SCM_FEEDBACK", 0, {
    "MAIN_ON": 1,
    "LEFT_BLINKER": left_blinker,
    "RIGHT_BLINKER": right_blinker,
  }))
  frames.append(packer.make_can_msg("POWERTRAIN_DATA", 0, {
    "ACC_STATUS": 0,
    "PEDAL_GAS": 0,
    "ENGINE_RPM": speed * 30.0,
    "GAS_PRESSED": 1 if speed > 1.0 else 0,
  }))
  frames.append(packer.make_can_msg("CAR_SPEED", 0, {
    "CAR_SPEED": speed,
    "ROUGH_CAR_SPEED": round(speed),
  }))

  # Camera bus frames expected by Honda radarless pipeline.
  frames.append(packer.make_can_msg("STEERING_CONTROL", 2, {}))
  frames.append(packer.make_can_msg("ACC_HUD", 2, {}))
  frames.append(packer.make_can_msg("LKAS_HUD", 2, {}))
  return frames


def build_toyota_can_frames(packer: CANPacker, speed_kph: float = 0.0, left_blinker: bool = False, right_blinker: bool = False):
  # Keep a stable set of Toyota PT/CAM frames flowing so card parses valid bus data.
  speed = max(0.0, speed_kph)
  frames = [
    packer.make_can_msg("VSC1S07", 0, {}),
    packer.make_can_msg("BODY_CONTROL_STATE", 0, {}),
    packer.make_can_msg("BRAKE_MODULE", 0, {}),
    packer.make_can_msg("ESP_CONTROL", 0, {}),
    packer.make_can_msg("PCM_CRUISE", 0, {}),
    packer.make_can_msg("GEAR_PACKET", 0, {
      "GEAR": 4,  # D
    }),
    packer.make_can_msg("ENGINE_RPM", 0, {
      "RPM": speed * 30.0,  # 粗略估算，100km/h ~ 3000rpm
    }),
    packer.make_can_msg("WHEEL_SPEEDS", 0, {
      "WHEEL_SPEED_FL": speed,
      "WHEEL_SPEED_FR": speed,
      "WHEEL_SPEED_RL": speed,
      "WHEEL_SPEED_RR": speed,
    }),
    packer.make_can_msg("STEER_ANGLE_SENSOR", 0, {}),
    packer.make_can_msg("STEER_TORQUE_SENSOR", 0, {}),
    packer.make_can_msg("EPS_STATUS", 0, {}),
    packer.make_can_msg("KINEMATICS", 0, {
      "YAW_RATE": 0.0,   # 必须，carstate.py 用它算 yawRate
      "ACCEL_X": 0.0,
      "ACCEL_Y": 0.0,
    }),
    packer.make_can_msg("BODY_CONTROL_STATE_2", 0, {}),
    packer.make_can_msg("PCM_CRUISE_2", 0, {}),
    packer.make_can_msg("PCM_CRUISE_SM", 0, {}),
    packer.make_can_msg("LIGHT_STALK", 0, {}),
    packer.make_can_msg("BLINKERS_STATE", 0, {
      "TURN_SIGNALS": 0,
    }),
    packer.make_can_msg("PRE_COLLISION", 2, {}),
    packer.make_can_msg("ACC_CONTROL", 2, {}),
    packer.make_can_msg("PCS_HUD", 2, {}),
    packer.make_can_msg("LKAS_HUD", 2, {}),
  ]
  return frames


def main() -> None:
  signal.signal(signal.SIGINT, handle_signal)
  signal.signal(signal.SIGTERM, handle_signal)

  panda_type = os.getenv("FAKE_PANDA_TYPE", "blackPanda")
  can_enabled = env_bool("FAKE_CAN_ENABLED", True)
  can_profile = os.getenv("FAKE_CAN_PROFILE", "toyota").strip().lower()

  safety_model_env = os.getenv("FAKE_PANDA_SAFETY_MODEL")
  if safety_model_env is None and can_enabled and can_profile == "honda":
    safety_model = "hondaBosch"
  else:
    safety_model = safety_model_env or "toyota"

  default_safety_param = "0"
  if safety_model == "hondaBosch":
    default_safety_param = str(HondaSafetyFlags.RADARLESS.value | HondaSafetyFlags.BOSCH_LONG.value)
  safety_param = int(os.getenv("FAKE_PANDA_SAFETY_PARAM", default_safety_param))
  alternative_experience = int(os.getenv("FAKE_PANDA_ALT_EXP", "0"))
  ignition = env_bool("FAKE_PANDA_IGNITION", True)

  panda_hz = float(os.getenv("FAKE_PANDA_HZ", "10"))
  periph_hz = float(os.getenv("FAKE_PERIPHERAL_HZ", "2"))
  can_hz = float(os.getenv("FAKE_CAN_HZ", "50"))
  can_speed_kph = float(os.getenv("FAKE_CAN_SPEED_KPH", "0"))
  can_left_blinker = env_bool("FAKE_CAN_LEFT_BLINKER", False)
  can_right_blinker = env_bool("FAKE_CAN_RIGHT_BLINKER", False)

  panda_interval = 1.0 / panda_hz if panda_hz > 0 else 0.1
  periph_interval = 1.0 / periph_hz if periph_hz > 0 else 0.5
  can_interval = 1.0 / can_hz if can_hz > 0 else 0.02

  services = ["pandaStates", "peripheralState"]
  if can_enabled:
    services.append("can")
  pm = messaging.PubMaster(services)

  can_packer = None
  if can_enabled:
    can_dbc = "toyota_nodsu_pt_generated" if can_profile != "honda" else "honda_civic_ex_2022_can_generated"
    can_packer = CANPacker(can_dbc)

  print(f"[fake_panda] start: pandaType={panda_type}, safetyModel={safety_model}, ignition={ignition}, can={can_enabled}, profile={can_profile}")

  last_periph = 0.0
  last_can = 0.0
  while RUNNING:
    now = time.monotonic()

    panda_msg = messaging.new_message("pandaStates", 1)
    panda_msg.valid = True
    panda_msg.pandaStates[0] = {
      "ignitionLine": ignition,
      "ignitionCan": ignition,
      "controlsAllowed": True,
      "pandaType": panda_type,
      "safetyModel": safety_model,
      "safetyParam": safety_param,
      "alternativeExperience": alternative_experience,
      "voltage": 12000,
      "current": 500,
      "powerSaveEnabled": False,
      "heartbeatLost": False,
      "safetyRxChecksInvalid": False,
    }
    pm.send("pandaStates", panda_msg)

    if now - last_periph >= periph_interval:
      periph_msg = messaging.new_message("peripheralState")
      periph_msg.valid = True
      periph_msg.peripheralState = {
        "pandaType": panda_type,
        "voltage": 12000,
        "current": 500,
        "fanSpeedRpm": 1000,
      }
      pm.send("peripheralState", periph_msg)
      last_periph = now

    if can_enabled and (now - last_can >= can_interval):
      if can_profile == "honda":
        can_frames = build_honda_can_frames(can_packer, can_speed_kph, can_left_blinker, can_right_blinker)
      else:
        # 传 speed/转向灯参数 (之前写死默认值, FAKE_CAN_SPEED_KPH 对 toyota 无效)
        can_frames = build_toyota_can_frames(can_packer, can_speed_kph, can_left_blinker, can_right_blinker)
      pm.send("can", can_list_to_can_capnp(can_frames))
      last_can = now

    time.sleep(panda_interval)

  print("[fake_panda] stopped")


if __name__ == "__main__":
  main()      

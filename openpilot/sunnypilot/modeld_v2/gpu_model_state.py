"""
GpuModelState — CUDA/TensorRT model state for openpilot.

Supports both new (sunnypilot) and old (openpilot) directory layouts.
All heavy imports are deferred to __init__ to avoid import-time failures.
"""
import os
import sys

import numpy as np


def _lazy_import(module_path, fallback_path=None):
  """Import a module lazily, trying new path first, then fallback."""
  try:
    return __import__(module_path, fromlist=[''])
  except ImportError:
    if fallback_path:
      return __import__(fallback_path, fromlist=[''])
    raise


def _import_attr(mod_path, attr, fallback_mod_path=None, fallback_attr=None):
  """Import an attribute from a module, with fallback path."""
  try:
    mod = __import__(mod_path, fromlist=[''])
    return getattr(mod, attr)
  except (ImportError, AttributeError):
    if fallback_mod_path:
      mod = __import__(fallback_mod_path, fromlist=[''])
      return getattr(mod, fallback_attr or attr)
    raise


class GpuModelState:
  """CUDA-first ModelState with tinygrad fallback."""

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False, model_name: str | None = None):
    # Lazy imports — only happen when GpuModelState is actually instantiated
    log = _import_attr("openpilot.cereal", "log", "cereal", "log")
    self._log = log

    Params = _import_attr("openpilot.common.params", "Params", "common.params", "Params")
    cloudlog = _import_attr("openpilot.common.swaglog", "cloudlog", "common.swaglog", "cloudlog")
    self._cloudlog = cloudlog

    drive_helpers = _lazy_import(
      "openpilot.selfdrive.controls.lib.drive_helpers",
      "selfdrive.controls.lib.drive_helpers",
    )
    self._get_accel_from_plan = drive_helpers.get_accel_from_plan
    self._get_curvature_from_plan = drive_helpers.get_curvature_from_plan
    self._smooth_value = drive_helpers.smooth_value
    # Some forks (e.g. Carrot) don't ship `should_stop`; fall back to a
    # minimal v_ego based stop heuristic so get_action_from_model works.
    self._should_stop = getattr(drive_helpers, "should_stop", None) or (
      lambda v_ego, a: a < -1.0 and v_ego < 1.0
    )

    ModelStateBase = _import_attr(
      "openpilot.sunnypilot.modeld_v2.modeld_base", "ModelStateBase",
      "openpilot.sunnypilot.modeld_v2.compat.modeld_base", "ModelStateBase",
    )

    # Import GPU backend (may be None if not on Orin)
    try:
      backend_mod = _lazy_import(
        "openpilot.sunnypilot.modeld_v2.gpu_backend.backend",
      )
      self._create_gpu_backend = backend_mod.create_gpu_backend
    except ImportError:
      self._create_gpu_backend = None

    # Import profile resolver
    try:
      registry = _lazy_import(
        "openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry",
      )
      self._resolve_profile = registry.resolve_profile
    except ImportError:
      try:
        compat_registry = _lazy_import(
          "openpilot.sunnypilot.modeld_v2.compat.registry",
        )
        self._resolve_profile = compat_registry.resolve_profile
      except ImportError:
        self._resolve_profile = None

    # Import constants
    # Priority: sunnypilot modeld_v2 -> fork's own selfdrive.modeld module ->
    # compat. Old-layout forks (Carrot / dp) keep ModelConstants/Plan in
    # openpilot.selfdrive.modeld.constants with DESIRE_LEN=8 etc.
    try:
      constants_mod = _lazy_import(
        "openpilot.sunnypilot.modeld_v2.constants",
      )
      self._ModelConstants = constants_mod.ModelConstants
      self._Plan = constants_mod.Plan
    except ImportError:
      try:
        legacy_constants = _lazy_import(
          "openpilot.selfdrive.modeld.constants",
          "selfdrive.modeld.constants",
        )
        self._ModelConstants = legacy_constants.ModelConstants
        self._Plan = legacy_constants.Plan
      except ImportError:
        compat_constants = _lazy_import(
          "openpilot.sunnypilot.modeld_v2.compat.constants",
        )
        self._ModelConstants = compat_constants.ModelConstants
        self._Plan = compat_constants.Plan

    # Import parsers
    # Old-layout forks ship their own MDN parsing in selfdrive.modeld; prefer
    # it over the pass-through compat parsers so outputs match fill_model_msg.
    try:
      parse_mod = _lazy_import(
        "openpilot.sunnypilot.modeld_v2.parse_model_outputs",
      )
      self._CombinedParser = parse_mod.Parser
    except ImportError:
      try:
        legacy_parse = _lazy_import(
          "openpilot.selfdrive.modeld.parse_model_outputs",
          "selfdrive.modeld.parse_model_outputs",
        )
        self._CombinedParser = legacy_parse.Parser
      except ImportError:
        compat_parse = _lazy_import(
          "openpilot.sunnypilot.modeld_v2.compat.parse_model_outputs",
        )
        self._CombinedParser = compat_parse.Parser

    try:
      parse_split_mod = _lazy_import(
        "openpilot.sunnypilot.modeld_v2.parse_model_outputs_split",
      )
      self._SplitParser = parse_split_mod.Parser
    except ImportError:
      try:
        legacy_split = _lazy_import(
          "openpilot.selfdrive.modeld.parse_model_outputs",
          "selfdrive.modeld.parse_model_outputs",
        )
        self._SplitParser = legacy_split.Parser
      except ImportError:
        compat_parse_split = _lazy_import(
          "openpilot.sunnypilot.modeld_v2.compat.parse_model_outputs_split",
        )
        self._SplitParser = compat_parse_split.Parser

    # --- Initialization ---
    self.chestnut = chestnut
    try:
      if model_name is None:
        model_name = os.getenv("MODEL_NAME") or Params().get("Model", encoding="utf-8")
    except Exception:
      # Some forks (e.g. Carrot) use the cython Params which lacks `encoding`
      # and has no "Model" key (the model is always the bundled classic one).
      if model_name is None:
        model_name = os.getenv("MODEL_NAME")

    if self._resolve_profile is None:
      raise RuntimeError("No profile resolver available")
    self.profile = self._resolve_profile(model_name)
    if self.profile is None:
      raise RuntimeError(f"No usable CUDA model profile for {model_name or 'default'}")

    if self._create_gpu_backend is None:
      raise RuntimeError("CUDA backend module not available")
    # TrtRunner + CudaTransform both talk to the CUDA driver directly and need
    # a live context on the current thread. Initialize tinygrad's CUDA device
    # (which creates a primary context with cuDevicePrimaryCtxRetain) so all the
    # raw driver calls below run inside it.
    try:
      os.environ["DEV"] = os.getenv("DEV", "CUDA")
      from tinygrad import Device as _TinygradDevice
      _TinygradDevice["CUDA"]
      cloudlog.warning("tinygrad CUDA device initialized")
    except Exception:
      cloudlog.exception("tinygrad CUDA device init failed; CUDA backend will fail")
    self.backend = self._create_gpu_backend(self.profile)
    if self.backend is None:
      raise RuntimeError("CUDA backend unavailable (not AGX Orin or TensorRT missing)")

    self.constants = self._ModelConstants
    self.vision_input_names = list(self.profile.vision_input_names)
    self.road_key = next(k for k in self.vision_input_names if "big" not in k)
    self.wide_key = next(k for k in self.vision_input_names if "big" in k)
    self.numpy_inputs: dict[str, np.ndarray] = {}
    for k, shape in self.profile.input_shapes.items():
      if k in self.vision_input_names:
        continue
      self.numpy_inputs[k] = np.zeros(shape, dtype=np.float32)
    desire_candidates = [k for k in self.numpy_inputs if k.startswith("desire")]
    self.desire_key = desire_candidates[0] if desire_candidates else "desire"
    # Size prev_desire from the profile's desire feature dim (8 for BigCombo/
    # FiletOFish), not the fork's ModelConstants.DESIRE_LEN (may differ).
    desire_shape = self.numpy_inputs.get(self.desire_key)
    desire_last = desire_shape.shape[-1] if desire_shape is not None else self.constants.DESIRE_LEN
    self.prev_desire = np.zeros(desire_last, dtype=np.float32)
    self.lat_delay = 0.0
    self.LAT_SMOOTH_SECONDS = 0.0
    self.LONG_SMOOTH_SECONDS = 0.0
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL = 1.0
    self.full_features_buffer: np.ndarray | None = None
    self.parser = self._SplitParser() if self.profile.mode == "split" else self._CombinedParser()
    cloudlog.warning(f"GpuModelState initialized: {model_name} mode={self.profile.mode}")

  @property
  def mlsim(self) -> bool:
    return False

  def slice_outputs(self, model_output: np.ndarray, slices: dict[str, slice]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in slices.items():
      if v.stop is None and v.start is not None and v.start < 0:
        continue
      if v.stop is None:
        continue
      out[k] = model_output[np.newaxis, v]
    return out

  def _update_temporal(self, raw_output: np.ndarray) -> None:
    # Classic Carrot keeps hidden_state in the vision output; backend.infer_split
    # already shifts+appends features_buffer for split mode. For merged models it
    # may live in policy slices. Only update if the key is present and for split
    # mode the backend already did the shift (here it would double-shift).
    if self.profile.mode == "split":
      return
    slices = self.profile.policy_slices
    if "hidden_state" not in slices:
      slices = self.profile.vision_slices
    if "hidden_state" not in slices:
      return
    hidden = raw_output[slices["hidden_state"]]
    feats = self.numpy_inputs.get("features_buffer")
    if feats is None:
      return
    feats[0, :-1] = feats[0, 1:]
    feats[0, -1] = hidden[: self.profile.temporal.features_len]

  def run(self, bufs: dict[str, object], transforms: dict[str, np.ndarray],
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    frames = {k: bufs[k] for k in self.vision_input_names if k in bufs}
    self.backend.preprocess(frames, transforms)

    # Dropped-frame tick: update image transform/temporal state but skip inference.
    if prepare_only:
      return None

    # modeld always publishes the canonical "desire" key; the profile may name
    # it differently (e.g. BigCombo uses "desire_pulse"). Map it explicitly so
    # the desire input is never silently zeroed.
    src_desire = "desire" if "desire" in inputs else self.desire_key
    if self.desire_key in self.numpy_inputs and src_desire in inputs:
      cur = np.asarray(inputs[src_desire]).reshape(-1)
      current = cur[-self.prev_desire.shape[0]:]
      pulse = np.where(current - self.prev_desire > 0.99, current, 0)
      target = self.numpy_inputs[self.desire_key]
      if target.ndim >= 2:
        target[..., :-1, :] = target[..., 1:, :]
        target[..., -1, :] = pulse
      else:
        target[:] = pulse
      self.prev_desire[:] = current

    for k, v in inputs.items():
      if k in self.numpy_inputs and k != self.desire_key:
        self.numpy_inputs[k][:] = v

    scalar_inputs = {k: v for k, v in self.numpy_inputs.items()}

    if self.profile.mode == "merged":
      raw = self.backend.infer_merged(scalar_inputs)
      # Merged engine: features_buffer is NOT updated inside the backend, so we
      # shift it here from the output hidden_state.
      self._update_temporal(raw)
    else:
      v_out, p_out = self.backend.infer_split(scalar_inputs)
      raw = np.concatenate([v_out, p_out])
      # Split path: infer_split() already shifted + wrote features_buffer from
      # the vision hidden_state — do NOT update again (would double-shift).

    if self.profile.mode == "merged":
      sliced = self.slice_outputs(raw, {**self.profile.vision_slices, **self.profile.policy_slices})
      outputs = self.parser.parse_outputs(sliced)
    else:
      vision_output = raw[: self._split_vision_size()]
      policy_output = raw[self._split_vision_size():]
      vision_sliced = self.slice_outputs(vision_output, self.profile.vision_slices)
      policy_sliced = self.slice_outputs(policy_output, self.profile.policy_slices)
      outputs = self.parser.parse_vision_outputs(vision_sliced)
      outputs.update(self.parser.parse_policy_outputs(policy_sliced))

    if self.chestnut and not np.all(np.isfinite(outputs.get("plan", np.array([0.0])))):
      self._cloudlog.error("model output not finite, dropping frame")
      return None
    return outputs

  def _split_vision_size(self) -> int:
    total = 0
    for s in self.profile.vision_slices.values():
      if s.stop is None:
        continue
      total += s.stop - s.start
    return total

  def get_action_from_model(self, model_output, prev_action, lat_action_t, long_action_t, v_ego):
    Plan = self._Plan
    if "action" not in model_output:
      plan = model_output["plan"][0]
      desired_accel = self._get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0],
                                          self.constants.T_IDXS, action_t=long_action_t)
      desired_curvature = self._get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                                  plan[:, Plan.ORIENTATION_RATE][:, 2],
                                                  self.constants.T_IDXS, v_ego, action_t=lat_action_t)
    else:
      desired_accel = model_output["action"][0, 1]
      desired_curvature = model_output["action"][0, 0] / (max(1.0, v_ego)) ** 2

    stop = self._should_stop(v_ego, desired_accel)
    desired_accel = self._smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)
    if v_ego > self.MIN_LAT_CONTROL_SPEED:
      desired_curvature = self._smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature
    return self._log.ModelDataV2.Action(
      desiredCurvature=float(desired_curvature),
      desiredAcceleration=float(desired_accel),
      shouldStop=bool(stop),
    )


# ═══════════════════════════════════════════════════════════════════════
# FailoverModelState — 大模型主跑 + 小模型热备,异常时小模型第一时间接管
#
# 官方思路:运行大模型时,小模型在内存热备;大模型输出异常(非有限值/
# 冻结/推理异常)时第一时间切小模型接管;同一帧只发布一个源的输出
# (modeld 是唯一 modelV2/drivingModelData 发布者,单源互斥天然成立)。
#
# 策略(与用户确认,2026-09-16):
#   - 备模型平时每 FB_SELF_CHECK_EVERY 帧跑一次(维持时序状态,不发布)
#   - 主异常 → 立即切备,备每帧推理发布;主模型降频复测
#   - 主连续 RECOVER_FRAMES 帧健康 → 自动切回
# ═══════════════════════════════════════════════════════════════════════

FB_SELF_CHECK_EVERY = 15    # 备模型低频自检间隔(帧)
FREEZE_FRAMES = 5           # 主连续 N 帧输出几乎不变 → 判冻结
FREEZE_EPS = 1e-6           # 冻结判定阈值(比特级冻结远小于此)
RECOVER_FRAMES = 50         # 主模型连续健康帧数 → 自动切回
RECOVER_CHECK_EVERY = 5     # fallback 期间主模型复测间隔(帧),控制双跑开销

_HEALTH_KEYS = ("plan", "lane_lines", "lane_lines_prob", "meta")


def _flatten_core(outputs: dict) -> np.ndarray | None:
  """把健康检测关心的核心输出拼成一个 flat 数组,用于非有限/冻结判定。"""
  parts = []
  for k in _HEALTH_KEYS:
    v = outputs.get(k)
    if v is None:
      continue
    a = np.asarray(v, dtype=np.float32)
    if a.size:
      parts.append(a.reshape(-1))
  return np.concatenate(parts) if parts else None


class FailoverModelState:
  """GpuModelState 的 failover 包装:主模型 + 热备小模型,单源发布。"""

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    try:
      Params = _import_attr("openpilot.common.params", "Params", "common.params", "Params")
    except Exception:
      Params = None
    cloudlog = _import_attr("openpilot.common.swaglog", "cloudlog", "common.swaglog", "cloudlog")
    self._cloudlog = cloudlog

    # ── 主模型 ──
    primary_name = None
    if Params is not None:
      try:
        primary_name = os.getenv("MODEL_NAME") or Params().get("Model", encoding="utf-8")
      except Exception:
        primary_name = os.getenv("MODEL_NAME")
    self.primary = GpuModelState(cam_w, cam_h, chestnut, model_name=primary_name)
    cloudlog.warning(f"FailoverModelState primary: {self.primary.profile.name}")

    # ── 备模型(热备)──
    fallback_name = ""
    if Params is not None:
      try:
        fallback_name = (Params().get("FallbackModel", encoding="utf-8") or "").strip()
      except Exception:
        fallback_name = ""
    if not fallback_name:
      # 默认组合:大模型主跑 → Classic 热备;Classic 自身主跑 → 无热备
      if self.primary.profile.name not in ("Classic",):
        fallback_name = "Classic"
    self.fallback = None
    if fallback_name and fallback_name != self.primary.profile.name:
      try:
        self.fallback = GpuModelState(cam_w, cam_h, chestnut, model_name=fallback_name)
        cloudlog.warning(f"FailoverModelState fallback ready: {self.fallback.profile.name}")
      except Exception:
        cloudlog.exception(f"Fallback model '{fallback_name}' init failed; running primary-only")
        self.fallback = None

    # ── modeld 主循环按此构造 bufs/transforms(主备并集,各取所需)──
    self.vision_input_names = list(self.primary.vision_input_names)
    if self.fallback is not None:
      for k in self.fallback.vision_input_names:
        if k not in self.vision_input_names:
          self.vision_input_names.append(k)

    self._state = "primary"      # "primary" | "fallback"
    self._tick = 0
    self._healthy_streak = 0
    self._freeze_streak = 0
    self._last_primary_core: np.ndarray | None = None
    self._fallback_fail_streak = 0

  # ── 代理主模型接口(modeld.py 使用)──
  @property
  def profile(self):
    return self.primary.profile

  @property
  def parser(self):
    return self.primary.parser

  @property
  def mlsim(self) -> bool:
    return self.primary.mlsim

  @property
  def desire_key(self):
    return self.primary.desire_key

  @property
  def numpy_inputs(self):
    return self.primary.numpy_inputs

  # ── 健康检测:非有限 + 冻结 ──
  def _primary_healthy(self, outputs: dict) -> bool:
    core = _flatten_core(outputs)
    if core is None or core.size == 0:
      return False
    if not np.all(np.isfinite(core)):
      return False
    if self._last_primary_core is not None and self._last_primary_core.shape == core.shape:
      if np.max(np.abs(core - self._last_primary_core)) < FREEZE_EPS:
        self._freeze_streak += 1
        if self._freeze_streak >= FREEZE_FRAMES:
          self._cloudlog.error(f"[Failover] primary {self.primary.profile.name} output frozen "
                               f"({self._freeze_streak} frames)")
          return False
      else:
        self._freeze_streak = 0
    self._last_primary_core = core
    return True

  # ── 备模型推理(不发布语义由调用方决定)──
  def _fb_run(self, bufs, transforms, inputs, prepare_only: bool) -> dict | None:
    if self.fallback is None:
      return None
    try:
      return self.fallback.run(bufs, transforms, inputs, prepare_only)
    except Exception:
      self._fallback_fail_streak += 1
      if self._fallback_fail_streak <= 2:
        self._cloudlog.exception("[Failover] fallback model run failed")
      return None

  def _maybe_fb_selfcheck(self, bufs, transforms, inputs) -> None:
    """主健康时,备模型低频自检,维持时序状态但不发布。"""
    if self.fallback is None:
      return
    if self._tick % FB_SELF_CHECK_EVERY == 0:
      try:
        self.fallback.run(bufs, transforms, inputs, False)
      except Exception:
        self._cloudlog.exception("[Failover] fallback self-check failed")

  # ── 主入口:单源发布 ──
  def run(self, bufs, transforms, inputs, prepare_only: bool) -> dict | None:
    self._tick += 1
    fb_active = self._state == "fallback"

    # 主模型本次是否推理:primary 状态每帧;fallback 状态降频复测(控制双跑开销)
    run_primary = (not fb_active) or (self._tick % RECOVER_CHECK_EVERY == 0)
    p_out = None
    if run_primary:
      try:
        p_out = self.primary.run(bufs, transforms, inputs, prepare_only)
      except Exception:
        self._cloudlog.exception("[Failover] primary model run failed")
        p_out = None

    if prepare_only:
      return None

    if not fb_active:
      if p_out is not None and self._primary_healthy(p_out):
        self._healthy_streak = 0
        self._maybe_fb_selfcheck(bufs, transforms, inputs)
        return p_out
      # 主模型异常 → 第一时间切备接管
      fb_name = self.fallback.profile.name if self.fallback is not None else "none"
      self._cloudlog.error(f"[Failover] primary {self.primary.profile.name} UNHEALTHY, "
                           f"switching to fallback ({fb_name})")
      self._state = "fallback"
      self._healthy_streak = 0
      self._fallback_fail_streak = 0
      fb = self._fb_run(bufs, transforms, inputs, False)
      if fb is not None:
        return fb
      # 备也不可用:宁可持续输出主结果,不断流(controls 需要连续输入)
      self._cloudlog.error("[Failover] fallback not available either; continuing with primary output")
      return p_out

    # ── fallback 状态:备每帧发布,主降频复测 ──
    fb = self._fb_run(bufs, transforms, inputs, False)
    if run_primary:
      if p_out is not None and self._primary_healthy(p_out):
        self._healthy_streak += 1
        if self._healthy_streak >= RECOVER_FRAMES:
          self._cloudlog.warning(f"[Failover] primary {self.primary.profile.name} recovered "
                                 f"({self._healthy_streak} frames), switching back")
          self._state = "primary"
          self._healthy_streak = 0
          self._freeze_streak = 0
          self._last_primary_core = None
      else:
        self._healthy_streak = 0
    if fb is not None:
      return fb
    return p_out

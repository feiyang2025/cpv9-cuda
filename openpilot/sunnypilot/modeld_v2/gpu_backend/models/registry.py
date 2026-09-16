import pickle
from pathlib import Path

from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile, TemporalMeta

MODELS_ROOT = Path(__file__).parents[4] / "selfdrive" / "modeld" / "models"

# ── merged 模型(单引擎,如 BigCombo)独有的输出头,切分时划给 policy 侧 ──
_MERGED_POLICY_KEYS = ('plan', 'lead', 'lead_prob', 'desire_state', 'action', 'hidden_state', 'pad')


def _load_pkl(path: Path) -> dict:
  with open(path, "rb") as f:
    return pickle.load(f)


def _load_slices(metadata_path: Path) -> dict[str, slice] | None:
  if not metadata_path.exists():
    return None
  try:
    metadata = _load_pkl(metadata_path)
    slices = metadata.get("output_slices")
    if isinstance(slices, dict):
      return slices
  except Exception:
    return None
  return None


def _probe_engine(engine_dir: Path, candidates: list[str]) -> str | None:
  for name in candidates:
    if (engine_dir / name).is_file():
      return name
  return None


def _temporal_from_shapes(policy_input_shapes: dict, merged_input_shapes: dict | None = None) -> TemporalMeta:
  """从 metadata input_shapes 推导时序窗口,避免硬编码 25/512 等常量。"""
  shapes = merged_input_shapes or policy_input_shapes
  feats = shapes.get("features_buffer", (1, 25, 512))
  desire = shapes.get("desire_pulse") or shapes.get("desire", (1, 25, 8))
  return TemporalMeta(
    features_len=feats[-1] if len(feats) >= 2 else 512,
    features_windows=feats[-2] if len(feats) >= 2 else 25,
    desire_len=desire[-1] if len(desire) >= 1 else 8,
    desire_windows=desire[-2] if len(desire) >= 2 else 25,
    features_includes_current=False,
    desire_includes_current=True,
  )


def _scan_dir(engine_dir: Path, name: str) -> ModelProfile | None:
  """扫描一个模型目录(或顶层),从 metadata 自动推导 profile。

  支持三种形态:
  - merged (supercombo 单引擎): <dir>/driving_supercombo_metadata.pkl + <dir>/driving_supercombo*.plan
  - split  (vision+policy 双引擎): <dir>/driving_vision_metadata.pkl + driving_policy_metadata.pkl
           + driving_vision*.plan + driving_policy*.plan
  - 只有 metadata(权重套件,如 0-tr16):注册但 is_ready=False,需先 build_plans.py
  """
  vision_meta_path = engine_dir / "driving_vision_metadata.pkl"
  policy_meta_path = engine_dir / "driving_policy_metadata.pkl"
  super_meta_path = engine_dir / "driving_supercombo_metadata.pkl"

  if super_meta_path.exists():
    # ── merged ──
    merged_slices = _load_slices(super_meta_path)
    if merged_slices is None:
      return None
    try:
      merged_input_shapes = _load_pkl(super_meta_path).get("input_shapes", {})
    except Exception:
      merged_input_shapes = {}
    vision_slices = {k: v for k, v in merged_slices.items() if k not in _MERGED_POLICY_KEYS}
    policy_slices = {k: v for k, v in merged_slices.items() if k in _MERGED_POLICY_KEYS}
    if not vision_slices or not policy_slices:
      return None
    merged_engine = _probe_engine(engine_dir, [
      "driving_supercombo_fp16.plan", "driving_supercombo.plan",
      "supercombo_fp16.plan", "supercombo.plan",
    ])
    vision_input_names = [k for k in merged_input_shapes if "img" in k] or list(merged_input_shapes)
    return ModelProfile(
      name=name, mode="merged", engine_dir=str(engine_dir),
      merged_engine=merged_engine,
      vision_input_names=vision_input_names,
      input_shapes=merged_input_shapes,
      vision_slices=vision_slices, policy_slices=policy_slices,
      temporal=_temporal_from_shapes({}, merged_input_shapes),
    )

  if not (vision_meta_path.exists() and policy_meta_path.exists()):
    return None

  # ── split ──
  vision_slices = _load_slices(vision_meta_path)
  policy_slices = _load_slices(policy_meta_path)
  if not vision_slices or not policy_slices:
    return None
  try:
    vision_meta = _load_pkl(vision_meta_path)
    policy_meta = _load_pkl(policy_meta_path)
  except Exception:
    return None
  vision_input_shapes = vision_meta.get("input_shapes", {})
  policy_input_shapes = policy_meta.get("input_shapes", {})
  vision_engine = _probe_engine(engine_dir, [
    "driving_vision_fp16.plan", "driving_vision.plan",
  ])
  policy_engine = _probe_engine(engine_dir, [
    "driving_policy_fp16.plan", "driving_policy.plan",
  ])
  vision_input_names = [k for k in vision_input_shapes if "img" in k] or list(vision_input_shapes)
  return ModelProfile(
    name=name, mode="split", engine_dir=str(engine_dir),
    vision_engine=vision_engine, policy_engine=policy_engine,
    vision_input_names=vision_input_names,
    input_shapes={**vision_input_shapes, **policy_input_shapes},
    vision_slices=vision_slices, policy_slices=policy_slices,
    temporal=_temporal_from_shapes(policy_input_shapes),
  )


def build_registry() -> dict[str, ModelProfile]:
  reg: dict[str, ModelProfile] = {}
  # 顶层 = "Classic"(原 registry 的 _classic)
  top = _scan_dir(MODELS_ROOT, "Classic")
  if top is not None:
    reg["Classic"] = top
  # 子目录:目录名即 Model 参数名
  for child in sorted(MODELS_ROOT.iterdir()):
    if not child.is_dir() or child.name.startswith(".") or child.name == "__pycache__":
      continue
    profile = _scan_dir(child, child.name)
    if profile is not None:
      reg[child.name] = profile
  return reg


REGISTRY: dict[str, ModelProfile] = build_registry()


def get_profile(name: str | None) -> ModelProfile | None:
  return REGISTRY.get(name or "Classic")


def resolve_profile(name: str | None = None) -> ModelProfile | None:
  """Pick the first profile that is actually usable on this machine.

  Priority:
    1. explicit `name` (from Params Model / env) if its engines exist;
    2. any ready profile in REGISTRY order (Classic first, then 0-tr16..);
    3. None -> caller falls back to tinygrad.
  """
  if name:
    profile = get_profile(name)
    if profile is not None and profile.is_ready():
      return profile
  for profile in REGISTRY.values():
    if profile.is_ready():
      return profile
  return None

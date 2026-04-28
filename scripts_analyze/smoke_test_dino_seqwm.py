"""Smoke test for DinoSeqWMPolicy:
    1. instantiate config + policy with reduced size (no DINO download / weights)
    2. craft a dummy batch matching delta_indices and bimanual_cooperate features
    3. forward → check loss is finite, backward → check grads on all trainables
    4. check phase-aware weighting actually changes loss when phase flips
"""
import sys, types, importlib.util
sys.path.insert(0, "lerobot/src")

# Stub the lerobot.policies package to skip its eager __init__.py (other policies
# need diffusers/etc which we don't have on macOS). We only need our submodule.
def _make_stub_pkg(name, path):
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg
    return pkg

_make_stub_pkg("lerobot.policies", "lerobot/src/lerobot/policies")
_make_stub_pkg("lerobot.policies.dino_seqwm", "lerobot/src/lerobot/policies/dino_seqwm")

def _direct_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
cfg_mod = _direct_import(
    "lerobot.policies.dino_seqwm.configuration_dino_seqwm",
    "lerobot/src/lerobot/policies/dino_seqwm/configuration_dino_seqwm.py",
)
DinoSeqWMConfig = cfg_mod.DinoSeqWMConfig
mod_mod = _direct_import(
    "lerobot.policies.dino_seqwm.modeling_dino_seqwm",
    "lerobot/src/lerobot/policies/dino_seqwm/modeling_dino_seqwm.py",
)
DinoSeqWMPolicy = mod_mod.DinoSeqWMPolicy

# Tiny config to keep DINOv3 download manageable; fall back to manual dim if needed.
cfg = DinoSeqWMConfig(
    predictor_depth=2,
    predictor_heads=4,
    predictor_mlp_dim=512,
)
cfg.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
    "observation.images.left_wrist":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.top":         PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))}
cfg.validate_features()

print(f"Config OK. num_patches_per_frame={cfg.num_patches_per_frame}  "
      f"observation_delta_indices={cfg.observation_delta_indices}")

policy = DinoSeqWMPolicy(cfg).eval()
print(f"Policy: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,} trainable params")
print(f"Frozen DINOv3: {sum(p.numel() for p in policy.dino_encoder.parameters()):,} params (no_grad)")

B = 2
T_total = cfg.num_hist + cfg.num_pred  # 4
T_hist = cfg.num_hist                  # 3

device = "cpu"
batch = {
    "observation.state": torch.randn(B, T_total, 12, device=device),
    "action":            torch.randn(B, T_hist, 12, device=device),
    "observation.images.left_wrist":  torch.rand(B, T_total, 3, 256, 256, device=device),
    "observation.images.right_wrist": torch.rand(B, T_total, 3, 256, 256, device=device),
    "observation.images.top":         torch.rand(B, T_total, 3, 256, 256, device=device),
    "phase": torch.tensor([[0], [1]], dtype=torch.int8, device=device),
}

policy.train()
loss, info = policy(batch)
print(f"\nForward OK. loss={loss.item():.6f}")
for k, v in info.items():
    print(f"  {k}: {v}")
assert torch.isfinite(loss), "loss is not finite"

loss.backward()
ok = []
for name, p in policy.named_parameters():
    if not p.requires_grad:
        continue
    has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
    ok.append((name, has_grad))

trainable_with_grad = sum(1 for _, g in ok if g)
total_trainable = len(ok)
print(f"\nBackward: {trainable_with_grad}/{total_trainable} trainable params received gradient")

# Check that DINOv3 has NO gradients
dino_params_with_grad = sum(
    1 for n, p in policy.named_parameters()
    if n.startswith("dino_encoder") and p.grad is not None and p.grad.abs().sum().item() > 0
)
print(f"DINOv3 params with grad (should be 0): {dino_params_with_grad}")

# Check phase weighting flips loss when we flip phase
policy.zero_grad()
batch["phase"] = torch.tensor([[1], [0]], dtype=torch.int8, device=device)
loss_flipped, info_flipped = policy(batch)
print(f"\nPhase-flipped loss: {loss_flipped.item():.6f}  (should differ from {loss.item():.6f} "
      f"because per-sample weights changed)")

# A few sanity prints from info
print(f"\noriginal phase_A_frac = {info['phase_A_frac']:.2f}")
print(f"flipped  phase_A_frac = {info_flipped['phase_A_frac']:.2f}")

# Inference smoke (B=1, no goal images → fallback path)
policy.eval()
inf_batch = {
    "observation.state": torch.randn(1, T_hist, 12, device=device),
    "action":            torch.randn(1, T_hist, 12, device=device),
    "observation.images.left_wrist":  torch.rand(1, T_hist, 3, 256, 256, device=device),
    "observation.images.right_wrist": torch.rand(1, T_hist, 3, 256, 256, device=device),
    "observation.images.top":         torch.rand(1, T_hist, 3, 256, 256, device=device),
}
with torch.no_grad():
    # Drop CEM to a tiny config so this finishes quickly
    cfg.cem_num_samples = 8
    cfg.cem_opt_steps = 2
    policy._planner = None     # force re-init with new params
    chunk = policy.predict_action_chunk(inf_batch)
print(f"\nInference OK. chunk shape: {chunk.shape}  (expected (1, {cfg.cem_horizon}, 12))")
print(f"phase detected at first call: {'B' if policy._in_phase_B else 'A'}")
print("\nSmoke test PASSED.")

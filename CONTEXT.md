# CONTEXT.md — Change Log

## 2026-03-23 02:08 — Phase 1: Single-Arm DINO-WM Policy Implementation

### What was done

Implemented the full Phase 1 `dino_wm_test` policy as a LeRobot plugin under `lerobot/src/lerobot/policies/dino_wm_test/`. This policy uses a frozen DINOv3 visual encoder and a causal ViT predictor to learn dynamics in DINO latent space for a single SO101 arm.

### Files created

| File | What it does |
|------|-------------|
| `dino_wm_test/__init__.py` | Package marker (empty). |
| `dino_wm_test/configuration_dino_wm_test.py` | `DinoWMTestConfig` — policy config registered as `"dino_wm_test"` via `@PreTrainedConfig.register_subclass`. Defines all hyperparameters: DINOv3 encoder settings (img_size=256, embed_dim=768, num_patches=256), ViTPredictor architecture (depth=6, heads=16, mlp_dim=2048), world model settings (num_hist=3, num_pred=1, frameskip=5, state_dim=6, action_dim=6, 2 cameras), CEM planner params (horizon=5, topk=10, 100 samples, 30 opt steps), and optimizer settings (AdamW lr=5e-4). Produces observation_delta_indices=[-10,-5,0,5] and action_delta_indices=[-10,-5,0]. |
| `dino_wm_test/modeling_dino_wm_test.py` | `DinoWMTestPolicy` — the core policy class inheriting `PreTrainedPolicy`. Training: encodes raw images inline with frozen DINOv3, embeds state/action via ProprioceptiveEmbedding, concatenates tokens per frame [visual_cam1, visual_cam2, proprio, action] (514 tokens/frame), runs ViTPredictor with causal attention, computes MSE loss on visual+proprio tokens (action token excluded) with stop-gradient on targets. Inference: CEM planner optimizes action sequences by rolling out the world model and comparing predicted latents to a goal image latent. Implements action chunking with a deque-based action queue. |
| `dino_wm_test/processor_dino_wm_test.py` | `make_dino_wm_test_pre_post_processors()` — standard LeRobot pre/post-processing pipelines. Preprocessor: rename → add batch dim → device → normalize (VISUAL=IDENTITY, STATE/ACTION=MEAN_STD). Postprocessor: unnormalize → CPU. |
| `dino_wm_test/dino_encoder.py` | `DINOv3Encoder` — wraps TIMM's `vit_base_patch16_dinov3.lvd1689m` model. Frozen (no gradients). Strips 5 prefix tokens (1 CLS + 4 register) and returns 256 patch tokens of dimension 768. Always stays in eval mode. |
| `dino_wm_test/vit_predictor.py` | `ViTPredictor` — causal Vision Transformer predictor adapted from `dino_wm/models/vit.py`. Key change: dim updated from 384→768 to match DINOv3. Uses block-causal attention masks (each frame attends only to current + past frames). Eliminates the original's global variable pattern by passing num_patches/num_frames to Attention layers directly via constructor. |
| `dino_wm_test/proprio_embedding.py` | `ProprioceptiveEmbedding` — Conv1d-based embedding adapted from `dino_wm/models/proprio.py`. Projects state vectors (6D) or action vectors (6D) to 768D tokens. emb_dim updated from 384→768. |
| `dino_wm_test/cem_planner.py` | `CEMPlanner` — simplified CEM planner adapted from `dino_wm/planning/cem.py`. Removed dependencies on BasePlanner, preprocessor, evaluator, and wandb. Takes a `rollout_fn` and `objective_fn` as callables. Iteratively samples action sequences, evaluates them, selects top-K elites, and updates the Gaussian distribution. |
| `dino_wm_test/objectives.py` | `create_objective_fn()` — copied from `dino_wm/planning/objectives.py`. Creates MSE-based objective functions for CEM that compare predicted visual+proprio latents against goal latents. Supports "last" (last frame only) and "all" (exponentially weighted) modes. |

### Files modified

| File | What changed |
|------|-------------|
| `lerobot/src/lerobot/policies/__init__.py` | Added `from .dino_wm_test.configuration_dino_wm_test import DinoWMTestConfig as DinoWMTestConfig` import and added `"DinoWMTestConfig"` to `__all__` list. This registers the policy so LeRobot's factory can discover it. |

### Environment setup

Created a virtual environment at `.venv/` using `uv venv` and installed all dependencies via `uv pip install`: torch, torchvision, torchaudio, timm, einops, draccus, safetensors, huggingface-hub, and lerobot (editable install).

### Verification

- All imports pass successfully.
- Forward pass with dummy data (B=2, 2 cameras, 256x256 images) produces correct loss with gradients.
- Backward pass confirms gradients flow to all trainable components (predictor, proprio_encoder, action_encoder).
- DINOv3 encoder remains frozen (no gradients).
- Config produces correct delta indices matching CLAUDE.md spec.
- ~7.5M trainable parameters (with reduced depth=2 test; full depth=6 config will be larger).

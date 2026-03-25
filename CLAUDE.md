# CLAUDE.md — Multi-Robot Cooperation (DINO-WM + SeqWM on LeRobot SO101)

## Project Overview

This project implements a **visual world model** for **multi-robot cooperative manipulation** using two LeRobot SO101 arms. The approach combines:
- **DINO-WM** — a visual world model that predicts dynamics in DINO latent space (no pixel decoding)
- **SeqWM** — a sequential multi-agent world model where agents predict autoregressively (Helper first, Leader conditioned on Helper)
- **DINOv3 (ViT-B/16)** via TIMM as the frozen visual encoder (replaces DINO-WM's DINOv2 ViT-S/14)
- **LeRobot** framework for data collection, training loop, dataset management, and policy integration

### How the Pieces Fit Together

- **DINO-WM** provides the core architecture: predict next-state in frozen DINO patch-token space using a causal ViTPredictor, plan actions via CEM
- **SeqWM** provides the multi-agent principle: decompose joint dynamics into sequential per-agent predictions (agent i conditions on agents 1..i-1)
- **Our contribution**: combine these — use DINO-WM's visual ViTPredictor architecture with SeqWM's sequential two-stage prediction for bimanual manipulation

## Repository Layout

```
Cooporate/
├── lerobot/                          # LeRobot framework (git repo)
├── dino_wm/                          # DINO-WM source code
├── seqwm/                            # SeqWM source code (ICLR 2026)
├── stable-pretraining/               # SSL framework (not used)
├── MuJoCo-Simulator-SO101-Cooperate/ # MuJoCo sim environment for SO101
├── Seq-WM.pdf                        # SeqWM paper
└── CLAUDE.md                         # This file
```

---

## Development Phases

### Phase 1: Single-Arm DINO-WM Test

**Goal:** Validate the full pipeline (data → features → train → CEM inference) on a single SO101 arm before scaling to bimanual.

**Task:** Pick up an object and place it into a box.

| Aspect | Value |
|--------|-------|
| Arms | 1 SO101 (follower + leader for teleop) |
| Cameras | 2 (wrist + global) |
| `observation.state` / `action` dim | **6** |
| Tokens per frame | **514** (2×256 patches + proprio + action) |
| World model | Standard DINO-WM (single ViTPredictor) |
| CEM planner | Single-stage |
| Policy name | `dino_wm_test` |
| Robot config | `so101_follower` / `so101_leader` |
| Target episodes | ~50 |

**Policy location:** `lerobot/src/lerobot/policies/dino_wm_test/`

### Phase 2: Bimanual SeqWM

**Goal:** Full bimanual cooperative manipulation with sequential world model.

| Aspect | Value |
|--------|-------|
| Arms | 2 SO101 (Left=Helper, Right=Leader) |
| Cameras | 3 (left_wrist, right_wrist, right_global) |
| `observation.state` / `action` dim | **14** (7 helper + 7 leader) |
| Tokens per frame | **770** (3×256 patches + proprio + action) |
| World model | SeqWM (2 ViTPredictors: Helper + Leader) |
| CEM planner | Sequential two-stage |
| Policy name | `dino_seqwm` |
| Loss | `z_loss_joint + α * z_loss_helper` (α=0.5) |
| Custom code | ~725 lines |

**Policy location:** `lerobot/src/lerobot/policies/dino_seqwm/`

---

## Visual Encoder: DINOv3 via TIMM

| Property | Value |
|----------|-------|
| TIMM model | `vit_base_patch16_dinov3.lvd1689m` |
| img_size | 256 |
| patch_size | 16 |
| embed_dim | **768** |
| num_patches | **256** (16×16) |
| prefix tokens | **5** (1 CLS + 4 register) → strip with `output[:, 5:, :]` |
| Status | Frozen; run inline during training |

### Feature Extraction Strategy

Run DINOv3 encoder **inline** inside `forward()` during training. The encoder is frozen (no gradients), so it acts as a deterministic transform. This avoids modifying LeRobot's dataset/parquet format.

The policy receives raw images from the dataloader and runs DINOv3 inside `forward()` to get patch tokens before passing them to the ViTPredictor.

---

## Architecture — Key Dimensions

All DINO-WM components have `emb_dim` updated from 384 → **768** to match DINOv3.

| Component | Source File | Key Change |
|-----------|------------|------------|
| `ViTPredictor` + `Transformer` | `dino_wm/models/vit.py` | `dim` 384→768 |
| `ProprioceptiveEmbedding` | `dino_wm/models/proprio.py` | `emb_dim` 384→768, `in_chans` matches arm config |
| `CEMPlanner` | `dino_wm/planning/cem.py` | Simplified wrapper for LeRobot |
| `create_objective_fn` | `dino_wm/planning/objectives.py` | Reuse as-is |

### DINO-WM Source Signatures (for copying)

**ViTPredictor** (`dino_wm/models/vit.py`):
```python
ViTPredictor(*, num_patches, num_frames, dim, depth, heads, mlp_dim,
             pool='cls', dim_head=64, dropout=0., emb_dropout=0.)
# forward(x) : (B, num_frames*num_patches, dim) -> (B, num_frames*num_patches, dim)
# Uses causal attention mask, learnable positional embeddings
# Sets global NUM_FRAMES and NUM_PATCHES (affects causal mask generation)
```

**ProprioceptiveEmbedding** (`dino_wm/models/proprio.py`):
```python
ProprioceptiveEmbedding(num_frames=16, tubelet_size=1, in_chans=8, emb_dim=384)
# forward(x) : (B, T, D) -> (B, T, emb_dim)
# Uses Conv1d(in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size)
```

**VWorldModel.encode()** (`dino_wm/models/visual_world_model.py`):
```python
# concat_dim=0 mode: appends proprio and action as extra tokens
# z = cat([visual_patches, proprio_emb.unsqueeze(2), action_emb.unsqueeze(2)], dim=2)
# Shape: (B, T, num_visual_patches + 2, emb_dim)
```

**VWorldModel.separate_emb()**: splits z back into visual, proprio, action:
```python
# concat_dim=0: z_visual = z[:, :, :-2, :], z_proprio = z[:, :, -2, :], z_act = z[:, :, -1, :]
```

**VWorldModel.rollout()**: iterative next-frame prediction for CEM:
```python
# encode initial obs -> iteratively predict + append -> separate embeddings
# Returns: z_obses = {"visual": (B,T,patches,dim), "proprio": (B,T,D)}, z_full
```

### Predictor Hyperparameters (both phases)

| Parameter | Value |
|-----------|-------|
| `predictor_depth` | 6 |
| `predictor_heads` | 16 |
| `predictor_mlp_dim` | 2048 |
| `num_hist` | 3 |
| `num_pred` | 1 |
| `frameskip` | 5 |

---

## LeRobot Policy Integration Details

### File Structure (Phase 1)

```
lerobot/src/lerobot/policies/dino_wm_test/
    configuration_dino_wm_test.py   # DinoWMTestConfig (with @register_subclass)
    modeling_dino_wm_test.py        # DinoWMTestPolicy
    processor_dino_wm_test.py       # make_dino_wm_test_pre_post_processors()
    dino_encoder.py                 # DINOv3Encoder class
    vit_predictor.py                # ViTPredictor (copied from dino_wm, dim→768)
    proprio_embedding.py            # ProprioceptiveEmbedding (copied, dim→768)
    cem_planner.py                  # Simplified CEMPlanner for LeRobot
    objectives.py                   # create_objective_fn (copied as-is)
```

### File Structure (Phase 2)

```
lerobot/src/lerobot/policies/dino_seqwm/
    configuration_dino_seqwm.py     # DinoSeqWMConfig (with @register_subclass)
    modeling_dino_seqwm.py          # DinoSeqWMPolicy
    processor_dino_seqwm.py         # make_dino_seqwm_pre_post_processors()
    dino_encoder.py                 # DINOv3Encoder (shared with Phase 1)
    vit_predictor.py                # ViTPredictor (shared with Phase 1)
    proprio_embedding.py            # ProprioceptiveEmbedding (shared, in_chans→14/7)
    seq_cem_planner.py              # SequentialCEMPlanner (two-stage)
    objectives.py                   # create_objective_fn (same as Phase 1)
```

### Policy Registration (How Discovery Works)

1. `@PreTrainedConfig.register_subclass("dino_wm_test")` on config class registers with draccus ChoiceRegistry
2. Add import in `lerobot/src/lerobot/policies/__init__.py`:
   ```python
   from .dino_wm_test.configuration_dino_wm_test import DinoWMTestConfig as DinoWMTestConfig
   from .dino_seqwm.configuration_dino_seqwm import DinoSeqWMConfig as DinoSeqWMConfig
   ```
3. `factory.py` auto-discovers via naming convention — no hardcoded list changes needed

### Processor (Mandatory)

Every LeRobot policy **must** have a processor. Minimum pipeline:

```python
# Preprocessor:  AddBatchDimensionProcessorStep → DeviceProcessorStep → NormalizerProcessorStep
# Postprocessor: UnnormalizerProcessorStep → DeviceProcessorStep("cpu")
```

### Config: input_features / output_features

Populated at runtime from dataset metadata. Controls normalization (type-based, not key-based):

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,   # Images: no normalization (DINOv3 handles its own)
    "STATE": NormalizationMode.MEAN_STD,    # Joint positions: standardize
    "ACTION": NormalizationMode.MEAN_STD,   # Joint targets: standardize
}
```

### validate_features() Best Practice

```python
def validate_features(self):
    if "observation.state" not in self.input_features:
        self.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(D,))
    if "action" not in self.output_features:
        self.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(D,))
    image_features = {k: v for k, v in self.input_features.items() if v.type == FeatureType.VISUAL}
    if not image_features:
        raise ValueError("Policy requires at least one visual input feature.")
```

### PreTrainedPolicy Abstract Methods

```python
get_optim_params(self) -> dict          # Return {"params": [...trainable params...]}
reset(self)                              # Clear caches on episode reset
forward(self, batch) -> (loss, dict)     # Training forward pass
predict_action_chunk(self, batch) -> Tensor  # Predict action chunk
select_action(self, batch) -> Tensor     # Single action for deployment
```

### Delta Indices and Batch Shapes

`observation_delta_indices` controls which frames the dataloader stacks. Applies to **all** `observation.*` keys.

**Phase 1:** `[-10, -5, 0, 5]` → 3 history + 1 target
- `observation.images.wrist`: `(B, 4, 3, H, W)`
- `observation.state`: `(B, 4, 6)`
- `action` (indices `[-10, -5, 0]`): `(B, 3, 6)`

**Phase 2:** `[-10, -5, 0, 5]` → same structure, larger dims
- `observation.images.left_wrist`: `(B, 4, 3, H, W)`
- `observation.images.right_wrist`: `(B, 4, 3, H, W)`
- `observation.images.right_global`: `(B, 4, 3, H, W)`
- `observation.state`: `(B, 4, 14)`
- `action` (indices `[-10, -5, 0]`): `(B, 3, 14)`

In `forward()`: frames 0-2 are history (source), frame 3 is target.

---

## Phase 2: SeqWM Architecture — Full Details

### Background: SeqWM Paper (ICLR 2026)

SeqWM ("Sequential World Models") decomposes joint multi-agent dynamics into **sequential, autoregressive per-agent world models**. The key idea:

1. Fix an agent ordering (e.g., Helper before Leader)
2. Agent i predicts its next state conditioned on predecessors' actions and predictions
3. Each agent has its own world model, but they communicate sequentially

**Original SeqWM formulation** (from paper):
```
Agent i receives message:  e_t^i  (predecessors' latent states + actions)
Encodes observation:       z_t^i = E^i(o_t^i)
Predicts next state:       ẑ_{t+1}^i = D^i(z_t^i, a_t^i, e_t^i)
Forwards to next agent:    e_t^{i+1} = e_t^i ⊕ (z_t^i, a_t^i)
```

**Original SeqWM uses MLP-based models** (encoder, dynamics, reward predictor per agent). Our adaptation replaces these with DINO-WM's **ViT-based visual world model** operating on frozen DINOv3 patch tokens.

### Our Adaptation: Visual SeqWM

Instead of per-agent MLP encoders/dynamics, we use:
- **Shared visual encoding**: all cameras encoded by frozen DINOv3 → shared patch tokens
- **Shared proprioceptive state**: full 14D joint state visible to both predictors
- **Per-agent ViTPredictors**: Helper and Leader each have independent ViTPredictor weights
- **Sequential conditioning via action token replacement**: Leader's predictor receives Helper's predicted output with the action token swapped to Leader's action

### Architecture Diagram

```
                    Per-camera DINOv3 patch tokens (frozen, inline)
                                |
                +---------------+---------------+
                |               |               |
          left_wrist      right_wrist      right_global
          (256, 768)      (256, 768)       (256, 768)
                |               |               |
                +-------+-------+
                        |
                Concatenate across cameras
                        |
                (768, 768)  [3 cameras × 256 patches]
                        |
               + ProprioEmb(state_14D)     → (1, 768)
               + ActionEmb(action_helper_7D) → (1, 768)
                        |
                (770, 768)  [768 visual + proprio + action]
                        |
              × num_hist=3 frames → flatten → (2310, 768)
                        |
            +-----------+-----------+
            |                       |
    ViTPredictor_Helper      [causal attention]
            |
    z_pred_helper (3, 770, 768)
            |
    Replace action token:  z_pred_helper[:, :, -1, :] = ActionEmb(action_leader_7D)
            |
    ViTPredictor_Leader      [conditioned on Helper's output]
            |
    z_pred_joint (3, 770, 768)
            |
    Loss: MSE on visual+proprio tokens vs ground-truth target
```

### Two-Stage Training Forward Pass

```python
def forward(self, batch):
    # 1. Encode images with frozen DINOv3
    v_h = dino_encoder(batch["observation.images.left_wrist"])    # (B, T, 256, 768)
    v_l = dino_encoder(batch["observation.images.right_wrist"])   # (B, T, 256, 768)
    v_g = dino_encoder(batch["observation.images.right_global"])  # (B, T, 256, 768)
    visual = cat([v_g, v_h, v_l], dim=2)                         # (B, T, 768, 768)

    proprio = batch["observation.state"]    # (B, T_total, 14)
    action = batch["action"]               # (B, T_hist, 14)
    action_helper = action[..., :7]
    action_leader = action[..., 7:]

    # 2. Split source (history) and target
    visual_src, visual_tgt = visual[:, :3], visual[:, 1:]  # 3 history, 3 shifted
    proprio_src, proprio_tgt = proprio[:, :3], proprio[:, 1:]

    # 3. Encode source with helper action (concat_dim=0)
    z_src = _encode(visual_src, proprio_src, action_helper)   # (B, 3, 770, 768)
    z_tgt = _encode(visual_tgt, proprio_tgt, action_helper)   # (B, 3, 770, 768)

    # 4. Stage 1: Helper prediction
    z_pred_helper = helper_predictor(z_src.flatten(1,2)).unflatten(1, (3, 770))

    # 5. Stage 2: Replace action token, Leader prediction
    z_leader_input = z_pred_helper.clone()
    z_leader_input[:, :, -1, :] = action_encoder(action_leader)  # swap action token
    z_pred_joint = leader_predictor(z_leader_input.flatten(1,2)).unflatten(1, (3, 770))

    # 6. Loss (exclude action token at position -1)
    z_loss_joint = MSE(z_pred_joint[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
    z_loss_helper = MSE(z_pred_helper[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
    loss = z_loss_joint + alpha * z_loss_helper

    return loss, {"z_loss_joint": ..., "z_loss_helper": ...}
```

### Why This Differs from Original SeqWM

| Aspect | Original SeqWM | Our Visual SeqWM |
|--------|---------------|------------------|
| Observation encoding | Per-agent MLP encoder → latent vector | Shared frozen DINOv3 → patch tokens (all cameras) |
| Dynamics model | Per-agent MLP predictor | Per-agent ViTPredictor (causal Transformer) |
| State representation | Learned latent vector (512D) | DINO patch tokens (770 tokens × 768D) |
| Sequential conditioning | Message passing (concat predecessors' z + a) | Action token replacement in ViT input |
| Reward model | Per-agent MLP → distributional (two-hot) | Not used (goal-conditioned CEM instead) |
| Actor | Per-agent squashed Gaussian MLP | Not used (CEM planning instead) |
| Critic | Shared distributional Q-network | Not used |
| Training | Online RL (SAC-style) with replay buffer | Offline imitation (demos via teleop) |
| Planning | CEM on reward model predictions | CEM on goal-latent MSE distance |

### Key Design Decision: Action Token Replacement

The sequential conditioning mechanism is simpler than SeqWM's message passing:

1. Helper's ViTPredictor receives `z_src` with `ActionEmb(action_helper)` as the last token
2. Helper predicts → `z_pred_helper` contains Helper's "intention" (predicted next state)
3. We **replace the action token** in `z_pred_helper` with `ActionEmb(action_leader)`
4. Leader's ViTPredictor receives this modified sequence → predicts the **joint** next state

This works because:
- The visual + proprio tokens in `z_pred_helper` encode Helper's predicted world state
- The Leader sees this prediction and conditions its own prediction on it
- This is analogous to SeqWM's `e_t^{i+1} = e_t^i ⊕ (z_t^i, a_t^i)` but in token space

### Trainable Parameters

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| DINOv3 encoder | Frozen | Frozen |
| ViTPredictor | 1 instance | 2 instances (Helper + Leader) |
| ProprioEmb (proprio) | 1 (in_chans=6) | 1 (in_chans=14) |
| ProprioEmb (action) | 1 (in_chans=6) | 1 (in_chans=14) |
| Optimizer | AdamW lr=5e-4 | AdamW lr=5e-4 |

Phase 2 `get_optim_params()`:
```python
params = (helper_predictor.parameters() + leader_predictor.parameters()
        + proprio_encoder.parameters() + action_encoder.parameters())
```

---

## Sequential CEM Planner (Phase 2)

### How Original SeqWM Plans

From `seqwm/runners/world_model_runner.py`:
1. For each agent in order: sample N action sequences
2. Roll forward each agent's dynamics model sequentially
3. Estimate value using reward model + terminal Q-value
4. Select top-K elite trajectories per agent
5. Update Gaussian distribution from elites
6. Communication: pass optimized trajectory + predicted latent to next agent

Key params from SeqWM: `num_samples=512, num_elites=64, plan_iter=6, horizon=1`

### Our Sequential CEM Adaptation

We adapt this for goal-conditioned visual planning with DINO-WM's CEM structure:

**Phase 1 CEM (single-stage):**
Standard DINO-WM CEM — sample action sequences, rollout, compare to goal latent via MSE.

**Phase 2 CEM (two-stage sequential):**

```
Phase 2a: Plan Helper Actions
  1. Encode current obs window (3 frames) with DINOv3
  2. Encode goal image with DINOv3
  3. Sample N=100 helper action sequences (horizon=5, dim=7)
  4. For each sample: rollout with ViTPredictor_Helper
  5. Evaluate: MSE(predicted_visual[-1], goal_visual) + α * MSE(predicted_proprio[-1], goal_proprio)
  6. Select top-10 elites, update Gaussian
  7. Repeat for 30 iterations
  8. Output: mu_helper (best helper trajectory)

Phase 2b: Plan Leader Actions (conditioned on Helper)
  1. Fix helper actions = mu_helper
  2. Sample N=100 leader action sequences (horizon=5, dim=7)
  3. For each sample: two-stage rollout
     a. Helper rollout with mu_helper → z_pred_helper
     b. Replace action token with leader action
     c. Leader rollout with ViTPredictor_Leader → z_pred_joint
  4. Evaluate: same MSE objective on z_pred_joint
  5. Select top-10, update Gaussian, repeat 30 iterations
  6. Output: mu_leader (best leader trajectory)

Final action: cat([mu_helper[0], mu_leader[0]], dim=-1)  → (14,)
```

### CEM Parameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| `num_samples` | 100 | 100 per stage |
| `horizon` | 5 | 5 |
| `topk` | 10 | 10 |
| `opt_steps` | 30 | 30 per stage |
| `action_dim` | 6 | 7 per stage |
| `objective` | `create_objective_fn(alpha=0.1, mode="last")` | same |

### Goal Image Handling

In DINO-WM, the goal comes from:
- Dataset's last frame of a successful demonstration
- A saved goal image file

Flow: `raw goal image → DINOv3 encoder → z_obs_g (latent)` → used in objective function.

The objective function computes:
```python
loss = MSE(z_pred_visual[:, -1], z_goal_visual) + alpha * MSE(z_pred_proprio[:, -1], z_goal_proprio)
```

---

## Training Loss

**Phase 1 (single arm):**
```
loss = MSE(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
```

**Phase 2 (bimanual SeqWM):**
```
loss = z_loss_joint + α * z_loss_helper
     = MSE(z_pred_joint[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
     + α * MSE(z_pred_helper[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
```
- `:-1` excludes the action token (only visual + proprio tokens contribute to loss)
- `α = 0.5` weights the helper auxiliary loss
- `z_tgt.detach()` — stop-gradient on target (no EMA needed, frozen visual backbone)

---

## Data Collection

### Phase 1
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    global: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --dataset.repo_id=your_username/pick_place_test
```

### Phase 2
```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.left_arm_config.cameras='{
    wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }' \
  --robot.right_arm_config.cameras='{
    wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
    global: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --dataset.repo_id=your_username/bimanual_task_dataset
```

### Dataset Key Mapping (Phase 2)

| LeRobot Key | Physical Camera | Logical Role |
|-------------|----------------|--------------|
| `observation.images.left_wrist` | Wrist cam on left follower | Helper wrist view |
| `observation.images.right_wrist` | Wrist cam on right follower | Leader wrist view |
| `observation.images.right_global` | Global scene cam | Workspace overview |
| `observation.state[:7]` | Left follower joints | Helper state |
| `observation.state[7:]` | Right follower joints | Leader state |
| `action[:7]` | Left follower targets | Helper action |
| `action[7:]` | Right follower targets | Leader action |

---

## Hardware

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Follower arms | 1 (`/dev/ttyACM0`) | 2 (left: `/dev/ttyACM0`, right: `/dev/ttyACM1`) |
| Leader arms | 1 (`/dev/ttyACM1`) | 2 (left: `/dev/ttyACM2`, right: `/dev/ttyACM3`) |
| Cameras | 2 (wrist + global) | 3 (left_wrist + right_wrist + right_global) |
| Training GPU | V100 | V100 |
| Inference GPU | RTX 4060 | RTX 4060 |

## Core Design Principles

1. **No pixel-level generation** — all dynamics prediction in DINO latent space (patch tokens)
2. **Minimize LeRobot modifications** — only add files under `policies/` + import lines in `__init__.py`
3. **Maximize reuse** — LeRobot infra + DINO-WM components + SeqWM's sequential principle
4. **LeRobot policy plugin** — inherit `PreTrainedPolicy`, register config via `@PreTrainedConfig.register_subclass()`
5. **Stop-gradient on targets** — `z_tgt.detach()`, no EMA needed
6. **DINOv3 inline in forward()** — run frozen encoder during training on raw images
7. **Sequential conditioning via action token replacement** — simpler than SeqWM's message passing, native to ViT token architecture

## SeqWM Source Reference

Key files in `seqwm/` repo for reference:
- `seqwm/runners/world_model_runner.py` — main training loop, `model_train()` (lines 758-872), `actor_train()` (lines 874-927), `plan()` (lines 462-603), `estimate_value()` (lines 606-668)
- `seqwm/models/base/wm_networks.py` — `MLPEncoder`, `MLPPredictor`, `TwoHotProcessor`, `SimNorm`, `RunningScale`
- `seqwm/algorithms/actors/world_model_actor.py` — `SeqWMPolicy`, `SeqWM` actor wrapper
- `seqwm/algorithms/critics/world_model_q_critic.py` — `EnsembleDisRegQCritic`
- `seqwm/common/buffers/world_model_buffer.py` — `OffPolicyBufferWM`

Sequential modes in SeqWM: `'seq'` (fixed order), `'cen'` (centralized), `'dec'` (decentralized), `'ran'` (random order). We use the equivalent of `'seq'` with fixed Helper→Leader ordering.

## MuJoCo Simulation Environment

Available at `MuJoCo-Simulator-SO101-Cooperate/` for testing without physical hardware:
- `gym_so100/env.py`: `SO100Env` (standard) and `SO100GoalEnv` (goal-conditioned)
- Tasks: `so100_touch_cube`, `so100_cube_to_bin`
- Obs types: `so100_pixels_agent_pos` (pixels + joint pos), `so100_state` (full state)
- Camera: single `top` camera rendering
- `scripts/evaluate_lerobot_policy.py`: evaluation script showing how LeRobot policies interface with the sim

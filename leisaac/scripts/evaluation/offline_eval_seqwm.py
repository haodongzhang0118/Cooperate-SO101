#!/usr/bin/env python3
"""Offline evaluation of DINO-SeqWM (Phase 2 bimanual world model).

Adapts offline_eval.py for the two-predictor sequential architecture:

  M1. Single-step prediction loss        — stratified by phase A/B
  M2. Multi-step rollout cumulative err  — pure-A, pure-B, handoff windows
  M3. CEM goal-distance reduction        — two-stage (helper→subgoal_A, leader→final_goal)
  M4. Action agreement w/ expert         — per-arm (helper, leader)
  N1. Phase detector accuracy + latency  — runtime heuristic vs ground-truth phase column

Goal handling matches dino_wm exactly: PER-EPISODE goals.
For each evaluated episode i:
  subgoal_A goal frame = ep_start[i] + stop_frame[i]   (lid moved, left arm idle)
  final_goal frame     = ep_end[i] - 1                  (cube in box)
We do NOT average goal latents across episodes — each episode plans toward
its own specific outcome. stop_frame[i] is read from the dataset's
embedded `phase` column (no sidecar JSON required).

Usage:
  python scripts/evaluation/offline_eval_seqwm.py \
      --checkpoint /path/to/pretrained_model \
      --dataset_repo_id haodoz0118/bimanual_cooperate \
      --dataset_root /path/to/bimanual_cooperate \
      --num_eval_samples 80 \
      --num_rollout_episodes 12 \
      --rollout_horizon 5 \
      --num_cem_episodes 5 \
      --output_json results.json
"""

import argparse
import json
import os
import struct
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import repeat


# ---------------------------------------------------------------------------
# Bypass lerobot.policies.__init__.py heavy imports
# ---------------------------------------------------------------------------

def _setup_lerobot_stub():
    import lerobot as _lr
    policies_dir = os.path.join(os.path.dirname(_lr.__file__), "policies")
    if "lerobot.policies" not in sys.modules:
        stub = types.ModuleType("lerobot.policies")
        stub.__path__ = [policies_dir]
        stub.__package__ = "lerobot.policies"
        sys.modules["lerobot.policies"] = stub


_setup_lerobot_stub()


# ---------------------------------------------------------------------------
# Normalization helpers (read MEAN_STD from checkpoint safetensors)
# ---------------------------------------------------------------------------

def load_norm_stats(checkpoint_path: str, device: torch.device) -> dict:
    """Load action / state mean+std from the policy preprocessor safetensors."""
    norm_file = os.path.join(
        checkpoint_path,
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    )
    with open(norm_file, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size

        def _read(name):
            info = header[name]
            f.seek(data_start + info["data_offsets"][0])
            n_bytes = info["data_offsets"][1] - info["data_offsets"][0]
            buf = np.frombuffer(f.read(n_bytes), dtype=np.float32).copy()
            return torch.from_numpy(buf).to(device)

        return {
            "action_mean": _read("action.mean"),
            "action_std": _read("action.std"),
            "state_mean": _read("observation.state.mean"),
            "state_std": _read("observation.state.std"),
        }


def normalize_state(state, stats):
    return (state - stats["state_mean"]) / (stats["state_std"] + 1e-8)


def normalize_action(action, stats):
    return (action - stats["action_mean"]) / (stats["action_std"] + 1e-8)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_episode_bounds(dataset, ep_idx):
    ep = dataset.meta.episodes[ep_idx]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def load_frame(dataset, idx, camera_names, device):
    sample = dataset[int(idx)]
    images = {}
    for cam in camera_names:
        img = sample[f"observation.images.{cam}"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        images[cam] = img.to(device)
    state = sample["observation.state"].to(device)
    action = sample["action"].to(device)
    phase = int(sample["phase"].item()) if "phase" in sample else -1
    return images, state, action, phase


def load_frame_sequence(dataset, indices, camera_names, device):
    images = {cam: [] for cam in camera_names}
    states, actions, phases = [], [], []
    for idx in indices:
        img, st, ac, ph = load_frame(dataset, idx, camera_names, device)
        for cam in camera_names:
            images[cam].append(img[cam])
        states.append(st)
        actions.append(ac)
        phases.append(ph)
    return (
        {cam: torch.stack(images[cam]) for cam in camera_names},
        torch.stack(states),
        torch.stack(actions),
        np.array(phases, dtype=np.int8),
    )


def get_episode_phase_array(dataset, ep_idx):
    """Return per-frame phase for the whole episode as np.int8 (T,)."""
    s, e = get_episode_bounds(dataset, ep_idx)
    phases = []
    for i in range(s, e):
        sample = dataset[i]
        phases.append(int(sample["phase"].item()) if "phase" in sample else -1)
    return np.array(phases, dtype=np.int8)


# ---------------------------------------------------------------------------
# Per-episode goal frame loader (mirrors dino_wm's offline eval semantics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def load_goal_latent_for_frame(model, dataset, frame_idx, stats, device):
    """Load all-camera images + real state at a single frame, encode them
    through the trained DINOv3 + proprio_encoder, return one goal latent.

    Returns: {"visual": (1, num_cam*256, 768), "proprio": (1, 768)}
    """
    cam_names = model.config.camera_names
    sample = dataset[int(frame_idx)]

    imgs_dict = {}
    for cam in cam_names:
        img = sample[f"observation.images.{cam}"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        imgs_dict[cam] = img.to(device).unsqueeze(0).unsqueeze(0)   # (1, 1, 3, H, W)

    state = sample["observation.state"].to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    visual = model._encode_images(imgs_dict, 1, 1)                  # (1, 1, num_cam*256, 768)
    proprio = model.proprio_encoder(normalize_state(state, stats))   # (1, 1, 768)
    return {"visual": visual[:, 0], "proprio": proprio[:, 0]}


# ---------------------------------------------------------------------------
# M1: Single-step prediction loss, stratified by phase
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_step_per_phase(model, dataset, stats, device, num_samples=80):
    """Per-phase mean of z_loss_helper and z_loss_joint, computed on
    randomly sampled (3 hist + 1 target) windows."""
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    cam_names = config.camera_names
    total_window = num_hist + config.num_pred  # 4
    min_ep_len = total_window * frameskip + 1
    num_eps = len(dataset.meta.episodes)

    by_phase = {0: defaultdict(list), 1: defaultdict(list)}  # 0=A, 1=B
    count = 0
    eps_used = 0

    for ep_idx in range(num_eps):
        if count >= num_samples:
            break
        ep_start, ep_end = get_episode_bounds(dataset, ep_idx)
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue
        max_base = ep_len - total_window * frameskip
        n_from_ep = min(8, max_base + 1, num_samples - count)
        bases = np.random.choice(max_base + 1, size=n_from_ep, replace=False)

        for base in bases:
            frame_ids = [ep_start + base + i * frameskip for i in range(total_window)]
            imgs, states, actions, phases = load_frame_sequence(
                dataset, frame_ids, cam_names, device
            )

            # phase of TARGET frame (last in window)
            tgt_phase = int(phases[-1])
            if tgt_phase not in (0, 1):
                continue

            batch = {}
            for cam in cam_names:
                batch[f"observation.images.{cam}"] = imgs[cam].unsqueeze(0)
            batch["observation.state"] = normalize_state(states.unsqueeze(0), stats)
            batch["action"] = normalize_action(
                actions[:num_hist].unsqueeze(0), stats
            )
            batch["phase"] = torch.tensor([[tgt_phase]], dtype=torch.int8, device=device)

            loss, info = model.forward(batch)
            by_phase[tgt_phase]["z_loss"].append(loss.item())
            by_phase[tgt_phase]["z_loss_joint"].append(info["z_loss_joint"])
            by_phase[tgt_phase]["z_loss_helper"].append(info["z_loss_helper"])
            count += 1
        eps_used += 1

    out = {}
    for phase_id, name in [(0, "A"), (1, "B")]:
        bucket = by_phase[phase_id]
        if not bucket:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": len(bucket["z_loss"]),
            **{k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in bucket.items()},
        }
    out["episodes_used"] = eps_used
    return out


# ---------------------------------------------------------------------------
# M2: Multi-step rollout cumulative error, three slicings
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_one_window(model, dataset, stats, device, frame_ids, rollout_horizon):
    """Run _seq_rollout on one window; return per-step (visual_mse, proprio_mse).

    Convention: step 0 = first predicted frame, step H-1 = last. We feed
    real first num_hist frames + real history actions + real future actions,
    and only score the predicted future latents.
    """
    config = model.config
    num_hist = config.num_hist
    cam_names = config.camera_names
    h_dim = config.helper_action_dim

    imgs, states, actions, _ = load_frame_sequence(dataset, frame_ids, cam_names, device)
    total_frames = len(frame_ids)

    imgs_b = {cam: imgs[cam].unsqueeze(0) for cam in cam_names}
    gt_visual = model._encode_images(imgs_b, 1, total_frames)
    gt_proprio = model.proprio_encoder(normalize_state(states.unsqueeze(0), stats))

    obs_visual = gt_visual[:, :num_hist]
    obs_proprio = gt_proprio[:, :num_hist]

    actions_norm = normalize_action(actions.unsqueeze(0), stats)
    a_helper_hist = actions_norm[:, :num_hist, :h_dim]                          # (1, T_h, 6)

    # Future: actions[num_hist-1 : num_hist-1+H], split into helper / leader 6D
    rollout_actions = actions_norm[:, num_hist - 1 : num_hist - 1 + rollout_horizon]
    a_helper_seq = rollout_actions[..., :h_dim]
    a_leader_seq = rollout_actions[..., h_dim:]

    result = model._seq_rollout(
        obs_visual, obs_proprio,
        a_helper_hist, a_helper_seq, a_leader_seq,
    )
    # result shapes: visual (1, num_hist+H, ...), proprio (1, num_hist+H, ...)
    # Indices 0..num_hist-1 = real history; num_hist..num_hist+H-1 = predicted

    mse = nn.MSELoss(reduction="mean")
    per_step = []
    for step in range(rollout_horizon):
        idx = num_hist + step
        pv = result["visual"][:, idx]
        pp = result["proprio"][:, idx]
        gv = gt_visual[:, idx]
        gp = gt_proprio[:, idx]
        per_step.append({"visual": mse(pv, gv).item(), "proprio": mse(pp, gp).item()})
    return per_step


@torch.no_grad()
def evaluate_multi_step_rollout(
    model, dataset, stats, device, rollout_horizon=5, num_episodes=12,
):
    """Three slicings: pure phase-A, pure phase-B, handoff (crosses stop_frame)."""
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    # Need num_hist real history frames + H frames to compare predictions against.
    total_frames = num_hist + rollout_horizon
    min_ep_len = total_frames * frameskip + 1
    num_eps = len(dataset.meta.episodes)

    slices = {"pure_A": [], "pure_B": [], "handoff": []}
    n_done = {"pure_A": 0, "pure_B": 0, "handoff": 0}
    target_per_slice = num_episodes

    rng = np.random.default_rng(42)
    ep_order = rng.permutation(num_eps)

    for ep_idx in ep_order:
        if all(n_done[k] >= target_per_slice for k in slices):
            break
        ep_start, ep_end = get_episode_bounds(dataset, int(ep_idx))
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue
        ep_phase = get_episode_phase_array(dataset, int(ep_idx))

        max_base = ep_len - total_frames * frameskip
        for base in rng.permutation(max_base + 1)[:6]:  # try a few bases per episode
            frame_offsets = [base + i * frameskip for i in range(total_frames)]
            frame_phases = ep_phase[frame_offsets]

            if (frame_phases == 0).all():
                slice_name = "pure_A"
            elif (frame_phases == 1).all():
                slice_name = "pure_B"
            elif frame_phases[0] == 0 and frame_phases[-1] == 1:
                slice_name = "handoff"
            else:
                continue
            if n_done[slice_name] >= target_per_slice:
                continue

            frame_ids = [ep_start + off for off in frame_offsets]
            try:
                per_step = _rollout_one_window(
                    model, dataset, stats, device, frame_ids, rollout_horizon
                )
            except Exception as ex:
                print(f"  [warn] ep {ep_idx} base {base} skipped: {ex}")
                continue
            slices[slice_name].append(per_step)
            n_done[slice_name] += 1
            print(
                f"  ep {ep_idx:3d} base {base:4d}  → {slice_name:8s} "
                f"({n_done[slice_name]}/{target_per_slice})"
            )

    out = {}
    for slice_name, runs in slices.items():
        if not runs:
            out[slice_name] = {"n": 0}
            continue
        H = max(len(r) for r in runs)
        per_step_agg = {f"step_{s}": {"visual": [], "proprio": []} for s in range(H)}
        for r in runs:
            for s, e in enumerate(r):
                per_step_agg[f"step_{s}"]["visual"].append(e["visual"])
                per_step_agg[f"step_{s}"]["proprio"].append(e["proprio"])
        out[slice_name] = {"n": len(runs)}
        for k, vals in per_step_agg.items():
            v = np.array(vals["visual"])
            p = np.array(vals["proprio"])
            out[slice_name][k] = {
                "visual_mse_mean": float(v.mean()), "visual_mse_std": float(v.std()),
                "proprio_mse_mean": float(p.mean()), "proprio_mse_std": float(p.std()),
            }
    return out


# ---------------------------------------------------------------------------
# M3: CEM goal-distance reduction (two-stage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cem_two_stage(
    model, dataset, stats, device,
    num_episodes=5, cem_num_samples=100, cem_opt_steps=30, cem_topk=10,
    mini_batch=10,
):
    """For each held-out episode (per-episode goals, NO averaging):
       - Phase A CEM: from episode start, plan helper 6D toward THIS
         episode's subgoal frame (ep_start + stop_frame).
       - Phase B CEM: from stop_frame, plan leader 6D toward THIS
         episode's last frame.
    """
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    cam_names = config.camera_names
    h_dim = config.helper_action_dim
    l_dim = config.leader_action_dim

    config.cem_num_samples = cem_num_samples
    config.cem_opt_steps = cem_opt_steps
    config.cem_topk = cem_topk
    model._planner = None
    model._objective_fn = None

    num_eps = len(dataset.meta.episodes)
    min_ep_len = (num_hist + 1) * frameskip + 1
    mse_fn = nn.MSELoss(reduction="mean")
    cos_fn = nn.CosineSimilarity(dim=-1)

    records = []
    for ep_idx in range(num_eps):
        if len(records) >= num_episodes:
            break
        ep_start, ep_end = get_episode_bounds(dataset, ep_idx)
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue
        ep_phase = get_episode_phase_array(dataset, ep_idx)
        if not ((ep_phase == 0).any() and (ep_phase == 1).any()):
            continue
        stop_frame = int(np.argmax(ep_phase == 1))

        rec = {"episode": int(ep_idx), "stop_frame": stop_frame}

        # Per-episode goals (no averaging across episodes)
        sub_frame_idx = ep_start + stop_frame
        fin_frame_idx = ep_end - 1
        try:
            subgoal_z = load_goal_latent_for_frame(
                model, dataset, sub_frame_idx, stats, device,
            )
            finalgoal_z = load_goal_latent_for_frame(
                model, dataset, fin_frame_idx, stats, device,
            )
        except Exception as ex:
            print(f"  [warn] ep {ep_idx} goal load failed: {ex}")
            continue
        episode_goals = {"subgoal": subgoal_z, "final": finalgoal_z}

        for stage, (frame_anchor, goal_key, action_dim) in [
            ("phase_A", (0,           "subgoal", h_dim)),
            ("phase_B", (stop_frame,  "final",   l_dim)),
        ]:
            # Pick num_hist history frames ending at the anchor (or starting from)
            hist_ids = [
                ep_start + max(0, frame_anchor - (num_hist - 1 - i) * frameskip)
                for i in range(num_hist)
            ]
            try:
                imgs, states, actions, _ = load_frame_sequence(
                    dataset, hist_ids, cam_names, device
                )
            except Exception:
                continue

            obs_visual = model._encode_images(
                {c: imgs[c].unsqueeze(0) for c in cam_names}, 1, num_hist,
            )
            obs_proprio = model.proprio_encoder(
                normalize_state(states.unsqueeze(0), stats)
            )
            actions_norm = normalize_action(actions.unsqueeze(0), stats)
            a_helper_hist = actions_norm[:, :, :h_dim]

            # Per-episode goal latent (this specific episode's subgoal / final)
            z_goal = episode_goals[goal_key]
            init_d_v = mse_fn(obs_visual[:, -1], z_goal["visual"]).item()
            init_d_p = mse_fn(obs_proprio[:, -1], z_goal["proprio"]).item()
            init_dist = init_d_v + config.cem_objective_alpha * init_d_p

            # The other arm gets frozen at its current (last history) value
            current = actions_norm[:, -1]                              # (1, 12)
            current_helper = current[:, :h_dim]
            current_leader = current[:, h_dim:]

            planner = model._get_planner(action_dim)
            objective_fn = model._objective_fn

            H = config.cem_horizon
            if stage == "phase_A":
                leader_frozen = current_leader.unsqueeze(1).expand(1, H, l_dim)

                def rollout_fn(action_helper_samples):
                    N = action_helper_samples.shape[0]
                    rs = []
                    for s in range(0, N, mini_batch):
                        e = min(s + mini_batch, N)
                        n = e - s
                        v = repeat(obs_visual, "b t p d -> (b n) t p d", n=n)
                        p = repeat(obs_proprio, "b t d -> (b n) t d", n=n)
                        h_hist = repeat(a_helper_hist, "b t d -> (b n) t d", n=n)
                        l_seq = repeat(leader_frozen, "b h d -> (b n) h d", n=n)
                        out = model._seq_rollout(
                            v, p, h_hist, action_helper_samples[s:e], l_seq
                        )
                        rs.append(out)
                    return {
                        "visual":  torch.cat([r["visual"]  for r in rs], 0),
                        "proprio": torch.cat([r["proprio"] for r in rs], 0),
                    }
            else:  # phase_B: optimize leader
                helper_frozen = current_helper.unsqueeze(1).expand(1, H, h_dim)

                def rollout_fn(action_leader_samples):
                    N = action_leader_samples.shape[0]
                    rs = []
                    for s in range(0, N, mini_batch):
                        e = min(s + mini_batch, N)
                        n = e - s
                        v = repeat(obs_visual, "b t p d -> (b n) t p d", n=n)
                        p = repeat(obs_proprio, "b t d -> (b n) t d", n=n)
                        h_hist = repeat(a_helper_hist, "b t d -> (b n) t d", n=n)
                        h_seq = repeat(helper_frozen, "b h d -> (b n) h d", n=n)
                        out = model._seq_rollout(
                            v, p, h_hist, h_seq, action_leader_samples[s:e]
                        )
                        rs.append(out)
                    return {
                        "visual":  torch.cat([r["visual"]  for r in rs], 0),
                        "proprio": torch.cat([r["proprio"] for r in rs], 0),
                    }

            t0 = time.time()
            best_actions = planner.plan(
                rollout_fn=rollout_fn, objective_fn=objective_fn,
                z_obs_goal=z_goal, device=device,
            )
            plan_time = time.time() - t0

            # Evaluate final-state distance
            if stage == "phase_A":
                final_result = model._seq_rollout(
                    obs_visual, obs_proprio, a_helper_hist,
                    best_actions.unsqueeze(0), leader_frozen,
                )
            else:
                final_result = model._seq_rollout(
                    obs_visual, obs_proprio, a_helper_hist,
                    helper_frozen, best_actions.unsqueeze(0),
                )
            final_v = final_result["visual"][:, -1]
            final_p = final_result["proprio"][:, -1]
            final_d_v = mse_fn(final_v, z_goal["visual"]).item()
            final_d_p = mse_fn(final_p, z_goal["proprio"]).item()
            final_dist = final_d_v + config.cem_objective_alpha * final_d_p
            reduction = 1.0 - final_dist / (init_dist + 1e-8)

            # M4: action agreement (per-arm)
            expert_action = actions_norm[0, -1]   # (12,)
            if stage == "phase_A":
                expert_arm = expert_action[:h_dim]
            else:
                expert_arm = expert_action[h_dim:]
            planned_a0 = best_actions[0]
            a_mse = mse_fn(planned_a0.unsqueeze(0), expert_arm.unsqueeze(0)).item()
            a_cos = cos_fn(planned_a0.unsqueeze(0), expert_arm.unsqueeze(0)).item()

            rec[stage] = {
                "init_dist": init_dist, "final_dist": final_dist,
                "init_d_visual": init_d_v, "final_d_visual": final_d_v,
                "init_d_proprio": init_d_p, "final_d_proprio": final_d_p,
                "dist_reduction": reduction,
                "action_mse_vs_expert": a_mse,
                "action_cos_sim_vs_expert": a_cos,
                "plan_time_s": plan_time,
            }
            print(
                f"  ep {ep_idx:3d}  {stage}: "
                f"init {init_dist:.4f} → final {final_dist:.4f} "
                f"reduction {reduction:+.2%}  "
                f"act_mse {a_mse:.4f}  cos {a_cos:+.3f}  "
                f"({plan_time:.1f}s)"
            )

        if "phase_A" in rec or "phase_B" in rec:
            records.append(rec)

    # Aggregate
    summary = {"phase_A": {}, "phase_B": {}}
    for stage in ["phase_A", "phase_B"]:
        keys = ["init_dist", "final_dist", "dist_reduction",
                "action_mse_vs_expert", "action_cos_sim_vs_expert", "plan_time_s"]
        for k in keys:
            vals = [r[stage][k] for r in records if stage in r]
            if vals:
                summary[stage][k] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n": len(vals),
                }
    return summary, records


# ---------------------------------------------------------------------------
# N1: Phase detector accuracy + transition latency
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_phase_detector(model, dataset, stats, device, num_episodes=20):
    """Replay each episode frame-by-frame through model._detect_phase() and
    compare to the ground-truth `phase` column. Report:
      - per-frame accuracy
      - transition latency (frames after true A→B switch until detector switches)
      - false switches in phase A (detector flips to B prematurely)
    """
    num_eps = len(dataset.meta.episodes)
    metrics = {
        "frame_accuracy": [],
        "switch_latency_frames": [],
        "false_switches_in_A": [],
        "missed_switch_eps": 0,
    }

    rng = np.random.default_rng(0)
    ep_idxs = rng.permutation(num_eps)[:num_episodes]

    for ep_idx in ep_idxs:
        ep_start, ep_end = get_episode_bounds(dataset, int(ep_idx))
        gt_phase = get_episode_phase_array(dataset, int(ep_idx))
        T = len(gt_phase)
        if (gt_phase == 1).any():
            true_switch = int(np.argmax(gt_phase == 1))
        else:
            true_switch = T

        # Reset detector state at start of episode
        model.reset()

        pred_phase = np.zeros(T, dtype=np.int8)
        switched_at = -1
        for t in range(T):
            sample = dataset[ep_start + t]
            left_state = sample["observation.state"][:6].to(device)
            phase = model._detect_phase(left_state)
            pred_phase[t] = 1 if phase == "B" else 0
            if pred_phase[t] == 1 and switched_at < 0:
                switched_at = t

        # Compare
        acc = float((pred_phase == gt_phase).mean())
        if true_switch < T and switched_at >= 0:
            latency = switched_at - true_switch    # negative = switched too early
            metrics["switch_latency_frames"].append(latency)
        elif true_switch < T and switched_at < 0:
            metrics["missed_switch_eps"] += 1
        # Count premature switches: predictions flipping to 1 before true_switch
        false_in_A = int(((pred_phase[:true_switch] == 1)).sum())
        metrics["false_switches_in_A"].append(false_in_A)
        metrics["frame_accuracy"].append(acc)

        print(
            f"  ep {int(ep_idx):3d}: T={T:4d}  true_switch={true_switch:4d}  "
            f"detected@={switched_at:4d}  acc={acc:.3f}  "
            f"false_A={false_in_A}"
        )

    out = {
        "n_episodes": int(num_episodes),
        "frame_accuracy_mean": float(np.mean(metrics["frame_accuracy"])),
        "frame_accuracy_std":  float(np.std(metrics["frame_accuracy"])),
        "missed_switch_eps":   int(metrics["missed_switch_eps"]),
        "false_switches_in_A_mean": float(np.mean(metrics["false_switches_in_A"])),
    }
    if metrics["switch_latency_frames"]:
        lat = np.array(metrics["switch_latency_frames"])
        out["switch_latency_mean_frames"] = float(lat.mean())
        out["switch_latency_median_frames"] = float(np.median(lat))
        out["switch_latency_p95_frames"] = float(np.percentile(lat, 95))
        out["switch_latency_p5_frames"] = float(np.percentile(lat, 5))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Offline DINO-SeqWM evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, default="haodoz0118/bimanual_cooperate")
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # M1
    p.add_argument("--num_eval_samples", type=int, default=80)
    # M2
    p.add_argument("--num_rollout_episodes", type=int, default=12,
                   help="Per-slice target sample count")
    p.add_argument("--rollout_horizon", type=int, default=5)
    # M3 & M4
    p.add_argument("--num_cem_episodes", type=int, default=5)
    p.add_argument("--cem_num_samples", type=int, default=100)
    p.add_argument("--cem_opt_steps", type=int, default=30)
    p.add_argument("--cem_topk", type=int, default=10)
    p.add_argument("--cem_mini_batch", type=int, default=10)
    # N1
    p.add_argument("--num_phase_detector_episodes", type=int, default=20)

    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated metrics to skip: M1,M2,M3,N1")
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print("DINO-SeqWM Offline Evaluation")
    print("=" * 70)

    print("\n[1/3] Loading model …")
    from lerobot.policies.dino_seqwm.modeling_dino_seqwm import DinoSeqWMPolicy
    model = DinoSeqWMPolicy.from_pretrained(args.checkpoint)
    model.to(device).eval()
    print(f"  cameras       : {model.config.camera_names}")
    print(f"  num_hist      : {model.config.num_hist}")
    print(f"  state_dim     : {model.config.state_dim}")
    print(f"  helper_dim    : {model.config.helper_action_dim}")
    print(f"  leader_dim    : {model.config.leader_action_dim}")
    print(f"  predictor_d   : {model.config.predictor_depth}")

    stats = load_norm_stats(args.checkpoint, device)

    print("\n[2/3] Loading dataset …")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"  frames    : {len(dataset)}")
    print(f"  episodes  : {len(dataset.meta.episodes)}")

    # Goals are loaded per-episode inside evaluate_cem_two_stage; nothing to
    # pre-compute here.

    all_results = {
        "config": vars(args),
        "model_config": {
            "camera_names": model.config.camera_names,
            "predictor_depth": model.config.predictor_depth,
            "predictor_heads": model.config.predictor_heads,
            "num_hist": model.config.num_hist,
            "frameskip": model.config.frameskip,
        },
    }

    print("\n[3/3] Running metrics …")

    if "M1" not in skip:
        print("\n" + "=" * 70)
        print("M1 — Single-step prediction loss (stratified by phase)")
        print("=" * 70)
        t0 = time.time()
        m1 = evaluate_single_step_per_phase(
            model, dataset, stats, device, num_samples=args.num_eval_samples,
        )
        print(f"  done in {time.time() - t0:.1f}s")
        for ph in ["A", "B"]:
            r = m1.get(ph, {})
            print(f"  phase {ph} (n={r.get('n', 0)}):")
            for k in ["z_loss", "z_loss_joint", "z_loss_helper"]:
                if k in r:
                    print(f"    {k:<18s}: {r[k]['mean']:.6f} ± {r[k]['std']:.6f}")
        all_results["M1_single_step_per_phase"] = m1

    if "M2" not in skip:
        print("\n" + "=" * 70)
        print("M2 — Multi-step rollout error (pure-A / pure-B / handoff)")
        print(f"  rollout_horizon = {args.rollout_horizon}")
        print("=" * 70)
        t0 = time.time()
        m2 = evaluate_multi_step_rollout(
            model, dataset, stats, device,
            rollout_horizon=args.rollout_horizon,
            num_episodes=args.num_rollout_episodes,
        )
        print(f"  done in {time.time() - t0:.1f}s")
        for slc in ["pure_A", "pure_B", "handoff"]:
            r = m2.get(slc, {})
            print(f"\n  {slc} (n={r.get('n', 0)}):")
            for k in sorted(r.keys()):
                if not k.startswith("step_"):
                    continue
                v = r[k]
                print(
                    f"    {k:<8s}: visual {v['visual_mse_mean']:.6f}±{v['visual_mse_std']:.6f}"
                    f"  proprio {v['proprio_mse_mean']:.6f}±{v['proprio_mse_std']:.6f}"
                )
        all_results["M2_rollout_by_slice"] = m2

    if "M3" not in skip:
        print("\n" + "=" * 70)
        print("M3 + M4 — Two-stage CEM goal-distance & action agreement")
        print(
            f"  samples={args.cem_num_samples} opt_steps={args.cem_opt_steps} "
            f"topk={args.cem_topk} mini_batch={args.cem_mini_batch}"
        )
        print("=" * 70)
        t0 = time.time()
        cem_summary, cem_details = evaluate_cem_two_stage(
            model, dataset, stats, device,
            num_episodes=args.num_cem_episodes,
            cem_num_samples=args.cem_num_samples,
            cem_opt_steps=args.cem_opt_steps,
            cem_topk=args.cem_topk,
            mini_batch=args.cem_mini_batch,
        )
        print(f"\n  done in {time.time() - t0:.1f}s")
        for stage in ["phase_A", "phase_B"]:
            r = cem_summary.get(stage, {})
            print(f"\n  {stage} (CEM):")
            for k in ["dist_reduction", "init_dist", "final_dist",
                      "action_mse_vs_expert", "action_cos_sim_vs_expert",
                      "plan_time_s"]:
                if k in r:
                    v = r[k]
                    print(f"    {k:<28s}: {v['mean']:>8.4f} ± {v['std']:.4f}  (n={v['n']})")
        all_results["M3_M4_cem_two_stage"] = {
            "summary": cem_summary, "per_episode": cem_details,
        }

    if "N1" not in skip:
        print("\n" + "=" * 70)
        print("N1 — Phase detector accuracy & latency")
        print("=" * 70)
        t0 = time.time()
        n1 = evaluate_phase_detector(
            model, dataset, stats, device,
            num_episodes=args.num_phase_detector_episodes,
        )
        print(f"\n  done in {time.time() - t0:.1f}s")
        print(f"  frame_accuracy:        {n1['frame_accuracy_mean']:.3f} ± {n1['frame_accuracy_std']:.3f}")
        print(f"  missed_switch_eps:     {n1['missed_switch_eps']} / {n1['n_episodes']}")
        print(f"  false_switches_in_A:   {n1['false_switches_in_A_mean']:.2f}  (mean per ep)")
        if "switch_latency_mean_frames" in n1:
            print(
                f"  switch_latency:        median {n1['switch_latency_median_frames']:+.1f} fr  "
                f"P95 {n1['switch_latency_p95_frames']:+.1f}  "
                f"P5 {n1['switch_latency_p5_frames']:+.1f}"
            )
        all_results["N1_phase_detector"] = n1

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda o: str(o))
        print(f"\nSaved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

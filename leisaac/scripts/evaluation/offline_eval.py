#!/usr/bin/env python3
"""Offline evaluation of DINO-WM world model and CEM planning quality.

No simulator needed — runs entirely on the recorded dataset.

Metrics:
  [World Model Quality]
  1. Single-step prediction loss (z_loss, z_visual_loss, z_proprio_loss) on held-out data
  2. Multi-step rollout cumulative error (MSE vs ground truth at each horizon step)

  [CEM Planning Quality]
  3. CEM goal latent distance reduction ratio
  4. Action agreement with expert (MSE + cosine similarity)

Usage:
  python scripts/evaluation/offline_eval.py \
      --checkpoint /path/to/pretrained_model \
      --dataset_repo_id haodoz0118/PickAndPlace \
      --device cuda \
      --num_eval_samples 50 \
      --num_rollout_episodes 10 \
      --rollout_horizon 5 \
      --num_cem_episodes 5
"""

import argparse
import json
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def load_norm_stats(checkpoint_path: str, device: torch.device) -> dict:
    """Load action / state mean+std from the checkpoint safetensors file."""
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


def normalize_state(state: torch.Tensor, stats: dict) -> torch.Tensor:
    return (state - stats["state_mean"]) / (stats["state_std"] + 1e-8)


def normalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    return (action - stats["action_mean"]) / (stats["action_std"] + 1e-8)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_episode_bounds(dataset, ep_idx: int) -> tuple[int, int]:
    """Return (start_frame_idx, end_frame_idx) for an episode.

    ``end`` is exclusive (one-past-last).
    """
    ep = dataset.meta.episodes[ep_idx]
    return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])


def load_frame(dataset, idx: int, camera_names: list[str], device: torch.device):
    """Load a single frame's images, state, and action from the dataset."""
    sample = dataset[idx]
    images = {}
    for cam in camera_names:
        key = f"observation.images.{cam}"
        img = sample[key]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        images[cam] = img.to(device)

    state = sample["observation.state"].to(device)
    action = sample["action"].to(device)
    return images, state, action


def load_frame_sequence(
    dataset, indices: list[int], camera_names: list[str], device: torch.device
):
    """Load a sequence of frames and stack them into (T, ...) tensors."""
    all_images = {cam: [] for cam in camera_names}
    all_states = []
    all_actions = []
    for idx in indices:
        images, state, action = load_frame(dataset, idx, camera_names, device)
        for cam in camera_names:
            all_images[cam].append(images[cam])
        all_states.append(state)
        all_actions.append(action)

    stacked_images = {cam: torch.stack(all_images[cam]) for cam in camera_names}
    stacked_states = torch.stack(all_states)
    stacked_actions = torch.stack(all_actions)
    return stacked_images, stacked_states, stacked_actions


# ---------------------------------------------------------------------------
# Metric 1: Single-step prediction loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_step(model, dataset, stats, device, num_samples=50):
    """Run model.forward() on dataset windows and collect z_loss.

    Each sample uses the same temporal structure as training:
    3 history frames + 1 target frame (spaced by frameskip).
    """
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    camera_names = config.camera_names
    total_window = num_hist + config.num_pred  # 3 + 1 = 4

    min_ep_len = total_window * frameskip + 1
    num_episodes = len(dataset.meta.episodes)

    losses = defaultdict(list)
    count = 0

    for ep_idx in range(num_episodes):
        if count >= num_samples:
            break

        ep_start, ep_end = get_episode_bounds(dataset, ep_idx)
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue

        max_base = ep_len - total_window * frameskip
        n_from_ep = min(5, max_base + 1, num_samples - count)
        bases = np.random.choice(max_base + 1, size=n_from_ep, replace=False)

        for base in bases:
            frame_ids = [
                ep_start + base + i * frameskip for i in range(total_window)
            ]

            images_seq, states_seq, actions_seq = load_frame_sequence(
                dataset, frame_ids, camera_names, device
            )

            batch = {}
            for cam in camera_names:
                batch[f"observation.images.{cam}"] = images_seq[cam].unsqueeze(0)
            batch["observation.state"] = normalize_state(
                states_seq.unsqueeze(0), stats
            )
            batch["action"] = normalize_action(
                actions_seq[:num_hist].unsqueeze(0), stats
            )

            loss, loss_dict = model.forward(batch)
            losses["total_loss"].append(loss.item())
            for k, v in loss_dict.items():
                losses[k].append(v)
            count += 1

    results = {}
    for k, vals in losses.items():
        arr = np.array(vals)
        results[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }
    return results


# ---------------------------------------------------------------------------
# Metric 2: Multi-step rollout cumulative error
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_multi_step_rollout(
    model, dataset, stats, device, rollout_horizon=5, num_episodes=10
):
    """Rollout with ground-truth actions and compare with ground-truth latents.

    At each rollout step t ∈ [0, rollout_horizon], report
    MSE(predicted_latent, ground_truth_latent).
    """
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    camera_names = config.camera_names
    total_frames = num_hist + rollout_horizon + 1
    min_ep_len = total_frames * frameskip + 1

    num_ep_total = len(dataset.meta.episodes)
    step_errors = defaultdict(list)
    mse_fn = nn.MSELoss(reduction="mean")

    eval_count = 0
    for ep_idx in range(num_ep_total):
        if eval_count >= num_episodes:
            break

        ep_start, ep_end = get_episode_bounds(dataset, ep_idx)
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue

        max_base = ep_len - total_frames * frameskip
        base = np.random.randint(0, max_base + 1)

        frame_ids = [
            ep_start + base + i * frameskip for i in range(total_frames)
        ]
        images_seq, states_seq, actions_seq = load_frame_sequence(
            dataset, frame_ids, camera_names, device
        )

        # Encode ALL frames with DINOv3 -> ground truth visual tokens
        imgs_batch = {
            cam: images_seq[cam].unsqueeze(0) for cam in camera_names
        }
        gt_visual = model._encode_images(imgs_batch, 1, total_frames)
        gt_proprio = model.proprio_encoder(
            normalize_state(states_seq.unsqueeze(0), stats)
        )

        obs_visual = gt_visual[:, :num_hist]
        obs_proprio = gt_proprio[:, :num_hist]

        # Actions for rollout: action at frame (num_hist-1) drives the
        # transition from last history frame to first predicted frame, etc.
        rollout_action_ids = list(range(num_hist - 1, num_hist - 1 + rollout_horizon))
        rollout_actions = normalize_action(
            actions_seq[rollout_action_ids].unsqueeze(0), stats
        )

        result = model._rollout(obs_visual, obs_proprio, rollout_actions)

        for step in range(rollout_horizon + 1):
            pred_v = result["visual"][:, num_hist + step]
            pred_p = result["proprio"][:, num_hist + step]
            gt_v = gt_visual[:, num_hist + step]
            gt_p = gt_proprio[:, num_hist + step]

            v_mse = mse_fn(pred_v, gt_v).item()
            p_mse = mse_fn(pred_p, gt_p).item()
            step_errors[step].append({"visual": v_mse, "proprio": p_mse})

        eval_count += 1
        print(f"  Episode {ep_idx} done")

    results = {}
    for step in sorted(step_errors.keys()):
        errs = step_errors[step]
        v = np.array([e["visual"] for e in errs])
        p = np.array([e["proprio"] for e in errs])
        results[f"step_{step}"] = {
            "visual_mse_mean": float(v.mean()),
            "visual_mse_std": float(v.std()),
            "proprio_mse_mean": float(p.mean()),
            "proprio_mse_std": float(p.std()),
        }
    return results


# ---------------------------------------------------------------------------
# Metric 3 & 4: CEM planning quality
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cem_planning(
    model,
    dataset,
    stats,
    device,
    num_episodes=5,
    cem_num_samples=100,
    cem_opt_steps=30,
    cem_topk=10,
    mini_batch=10,
    goal_frame=-1,
):
    """Plan with CEM from episode start toward episode end and measure
    goal distance reduction plus action agreement with the expert.
    """
    config = model.config
    frameskip = config.frameskip
    num_hist = config.num_hist
    camera_names = config.camera_names

    config.cem_num_samples = cem_num_samples
    config.cem_opt_steps = cem_opt_steps
    config.cem_topk = cem_topk
    model._planner = None  # force planner re-init
    model._objective_fn = None  # force objective re-init

    num_ep_total = len(dataset.meta.episodes)
    min_ep_len = (num_hist + 1) * frameskip + 1
    mse_fn = nn.MSELoss(reduction="mean")
    cos_fn = nn.CosineSimilarity(dim=-1)

    records = []
    eval_count = 0

    for ep_idx in range(num_ep_total):
        if eval_count >= num_episodes:
            break

        ep_start, ep_end = get_episode_bounds(dataset, ep_idx)
        ep_len = ep_end - ep_start
        if ep_len < min_ep_len:
            continue

        # History: first num_hist frames (frameskipped) of the episode
        hist_ids = [ep_start + i * frameskip for i in range(num_hist)]

        # Goal frame: -1 = last frame, positive = offset from episode start
        if goal_frame < 0:
            goal_idx = ep_end - 1
        else:
            goal_idx = min(ep_start + goal_frame, ep_end - 1)

        # Expert action at last history frame
        expert_action = normalize_action(
            dataset[hist_ids[-1]]["action"].to(device), stats
        )

        hist_imgs, hist_states, _ = load_frame_sequence(
            dataset, hist_ids, camera_names, device
        )
        goal_imgs, goal_state, _ = load_frame_sequence(
            dataset, [goal_idx], camera_names, device
        )

        # Encode history
        obs_visual = model._encode_images(
            {c: hist_imgs[c].unsqueeze(0) for c in camera_names}, 1, num_hist
        )
        obs_proprio = model.proprio_encoder(
            normalize_state(hist_states.unsqueeze(0), stats)
        )

        # Encode goal
        goal_visual = model._encode_images(
            {c: goal_imgs[c].unsqueeze(0) for c in camera_names}, 1, 1
        )
        goal_proprio = model.proprio_encoder(
            normalize_state(goal_state.unsqueeze(0), stats)
        )

        z_goal = {
            "visual": goal_visual[:, 0],
            "proprio": goal_proprio[:, 0],
        }

        # Initial distance
        init_d_v = mse_fn(obs_visual[:, -1], z_goal["visual"]).item()
        init_d_p = mse_fn(obs_proprio[:, -1], z_goal["proprio"]).item()
        init_dist = init_d_v + config.cem_objective_alpha * init_d_p

        # CEM planning
        planner = model._get_planner()
        objective_fn = model._objective_fn

        def rollout_fn(action_batch):
            N = action_batch.shape[0]
            out_v, out_p = [], []
            for s in range(0, N, mini_batch):
                e = min(s + mini_batch, N)
                n = e - s
                v = repeat(obs_visual, "b t p d -> (b n) t p d", n=n)
                p = repeat(obs_proprio, "b t d -> (b n) t d", n=n)
                r = model._rollout(v, p, action_batch[s:e])
                out_v.append(r["visual"])
                out_p.append(r["proprio"])
            return {
                "visual": torch.cat(out_v, 0),
                "proprio": torch.cat(out_p, 0),
            }

        t0 = time.time()
        best_actions = planner.plan(
            rollout_fn=rollout_fn,
            objective_fn=objective_fn,
            z_obs_goal=z_goal,
            device=device,
        )
        plan_time = time.time() - t0

        # Rollout best actions
        best_result = model._rollout(
            obs_visual, obs_proprio, best_actions.unsqueeze(0)
        )
        final_v = best_result["visual"][:, -1]
        final_p = best_result["proprio"][:, -1]
        final_d_v = mse_fn(final_v, z_goal["visual"]).item()
        final_d_p = mse_fn(final_p, z_goal["proprio"]).item()
        final_dist = final_d_v + config.cem_objective_alpha * final_d_p

        dist_reduction = 1.0 - final_dist / (init_dist + 1e-8)

        # Action agreement
        planned_a0 = best_actions[0]
        a_mse = mse_fn(planned_a0.unsqueeze(0), expert_action.unsqueeze(0)).item()
        a_cos = cos_fn(planned_a0.unsqueeze(0), expert_action.unsqueeze(0)).item()

        rec = {
            "episode": ep_idx,
            "init_dist": init_dist,
            "init_dist_visual": init_d_v,
            "init_dist_proprio": init_d_p,
            "final_dist": final_dist,
            "final_dist_visual": final_d_v,
            "final_dist_proprio": final_d_p,
            "dist_reduction": dist_reduction,
            "action_mse": a_mse,
            "action_cos_sim": a_cos,
            "plan_time_s": plan_time,
        }
        records.append(rec)
        eval_count += 1

        print(
            f"  Episode {ep_idx}: "
            f"init={init_dist:.4f} -> final={final_dist:.4f} "
            f"reduction={dist_reduction:+.2%} | "
            f"act_mse={a_mse:.4f} cos={a_cos:.4f} | "
            f"{plan_time:.1f}s"
        )

    summary = {}
    if records:
        for key in records[0]:
            if key == "episode":
                continue
            vals = np.array([r[key] for r in records])
            summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    return summary, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline DINO-WM evaluation (no simulator needed)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to pretrained_model directory",
    )
    parser.add_argument(
        "--dataset_repo_id", type=str, default="haodoz0118/PickAndPlace",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Metric 1
    parser.add_argument(
        "--num_eval_samples", type=int, default=50,
        help="Number of samples for single-step prediction evaluation",
    )
    # Metric 2
    parser.add_argument(
        "--num_rollout_episodes", type=int, default=10,
        help="Number of episodes for multi-step rollout evaluation",
    )
    parser.add_argument("--rollout_horizon", type=int, default=5)
    # Metric 3 & 4
    parser.add_argument(
        "--num_cem_episodes", type=int, default=5,
        help="Number of episodes for CEM planning evaluation",
    )
    parser.add_argument("--cem_num_samples", type=int, default=100)
    parser.add_argument("--cem_opt_steps", type=int, default=30)
    parser.add_argument("--cem_topk", type=int, default=10)
    parser.add_argument("--cem_mini_batch", type=int, default=10)
    parser.add_argument(
        "--cem_objective_alpha", type=float, default=None,
        help="Proprio weight in CEM objective. None = use model default (0.1). Set 0 to disable.",
    )
    parser.add_argument(
        "--goal_frame", type=int, default=-1,
        help="Goal frame offset from episode start. -1 = last frame. e.g. 50 = frame 50.",
    )

    # Output
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="If set, save all metrics to this JSON file",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("DINO-WM Offline Evaluation")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model …")
    from lerobot.policies.dino_wm_test.modeling_dino_wm_test import DinoWMTestPolicy

    model = DinoWMTestPolicy.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    print(f"  camera_names : {model.config.camera_names}")
    print(f"  num_hist     : {model.config.num_hist}")
    print(f"  frameskip    : {model.config.frameskip}")
    print(f"  state_dim    : {model.config.state_dim}")
    print(f"  action_dim   : {model.config.action_dim}")

    stats = load_norm_stats(args.checkpoint, device)
    print(f"  action  mean : {stats['action_mean'].cpu().numpy()}")
    print(f"  action  std  : {stats['action_std'].cpu().numpy()}")
    print(f"  state   mean : {stats['state_mean'].cpu().numpy()}")
    print(f"  state   std  : {stats['state_std'].cpu().numpy()}")

    # Load dataset
    print("\n[2/3] Loading dataset …")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id)
    num_frames = len(dataset)
    num_episodes = len(dataset.meta.episodes)
    print(f"  repo_id   : {args.dataset_repo_id}")
    print(f"  frames    : {num_frames}")
    print(f"  episodes  : {num_episodes}")

    all_results = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "goal_frame": args.goal_frame,
            "cem_objective_alpha": model.config.cem_objective_alpha,
        },
    }

    # ------------------------------------------------------------------
    # Metric 1: Single-step prediction loss
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Metric 1 — Single-step Prediction Loss")
    print("=" * 70)
    t0 = time.time()
    ss = evaluate_single_step(
        model, dataset, stats, device, num_samples=args.num_eval_samples
    )
    print(f"  (completed in {time.time() - t0:.1f}s)")
    for k in sorted(ss.keys()):
        v = ss[k]
        print(
            f"  {k:<20s}: "
            f"{v['mean']:.6f} ± {v['std']:.6f}  "
            f"[{v['min']:.6f}, {v['max']:.6f}]  n={v['n']}"
        )
    all_results["single_step"] = ss

    # ------------------------------------------------------------------
    # Metric 2: Multi-step rollout cumulative error
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Metric 2 — Multi-step Rollout Cumulative Error")
    print(f"  rollout_horizon = {args.rollout_horizon}")
    print("=" * 70)
    t0 = time.time()
    ms = evaluate_multi_step_rollout(
        model,
        dataset,
        stats,
        device,
        rollout_horizon=args.rollout_horizon,
        num_episodes=args.num_rollout_episodes,
    )
    print(f"  (completed in {time.time() - t0:.1f}s)\n")
    header = f"  {'Step':<6} {'Visual MSE':>22}  {'Proprio MSE':>22}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for key in sorted(ms.keys(), key=lambda x: int(x.split("_")[1])):
        r = ms[key]
        step = key.split("_")[1]
        print(
            f"  {step:<6} "
            f"{r['visual_mse_mean']:>10.6f} ± {r['visual_mse_std']:<9.6f}  "
            f"{r['proprio_mse_mean']:>10.6f} ± {r['proprio_mse_std']:<9.6f}"
        )
    all_results["multi_step_rollout"] = ms

    # ------------------------------------------------------------------
    # Metric 3 & 4: CEM planning quality
    # ------------------------------------------------------------------
    if args.cem_objective_alpha is not None:
        model.config.cem_objective_alpha = args.cem_objective_alpha
        print(f"\n  [Override] cem_objective_alpha = {args.cem_objective_alpha}")

    print("\n" + "=" * 70)
    print("Metric 3 & 4 — CEM Planning Quality")
    goal_label = "last" if args.goal_frame < 0 else f"frame {args.goal_frame}"
    print(
        f"  samples={args.cem_num_samples}  "
        f"opt_steps={args.cem_opt_steps}  "
        f"topk={args.cem_topk}  "
        f"mini_batch={args.cem_mini_batch}  "
        f"alpha={model.config.cem_objective_alpha}  "
        f"goal={goal_label}"
    )
    print("=" * 70)
    t0 = time.time()
    cem_summary, cem_details = evaluate_cem_planning(
        model,
        dataset,
        stats,
        device,
        num_episodes=args.num_cem_episodes,
        cem_num_samples=args.cem_num_samples,
        cem_opt_steps=args.cem_opt_steps,
        cem_topk=args.cem_topk,
        mini_batch=args.cem_mini_batch,
        goal_frame=args.goal_frame,
    )
    print(f"\n  (completed in {time.time() - t0:.1f}s)")
    print("\n  Aggregated:")
    for k in [
        "dist_reduction",
        "init_dist",
        "final_dist",
        "init_dist_visual",
        "final_dist_visual",
        "init_dist_proprio",
        "final_dist_proprio",
        "action_mse",
        "action_cos_sim",
        "plan_time_s",
    ]:
        if k in cem_summary:
            v = cem_summary[k]
            print(f"    {k:<22s}: {v['mean']:>10.4f} ± {v['std']:.4f}")
    all_results["cem_planning"] = {
        "summary": cem_summary,
        "per_episode": cem_details,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

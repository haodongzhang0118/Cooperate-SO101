"""Final per-episode phase analysis + filtering for bimanual_cooperate.

Outputs (in `bimanual_cooperate_meta/`):
  episode_phases.json    Per-episode {stop_frame, left_freeze_pose, std stats,
                         keep, filter_reasons}.
  kept_episodes.txt      Whitespace-separated kept episode indices.
  frame_phases.npy       (N,) int8 — per-frame phase label (0=A, 1=B). Use as
                         a sidecar at training time, indexed by `index`.
  phase_summary.md       Human-readable report.

Design summary the metadata enforces:
  - Per-episode adaptive STILL threshold (5% of that episode's left peak speed)
  - stop_frame = first frame whose 1s window is >=90% still AND remainder of
    episode is >=80% still
  - LEFT_FREEZE_POSE = mean of left-arm position over phase B (per episode,
    NOT a global constant — handles the manual-recording variability)
  - Filter out episodes where phase B is too short, left arm wasn't really
    locked, or stop_frame too early
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/Users/zhanghaodong/Desktop/CSCI2951K/Cooperate-SO101")
DATASET = ROOT / "bimanual_cooperate"
OUT = ROOT / "bimanual_cooperate_meta"
OUT.mkdir(exist_ok=True)

# ----- Filter thresholds (tune here) -----
MIN_PHASE_B_S         = 3.0    # Phase B must last at least 3 seconds
MIN_PHASE_A_S         = 4.0    # Phase A must last at least 4 seconds
MAX_LEFT_STD_PHASE_B  = 1.0    # Left arm std during phase B (non-gripper, deg)
MAX_RIGHT_STD_PHASE_A = 10.0   # Right arm should stay roughly idle during phase A
MAX_PHASE_A_S         = 13.0   # Drop episodes where left arm took too long
SPEED_THR_FRAC_OF_PEAK = 0.05  # "Moving" if speed > 5% of episode peak
SPEED_THR_FLOOR       = 0.005  # Floor for very-slow episodes
SMOOTH_WINDOW         = 15     # Frames for moving-avg speed smoothing (0.5s)
FPS                   = 30

# ----- Load parquet -----
print("Loading parquet ...")
tbl = pq.read_table(DATASET / "data/chunk-000/file-000.parquet",
                    columns=["episode_index", "frame_index", "index",
                             "observation.state", "action"])
df = tbl.to_pandas()
state = np.stack(df["observation.state"].to_numpy())   # (N, 12)
action = np.stack(df["action"].to_numpy())             # (N, 12)
ep_idx = df["episode_index"].to_numpy()
global_idx = df["index"].to_numpy()
N = len(df)
print(f"  rows={N}  episodes={int(ep_idx.max())+1}")

# Per-episode finite-difference and smoothed speed
left_pos = state[:, :5]    # exclude gripper noise
right_pos = state[:, 6:11]
left_speed = np.zeros(N)
right_speed = np.zeros(N)
ep_unique = np.unique(ep_idx)

for ep in ep_unique:
    m = ep_idx == ep
    lp, rp = left_pos[m], right_pos[m]
    if len(lp) < 2:
        continue
    dl = np.diff(lp, axis=0, prepend=lp[:1])
    dr = np.diff(rp, axis=0, prepend=rp[:1])
    sl = np.linalg.norm(dl, axis=1)
    sr = np.linalg.norm(dr, axis=1)
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    left_speed[m] = np.convolve(sl, kernel, mode="same")
    right_speed[m] = np.convolve(sr, kernel, mode="same")

# Per-frame phase label (filled in below)
frame_phase = np.zeros(N, dtype=np.int8)

per_ep = []

for ep in ep_unique:
    m = ep_idx == ep
    rows = np.where(m)[0]
    T = len(rows)
    ls = left_speed[rows]
    rs = right_speed[rows]
    st = state[rows]

    peak = float(ls.max()) if T > 0 else 0.0
    thr = max(peak * SPEED_THR_FRAC_OF_PEAK, SPEED_THR_FLOOR)
    moving = ls > thr

    # ----- Backward search: find LAST moving frame, stop_frame = right after it.
    # Robust against transient mid-episode pauses (which a forward search would
    # mistakenly latch onto). Adds a small post-buffer so we don't include the
    # tail of the deceleration in phase B.
    POST_BUFFER_S = 0.3
    POST_BUFFER_FR = int(POST_BUFFER_S * FPS)
    if moving.any():
        last_moving = int(np.where(moving)[0][-1])
        stop_frame = min(last_moving + 1 + POST_BUFFER_FR, T)
    else:
        # Left arm never moved at all — odd, treat whole episode as phase A.
        stop_frame = T

    # Sanity: if stop_frame == T, phase B is empty (will be filtered).
    # If stop_frame < still_window_fr, the implied phase A is too short
    # (handled by MIN_PHASE_A_S filter).

    pA = slice(0, stop_frame)
    pB = slice(stop_frame, T)
    nA, nB = pA.stop - pA.start, pB.stop - pB.start

    # Per-episode left freeze pose (the position this specific run settled at)
    left_freeze = st[pB, :6].mean(axis=0) if nB > 0 else np.full(6, np.nan)
    left_freeze_std = st[pB, :6].std(axis=0) if nB > 1 else np.zeros(6)

    # Phase B left arm stability — used for filtering
    left_std_max_nongrip = float(left_freeze_std[:5].max()) if nB > 1 else 0.0

    # Right arm pose drift during phase A (informational, not filter)
    right_std_A = st[pA, 6:].std(axis=0) if nA > 1 else np.zeros(6)

    # ----- Filter rules -----
    right_std_max_nongrip = float(right_std_A[:5].max()) if nA > 1 else 0.0
    reasons = []
    if nB / FPS < MIN_PHASE_B_S:
        reasons.append(f"phase_B_short({nB/FPS:.2f}s)")
    if nA / FPS < MIN_PHASE_A_S:
        reasons.append(f"phase_A_short({nA/FPS:.2f}s)")
    if nA / FPS > MAX_PHASE_A_S:
        reasons.append(f"phase_A_long({nA/FPS:.2f}s)")
    if left_std_max_nongrip > MAX_LEFT_STD_PHASE_B:
        reasons.append(f"left_phaseB_unstable({left_std_max_nongrip:.2f}deg)")
    if right_std_max_nongrip > MAX_RIGHT_STD_PHASE_A:
        reasons.append(f"right_phaseA_drift({right_std_max_nongrip:.2f}deg)")
    keep = len(reasons) == 0

    # Speed stats per phase (informational)
    rec = {
        "ep": int(ep),
        "T": int(T),
        "stop_frame": int(stop_frame),
        "stop_time_s": float(stop_frame / FPS),
        "phase_A_s": float(nA / FPS),
        "phase_B_s": float(nB / FPS),
        "left_freeze_pose": left_freeze.tolist(),
        "left_phaseB_std": left_freeze_std.tolist(),
        "left_phaseB_max_std_nongrip": left_std_max_nongrip,
        "right_phaseA_std": right_std_A.tolist(),
        "right_phaseA_max_std_nongrip": right_std_max_nongrip if nA > 1 else 0.0,
        "left_speed_A_mean": float(ls[pA].mean()) if nA > 0 else 0.0,
        "right_speed_A_mean": float(rs[pA].mean()) if nA > 0 else 0.0,
        "left_speed_B_mean": float(ls[pB].mean()) if nB > 0 else 0.0,
        "right_speed_B_mean": float(rs[pB].mean()) if nB > 0 else 0.0,
        "speed_threshold_used": float(thr),
        "left_peak_speed": peak,
        "keep": keep,
        "filter_reasons": reasons,
        "global_idx_start": int(global_idx[rows[0]]),
        "global_idx_end": int(global_idx[rows[-1]]),
    }
    per_ep.append(rec)

    # Fill frame phase
    frame_phase[rows[:stop_frame]] = 0
    frame_phase[rows[stop_frame:]] = 1

# ----- Aggregate -----
kept = [r for r in per_ep if r["keep"]]
filtered = [r for r in per_ep if not r["keep"]]

# Reason histogram
reason_counts = {}
for r in filtered:
    for reason in r["filter_reasons"]:
        key = reason.split("(")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1

print(f"\nKept episodes: {len(kept)}/{len(per_ep)} ({len(kept)/len(per_ep)*100:.1f}%)")
print(f"Filtered out: {len(filtered)}")
for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:35s}: {v}")

# Frame-level kept stats
kept_eps_set = {r["ep"] for r in kept}
kept_frame_mask = np.isin(ep_idx, list(kept_eps_set))
print(f"\nFrames kept: {kept_frame_mask.sum()}/{N} ({kept_frame_mask.sum()/N*100:.1f}%)")
print(f"  Phase A frames (kept): {((frame_phase==0) & kept_frame_mask).sum()}")
print(f"  Phase B frames (kept): {((frame_phase==1) & kept_frame_mask).sum()}")

# Median freeze pose across KEPT episodes
homes = np.array([r["left_freeze_pose"] for r in kept])
print(f"\nLeft freeze pose distribution across {len(kept)} KEPT episodes:")
print(f"  median: {np.array2string(np.median(homes, axis=0), precision=2)}")
print(f"  std:    {np.array2string(np.std(homes, axis=0), precision=2)}")
print(f"  width (5–95%): {np.array2string(np.percentile(homes, 95, axis=0) - np.percentile(homes, 5, axis=0), precision=2)}")
print("  → Use PER-EPISODE freeze pose, not a global constant.")

# ----- Save -----
phases_path = OUT / "episode_phases.json"
with open(phases_path, "w") as f:
    json.dump({
        "config": {
            "MIN_PHASE_B_S": MIN_PHASE_B_S,
            "MIN_PHASE_A_S": MIN_PHASE_A_S,
            "MAX_LEFT_STD_PHASE_B": MAX_LEFT_STD_PHASE_B,
            "SPEED_THR_FRAC_OF_PEAK": SPEED_THR_FRAC_OF_PEAK,
            "SPEED_THR_FLOOR": SPEED_THR_FLOOR,
            "SMOOTH_WINDOW": SMOOTH_WINDOW,
            "FPS": FPS,
        },
        "summary": {
            "total_episodes": len(per_ep),
            "kept": len(kept),
            "filtered": len(filtered),
            "filter_reason_counts": reason_counts,
            "kept_frames": int(kept_frame_mask.sum()),
            "kept_phase_A_frames": int(((frame_phase==0) & kept_frame_mask).sum()),
            "kept_phase_B_frames": int(((frame_phase==1) & kept_frame_mask).sum()),
            "kept_left_freeze_median": np.median(homes, axis=0).tolist(),
            "kept_left_freeze_std": np.std(homes, axis=0).tolist(),
        },
        "episodes": per_ep,
    }, f, indent=2)
print(f"\nSaved: {phases_path}")

kept_path = OUT / "kept_episodes.txt"
with open(kept_path, "w") as f:
    f.write(" ".join(str(r["ep"]) for r in kept) + "\n")
print(f"Saved: {kept_path}  ({len(kept)} ids)")

frames_path = OUT / "frame_phases.npy"
np.save(frames_path, frame_phase)
print(f"Saved: {frames_path}  shape={frame_phase.shape}  dtype={frame_phase.dtype}")

# ----- Markdown summary -----
md = []
md.append("# Phase Metadata for `bimanual_cooperate`\n")
md.append(f"Built from {N} frames across {len(per_ep)} episodes (30 fps).\n")
md.append("## Filter\n")
md.append(f"- min Phase A duration:    {MIN_PHASE_A_S} s")
md.append(f"- max Phase A duration:    {MAX_PHASE_A_S} s")
md.append(f"- min Phase B duration:    {MIN_PHASE_B_S} s")
md.append(f"- max left-arm std (phase B, non-gripper joints):  {MAX_LEFT_STD_PHASE_B}°")
md.append(f"- max right-arm std (phase A, non-gripper joints): {MAX_RIGHT_STD_PHASE_A}°\n")
md.append(f"## Result\n")
md.append(f"- **Kept**: {len(kept)} / {len(per_ep)}  ({len(kept)/len(per_ep)*100:.1f}%)")
md.append(f"- **Filtered**: {len(filtered)}")
md.append("- Reasons:")
for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
    md.append(f"  - `{k}`: {v}")
md.append(f"- Kept frames: {int(kept_frame_mask.sum()):,} / {N:,}  "
          f"({kept_frame_mask.sum()/N*100:.1f}%)")
md.append(f"  - Phase A: {int(((frame_phase==0) & kept_frame_mask).sum()):,}")
md.append(f"  - Phase B: {int(((frame_phase==1) & kept_frame_mask).sum()):,}\n")
md.append("## Per-episode left freeze pose (across kept episodes)\n")
md.append("Each kept episode has its own `left_freeze_pose` in `episode_phases.json`. "
          "Use it instead of a global constant.\n")
md.append("| Joint | median | std | 5–95% width |")
md.append("|---|---|---|---|")
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
for j, name in enumerate(joint_names):
    p5, p95 = np.percentile(homes[:, j], [5, 95])
    md.append(f"| {name} | {np.median(homes[:, j]):.2f} | {np.std(homes[:, j]):.2f} | {p95-p5:.2f} |")
md.append("\n## Files\n")
md.append("- `episode_phases.json` — per-episode dict (stop_frame, left_freeze_pose, keep, ...)")
md.append("- `kept_episodes.txt`   — whitespace-separated kept episode indices")
md.append("- `frame_phases.npy`    — (N,) int8, per-frame phase {0=A, 1=B}, indexed by `index` column")
md.append("- `phase_panel.png`     — visual sanity check of detection on 12 episodes\n")

(OUT / "phase_summary.md").write_text("\n".join(md))
print(f"Saved: {OUT / 'phase_summary.md'}")

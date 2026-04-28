"""Visualize per-episode phase detection on a sample of episodes.
Generates phase_panel.png with 16 episodes (8 kept, 8 filtered) showing:
  - left arm speed (smoothed) — used for stop detection
  - right arm speed
  - detected stop_frame as vertical line
  - per-joint left position over time, with phase B region shaded
"""
from pathlib import Path
import numpy as np
import json
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/zhanghaodong/Desktop/CSCI2951K/Cooperate-SO101")
DATASET = ROOT / "bimanual_cooperate"
META = ROOT / "bimanual_cooperate_meta"

phases = json.load(open(META / "episode_phases.json"))
records = phases["episodes"]
FPS = 30

tbl = pq.read_table(DATASET / "data/chunk-000/file-000.parquet",
                    columns=["episode_index", "observation.state"])
df = tbl.to_pandas()
state = np.stack(df["observation.state"].to_numpy())
ep_idx = df["episode_index"].to_numpy()

def smoothed_speed(x, w=15):
    d = np.diff(x, axis=0, prepend=x[:1])
    s = np.linalg.norm(d, axis=1)
    return np.convolve(s, np.ones(w)/w, mode="same")

# Pick samples
kept = [r for r in records if r["keep"]]
dropped = [r for r in records if not r["keep"]]
np.random.seed(0)
sample_kept = list(np.random.choice(kept, 8, replace=False))
sample_drop = list(np.random.choice(dropped, min(8, len(dropped)), replace=False))
samples = sample_kept + sample_drop

fig, axes = plt.subplots(4, 4, figsize=(20, 14))
axes = axes.flatten()
for ax, rec in zip(axes, samples):
    ep = rec["ep"]
    m = ep_idx == ep
    st = state[m]
    T = len(st)
    ts = np.arange(T) / FPS
    sp_l = smoothed_speed(st[:, :5])
    sp_r = smoothed_speed(st[:, 6:11])
    stop_t = rec["stop_frame"] / FPS

    ax2 = ax.twinx()
    ax.plot(ts, sp_l, color="C0", lw=1.2, label="left speed")
    ax.plot(ts, sp_r, color="C1", lw=1.2, alpha=0.7, label="right speed")
    ax.axvline(stop_t, color="red", ls="--", lw=1.2, label=f"stop@{stop_t:.1f}s")

    # Shade phase B
    if rec["stop_frame"] < T:
        ax.axvspan(stop_t, T/FPS, alpha=0.10, color="green")

    # Plot left non-gripper joints (raw, normalized to fit axis)
    left_pos = st[:, :5]
    left_norm = (left_pos - left_pos.mean(axis=0)) / (left_pos.std(axis=0) + 1e-3)
    for j in range(5):
        ax2.plot(ts, left_norm[:, j], lw=0.6, alpha=0.5)
    ax2.set_ylim(-4, 4)
    ax2.set_ylabel("left joints (z)", fontsize=7)
    ax2.tick_params(labelsize=6)

    title = f"ep {ep}  T={T/FPS:.1f}s  pB={rec['phase_B_s']:.1f}s"
    if rec["keep"]:
        title += "  ✓"
        ax.set_title(title, fontsize=9, color="green")
    else:
        title += f"  ✗ ({','.join(r.split('(')[0] for r in rec['filter_reasons'])})"
        ax.set_title(title, fontsize=9, color="red")

    ax.set_xlabel("time (s)", fontsize=7)
    ax.set_ylabel("speed", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")

fig.suptitle(
    "Phase detection sanity check  —  green shaded = phase B (left arm idle, right arm working)\n"
    "Top 2 rows: kept episodes  |  Bottom 2 rows: filtered episodes",
    fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = META / "phase_panel.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved: {out}")

# Also produce a histogram of left-phase-B std to show the bimodal split
fig2, ax = plt.subplots(figsize=(8, 4))
stds = np.array([r["left_phaseB_max_std_nongrip"] for r in records])
ax.hist(stds, bins=60, color="steelblue", edgecolor="white")
ax.axvline(phases["config"]["MAX_LEFT_STD_PHASE_B"], color="red", ls="--",
           label=f"filter cutoff = {phases['config']['MAX_LEFT_STD_PHASE_B']}°")
ax.set_xlabel("Left arm joint std (deg, max non-gripper) during phase B")
ax.set_ylabel("# episodes")
ax.set_title("Bimodal: episodes either have a clean stop or they don't")
ax.legend()
out2 = META / "phase_b_std_hist.png"
fig2.tight_layout()
fig2.savefig(out2, dpi=110)
print(f"Saved: {out2}")

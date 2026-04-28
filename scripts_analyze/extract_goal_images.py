"""Extract per-episode subgoal_A and final_goal frames from the top camera.

For each kept episode (stop_frame established by build_phase_metadata.py):
  subgoal_A:   the moment left arm finishes and right arm is about to start.
               This is the visual state of "lid removed, cube exposed, left arm at home."
  final_goal:  the last frame of the episode = "cube placed in box."

Outputs:
  bimanual_cooperate_meta/goal_frames.npz    {ep_idx, subgoal_imgs, final_imgs}
                                              shapes (K,), (K,240,320,3), (K,240,320,3)
  bimanual_cooperate_meta/goal_preview.png   16-episode visual sanity check

Notes:
  - Videos use AV1 codec; we use PyAV for decoding (works on macOS without ffmpeg).
  - Episode video lives inside one big concatenated mp4; the per-episode
    from_timestamp lookup comes from meta/episodes/.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/zhanghaodong/Desktop/CSCI2951K/Cooperate-SO101")
DATASET = ROOT / "bimanual_cooperate"
META = ROOT / "bimanual_cooperate_meta"
CAM = "observation.images.top"
FPS = 30

print("Loading episode metadata...")
phases = json.load(open(META / "episode_phases.json"))
ep_records = {r["ep"]: r for r in phases["episodes"]}
kept_eps = sorted(int(x) for x in (META / "kept_episodes.txt").read_text().split())
print(f"  kept episodes: {len(kept_eps)}")

print("Loading episode video offsets...")
ep_meta = pq.read_table(DATASET / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
ep_meta = ep_meta.set_index("episode_index")
top_from = ep_meta[f"videos/{CAM}/from_timestamp"].to_dict()
top_to = ep_meta[f"videos/{CAM}/to_timestamp"].to_dict()
top_chunk = ep_meta[f"videos/{CAM}/chunk_index"].to_dict()
top_file = ep_meta[f"videos/{CAM}/file_index"].to_dict()


def decode_frame_at(container, target_pts_sec, time_base):
    """Seek to and decode the first frame at/after target_pts_sec. Returns
    (H, W, 3) uint8 numpy array."""
    target_pts = int(target_pts_sec / time_base)
    # Seek a bit before target to land on a keyframe.
    container.seek(max(0, target_pts - int(2.0 / time_base)), backward=True,
                   stream=container.streams.video[0])
    last = None
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        if frame.pts >= target_pts:
            return frame.to_ndarray(format="rgb24")
        last = frame
    return last.to_ndarray(format="rgb24") if last is not None else None


def open_for_episode(ep):
    chunk = top_chunk[ep]
    file_idx = top_file[ep]
    path = DATASET / "videos" / CAM / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
    container = av.open(str(path))
    return container


# Group episodes by their video file so we open each file once.
by_file = {}
for ep in kept_eps:
    key = (top_chunk[ep], top_file[ep])
    by_file.setdefault(key, []).append(ep)

subgoal_imgs = []
final_imgs = []
ep_order = []

print("Decoding frames ...")
for (ck, fi), eps in by_file.items():
    path = DATASET / "videos" / CAM / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
    container = av.open(str(path))
    stream = container.streams.video[0]
    tb = float(stream.time_base)

    # Sort by from_timestamp so seeks are mostly forward.
    eps_sorted = sorted(eps, key=lambda e: top_from[e])
    for i, ep in enumerate(eps_sorted):
        rec = ep_records[ep]
        start_ts = float(top_from[ep])
        end_ts = float(top_to[ep])

        # subgoal_A timestamp: stop_frame within episode
        sub_ts = start_ts + rec["stop_frame"] / FPS
        # final_goal timestamp: last frame
        fin_ts = end_ts - 1.0 / FPS

        sub_img = decode_frame_at(container, sub_ts, tb)
        fin_img = decode_frame_at(container, fin_ts, tb)

        if sub_img is None or fin_img is None:
            print(f"  ep {ep}: decode failed, skipping")
            continue

        subgoal_imgs.append(sub_img)
        final_imgs.append(fin_img)
        ep_order.append(ep)

        if (len(ep_order) % 25) == 0:
            print(f"  ... {len(ep_order)}/{len(kept_eps)}")
    container.close()

subgoal_imgs = np.stack(subgoal_imgs)
final_imgs = np.stack(final_imgs)
ep_order = np.array(ep_order, dtype=np.int32)
print(f"\n  subgoal_imgs: {subgoal_imgs.shape}  ({subgoal_imgs.nbytes/1e6:.1f} MB)")
print(f"  final_imgs:   {final_imgs.shape}    ({final_imgs.nbytes/1e6:.1f} MB)")

out_npz = META / "goal_frames.npz"
np.savez_compressed(out_npz,
                    ep_idx=ep_order,
                    subgoal_imgs=subgoal_imgs,
                    final_imgs=final_imgs)
print(f"Saved: {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB compressed)")

# Visual sanity preview: 8 random episodes' subgoal vs final
np.random.seed(42)
sample = np.random.choice(len(ep_order), 6, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(22, 9))
for col, idx in enumerate(sample):
    axes[0, col].imshow(subgoal_imgs[idx])
    axes[0, col].set_title(f"ep {ep_order[idx]}  subgoal_A\n(lid moved, left at home)", fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(final_imgs[idx])
    axes[1, col].set_title(f"ep {ep_order[idx]}  final_goal\n(cube in box)", fontsize=10)
    axes[1, col].axis("off")
fig.suptitle("Per-episode goal frames (top camera, 320×240)", fontsize=13)
fig.tight_layout()
out_png = META / "goal_preview.png"
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"Saved: {out_png}")

# Also save 3 individual PNGs at full size for the user to inspect
from PIL import Image
for k in range(3):
    idx = sample[k]
    Image.fromarray(subgoal_imgs[idx]).save(META / f"sample_subgoal_ep{ep_order[idx]}.png")
    Image.fromarray(final_imgs[idx]).save(META / f"sample_final_ep{ep_order[idx]}.png")
print(f"Saved: 6 individual sample PNGs in {META}/")

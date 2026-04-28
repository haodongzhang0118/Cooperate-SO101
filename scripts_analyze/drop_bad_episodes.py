"""Permanently remove the 10 bad episodes from `bimanual_cooperate`.

After running, the dataset is self-cleaning: total_episodes=191, episode_index
becomes 0..190 contiguous, total_frames is updated, and you no longer need
--dataset.episodes=... at training time.

What's modified (with .bak2 backup the first time):
  data/chunk-000/file-000.parquet       (drop rows; reindex episode_index, index)
  meta/episodes/chunk-000/file-000.parquet  (drop rows; recompute dataset_from/to_index)
  meta/info.json                        (total_episodes, total_frames, splits)
  bimanual_cooperate_meta/frame_phases.npy   (regenerate to match new layout)
  bimanual_cooperate_meta/kept_episodes.txt  (now 0..190)

NOT modified:
  videos/**.mp4                         (untouched; dropped segments become
                                         unreachable bytes ≈ 5% of file size,
                                         which is fine — re-encoding is slow
                                         and risky for AV1)
  meta/episodes' video timestamps       (still reference correct positions
                                         in the unchanged mp4)

Idempotent: refuses to run if total_episodes is already 191.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/Users/zhanghaodong/Desktop/CSCI2951K/Cooperate-SO101")
DATASET = ROOT / "bimanual_cooperate"
META = ROOT / "bimanual_cooperate_meta"

DATA_PARQ = DATASET / "data/chunk-000/file-000.parquet"
EPS_PARQ  = DATASET / "meta/episodes/chunk-000/file-000.parquet"
INFO_JSON = DATASET / "meta/info.json"


def backup2(p: Path):
    """Use .bak2 so we don't clobber the first .bak created by phase embedding."""
    bak = p.with_suffix(p.suffix + ".bak2")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backup -> {bak.name}")


# ---------- 0. read filter list ----------
ep_phases = json.load(open(META / "episode_phases.json"))
kept_old = sorted(r["ep"] for r in ep_phases["episodes"] if r["keep"])
dropped_old = sorted(r["ep"] for r in ep_phases["episodes"] if not r["keep"])
print(f"Keeping {len(kept_old)} episodes; dropping {len(dropped_old)}: {dropped_old}")

info = json.loads(INFO_JSON.read_text())
if info["total_episodes"] == len(kept_old):
    print(f"\nDataset already pruned (total_episodes={info['total_episodes']}). "
          f"Nothing to do.")
    raise SystemExit(0)
if info["total_episodes"] != 201:
    raise RuntimeError(
        f"info.json total_episodes={info['total_episodes']} but expected 201. "
        f"Refusing to run on unfamiliar state.")

# Old-ep -> new-ep map (compact 0..190)
old_to_new = {old: new for new, old in enumerate(kept_old)}

# ---------- 1. data parquet ----------
print(f"\n[1/4] data parquet")
backup2(DATA_PARQ)
tbl = pq.read_table(DATA_PARQ)
print(f"  before: {tbl.num_rows} rows × {len(tbl.column_names)} cols")

old_ep = tbl["episode_index"].to_numpy()
keep_mask = np.isin(old_ep, kept_old)
tbl = tbl.filter(pa.array(keep_mask))

# Reindex episode_index
new_ep = np.array([old_to_new[e] for e in tbl["episode_index"].to_numpy()],
                   dtype=np.int64)
tbl = tbl.set_column(tbl.column_names.index("episode_index"),
                     "episode_index", pa.array(new_ep))

# Reindex global `index` to 0..N-1
new_idx = np.arange(tbl.num_rows, dtype=np.int64)
tbl = tbl.set_column(tbl.column_names.index("index"), "index", pa.array(new_idx))

# `frame_index` (per-episode 0..T-1) is already correct — no change needed.
pq.write_table(tbl, DATA_PARQ, compression="snappy")
print(f"  after:  {tbl.num_rows} rows × {len(tbl.column_names)} cols")

# ---------- 2. meta/episodes parquet ----------
print(f"\n[2/4] meta/episodes parquet")
backup2(EPS_PARQ)
ep_tbl = pq.read_table(EPS_PARQ)
print(f"  before: {ep_tbl.num_rows} rows × {len(ep_tbl.column_names)} cols")

old_idx = ep_tbl["episode_index"].to_numpy()
keep_mask_ep = np.isin(old_idx, kept_old)
ep_tbl = ep_tbl.filter(pa.array(keep_mask_ep))

# Reindex episode_index
new_ep_idx = np.array([old_to_new[e] for e in ep_tbl["episode_index"].to_numpy()],
                       dtype=np.int64)
ep_tbl = ep_tbl.set_column(ep_tbl.column_names.index("episode_index"),
                           "episode_index", pa.array(new_ep_idx))

# Recompute dataset_from_index / dataset_to_index from lengths (in NEW order).
# After filter+sort, the rows are in the order of the new episode_index.
# Sort to be safe.
df = ep_tbl.to_pandas().sort_values("episode_index").reset_index(drop=True)
lens = df["length"].astype(np.int64).to_numpy()
ds_from = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
ds_to = np.cumsum(lens).astype(np.int64)
df["dataset_from_index"] = ds_from
df["dataset_to_index"]   = ds_to
ep_tbl_new = pa.Table.from_pandas(df, preserve_index=False)

pq.write_table(ep_tbl_new, EPS_PARQ, compression="snappy")
print(f"  after:  {ep_tbl_new.num_rows} rows × {len(ep_tbl_new.column_names)} cols")
assert int(ds_to[-1]) == int(tbl.num_rows), \
    f"row count mismatch: episodes sum {ds_to[-1]} != data rows {tbl.num_rows}"

# ---------- 3. info.json ----------
print(f"\n[3/4] info.json")
backup2(INFO_JSON)
info["total_episodes"] = len(kept_old)
info["total_frames"] = int(tbl.num_rows)
info["splits"] = {"train": f"0:{len(kept_old)}"}
INFO_JSON.write_text(json.dumps(info, indent=4))
print(f"  total_episodes: 201 -> {len(kept_old)}")
print(f"  total_frames:   116231 -> {tbl.num_rows}")
print(f"  splits.train:   0:201 -> 0:{len(kept_old)}")

# ---------- 4. sidecars ----------
print(f"\n[4/4] sidecars (frame_phases.npy + kept_episodes.txt)")
phase_col = tbl["phase"].to_numpy().astype(np.int8)
np.save(META / "frame_phases.npy", phase_col)
print(f"  frame_phases.npy regenerated: shape {phase_col.shape}")

(META / "kept_episodes.txt").write_text(" ".join(str(i) for i in range(len(kept_old))) + "\n")
print(f"  kept_episodes.txt: now 0..{len(kept_old)-1} (all good)")

# ---------- final verification ----------
print(f"\n=== Verification ===")
tbl_v = pq.read_table(DATA_PARQ, columns=["index", "episode_index", "phase"])
ep_min, ep_max = int(tbl_v["episode_index"].to_numpy().min()), int(tbl_v["episode_index"].to_numpy().max())
print(f"  episode_index range: {ep_min}..{ep_max}")
idx = tbl_v["index"].to_numpy()
ok = (idx == np.arange(len(idx))).all()
print(f"  index column is 0..N-1: {ok}")
print(f"  phase column distribution: A={(tbl_v['phase'].to_numpy()==0).sum()}, "
      f"B={(tbl_v['phase'].to_numpy()==1).sum()}")
print(f"\nDone. Train with simply:")
print(f"  lerobot-train --policy.type=dino_seqwm --dataset.repo_id=... "
      f"(no --dataset.episodes needed)")

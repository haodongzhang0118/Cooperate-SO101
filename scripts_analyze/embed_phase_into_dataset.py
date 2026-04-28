"""Embed `phase` (0/1 per frame) as a first-class feature in the LeRobot
dataset so the dataloader yields it automatically — no policy-side hack.

Mutations:
  1. data/chunk-000/file-000.parquet     → add `phase` int8 column
  2. meta/info.json                      → add `phase` to features dict
  3. meta/stats.json (if exists)         → add trivial stats for `phase`

Idempotent: re-running detects existing column and short-circuits.

Backups (safety): writes ".bak" copies of any file it touches the first time.
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

DATA_PARQUET = DATASET / "data/chunk-000/file-000.parquet"
INFO_JSON    = DATASET / "meta/info.json"
STATS_JSON   = DATASET / "meta/stats.json"
EP_META_PARQ = DATASET / "meta/episodes/chunk-000/file-000.parquet"

FRAME_PHASES_NPY = META / "frame_phases.npy"


def backup_once(p: Path):
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  backup -> {bak.name}")


# ---------- Step 1: parquet ----------
print(f"\n[1/3] data parquet: {DATA_PARQUET.relative_to(ROOT)}")
tbl = pq.read_table(DATA_PARQUET)
print(f"  rows={tbl.num_rows}  columns={tbl.column_names}")

if "phase" in tbl.column_names:
    print("  phase column already present — skipping")
else:
    backup_once(DATA_PARQUET)
    frame_phases = np.load(FRAME_PHASES_NPY)
    assert len(frame_phases) == tbl.num_rows, \
        f"frame_phases length {len(frame_phases)} != rows {tbl.num_rows}"

    # frame_phases is indexed by global `index` column. Verify alignment by
    # checking that the parquet's `index` column is monotonically 0..N-1.
    idx_col = tbl["index"].to_numpy()
    assert (idx_col == np.arange(len(idx_col))).all(), \
        "parquet `index` column is not monotonic 0..N-1; frame_phases alignment unsafe"

    phase_arr = pa.array(frame_phases.astype(np.int8), type=pa.int8())
    tbl_new = tbl.append_column("phase", phase_arr)
    pq.write_table(tbl_new, DATA_PARQUET, compression="snappy")
    print(f"  wrote {tbl_new.num_rows} rows × {len(tbl_new.column_names)} cols "
          f"(added 'phase' int8)")

# ---------- Step 2: info.json ----------
print(f"\n[2/3] info.json")
info = json.loads(INFO_JSON.read_text())
if "phase" in info["features"]:
    print("  'phase' feature already registered — skipping")
else:
    backup_once(INFO_JSON)
    info["features"]["phase"] = {
        "dtype": "int8",
        "shape": [1],
        "names": ["phase"],
    }
    INFO_JSON.write_text(json.dumps(info, indent=4))
    print("  added 'phase' int8 [1] feature")

# ---------- Step 3: stats.json (optional) ----------
print(f"\n[3/3] stats.json")
if not STATS_JSON.exists():
    print(f"  {STATS_JSON.name} not present — nothing to update (dataset uses per-episode stats)")
else:
    stats = json.loads(STATS_JSON.read_text())
    if "phase" in stats:
        print("  'phase' stats already present — skipping")
    else:
        backup_once(STATS_JSON)
        stats["phase"] = {
            "min":  [0.0],
            "max":  [1.0],
            "mean": [0.5],
            "std":  [0.5],
            "count": [int(tbl.num_rows)],
        }
        STATS_JSON.write_text(json.dumps(stats, indent=2))
        print("  added trivial phase stats")

# ---------- Sanity: re-read and print head ----------
tbl2 = pq.read_table(DATA_PARQUET, columns=["index", "episode_index", "phase"])
print("\nVerification — first 5 rows of new columns:")
print(tbl2.slice(0, 5).to_pandas())

# Spot-check phase boundaries by comparing against frame_phases.npy
fp = np.load(FRAME_PHASES_NPY)
mismatch = (tbl2["phase"].to_numpy() != fp).sum()
print(f"\nrow-wise mismatch with frame_phases.npy: {mismatch}/{len(fp)}  "
      f"{'OK' if mismatch == 0 else 'BROKEN'}")

# Phase counts
n_A = int((tbl2["phase"].to_numpy() == 0).sum())
n_B = int((tbl2["phase"].to_numpy() == 1).sum())
print(f"phase==0 (A, left arm working): {n_A:,}")
print(f"phase==1 (B, right arm working): {n_B:,}")

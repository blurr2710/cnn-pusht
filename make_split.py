"""Step 4 implementation — build the 80/20 episode split (seed 42) and save it.

Reads ONLY metadata (no video decoding). Produces cnn_pusht/split.json:
one canonical file that every later script (training, eval) must load.

Usage:
    source .venv/bin/activate
    python cnn_pusht/make_split.py
"""
import json
import random

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/pusht"
SEED = 42
TRAIN_RATIO = 0.8
VIDEO_BACKEND = "pyav"  # per notes/env.md — torchcodec is broken on this machine
OUT_FILE = "cnn_pusht/split.json"


def main():
    # Metadata only — no frame is decoded in this script.
    ds = LeRobotDataset(REPO_ID, video_backend=VIDEO_BACKEND)
    episodes = list(ds.meta.episodes)  # one entry per episode

    # per-episode frame counts: verified key names from notes/schema.md
    counts = [row["dataset_to_index"] - row["dataset_from_index"] for row in episodes]
    total_rows = len(ds)

    # --- the split: shuffle episode IDs with a dedicated seeded RNG ---
    # random.Random(SEED) instead of random.seed(): does not pollute global
    # RNG state, so any other code using random stays unaffected.
    rng = random.Random(SEED)
    ids = list(range(len(counts)))
    rng.shuffle(ids)
    n_train = int(round(len(ids) * TRAIN_RATIO))
    train_ids = sorted(ids[:n_train])
    eval_ids = sorted(ids[n_train:])

    train_rows = sum(counts[i] for i in train_ids)
    eval_rows = sum(counts[i] for i in eval_ids)

    # --- sanity checks ---
    # 1. boundary is episode-level: no id on both sides
    assert set(train_ids) & set(eval_ids) == set(), "episode crossed the boundary"
    # 2. every episode assigned exactly once
    assert len(train_ids) + len(eval_ids) == len(counts)
    # 3. rows add up
    assert train_rows + eval_rows == total_rows == sum(counts), "row counts disagree"

    print(f"repo_id       : {REPO_ID}")
    print(f"num_episodes  : {ds.num_episodes}")
    print(f"fps           : {ds.fps}")
    print(f"total rows    : {total_rows}")
    print(f"train episodes: {len(train_ids)}  rows: {train_rows}")
    print(f"eval episodes : {len(eval_ids)}   rows: {eval_rows}")
    print(f"split file    : {OUT_FILE}")

    # --- save ONE canonical split file ---
    # Train/eval scripts load this file; membership is the contract, never re-derived.
    split = {
        "repo_id": REPO_ID,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "fps": ds.fps,
        "total_rows": total_rows,
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        # index ranges let later scripts build train examples without re-loading metadata
        "episode_index_ranges": {
            str(ep["episode_index"]): [ep["dataset_from_index"], ep["dataset_to_index"]]
            for ep in episodes
        },
    }
    with open(OUT_FILE, "w") as f:
        json.dump(split, f, indent=2)
    print("split.json written. Checks: ALL PASS")


if __name__ == "__main__":
    main()

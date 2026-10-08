"""Load the LeRobot PushT dataset and prepare tensors for training.

Data preparation:
  input image : decoded video frame of stream observation.image, 96x96 RGB
                -> float 0-1, permute HWC -> CHW  -> (3,96,96)
  input state : observation.state (x,y) raw ints around 256
                -> (v - 256) / 256  -> roughly (-1..1)
  label action: action (x,y) raw ints around 256
                -> (v - 256) / 256  -> roughly (-1..1)
  split       : membership comes ONLY from cnn_pusht/split.json (run make_split.py first)
                we pass the episode id lists to LeRobotDataset(episodes=...) so the
                train/eval boundary stays at the episode level.

Usage:
    source .venv/bin/activate
    python cnn_pusht/dataset.py     # sanity check: prints one preprocessed example
"""
import json

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/pusht"
VIDEO_BACKEND = "pyav"  # per make_split.py — torchcodec is broken on this machine
SPLIT_FILE = "cnn_pusht/split.json"

# normalization constants verified in STEP2 (notes/schema.md): PushT state and
# action are stored as ints whose values sit around the middle of 0..512,
# i.e. around 256. (v-256)/256 maps 0..512 to (-1..1), with 256 -> 0.
STATE_RAW_CENTER = 256.0
STATE_RAW_RANGE = 256.0


def normalize_state_action(v):
    """raw int (x or y, single value or tensor) -> roughly (-1..1)."""
    return (v - STATE_RAW_CENTER) / STATE_RAW_RANGE


class PushTTrainExample(Dataset):
    """Wrap a LeRobotDataset and return model-ready tensors."""

    def __init__(self, lerobot_ds):
        self.ds = lerobot_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # --- image: decoded frame of the observation.image stream ---
        img = row["observation.image"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        # Older lerobot returns (H,W,C) -> permute to (C,H,W). Newer versions
        # already return (C,H,W) float — permuting again would scramble it,
        # so only flip when the channels are last.
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1).contiguous()

        # --- state: pusher (x,y) at row t, normalized ---
        state = normalize_state_action(row["observation.state"].float())

        # --- label: action (absolute next target) at same row t, normalized ---
        action = normalize_state_action(row["action"].float())

        return {
            "image": img,          # (3,96,96) float 0-1
            "state": state,        # (2,)      float ~(-1..1)
            "action": action,      # (2,)      float ~(-1..1)
        }


def _load_split():
    """Load the saved episode-level split."""
    try:
        with open(SPLIT_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"{SPLIT_FILE} not found.\n"
            "Run the Step 4 script first:\n"
            "    source .venv/bin/activate\n"
            "    python cnn_pusht/make_split.py"
        )


def create_train_eval_datasets():
    """Build the train and evaluation datasets from the saved episode lists."""
    split = _load_split()
    train_ds = LeRobotDataset(
        REPO_ID, episodes=split["train_episode_ids"], video_backend=VIDEO_BACKEND
    )
    eval_ds = LeRobotDataset(
        REPO_ID, episodes=split["eval_episode_ids"], video_backend=VIDEO_BACKEND
    )
    return (
        PushTTrainExample(train_ds),
        PushTTrainExample(eval_ds),
        split,
    )


if __name__ == "__main__":
    train_ex, eval_ex, split = create_train_eval_datasets()
    print(f"split seed         : {split['seed']}")
    print(f"train examples     : {len(train_ex)}")
    print(f"eval examples      : {len(eval_ex)}")
    print(f"split.json rows    : {split['train_rows']} + {split['eval_rows']}")
    assert len(train_ex) == split["train_rows"], "train row count disagrees with split.json"
    assert len(eval_ex) == split["eval_rows"], "eval row count disagrees with split.json"

    ex = train_ex[0]
    print(f"\none example:")
    print(f"  image  {tuple(ex['image'].shape)} dtype={ex['image'].dtype}"
          f" min={ex['image'].min():.3f} max={ex['image'].max():.3f}")
    print(f"  state  {tuple(ex['state'].shape)} values={ex['state'].tolist()}")
    print(f"  action {tuple(ex['action'].shape)} values={ex['action'].tolist()}")
    assert ex["image"].shape == (3, 96, 96)
    assert ex["state"].shape == (2,) and ex["action"].shape == (2,)
    assert -1.01 <= ex["state"].min() and ex["state"].max() <= 1.01
    assert -1.01 <= ex["action"].min() and ex["action"].max() <= 1.01
    print("\ndataset.py sanity check: PASS")

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

For chunked behavioral cloning, :class:`PushTActionChunkExample` keeps the
same observation at time ``t`` and uses recorded actions ``t`` through
``t + chunk_size - 1`` as its target. Starts that could run past an episode
are deliberately excluded; action chunks are never padded or joined across
games.
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
DEFAULT_CHUNK_SIZE = 8


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


class PushTActionChunkExample(Dataset):
    """Return one observation and a fixed-length, within-episode action plan.

    ``LeRobotDataset`` stores an absolute ``index`` for every row, including
    when it has been filtered to a subset of episodes. We build valid relative
    starts from the episode metadata once, then fetch future actions directly
    from its tabular data. This avoids decoding eight future video frames per
    training example and, more importantly, makes crossing an episode boundary
    impossible.
    """

    def __init__(self, lerobot_ds, chunk_size=DEFAULT_CHUNK_SIZE):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if lerobot_ds.episodes is None:
            raise ValueError(
                "PushTActionChunkExample requires a LeRobotDataset restricted "
                "to explicit episode IDs."
            )

        self.ds = lerobot_ds
        self.chunk_size = chunk_size

        # The filtered HF table remains ordered by episode and retains each
        # row's original absolute index. Map those absolute starts back to the
        # relative index accepted by self.ds[...].
        absolute_to_relative = {
            int(absolute_index): relative_index
            for relative_index, absolute_index in enumerate(self.ds.hf_dataset["index"])
        }

        self.start_indices = []
        for episode_id in self.ds.episodes:
            episode = self.ds.meta.episodes[episode_id]
            first = episode["dataset_from_index"]
            last_exclusive = episode["dataset_to_index"]

            # For labels [t, ..., t + chunk_size - 1], the largest legal t is
            # last_exclusive - chunk_size. Therefore every episode loses its
            # final chunk_size - 1 starts (seven starts for chunk size eight).
            for absolute_start in range(first, last_exclusive - chunk_size + 1):
                try:
                    self.start_indices.append(absolute_to_relative[absolute_start])
                except KeyError as exc:
                    raise RuntimeError(
                        f"Episode {episode_id} row {absolute_start} was not loaded "
                        "into the requested LeRobotDataset."
                    ) from exc

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        row = self.ds[start]

        # Decode only the image at observation time t, with the exact same
        # preprocessing as the one-step dataset.
        img = row["observation.image"]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1).contiguous()

        state = normalize_state_action(row["observation.state"].float())

        # Actions live in the parquet table, so this slice does not decode
        # extra videos. The valid-start construction above guarantees this
        # whole contiguous slice belongs to the same episode.
        raw_actions = self.ds.hf_dataset[start : start + self.chunk_size]["action"]
        action_chunk = normalize_state_action(torch.stack(raw_actions).float())

        return {
            "image": img,                  # (3,96,96) float 0-1
            "state": state,                # (2,)      float ~(-1..1)
            "action": action_chunk,        # (chunk_size,2) normalized targets
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


def create_train_eval_chunk_datasets(chunk_size=DEFAULT_CHUNK_SIZE):
    """Build episode-safe action-chunk datasets from the saved split.

    This deliberately leaves :func:`create_train_eval_datasets` unchanged for
    the existing single-action experiment.
    """
    split = _load_split()
    train_ds = LeRobotDataset(
        REPO_ID, episodes=split["train_episode_ids"], video_backend=VIDEO_BACKEND
    )
    eval_ds = LeRobotDataset(
        REPO_ID, episodes=split["eval_episode_ids"], video_backend=VIDEO_BACKEND
    )
    return (
        PushTActionChunkExample(train_ds, chunk_size=chunk_size),
        PushTActionChunkExample(eval_ds, chunk_size=chunk_size),
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

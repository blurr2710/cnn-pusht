"""Make picture-based comparisons between the CNN policy and human demonstrations.

Each saved board has:
  * blue dot:   the pusher's current position
  * green dot:  the target position chosen by the human demonstrator
  * red dot:    the target position predicted by the CNN policy

This does not play a game and does not change the model.  It asks both the
human recording and the model the same question for the same frozen board.

Run from the repository root:
    .venv/bin/python cnn_pusht/evals/visualize_predictions.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# This file lives one folder below dataset.py and model.py.  Add that parent
# folder so the documented `python cnn_pusht/evals/...` command works.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import (
    STATE_RAW_CENTER,
    STATE_RAW_RANGE,
    create_train_eval_datasets,
)
from model import PushTCNNMLP

DEFAULT_WEIGHTS = "cnn_pusht/models/pusht_cnn_mlp_v1.pt"
DEFAULT_OUTPUT = "cnn_pusht/evals/prediction_gallery"
BOARD_SIZE = 96
SCALE = 5


def choose_device(requested):
    """Choose a device, while letting --device cpu be useful for debugging."""
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available on this machine. Use --device cpu.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available on this machine. Use --device mps or cpu.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def raw_to_pixel(point):
    """Convert a PushT coordinate (0..512) to a point on its 96px picture."""
    return tuple(float(value) * BOARD_SIZE / 512.0 for value in point)


def model_to_raw(point):
    """Undo the same (value - 256) / 256 normalization used at training."""
    return point * STATE_RAW_RANGE + STATE_RAW_CENTER


def draw_target(draw, current, target, color, label):
    """Draw a clear line and dot from the current pusher to a chosen target."""
    draw.line([current, target], fill=color, width=3)
    radius = 5
    draw.ellipse(
        (target[0] - radius, target[1] - radius, target[0] + radius, target[1] + radius),
        fill=color,
        outline="white",
        width=1,
    )
    draw.text((target[0] + 7, target[1] - 8), label, fill=color, stroke_width=1, stroke_fill="black")


def render_comparison(image_tensor, state_raw, human_raw, prediction_raw, filename, sample_index):
    """Overlay one prediction-versus-human comparison and write one PNG file."""
    image = image_tensor.detach().cpu()
    if image.shape[0] == 3:  # training example is CHW
        image = image.permute(1, 2, 0)
    image_uint8 = image.clamp(0, 1).mul(255).byte().numpy()
    board = Image.fromarray(image_uint8, mode="RGB").resize(
        (BOARD_SIZE * SCALE, BOARD_SIZE * SCALE), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(board)

    current = tuple(value * SCALE for value in raw_to_pixel(state_raw))
    human = tuple(value * SCALE for value in raw_to_pixel(human_raw))
    prediction = tuple(value * SCALE for value in raw_to_pixel(prediction_raw))
    draw_target(draw, current, human, "#18d86a", "human")
    draw_target(draw, current, prediction, "#ff4545", "CNN")

    radius = 5
    draw.ellipse(
        (current[0] - radius, current[1] - radius, current[0] + radius, current[1] + radius),
        fill="#3ca7ff",
        outline="white",
        width=1,
    )
    draw.text((8, 8), f"example {sample_index}", fill="white", stroke_width=2, stroke_fill="black")
    board.save(filename)


def make_contact_sheet(image_paths, output_path):
    """Make one easy-to-open overview image from the individual comparisons."""
    columns = 4
    rows = (len(image_paths) + columns - 1) // columns
    tile_size = BOARD_SIZE * SCALE
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "#202020")
    for position, image_path in enumerate(image_paths):
        tile = Image.open(image_path).convert("RGB")
        x = (position % columns) * tile_size
        y = (position // columns) * tile_size
        sheet.paste(tile, (x, y))
    sheet.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Draw human targets and CNN predictions on real PushT dataset images."
    )
    parser.add_argument("--split", choices=("train", "eval"), default="eval",
                        help="which saved data split to inspect (default: eval)")
    parser.add_argument("--examples", type=int, default=20,
                        help="number of boards to draw (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="controls which boards are selected (default: 42)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"CNN weights file (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"folder for PNG files (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.examples < 1:
        raise SystemExit("--examples must be at least 1")

    device = choose_device(args.device)
    train_dataset, eval_dataset, _ = create_train_eval_datasets()
    dataset = eval_dataset if args.split == "eval" else train_dataset
    count = min(args.examples, len(dataset))
    selected_indices = random.Random(args.seed).sample(range(len(dataset)), count)

    model = PushTCNNMLP()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    manifest = []
    print(f"device       : {device}")
    print(f"data split   : {args.split} ({len(dataset):,} examples)")
    print(f"drawing      : {count} boards")

    for number, dataset_index in enumerate(selected_indices, start=1):
        example = dataset[dataset_index]
        image = example["image"]
        state = example["state"]
        human_action = example["action"]
        with torch.no_grad():
            prediction = model(image.unsqueeze(0).to(device), state.unsqueeze(0).to(device))[0].cpu()

        state_raw = model_to_raw(state).tolist()
        human_raw = model_to_raw(human_action).tolist()
        prediction_raw = model_to_raw(prediction).clamp(0, 512).tolist()
        filename = output_dir / f"{number:02d}_dataset_{dataset_index:05d}.png"
        render_comparison(image, state_raw, human_raw, prediction_raw, filename, dataset_index)
        written.append(filename)
        manifest.append({
            "image": filename.name,
            "dataset_index": dataset_index,
            "current_pusher": state_raw,
            "human_target": human_raw,
            "cnn_target": prediction_raw,
        })
        print(f"  {number:2d}/{count}: {filename.name}")

    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(written, contact_sheet)
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nSaved individual images in: {output_dir}")
    print(f"Open this overview first:    {contact_sheet}")
    print("blue = current pusher; green = human target; red = CNN target")


if __name__ == "__main__":
    main()

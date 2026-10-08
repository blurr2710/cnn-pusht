"""Train the chunk-8 CNN+MLP PushT behavioral-cloning policy.

Unlike train.py's one-step policy, each example predicts eight consecutive
absolute pusher targets from one image and current pusher state. The final
seven frames of every episode are excluded because they cannot supply a full
eight-action plan without crossing into another game.

Usage (run from the pusht directory):
    source .venv/bin/activate
    python cnn_pusht/train_chunk8.py --device mps
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_CHUNK_SIZE, create_train_eval_chunk_datasets
from model import PushTCNNMLPChunk

BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
OUT_FILE = "cnn_pusht/models/pusht_cnn_mlp_chunk8_v1.pt"


def default_device():
    """Choose the requested CUDA -> MPS -> CPU fallback order."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested):
    if requested == "auto":
        return default_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps was requested, but MPS is unavailable.")
    return requested


def mean_plan_mse(model, loader, device):
    """MSE averaged over every batch element, plan point, and x/y coordinate."""
    model.eval()
    squared_error_sum = 0.0
    n_values = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["action"].to(device)
            prediction = model(batch["image"].to(device), batch["state"].to(device))
            squared_error_sum += torch.nn.functional.mse_loss(
                prediction, target, reduction="sum"
            ).item()
            n_values += target.numel()
    model.train()
    return squared_error_sum / n_values


def main():
    parser = argparse.ArgumentParser(
        description="Train an episode-safe 8-target PushT CNN+MLP policy."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--weights", type=str, default=OUT_FILE)
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least one.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least one.")

    device = resolve_device(args.device)
    print(f"device: {device}")

    train_ex, eval_ex, split = create_train_eval_chunk_datasets(DEFAULT_CHUNK_SIZE)
    print(
        f"chunk size: {DEFAULT_CHUNK_SIZE} targets  "
        f"(excluding final {DEFAULT_CHUNK_SIZE - 1} start frames per episode)"
    )
    print(
        f"train examples: {len(train_ex)}  eval examples: {len(eval_ex)}  "
        f"(seed {split['seed']}, episode-level split)"
    )

    train_loader = DataLoader(train_ex, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ex, batch_size=args.batch_size, shuffle=False)

    model = PushTCNNMLPChunk(chunk_size=DEFAULT_CHUNK_SIZE).to(device)
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        squared_error_sum = 0.0
        n_values = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            target = batch["action"].to(device)

            optimizer.zero_grad()
            prediction = model(image, state)
            loss = loss_fn(prediction, target)  # mean over all 8 x/y targets
            loss.backward()
            optimizer.step()

            squared_error_sum += loss.item() * target.numel()
            n_values += target.numel()

        train_mse = squared_error_sum / n_values
        eval_mse = mean_plan_mse(model, eval_loader, device)
        print(
            f"epoch {epoch:3d}/{args.epochs}  train_mse {train_mse:.6f}  "
            f"eval_mse {eval_mse:.6f}  ({time.time() - started_at:.1f}s)"
        )

    output_path = Path(args.weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()

"""Training script — behavioral cloning of PushTCNNMLP on the PushT train episodes.

What this script does:
  1. load the episode-level train/eval split (80/20, seed 42)
  2. build train + eval datasets
  3. train PushTCNNMLP (model.py) with MSE on normalized actions, Adam
  4. after each epoch: eval loss on eval episodes (monitoring only)
  5. save weights to cnn_pusht/models/pusht_cnn_mlp_v1.pt

Usage:
    source .venv/bin/activate
    python cnn_pusht/train.py
"""
import time

import torch
from torch.utils.data import DataLoader

from dataset import create_train_eval_datasets  # cnn_pusht is the cwd when running
from model import PushTCNNMLP

BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
OUT_FILE = "cnn_pusht/models/pusht_cnn_mlp_v1.pt"


def run_eval_loss(model, eval_loader):
    """Average MSE over eval episodes. Monitoring only — no gradients, no training."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():  # never touch eval data with gradients
        for batch in eval_loader:
            pred = model(batch["image"].to(DEVICE), batch["state"].to(DEVICE))
            loss = torch.nn.functional.mse_loss(
                pred, batch["action"].to(DEVICE), reduction="sum"
            )
            total += loss.item()
            n += pred.shape[0]
    model.train()
    return total / n  # mean squared error per example


def main():
    print(f"device: {DEVICE}")

    # --- datasets: episode-level split from split.json (Step 4) ---
    train_ex, eval_ex, split = create_train_eval_datasets()
    print(f"train examples: {len(train_ex)}  eval examples: {len(eval_ex)}"
          f"  (seed {split['seed']}, split.json)")

    train_loader = DataLoader(train_ex, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ex, batch_size=BATCH_SIZE, shuffle=False)

    # --- model, loss, optimizer ---
    model = PushTCNNMLP().to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # MSE on normalized actions (Step 3 contract). reduction="mean" = the
    # plain average over batch + dims — the single number training shrinks.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- training loop ---
    model.train()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running, n_seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["image"].to(DEVICE), batch["state"].to(DEVICE))
            loss = loss_fn(pred, batch["action"].to(DEVICE))
            loss.backward()      # gradients: how wrong each weight was
            optimizer.step()     # nudge every weight to shrink the loss
            running += loss.item() * pred.shape[0]
            n_seen += pred.shape[0]
        train_mse = running / n_seen

        eval_mse = run_eval_loss(model, eval_loader)
        print(f"epoch {epoch:3d}/{EPOCHS}  train_mse {train_mse:.5f}"
              f"  eval_mse {eval_mse:.5f}  ({time.time() - t0:.1f}s)")

    # --- save weights ---
    torch.save(model.state_dict(), OUT_FILE)
    print(f"saved {OUT_FILE}")


if __name__ == "__main__":
    main()

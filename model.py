"""CNN + MLP policy for PushT.

Architecture (exactly as explained in STEP1 contract + STEP3 preprocessing):
  image (B,3,96,96) -> CNN encoder -> features (B,128*6*6 = 4608)
  state (B,2)  [normalized to roughly (-1..1)]
  concat -> (B,4610) -> MLP head -> (B,2) predicted next target (x,y) [normalized]

Loss: MSE between predicted and demonstrated actions, both normalized.
"""
import torch
import torch.nn as nn


class PushTCNNMLP(nn.Module):
    """CNN image encoder -> concat pusher (x,y) -> MLP -> next target (x,y)."""

    def __init__(self):
        super().__init__()

        # --- image encoder: 4 conv layers, each halves the board ---
        # kernel 3x3, stride 2 downsizes (96 -> 48 -> 24 -> 12 -> 6),
        # padding 1 keeps the border. ReLU after each conv.
        # Channels grow 3 -> 32 -> 64 -> 128 -> 128.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # (B,128,6,6) -> (B, 128*6*6) = (B, 4608)
        )

        # --- decision head: plain MLP ---
        # input = 4608 visual features + 2 state numbers = 4610
        # shrink 4610 -> 128 -> 64 -> 2 (the output: next target x,y)
        self.head = nn.Sequential(
            nn.Linear(128 * 6 * 6 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, image, state):
        feat = self.encoder(image)              # (B, 4608)
        joined = torch.cat([feat, state], dim=1)  # (B, 4610)
        return self.head(joined)                # (B, 2)


class PushTCNNMLPChunk(nn.Module):
    """CNN + MLP policy that predicts a connected plan of future targets.

    The image encoder matches :class:`PushTCNNMLP`. Only the last linear layer
    changes: it emits ``chunk_size * 2`` values which are reshaped into
    ``(batch, chunk_size, 2)`` absolute pusher targets in normalized space.
    """

    def __init__(self, chunk_size=8):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 6 * 6 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, chunk_size * 2),
        )

    def forward(self, image, state):
        feat = self.encoder(image)
        joined = torch.cat([feat, state], dim=1)
        return self.head(joined).view(-1, self.chunk_size, 2)


if __name__ == "__main__":
    # sanity check: run one fake example through the net, print shapes
    model = PushTCNNMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params:,}")

    fake_image = torch.randn(2, 3, 96, 96)
    fake_state = torch.randn(2, 2)
    out = model(fake_image, fake_state)
    print(f"image {tuple(fake_image.shape)} + state {tuple(fake_state.shape)}"
          f" -> action {tuple(out.shape)}")
    assert out.shape == (2, 2), "output shape wrong"
    print("model.py sanity check: PASS")

"""Step 5 — rollout: load the trained brain, play real games in the simulator.

Why this exists: train.py's eval replays *recorded* moments and compares the
model's guess to the recorded human answer. It never lets the model touch the
controls. A rollout is the real test: the model plays whole games, and its own
moves create the next situations it sees (errors can compound — the quiz
cannot measure that).

What this script does:
  1. load weights from cnn_pusht/models/pusht_cnn_mlp_v1.pt (saved by train.py)
  2. start N games in the real PushT simulator (gym_pusht, seeded)
  3. each step: render the board (96x96, same pipeline as the training
     videos), predict the next target, move there, repeat
  4. stop when the block reaches the goal (>=95% coverage) or after 300 moves
  5. report per-game result + summary (success rate, best coverage)

Usage:
    source .venv/bin/activate
    python cnn_pusht/rollout.py                 # default: 10 games
    python cnn_pusht/rollout.py --episodes 5 --seed 1000
"""
import argparse

import gymnasium as gym
import numpy as np
import torch

import gym_pusht  # noqa: F401  (importing registers the PushT env with gymnasium)

from dataset import normalize_state_action  # same math as training data
from model import PushTCNNMLP

WEIGHTS_FILE = "cnn_pusht/models/pusht_cnn_mlp_v1.pt"
EPISODES = 10
SEED = 1000
MAX_STEPS = 300       # same cap the gym registration enforces (TimeLimit)
SUCCESS_THRESHOLD = 0.95  # 95% of the T-block inside the goal zone = win

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def load_policy(weights_file):
    """Rebuild the brain and pour the saved numbers back in."""
    model = PushTCNNMLP()
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.to(DEVICE)
    model.eval()  # exam mode: no gradient bookkeeping, behavior identical
    return model


def preprocess(pixels, agent_pos):
    """Raw sim outputs -> exactly the tensors training saw.

    pixels     (96,96,3) uint8 0-255 -> (1,3,96,96) float 0-1
    agent_pos  (2,) float 0-512      -> (1,2)     float ~(-1..1)

    The env already renders its pixels with the same 512->96 cv2.resize the
    recorded dataset videos were made with, so no further resizing is needed.
    """
    img = torch.from_numpy(np.ascontiguousarray(pixels)).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # (3,96,96) -> add batch dim
    state = normalize_state_action(torch.from_numpy(agent_pos).float()).unsqueeze(0)
    return img.to(DEVICE), state.to(DEVICE)


def predict_target(model, img, state):
    """Run the brain once, un-normalize its answer back to 0..512 sim units."""
    with torch.no_grad():
        pred = model(img, state)  # (1,2), normalized ~(-1..1)
    raw = pred[0].cpu().numpy() * 256.0 + 256.0  # back to 512x512 board space
    return np.clip(raw, 0.0, 512.0).astype(np.float32)


def play_one_game(env, model, seed):
    """Play a single game from a fresh random board. Returns a result dict."""
    obs, info = env.reset(seed=seed)
    best_coverage = 0.0
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        img, state = preprocess(obs["pixels"], obs["agent_pos"])
        target = predict_target(model, img, state)  # where the model wants to go
        obs, reward, terminated, truncated, info = env.step(target)
        best_coverage = max(best_coverage, info["coverage"])
        if terminated:
            break
        if truncated:  # TimeLimit fired at MAX_STEPS
            break
    return {
        "success": bool(info["is_success"]),
        "coverage": info["coverage"],        # coverage at the last step
        "best_coverage": best_coverage,      # highest coverage reached in the game
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Play PushT with the trained policy.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weights", type=str, default=WEIGHTS_FILE)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    model = load_policy(args.weights)
    print(f"weights : {args.weights}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # pixels_agent_pos = the env gives us the 96x96 board picture and the
    # pusher (x,y) each step — exactly the two inputs of our policy.
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")

    results = []
    for i in range(args.episodes):
        seed = args.seed + i
        res = play_one_game(env, model, seed)
        results.append(res)
        outcome = "SUCCESS" if res["success"] else "no success"
        print(f"game {i + 1:2d}/{args.episodes} (seed {seed})  {outcome:10s}"
              f"  final coverage {res['coverage']:.3f}"
              f"  best coverage {res['best_coverage']:.3f}"
              f"  steps {res['steps']:3d}")

    env.close()

    n_success = sum(r["success"] for r in results)
    mean_best = float(np.mean([r["best_coverage"] for r in results]))
    mean_steps = float(np.mean([r["steps"] for r in results]))
    print(f"\nsummary: success {n_success}/{args.episodes}"
          f"  mean best coverage {mean_best:.3f}"
          f"  mean steps {mean_steps:.0f}")
    print(f"(success = >=95% of the T-block inside the goal zone within 300 steps)")


if __name__ == "__main__":
    main()

"""Roll out the chunk-8 CNN+MLP PushT policy in the real simulator.

The policy observes once, creates an eight-position plan, sends all eight
targets in order, then observes again to replan. A game succeeds only when at
least 95% coverage is reached within the environment's 300 simulation steps.

Usage (run from the pusht directory):
    source .venv/bin/activate
    python cnn_pusht/rollout_chunk8.py --episodes 2 --seed 1000 --device mps
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import gym_pusht  # noqa: F401  (registers the PushT environment)

from dataset import DEFAULT_CHUNK_SIZE, normalize_state_action
from model import PushTCNNMLPChunk

WEIGHTS_FILE = "cnn_pusht/models/pusht_cnn_mlp_chunk8_v1.pt"
EPISODES = 10
SEED = 1000
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.95


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


def load_policy(weights_file, device):
    """Load a trained chunk-8 policy onto the selected device."""
    weights_path = Path(weights_file)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Chunk-8 checkpoint not found: {weights_path}\n"
            "Train it first with: python cnn_pusht/train_chunk8.py"
        )
    model = PushTCNNMLPChunk(chunk_size=DEFAULT_CHUNK_SIZE)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def preprocess(pixels, agent_pos, device):
    """Apply the same image and (value - 256) / 256 preprocessing as training."""
    image = torch.from_numpy(np.ascontiguousarray(pixels)).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    state = normalize_state_action(torch.from_numpy(agent_pos).float()).unsqueeze(0)
    return image.to(device), state.to(device)


def predict_plan(model, image, state):
    """Un-normalize one predicted ``(8, 2)`` plan into simulator coordinates."""
    with torch.no_grad():
        normalized_plan = model(image, state)[0]
    plan = normalized_plan.cpu().numpy() * 256.0 + 256.0
    return np.clip(plan, 0.0, 512.0).astype(np.float32)


def play_one_game(env, model, seed, device):
    """Replan after each eight simulator steps (or earlier if the game ends)."""
    observation, info = env.reset(seed=seed)
    best_coverage = float(info.get("coverage", 0.0))
    steps = 0
    terminated = truncated = False

    while steps < MAX_STEPS and not (terminated or truncated):
        image, state = preprocess(observation["pixels"], observation["agent_pos"], device)
        plan = predict_plan(model, image, state)

        # Do not render or re-infer between targets: this is intentionally a
        # connected open-loop eight-action plan, followed by a new observation.
        for target in plan:
            observation, reward, terminated, truncated, info = env.step(target)
            steps += 1
            best_coverage = max(best_coverage, float(info["coverage"]))
            if terminated or truncated or steps >= MAX_STEPS:
                break

    return {
        "success": bool(info["is_success"]),
        "coverage": float(info["coverage"]),
        "best_coverage": best_coverage,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the chunk-8 PushT CNN+MLP policy.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weights", type=str, default=WEIGHTS_FILE)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least one.")

    device = resolve_device(args.device)
    print(f"device: {device}")
    model = load_policy(args.weights, device)
    print(f"weights: {args.weights}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    results = []
    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            result = play_one_game(env, model, seed, device)
            results.append(result)
            outcome = "SUCCESS" if result["success"] else "FAILURE"
            print(
                f"game {episode_index + 1:2d}/{args.episodes}  seed {seed}  {outcome:7s}  "
                f"best coverage {result['best_coverage']:.3f}  "
                f"final coverage {result['coverage']:.3f}  steps {result['steps']:3d}"
            )
    finally:
        env.close()

    n_success = sum(result["success"] for result in results)
    mean_best_coverage = float(np.mean([result["best_coverage"] for result in results]))
    print(
        f"\nsummary: success {n_success}/{args.episodes}  "
        f"mean best coverage {mean_best_coverage:.3f}"
    )
    print(f"(success = >= {SUCCESS_THRESHOLD:.0%} coverage within {MAX_STEPS} simulation steps)")


if __name__ == "__main__":
    main()

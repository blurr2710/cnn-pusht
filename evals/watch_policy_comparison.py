"""Watch the CNN and diffusion policies play the same PushT starting board.

The two policies play one after another.  Both environments are reset with the
same seed, so the pusher, T block, and goal begin in exactly the same places.
This is a visual comparison of full closed-loop games, not a dataset test.

Run from the repository root:
    .venv/bin/python cnn_pusht/evals/watch_policy_comparison.py --seed 1000 --device mps

Close the simulator window or press Ctrl-C in the terminal to stop early.
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import gym_pusht  # noqa: F401  (registers PushT with Gymnasium)

# The project modules live one folder above this file.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dataset import normalize_state_action
from model import PushTCNNMLP
from eval_hf_baseline import find_model_dir, load_hf_policy as load_diffusion_policy
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.transition import move_state_dict_to_device


CNN_WEIGHTS = PROJECT_DIR / "models" / "pusht_cnn_mlp_v1.pt"
MAX_STEPS = 300


def choose_device(requested):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available. Use --device cpu.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is not available. Use --device mps or cpu.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cnn(weights_file, device):
    model = PushTCNNMLP()
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    return model.to(device).eval()


def cnn_action(model, observation, device):
    """Turn simulator observation into one absolute PushT target position."""
    pixels = torch.from_numpy(np.ascontiguousarray(observation["pixels"])).float() / 255.0
    image = pixels.permute(2, 0, 1).unsqueeze(0).to(device)
    state = normalize_state_action(
        torch.from_numpy(observation["agent_pos"]).float()
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        normalized_target = model(image, state)[0].cpu().numpy()
    return np.clip(normalized_target * 256.0 + 256.0, 0.0, 512.0).astype(np.float32)


def diffusion_action(policy, preprocessor, postprocessor, observation, device):
    """Run the Hugging Face policy through its official pre/post-processing."""
    raw_observation = preprocess_observation(observation)
    with torch.no_grad():
        model_input = move_state_dict_to_device(preprocessor(raw_observation), device)
        normalized_action = policy.select_action(model_input)
        return postprocessor(normalized_action)[0].cpu().numpy()


def watch_one_policy(name, action_fn, seed):
    """Open a real PushT window and let one policy play a complete game."""
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="human",
    )
    observation, _ = env.reset(seed=seed)
    env.render()  # creates the visible window
    try:
        # The title makes it obvious which policy is currently in control.
        import pygame

        pygame.display.set_caption(f"PushT: {name} — seed {seed}")
    except Exception:
        pass

    print(f"\nWatching {name} from starting seed {seed}...")
    best_coverage = 0.0
    try:
        for step in range(1, MAX_STEPS + 1):
            action = action_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            best_coverage = max(best_coverage, float(info["coverage"]))
            env.render()
            if step % 25 == 0 or terminated or truncated:
                print(f"  {name:20s} step {step:3d}: coverage {info['coverage']:.3f}")
            if terminated or truncated:
                break
    finally:
        env.close()

    outcome = "SUCCESS" if info["is_success"] else "no success"
    print(
        f"{name}: {outcome}; best coverage {best_coverage:.3f}; "
        f"finished after {step} moves."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Watch both PushT policies play the identical seeded starting board."
    )
    parser.add_argument("--seed", type=int, default=1000,
                        help="simulator starting board; both policies use this seed")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--cnn-weights", type=Path, default=CNN_WEIGHTS)
    parser.add_argument("--hf-model-dir", type=Path,
                        help="migrated Hugging Face model folder; uses common local location by default")
    parser.add_argument("--only", choices=("both", "cnn", "diffusion"), default="both",
                        help="watch only one policy, or both in sequence (default: both)")
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"device: {device}")
    print("Both policies will start from the same simulator seed.")

    if args.only in ("both", "cnn"):
        if not args.cnn_weights.is_file():
            raise SystemExit(f"CNN weights not found: {args.cnn_weights}")
        cnn = load_cnn(args.cnn_weights, device)
        watch_one_policy("CNN + MLP", lambda obs: cnn_action(cnn, obs, device), args.seed)

    if args.only in ("both", "diffusion"):
        model_dir = args.hf_model_dir or find_model_dir()
        if not model_dir.is_dir():
            raise SystemExit(
                "Hugging Face model folder not found. Pass --hf-model-dir with the "
                "migrated model folder used by eval_hf_baseline.py."
            )
        diffusion, preprocessor, postprocessor = load_diffusion_policy(model_dir, device)
        torch.manual_seed(args.seed)
        diffusion.reset()
        watch_one_policy(
            "Hugging Face diffusion",
            lambda obs: diffusion_action(diffusion, preprocessor, postprocessor, obs, device),
            args.seed,
        )


if __name__ == "__main__":
    main()

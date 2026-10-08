"""Evaluate the local Hugging Face PushT diffusion policy in the PushT simulator.

Example:
    .venv/bin/python cnn_pusht/evals/eval_hf_baseline.py --episodes 20
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the environment)
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.transition import move_state_dict_to_device


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "lerobot_diffusion_pusht_0p4"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "hf_baseline_eval.json"
MAX_STEPS = 300


def load_policy(device):
    """Load the already-downloaded Hugging Face PushT policy."""
    config = PreTrainedConfig.from_pretrained(MODEL_DIR)
    config.device = device
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=MODEL_DIR)
    policy = DiffusionPolicy.from_pretrained(MODEL_DIR, config=config)
    policy.to(device)
    policy.eval()
    return policy, preprocessor, postprocessor


def play_episode(env, policy, preprocessor, postprocessor, seed, device):
    """Play one full game and return simple, human-readable measurements."""
    torch.manual_seed(seed)
    policy.reset()
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    best_coverage = 0.0

    for step in range(1, MAX_STEPS + 1):
        raw_observation = preprocess_observation(observation)
        with torch.no_grad():
            model_input = preprocessor(raw_observation)
            model_input = move_state_dict_to_device(model_input, device)
            normalized_action = policy.select_action(model_input)
            action = postprocessor(normalized_action)[0].cpu().numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        best_coverage = max(best_coverage, float(info["coverage"]))
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "success": bool(info["is_success"]),
        "steps": step,
        "final_coverage": float(info["coverage"]),
        "best_coverage": best_coverage,
        "total_reward": total_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face PushT baseline.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of games to play.")
    parser.add_argument("--seed", type=int, default=1000, help="First game seed.")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Where to run the model: cuda for an NVIDIA GPU, mps for an Apple GPU, or cpu.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine; use --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine; use --device cpu or mps")
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model folder not found: {MODEL_DIR}")

    policy, preprocessor, postprocessor = load_policy(args.device)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    results = []
    try:
        for episode_index in range(args.episodes):
            result = play_episode(
                env, policy, preprocessor, postprocessor, args.seed + episode_index, args.device
            )
            results.append(result)
            outcome = "SUCCESS" if result["success"] else "no success"
            print(
                f"game {episode_index + 1:2d}/{args.episodes} ({outcome})  "
                f"best coverage {result['best_coverage']:.3f}  steps {result['steps']}"
            )
    finally:
        env.close()

    successes = sum(result["success"] for result in results)
    report = {
        "model": str(MODEL_DIR),
        "device": args.device,
        "episode_count": args.episodes,
        "first_seed": args.seed,
        "success_rate_percent": 100.0 * successes / args.episodes,
        "successes": successes,
        "mean_best_coverage": float(np.mean([result["best_coverage"] for result in results])),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nSuccess: {successes}/{args.episodes} ({report['success_rate_percent']:.1f}%)")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

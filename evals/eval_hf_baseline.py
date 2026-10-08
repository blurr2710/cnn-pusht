"""Evaluate any of the three PushT policies with the same rollout rules.

The ``--policy`` flag chooses the brain being tested:
  single  : the original CNN+MLP, one target at a time
  chunk8  : the CNN+MLP that plans eight targets, then replans
  hf      : the downloaded Hugging Face diffusion baseline

All choices use the real PushT simulator, the same 300-step cap, the same
starting seeds, and the same 95% coverage success threshold.

Examples (run from the cnn_pusht folder):
    python evals/eval_hf_baseline.py --policy single --episodes 20 --device cuda
    python evals/eval_hf_baseline.py --policy chunk8 --episodes 20 --device cuda
    python evals/eval_hf_baseline.py --policy hf --episodes 20 --device cuda \
        --model-dir models/hf_baseline_migrated
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers PushT with gymnasium)
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from dataset import DEFAULT_CHUNK_SIZE, normalize_state_action
from model import PushTCNNMLP, PushTCNNMLPChunk
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.transition import move_state_dict_to_device


MODEL_NAME = "lerobot_diffusion_pusht_0p4"
MAX_STEPS = 300
SINGLE_WEIGHTS = PROJECT_DIR / "models" / "pusht_cnn_mlp_v1.pt"
CHUNK8_WEIGHTS = PROJECT_DIR / "models" / "pusht_cnn_mlp_chunk8_v1.pt"


def resolve_device(requested):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS is unavailable. Use --device cpu.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable. Use --device mps or cpu.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_model_dir():
    """Find the migrated Hugging Face model in either supported project layout."""
    candidates = [
        PROJECT_DIR / "models" / "hf_baseline_migrated",
        PROJECT_DIR / "models" / MODEL_NAME,
        ROOT / "models" / MODEL_NAME,
    ]
    for directory in candidates:
        if (directory / "policy_preprocessor.json").is_file():
            return directory
    return candidates[0]


def load_hf_policy(model_dir, device):
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = device
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=model_dir)
    policy = DiffusionPolicy.from_pretrained(model_dir, config=config)
    return policy.to(device).eval(), preprocessor, postprocessor


def load_cnn_policy(policy_name, weights_file, device):
    model = PushTCNNMLP() if policy_name == "single" else PushTCNNMLPChunk(
        chunk_size=DEFAULT_CHUNK_SIZE
    )
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    return model.to(device).eval()


def cnn_inputs(observation, device):
    image = torch.from_numpy(np.ascontiguousarray(observation["pixels"])).float() / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    state = normalize_state_action(
        torch.from_numpy(observation["agent_pos"]).float()
    ).unsqueeze(0).to(device)
    return image, state


def make_single_action_fn(model, device):
    def choose_action(observation):
        image, state = cnn_inputs(observation, device)
        with torch.no_grad():
            action = model(image, state)[0].cpu().numpy()
        return np.clip(action * 256.0 + 256.0, 0.0, 512.0).astype(np.float32)

    return choose_action, lambda: None


def make_chunk8_action_fn(model, device):
    plan = []

    def reset():
        plan.clear()

    def choose_action(observation):
        if not plan:
            image, state = cnn_inputs(observation, device)
            with torch.no_grad():
                predicted_plan = model(image, state)[0].cpu().numpy()
            raw_plan = np.clip(predicted_plan * 256.0 + 256.0, 0.0, 512.0)
            plan.extend(raw_plan.astype(np.float32))
        return plan.pop(0)

    return choose_action, reset


def make_hf_action_fn(policy, preprocessor, postprocessor, device):
    def reset():
        policy.reset()

    def choose_action(observation):
        raw_observation = preprocess_observation(observation)
        with torch.no_grad():
            model_input = move_state_dict_to_device(preprocessor(raw_observation), device)
            normalized_action = policy.select_action(model_input)
            return postprocessor(normalized_action)[0].cpu().numpy()

    return choose_action, reset


def play_episode(env, choose_action, reset_policy, seed):
    reset_policy()
    observation, _ = env.reset(seed=seed)
    best_coverage = 0.0
    total_reward = 0.0
    for step in range(1, MAX_STEPS + 1):
        observation, reward, terminated, truncated, info = env.step(choose_action(observation))
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
    parser = argparse.ArgumentParser(description="Evaluate a PushT policy in the real simulator.")
    parser.add_argument("--policy", choices=("hf", "single", "chunk8"), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--weights", type=Path,
                        help="CNN checkpoint; defaults to the matching single or chunk-8 checkpoint")
    parser.add_argument("--model-dir", type=Path,
                        help="migrated Hugging Face model folder; only used with --policy hf")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.policy == "hf" and args.weights:
        raise SystemExit("--weights only applies to --policy single or --policy chunk8")
    if args.policy != "hf" and args.model_dir:
        raise SystemExit("--model-dir only applies to --policy hf")

    device = resolve_device(args.device)
    if args.policy == "hf":
        model_path = args.model_dir or find_model_dir()
        if not (model_path / "policy_preprocessor.json").is_file():
            raise SystemExit(
                f"Hugging Face model is not ready at {model_path}. "
                "Run bash scripts/download_hf_baseline.sh first."
            )
        model, preprocessor, postprocessor = load_hf_policy(model_path, device)
        choose_action, reset_policy = make_hf_action_fn(model, preprocessor, postprocessor, device)
    else:
        default_weights = SINGLE_WEIGHTS if args.policy == "single" else CHUNK8_WEIGHTS
        model_path = args.weights or default_weights
        if not model_path.is_file():
            raise SystemExit(f"CNN checkpoint not found: {model_path}")
        model = load_cnn_policy(args.policy, model_path, device)
        if args.policy == "single":
            choose_action, reset_policy = make_single_action_fn(model, device)
        else:
            choose_action, reset_policy = make_chunk8_action_fn(model, device)

    print(f"policy: {args.policy}")
    print(f"device: {device}")
    print(f"model:  {model_path}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    results = []
    try:
        for index in range(args.episodes):
            result = play_episode(env, choose_action, reset_policy, args.seed + index)
            results.append(result)
            outcome = "SUCCESS" if result["success"] else "no success"
            print(
                f"game {index + 1:2d}/{args.episodes} (seed {result['seed']}) {outcome:10s} "
                f"best coverage {result['best_coverage']:.3f} steps {result['steps']}"
            )
    finally:
        env.close()

    successes = sum(result["success"] for result in results)
    report = {
        "policy": args.policy,
        "model": str(model_path),
        "device": device,
        "episode_count": args.episodes,
        "first_seed": args.seed,
        "success_rate_percent": 100.0 * successes / args.episodes,
        "successes": successes,
        "mean_best_coverage": float(np.mean([result["best_coverage"] for result in results])),
        "episodes": results,
    }
    output = args.output or Path("evals/results") / f"{args.policy}_{args.episodes}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nSuccess: {successes}/{args.episodes} ({report['success_rate_percent']:.1f}%)")
    print(f"Mean best coverage: {report['mean_best_coverage']:.3f}")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()

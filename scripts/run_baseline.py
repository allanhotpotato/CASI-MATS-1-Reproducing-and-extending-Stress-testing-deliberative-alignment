"""
Run a baseline evaluation: N rollouts of a model across all environments.

Saves each trajectory as a JSON record to results/<run_id>.json.

Usage:
    python scripts/run_baseline.py --model deepseek-r1 --rollouts 30
    python scripts/run_baseline.py --model qwen-qwq-32b --rollouts 10 --env moderator_task
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from environments.moderator_task import ModeratorTask
from environments.duplicate_check_task import DuplicateCheckTask
from models.deepseek_r1 import DeepSeekR1
from models.qwen_qwq import QwenQwQ
from evaluation.classifier import classify

MODELS = {
    "deepseek-r1": DeepSeekR1,
    "qwen-qwq-32b": QwenQwQ,
}

ENVIRONMENTS = {
    "moderator_task": ModeratorTask,
    "duplicate_check_task": DuplicateCheckTask,
}


def run_baseline(model_name: str, n_rollouts: int, env_filter: str | None, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    model = MODELS[model_name]()
    run_id = f"{model_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    records = []

    envs = {k: v for k, v in ENVIRONMENTS.items() if env_filter is None or k == env_filter}

    for env_name, EnvClass in envs.items():
        print(f"\n=== {env_name} ({n_rollouts} rollouts) ===")
        for i in range(n_rollouts):
            env = EnvClass()   # each environment manages its own reset and topic selection
            env.reset()
            trajectory = env.run(model)
            label = classify(trajectory)
            print(f"  rollout {i+1:3d}: {label}")
            records.append(asdict(trajectory))

    output_path = results_dir / f"{run_id}.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} trajectories to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="deepseek-r1")
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--env", choices=list(ENVIRONMENTS), default=None)
    parser.add_argument("--results_dir", type=Path, default=Path("results/"))
    args = parser.parse_args()

    run_baseline(args.model, args.rollouts, args.env, args.results_dir)


if __name__ == "__main__":
    main()

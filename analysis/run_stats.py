"""
Compute summary statistics from results/ JSON files.

Reads all trajectory JSON files in a results directory and prints:
    - Covert action rate per environment and per model
    - Escalation rate and compliance rate
    - Breakdown table (environment × model × label counts)

Usage:
    python analysis/run_stats.py --results_dir results/
    python analysis/run_stats.py --results_dir results/ --model deepseek-r1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABELS = ["covert_action", "escalated", "complied", "refused"]


def load_results(results_dir: Path) -> list[dict]:
    """Load all trajectory JSON files from a directory."""
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


def compute_rates(records: list[dict]) -> dict:
    """
    Returns a nested dict: rates[environment][model][label] = count.
    Also includes a 'total' key per (environment, model).
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals: dict = defaultdict(lambda: defaultdict(int))

    for r in records:
        env = r.get("environment", "unknown")
        model = r.get("model", "unknown")
        label = r.get("label", "unknown")
        counts[env][model][label] += 1
        totals[env][model] += 1

    rates = {}
    for env, models in counts.items():
        rates[env] = {}
        for model, label_counts in models.items():
            total = totals[env][model]
            rates[env][model] = {
                label: label_counts[label] / total for label in LABELS
            }
            rates[env][model]["total"] = total

    return rates


def print_table(rates: dict) -> None:
    header = f"{'Environment':<30} {'Model':<20} {'N':>5} " + " ".join(f"{l:>14}" for l in LABELS)
    print(header)
    print("-" * len(header))
    for env, models in rates.items():
        for model, r in models.items():
            row = f"{env:<30} {model:<20} {r['total']:>5} " + " ".join(
                f"{r.get(l, 0.0):>13.1%}" for l in LABELS
            )
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results/"))
    parser.add_argument("--model", type=str, default=None, help="Filter to a single model")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if args.model:
        records = [r for r in records if r.get("model") == args.model]

    if not records:
        print("No records found.")
        return

    rates = compute_rates(records)
    print_table(rates)


if __name__ == "__main__":
    main()

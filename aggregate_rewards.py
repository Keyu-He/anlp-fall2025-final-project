#!/usr/bin/env python
"""
Aggregate Sotopia reward logs into per-agent averages.

Usage:
    python aggregate_rewards.py output.jsonl results/file1.jsonl [results/file2.jsonl ...]
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def _extract_agent_dims(agent_reward: Any) -> Dict[str, float]:
    """
    agent_reward is expected to be:
      [overall_score, {dim: score, ...}]  or  (overall_score, {..})
    """
    if agent_reward is None:
        return {}
    if isinstance(agent_reward, (list, tuple)) and len(agent_reward) == 2:
        _, dims = agent_reward
        if isinstance(dims, dict):
            return {k: float(v) for k, v in dims.items()}
    if isinstance(agent_reward, dict):
        return {k: float(v) for k, v in agent_reward.items()}
    return {}


def aggregate_file(path: str) -> Dict[str, Any]:
    agent_stats: List[Dict[str, Dict[str, Tuple[float, int]]]] = [
        defaultdict(lambda: (0.0, 0)),  # agent 1
        defaultdict(lambda: (0.0, 0)),  # agent 2
    ]
    agent_names = ["agent1", "agent2"]
    agent_models = ["", ""]
    episode_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            episode_count += 1

            # names
            env = rec.get("environment", {})
            names = env.get("agent_names") or []
            for idx in (0, 1):
                if idx < len(names) and names[idx]:
                    agent_names[idx] = names[idx]

            # models (assume consistent per file; models[1] and models[2] are agent1/2)
            models = rec.get("meta", {}).get("models") or []
            if len(models) >= 3:
                if not agent_models[0]:
                    agent_models[0] = models[1]
                if not agent_models[1]:
                    agent_models[1] = models[2]

            rewards = rec.get("rewards") or []
            for idx in (0, 1):
                if idx >= len(rewards):
                    continue
                dims = _extract_agent_dims(rewards[idx])
                for dim, val in dims.items():
                    s, n = agent_stats[idx][dim]
                    agent_stats[idx][dim] = (s + val, n + 1)

    def summarize(idx: int) -> Dict[str, Any]:
        avg_scores = {
            dim: (s / n if n > 0 else 0.0) for dim, (s, n) in agent_stats[idx].items()
        }
        return {
            "name": agent_names[idx],
            "model": agent_models[idx],
            "avg_scores": avg_scores,
            "episode_count": episode_count,
        }

    return {
        "source_file": os.path.basename(path),
        "agent1": summarize(0),
        "agent2": summarize(1),
    }


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print(
            "Usage: python aggregate_rewards.py output.jsonl results/file1.jsonl [results/file2.jsonl ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = argv[1]
    in_paths = argv[2:]

    records: List[Dict[str, Any]] = []
    for p in in_paths:
        if not os.path.isfile(p):
            print(f"Warning: skip non-existent file {p}", file=sys.stderr)
            continue
        records.append(aggregate_file(p))

    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            json.dump(rec, out_f, ensure_ascii=False)
            out_f.write("\n")


if __name__ == "__main__":
    main(sys.argv)


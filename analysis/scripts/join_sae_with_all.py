import json
from pathlib import Path


def get_scenario(block: str) -> str:
    """Extract the Scenario: ... line content from a context/prompt block."""
    if "Scenario:" not in block:
        return ""
    part = block.split("Scenario:", 1)[1]
    if "\nParticipants:" in part:
        part = part.split("\nParticipants:", 1)[0]
    return part.strip()


def index_tasks_by_scenario(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            ctx = ep["meta"]["messages"][0][0][2]
            scenario = get_scenario(ctx)
            if not scenario:
                continue
            # 如果同一个 Scenario 出现多次，这里只保留第一条
            if scenario not in mapping:
                mapping[scenario] = ep
    return mapping


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir.parent / "results"
    sae_dir = results_dir / "sae"

    sae_path = (
        sae_dir
        / "sae_server_logs_l15_k64_20251201_133447_all_only.jsonl"
    )
    all_path = (
        results_dir
        / "sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl"
    )

    if not sae_path.exists():
        raise SystemExit(f"SAE file not found: {sae_path}")
    if not all_path.exists():
        raise SystemExit(f"All-results file not found: {all_path}")

    tasks_by_scenario = index_tasks_by_scenario(all_path)

    out_path = sae_path.with_name(
        sae_path.stem + "_joined_all" + sae_path.suffix
    )

    matched = 0
    total = 0

    with sae_path.open("r") as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            log = json.loads(line)
            scenario = get_scenario(log.get("prompt", ""))
            task = tasks_by_scenario.get(scenario)
            if task is not None:
                env = task.get("environment", {})
                meta = task.get("meta", {})
                rewards = task.get("rewards", [])
                log["task_info"] = {
                    "environment_id": env.get("id"),
                    "agent_names": env.get("agent_names"),
                    "tag": meta.get("tag"),
                    "rewards": rewards,
                }
                matched += 1
            fout.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"total SAE records  : {total}")
    print(f"matched to episodes: {matched}")
    print(f"output written to  : {out_path}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR.parent / "results"
SAE_DIR = RESULTS_DIR / "sae"

SAE_PATH = SAE_DIR / "sae_server_logs_l15_k64_20251201_133447.jsonl"
SOTOPIA_PATH = RESULTS_DIR / "sotopia_hard_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_133556.jsonl"
OUT_PATH = SAE_PATH.with_name(SAE_PATH.stem + "_joined" + SAE_PATH.suffix)


def get_scenario(block: str) -> str:
    """Extract the Scenario: ... line content from a context/prompt block."""
    if "Scenario:" not in block:
        return ""
    part = block.split("Scenario:", 1)[1]
    if "\nParticipants:" in part:
        part = part.split("\nParticipants:", 1)[0]
    return part.strip()


def index_tasks_by_scenario(path: Path):
    mapping = {}
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
            mapping.setdefault(scenario, []).append(ep)
    return mapping


def main():
    tasks_by_scenario = index_tasks_by_scenario(SOTOPIA_PATH)

    with SAE_PATH.open("r") as fin, OUT_PATH.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            log = json.loads(line)
            scenario = get_scenario(log.get("prompt", ""))
            tasks = tasks_by_scenario.get(scenario)
            if tasks:
                task = tasks[0]
                env = task.get("environment", {})
                meta = task.get("meta", {})
                rewards = task.get("rewards", [])
                log["task_info"] = {
                    "environment_id": env.get("id"),
                    "agent_names": env.get("agent_names"),
                    "tag": meta.get("tag"),
                    "rewards": rewards,
                }
            fout.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

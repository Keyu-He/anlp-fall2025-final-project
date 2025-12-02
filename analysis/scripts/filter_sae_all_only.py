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


def build_scenario_to_env(all_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with all_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            env = ep.get("environment", {})
            env_id = env.get("id")
            if not env_id:
                continue
            ctx = ep["meta"]["messages"][0][0][2]
            scenario = get_scenario(ctx)
            if not scenario:
                continue
            mapping[scenario] = env_id
    return mapping


def load_hard_env_ids(hard_path: Path) -> set[str]:
    ids: set[str] = set()
    with hard_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            env = ep.get("environment", {})
            env_id = env.get("id")
            if env_id:
                ids.add(env_id)
    return ids


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir.parent / "results"
    sae_dir = results_dir / "sae"

    sae_path = sae_dir / "sae_server_logs_l15_k64_20251201_133447.jsonl"
    all_path = (
        results_dir
        / "sotopia_all_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_merged.jsonl"
    )
    hard_path = (
        results_dir
        / "sotopia_hard_gpt-4o_Qwen_Qwen2.5-7B-Instruct_20251201_133556.jsonl"
    )

    if not sae_path.exists():
        raise SystemExit(f"SAE file not found: {sae_path}")
    if not all_path.exists():
        raise SystemExit(f"All-results file not found: {all_path}")
    if not hard_path.exists():
        raise SystemExit(f"Hard-results file not found: {hard_path}")

    scenario_to_env = build_scenario_to_env(all_path)
    hard_ids = load_hard_env_ids(hard_path)

    # First pass: find the earliest timestamp of any SAE entry
    # that maps to a non-hard environment. This approximates the
    # start of the 'all' run, assuming 'hard' was run completely
    # before 'all'.
    all_start_ts = None
    records: list[dict] = []

    with sae_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            log = json.loads(line)
            records.append(log)
            scenario = get_scenario(log.get("prompt", ""))
            env_id = scenario_to_env.get(scenario)
            if not env_id:
                continue
            if env_id in hard_ids:
                continue
            ts = log.get("timestamp")
            if isinstance(ts, (int, float)):
                if all_start_ts is None or ts < all_start_ts:
                    all_start_ts = ts

    if all_start_ts is None:
        raise SystemExit(
            "Could not infer all_start_ts; no SAE entries mapped to non-hard envs."
        )

    out_path = sae_path.with_name(
        sae_path.stem + "_all_only" + sae_path.suffix
    )

    kept = 0
    with out_path.open("w") as fout:
        for log in records:
            scenario = get_scenario(log.get("prompt", ""))
            env_id = scenario_to_env.get(scenario)
            if not env_id:
                # If we cannot map to an env, keep it as-is.
                fout.write(json.dumps(log, ensure_ascii=False) + "\n")
                kept += 1
                continue

            ts = log.get("timestamp")
            if env_id in hard_ids and isinstance(ts, (int, float)):
                # Drop SAE entries for hard envs that were logged
                # before the 'all' run started.
                if ts < all_start_ts:
                    continue

            fout.write(json.dumps(log, ensure_ascii=False) + "\n")
            kept += 1

    print(f"all_start_ts: {all_start_ts}")
    print(f"wrote {kept} SAE entries to {out_path}")


if __name__ == "__main__":
    main()

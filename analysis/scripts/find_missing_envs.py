import argparse
import json
from pathlib import Path


def parse_all_env_ids(sh_path: Path) -> list[str]:
    """Extract ALL_ENV_IDS from run_sotopia_eval.sh."""
    env_ids: list[str] = []
    in_block = False

    with sh_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("ALL_ENV_IDS=("):
                in_block = True
                continue
            if in_block:
                if line.startswith(")"):
                    break
                if not line:
                    continue
                # lines like:   "01H7VFHP1JEP91TTK5PEK39D2S"
                if line.startswith('"') or line.startswith("'"):
                    env_ids.append(line.strip('",\' '))

    if not env_ids:
        raise ValueError("No ALL_ENV_IDS found in run_sotopia_eval.sh")
    return env_ids


def parse_finished_env_ids(results_path: Path) -> list[str]:
    """Read environment.id from a sotopia results jsonl file."""
    env_ids: list[str] = []
    with results_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            env = obj.get("environment", {})
            env_id = env.get("id")
            if env_id:
                env_ids.append(env_id)
    return env_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ALL_ENV_IDS from run_sotopia_eval.sh with a "
            "sotopia_all results jsonl and report missing environment ids."
        )
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to sotopia_all_gpt-4o_*.jsonl (partial results).",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parents[1]
    sh_path = project_root / "run_sotopia_eval.sh"

    if not sh_path.exists():
        raise SystemExit(f"run_sotopia_eval.sh not found at {sh_path}")
    if not args.results_path.exists():
        raise SystemExit(f"Results file not found: {args.results_path}")

    all_ids = parse_all_env_ids(sh_path)
    finished_ids = parse_finished_env_ids(args.results_path)

    all_set = set(all_ids)
    fin_set = set(finished_ids)
    missing = [env_id for env_id in all_ids if env_id not in fin_set]

    print(f"Total expected (ALL_ENV_IDS): {len(all_ids)}")
    print(f"Finished in {args.results_path.name}: {len(fin_set)}")
    print(f"Missing: {len(missing)}")
    if missing:
        print("Missing env_ids (in order):")
        for env_id in missing:
            print(env_id)


if __name__ == "__main__":
    main()


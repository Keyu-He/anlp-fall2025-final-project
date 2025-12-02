import argparse
import json
import re
from pathlib import Path


def sanitize_run_id(run_id: str) -> str:
    """Turn a run_id into a filesystem-safe suffix."""
    if run_id is None:
        return "no_run_id"
    # Replace non-alphanum with underscore, collapse repeats
    s = re.sub(r"[^0-9a-zA-Z]+", "_", run_id).strip("_")
    return s or "empty_run_id"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a SAE jsonl log into multiple files by run_id.",
    )
    parser.add_argument(
        "sae_path",
        type=Path,
        help="Path to sae_server_logs_*.jsonl",
    )
    args = parser.parse_args()

    sae_path: Path = args.sae_path
    if not sae_path.exists():
        raise SystemExit(f"SAE file not found: {sae_path}")

    base_dir = sae_path.parent
    stem = sae_path.stem
    suffix = sae_path.suffix

    writers: dict[str, tuple[Path, "io.TextIOWrapper"]] = {}
    counts: dict[str, int] = {}

    try:
        with sae_path.open("r") as fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                run_id = obj.get("run_id") or "no_run_id"
                counts[run_id] = counts.get(run_id, 0) + 1
                key = sanitize_run_id(run_id)
                if key not in writers:
                    out_path = base_dir / f"{stem}_{key}{suffix}"
                    fout = out_path.open("w")
                    writers[key] = (out_path, fout)
                _, fout = writers[key]
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        for _, fout in writers.values():
            fout.close()

    print("Found run_ids:")
    for rid, c in counts.items():
        print(f"  {rid!r}: {c} lines")
    print("Written files:")
    for key, (out_path, _) in writers.items():
        print(f"  {out_path}")


if __name__ == "__main__":
    main()


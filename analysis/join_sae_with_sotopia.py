import argparse
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join Sotopia result logs with SAE server logs.\n"
            "Matches each agent2 utterance with its SAE features using the "
            "utterance_key stored in the SAE server log."
        )
    )
    parser.add_argument(
        "--sotopia-log",
        required=True,
        help="Path to Sotopia result JSONL (e.g. results/sotopia_all_*.jsonl).",
    )
    parser.add_argument(
        "--sae-log",
        required=True,
        help="Path to SAE server JSONL (e.g. results/sae_server_logs_l15_k64.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output joined JSONL.",
    )
    return parser.parse_args()


def load_sae_log(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load SAE server log and index by utterance_key.

    We allow multiple records per key (rare but possible); consumers will pop
    from these lists as they match Sotopia utterances.
    """
    key_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            utterance_key = rec.get("utterance_key")
            if not isinstance(utterance_key, str):
                # Fallback: derive from utterance_id if needed
                utterance_key = ""
            key_to_records[utterance_key].append(rec)
    return key_to_records


def derive_utterance_key_from_history_line(history_line: str) -> str:
    """
    In the server, utterance_key is set to the last 'Turn #x: ...' line from
    the prompt. For joining, we can directly reuse the full observation line
    seen by agent2, which has the same format.
    """
    return history_line.rstrip("\n")


def main() -> None:
    args = parse_args()

    sae_index = load_sae_log(args.sae_log)

    joined_count = 0
    missing_sae = 0

    with open(args.sotopia_log, "r", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        for episode_index, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            env = rec.get("environment", {})
            env_id = env.get("id")
            agent_names = env.get("agent_names") or []
            agent2_name = agent_names[1] if len(agent_names) > 1 else "agent2"

            rewards = rec.get("rewards") or []
            agent2_reward = rewards[1] if len(rewards) > 1 else None

            meta = rec.get("meta", {})
            messages_all = meta.get("messages", [])
            if len(messages_all) < 2:
                continue
            # messages_all[1]: agent2's view of the conversation
            messages = messages_all[1]

            for turn_idx, msg in enumerate(messages):
                if not isinstance(msg, list) or len(msg) < 3:
                    continue
                speaker, _receiver, text = msg
                if speaker != agent2_name:
                    continue

                # Derive argument text from AgentAction.to_natural_language()
                argument_text = text
                if argument_text.startswith("said: "):
                    argument_text = argument_text[len("said: ") :].strip()
                    if (
                        argument_text.startswith('"')
                        and argument_text.endswith('"')
                        and len(argument_text) >= 2
                    ):
                        argument_text = argument_text[1:-1]

                # Find the most recent Environment -> agent2 observation line
                history_line: Optional[str] = None
                for back_idx in range(turn_idx - 1, -1, -1):
                    prev = messages[back_idx]
                    if not isinstance(prev, list) or len(prev) < 3:
                        continue
                    prev_speaker, prev_receiver, prev_text = prev
                    if prev_speaker == "Environment" and prev_receiver == agent2_name:
                        if prev_text.strip().startswith("Turn #"):
                            history_line = prev_text
                            break

                if history_line is None:
                    missing_sae += 1
                    continue

                utterance_key = derive_utterance_key_from_history_line(history_line)
                candidates = sae_index.get(utterance_key)
                if not candidates:
                    missing_sae += 1
                    continue

                sae_rec = candidates.pop(0)

                out_rec: Dict[str, Any] = {
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "turn_idx": turn_idx,
                    "speaker": speaker,
                    "text": text,
                    "argument": argument_text,
                    "sae_layer": sae_rec.get("sae_layer"),
                    "sae_trainer": sae_rec.get("sae_trainer"),
                    "sae_features": sae_rec.get("sae_features"),
                    "sae_recon_error": sae_rec.get("sae_recon_error"),
                    "sae_relative_error": sae_rec.get("sae_relative_error"),
                    "sae_request_index": sae_rec.get("request_index"),
                    "sae_utterance_id": sae_rec.get("utterance_id"),
                    "sae_utterance_key": sae_rec.get("utterance_key"),
                    "agent2_reward": agent2_reward,
                }
                f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                joined_count += 1

    print(f"Joined {joined_count} utterances with SAE features.")
    if missing_sae:
        print(f"Warning: {missing_sae} utterances had no matching SAE record.")


if __name__ == "__main__":
    main()


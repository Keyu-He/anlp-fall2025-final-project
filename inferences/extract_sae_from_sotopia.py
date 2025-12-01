import argparse
import json
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferences.qwen_sae_inference import (
    attach_residual_hook,
    build_sae_dir,
    load_sae_checkpoint,
    load_sae_config,
    prepare_sae,
    sae_encode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline SAE feature extraction for existing Sotopia result logs.\n"
            "For each agent2 utterance, run Qwen2.5-7B-Instruct with a forward hook "
            "and log top SAE features together with Sotopia rewards."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more Sotopia JSONL result files (e.g., results/sotopia_all_*.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL with SAE features per utterance.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct",
        help="Local path to Qwen2.5-7B-Instruct weights.",
    )
    parser.add_argument(
        "--sae-root",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct",
        help="Root directory of the SAE checkpoints.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Transformer block index for the residual stream SAE.",
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=1,
        help="Which SAE variant to use for the layer (0/1/2/3 ~ k=32/64/128/256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=64,
        help="How many top SAE features to log per utterance.",
    )
    return parser.parse_args()


def extract_agent2_reward(rec: Dict[str, Any]) -> Dict[str, Any]:
    rewards = rec.get("rewards") or []
    if len(rewards) < 2:
        return {}
    agent2_reward = rewards[1]
    if isinstance(agent2_reward, (list, tuple)) and len(agent2_reward) == 2:
        overall, dims = agent2_reward
        dims = dims or {}
        if isinstance(dims, dict):
            return {
                "overall": float(overall),
                "dims": {k: float(v) for k, v in dims.items()},
            }
    if isinstance(agent2_reward, dict):
        return {
            "overall": float(agent2_reward.get("overall", 0.0)),
            "dims": {
                k: float(v)
                for k, v in agent2_reward.items()
                if k != "overall"
            },
        }
    return {}


def iter_agent2_messages(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    env = rec.get("environment", {})
    agent_names = env.get("agent_names") or []
    agent2_name = agent_names[1] if len(agent_names) > 1 else "agent2"

    meta = rec.get("meta", {})
    messages_all = meta.get("messages", [])
    messages = messages_all[1] if len(messages_all) > 1 else []

    utterances: List[Dict[str, Any]] = []
    for turn_idx, msg in enumerate(messages):
        if not isinstance(msg, list) or len(msg) < 3:
            continue
        speaker = msg[0]
        content = msg[2]
        if speaker != agent2_name:
            continue
        utterances.append(
            {
                "turn_idx": turn_idx,
                "speaker": speaker,
                "text": content,
            }
        )
    return utterances


def main() -> None:
    args = parse_args()

    print(f"Loading base model from: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        device_map=args.device,
    )
    model.eval()

    sae_dir = build_sae_dir(args.sae_root, args.layer, args.trainer)
    print(f"Using SAE directory: {sae_dir}")

    sae_cfg = load_sae_config(sae_dir)
    if sae_cfg is not None:
        trainer_cfg = sae_cfg.get("trainer", {})
        print(
            "SAE config summary:",
            {
                "layer": trainer_cfg.get("layer"),
                "activation_dim": trainer_cfg.get("activation_dim"),
                "dict_size": trainer_cfg.get("dict_size"),
                "k": trainer_cfg.get("k"),
                "submodule_name": trainer_cfg.get("submodule_name"),
            },
        )

    ckpt = load_sae_checkpoint(sae_dir, args.device)
    if ckpt is None:
        raise RuntimeError("SAE checkpoint ae.pt not found; cannot proceed.")
    sae = prepare_sae(ckpt, args.device)

    device = args.device

    with open(args.output, "w", encoding="utf-8") as out_f:
        episode_index = 0
        for path in args.input:
            print(f"Processing log file: {path}")
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    env = rec.get("environment", {})
                    env_id = env.get("id")

                    reward_info = extract_agent2_reward(rec)
                    utterances = iter_agent2_messages(rec)

                    for utter in utterances:
                        text = utter["text"]

                        activations: Dict[str, torch.Tensor] = {}
                        handle = attach_residual_hook(model, args.layer, activations)

                        inputs = tokenizer(
                            text,
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        with torch.no_grad():
                            _ = model(**inputs)

                        handle.remove()

                        resid = activations.get("resid_post")
                        if resid is None:
                            continue

                        feats_mean, top_indices, top_values = sae_encode(
                            resid,
                            sae,
                            topn=args.topn,
                        )

                        out_rec: Dict[str, Any] = {
                            "episode_index": episode_index,
                            "env_id": env_id,
                            "turn_idx": utter["turn_idx"],
                            "speaker": utter["speaker"],
                            "text": text,
                            "sae_layer": args.layer,
                            "sae_trainer": args.trainer,
                            "sae_top_feature_indices": top_indices.cpu().tolist(),
                            "sae_top_feature_values": [float(v) for v in top_values.cpu().tolist()],
                            "agent2_reward": reward_info,
                        }
                        out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

                    episode_index += 1


if __name__ == "__main__":
    main()


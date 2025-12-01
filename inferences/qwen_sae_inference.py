import argparse
import json
import os
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen2.5-7B-Instruct and capture activations from a given layer, "
            "optionally loading the matching SAE checkpoint."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct",
        help="Path to Qwen2.5-7B-Instruct weights (local directory).",
    )
    parser.add_argument(
        "--sae-root",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct",
        help="Root directory of the SAE checkpoints (resid_post_layer_*/trainer_*).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Transformer block index for the residual stream SAE (e.g., 3, 7, 11, 15, 19, 23, 27).",
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=1,
        help="Which SAE variant to use for the layer (0/1/2/3 ~ k=32/64/128/256).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好，简单介绍一下你自己。",
        help="Input text to run through the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    return parser.parse_args()


def build_sae_dir(root: str, layer: int, trainer: int) -> str:
    layer_dir = f"resid_post_layer_{layer}"
    trainer_dir = f"trainer_{trainer}"
    return os.path.join(root, layer_dir, trainer_dir)


def load_sae_config(sae_dir: str) -> Optional[Dict]:
    cfg_path = os.path.join(sae_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sae_checkpoint(sae_dir: str, device: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load the SAE checkpoint object as a dict of tensors.

    The expected keys for the Qwen2.5 SAEs are:
      - 'encoder.weight'  [dict_size, d]
      - 'encoder.bias'    [dict_size]
      - 'decoder.weight'  [dict_size, d]
      - 'b_dec'           [d]
      - 'threshold'       [dict_size] or scalar
      - 'k'               (int)
    """
    ckpt_path = os.path.join(sae_dir, "ae.pt")
    if not os.path.exists(ckpt_path):
        return None
    raw = torch.load(ckpt_path, map_location=device)
    if not isinstance(raw, dict):
        raise ValueError("Unexpected SAE checkpoint type, expected dict.")
    return raw


def prepare_sae(ckpt: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Prepare SAE weights for encoding/residual reconstruction.
    """
    sae_device = torch.device(device)
    encoder_weight = ckpt["encoder.weight"].to(sae_device)
    encoder_bias = ckpt["encoder.bias"].to(sae_device)
    decoder_weight = ckpt["decoder.weight"].to(sae_device)
    decoder_bias = ckpt["b_dec"].to(sae_device)

    threshold = ckpt["threshold"]
    threshold_tensor = torch.as_tensor(threshold, device=sae_device, dtype=encoder_weight.dtype)
    k = int(ckpt["k"])

    return {
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "threshold": threshold_tensor,
        "k": k,
    }


def attach_residual_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
    """
    Register a forward hook on model.model.layers[layer_index] to capture
    the post-residual activations.
    """
    try:
        layer_module = model.model.layers[layer_index]
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Cannot access model.model.layers[{layer_index}]") from exc

    def hook(_module, _inputs, output):
        storage["resid_post"] = output.detach().to("cpu")

    handle = layer_module.register_forward_hook(hook)
    return handle


def sae_encode(
    resid: torch.Tensor,
    sae: Dict[str, torch.Tensor],
    topn: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode residual activations using the SAE and return:
      - feats_mean: mean activation over batch and sequence [dict_size]
      - top_indices: indices of topn features [topn]
      - top_values: values of topn features [topn]
    """
    encoder_weight = sae["encoder_weight"]
    encoder_bias = sae["encoder_bias"]
    decoder_weight = sae["decoder_weight"]
    decoder_bias = sae["decoder_bias"]
    threshold = sae["threshold"]
    k = int(sae["k"])

    device = encoder_weight.device
    x = resid.to(device)

    pre = torch.nn.functional.linear(x, encoder_weight, encoder_bias)

    if threshold.dim() == 0:
        thr = threshold.view(1, 1, 1)
    else:
        thr = threshold.view(1, 1, -1)

    acts = torch.clamp(pre - thr, min=0.0)

    if k > 0 and k < acts.shape[-1]:
        values, indices = acts.topk(k, dim=-1)
        mask = torch.zeros_like(acts, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        acts = acts * mask

    _ = torch.nn.functional.linear(acts, decoder_weight, decoder_bias)

    feats_mean = acts.mean(dim=(0, 1))

    topn_eff = min(topn, feats_mean.shape[0])
    top_values, top_indices = torch.topk(feats_mean, topn_eff)

    return feats_mean, top_indices, top_values


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
    else:
        print("Warning: SAE config.json not found; continuing without config metadata.")

    ckpt = load_sae_checkpoint(sae_dir, args.device)
    if ckpt is None:
        print("Warning: SAE checkpoint ae.pt not found; will only capture activations.")
        sae = None
    else:
        top_keys = list(ckpt.keys())
        print("Loaded SAE checkpoint (dict), top-level keys:", top_keys[:10])
        sae = prepare_sae(ckpt, args.device)

    activations: Dict[str, torch.Tensor] = {}
    handle = attach_residual_hook(model, args.layer, activations)

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    if "resid_post" not in activations:
        raise RuntimeError("Hook did not capture any residual activations.")

    resid = activations["resid_post"]
    print(f"Captured residual activations at layer {args.layer}: shape={tuple(resid.shape)}")

    if sae is not None:
        feats_mean, top_indices, top_values = sae_encode(resid, sae)
        print(f"Mean SAE feature vector shape: {tuple(feats_mean.shape)}")
        print("Top SAE feature indices:", top_indices.tolist())
        print("Top SAE feature values:", top_values.tolist())


if __name__ == "__main__":
    main()


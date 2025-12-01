import argparse
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dictionary_learning import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen2.5-7B-Instruct, capture residual-stream activations at a given layer, "
            "mean-pool over tokens, and encode with the matching SAE."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct",
        help="Path to Qwen2.5-7B-Instruct weights (local directory).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Transformer block index for the residual stream SAE (e.g., 3, 7, 11, 15, 19, 23, 27).",
    )
    parser.add_argument(
        "--sae-dir",
        type=str,
        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1",
        help="Path to the trained SAE/dictionary directory (the folder containing ae.pt and config.json).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, briefly introduce yourself.",
        help="Input text to run through the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device for inference, e.g. 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=64,
        help="How many top SAE features to report.",
    )
    return parser.parse_args()


def attach_resid_post_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
    """
    Register a forward hook that reconstructs resid_post at model.model.layers[layer_index].

    对齐 'resid_post_layer_{layer}'：
    resid_post = pre_mlp_hidden + mlp(pre_mlp_hidden)
    """
    try:
        layer_module = model.model.layers[layer_index]
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Cannot access model.model.layers[{layer_index}]") from exc

    if not hasattr(layer_module, "mlp"):
        raise ValueError(f"Layer {layer_index} has no 'mlp' submodule, cannot build resid_post hook.")

    def mlp_hook(_module, inputs, output):
        # inputs[0]: hidden states BEFORE MLP
        pre_mlp = inputs[0]          # [batch, seq_len, d_model]
        mlp_out = output             # [batch, seq_len, d_model]
        resid_post = pre_mlp + mlp_out
        storage["resid_post"] = resid_post.detach().cpu()

    handle = layer_module.mlp.register_forward_hook(mlp_hook)
    return handle


def load_sae_dictionary(sae_dir: str, device: str):
    """
    Load a trained dictionary/SAE using dictionary_learning.utils.load_dictionary.
    使用官方 README 推荐的通用加载方式。
    """
    print(f"Loading SAE dictionary from: {sae_dir}")
    ae, config = utils.load_dictionary(sae_dir, device=device)
    trainer_cfg = config.get("trainer", {})
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
    return ae, config


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

    # 先加载 SAE，看一下 config 是否和 layer 对齐
    ae, sae_config = load_sae_dictionary(args.sae_dir, device=args.device)
    trainer_cfg = sae_config.get("trainer", {})
    sae_layer = trainer_cfg.get("layer", None)
    submodule_name = trainer_cfg.get("submodule_name", None)
    print(f"SAE trainer layer={sae_layer}, submodule_name={submodule_name}")
    if sae_layer is not None and sae_layer != args.layer:
        print(f"[WARN] SAE trained on layer={sae_layer}, but --layer={args.layer}. "
              f"Consider setting --layer={sae_layer} for best alignment.")

    activations: Dict[str, torch.Tensor] = {}
    handle = attach_resid_post_hook(model, args.layer, activations)

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
    print(f"Captured resid_post at layer {args.layer}: shape={tuple(resid.shape)}")

    # resid: [batch=1, seq_len, d_model]
    resid_tokens = resid[0].to(args.device, dtype=torch.float32)  # [seq_len, d_model]
    print(f"Token-level residual shape: {tuple(resid_tokens.shape)}")
    rms = resid_tokens.pow(2).mean().sqrt().item()
    print(f"resid_post RMS: {rms:.6f}")

    # Encode and decode using the AutoEncoder, exactly as in the demo.
    # 也可以 ae(resid_tokens, output_features=True) 一步拿到。
    reconstructed, features = ae(resid_tokens, output_features=True)  # [seq_len, d_model], [seq_len, dict_size]

    # Mean-pool SAE features over tokens
    feats_mean = features.mean(dim=0).detach().cpu()
    dict_size = feats_mean.shape[0]
    print(f"Mean SAE feature vector shape: ({dict_size},)")

    topn_eff = min(args.topn, dict_size)
    top_values, top_indices = torch.topk(feats_mean, topn_eff)
    print("Top SAE feature indices:", top_indices.tolist())
    print("Top SAE feature values:", top_values.tolist())

    # Reconstruction error metrics over all tokens
    diff = resid_tokens - reconstructed
    recon_error = float(torch.linalg.norm(diff).item())
    x_norm = float(torch.linalg.norm(resid_tokens).item())
    relative_error = recon_error / x_norm if x_norm > 0 else 0.0
    print(f"Reconstruction error: {recon_error:.6f} (relative {relative_error:.6f})")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors


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
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Device for inference, e.g. 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--sae-repo",
        type=str,
        default=None,
        help="Hugging Face repo ID for the SAE (overrides --sae-root).",
    )
    parser.add_argument(
        "--steer-feature",
        type=int,
        default=None,
        help="SAE feature index to steer with (requires --sae-root or --sae-repo).",
    )
    parser.add_argument(
        "--steer-strength",
        type=float,
        default=50.0,
        help="Strength of the steering vector injection.",
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


def load_sae_from_hub(repo_id: str, layer: int, trainer: int, device: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Attempt to load SAE checkpoint from Hugging Face Hub.
    Tries 'ae.pt' first, then 'sae_weights.safetensors'.
    """
    # Try 1: Exact structure as local
    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="ae.pt", subfolder=subfolder)
        print(f"Downloaded SAE from {repo_id}/{subfolder}/ae.pt")
        return torch.load(ckpt_path, map_location=device)
    except Exception:
        pass

    # Try 2: Root ae.pt
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="ae.pt")
        print(f"Downloaded SAE from {repo_id}/ae.pt")
        return torch.load(ckpt_path, map_location=device)
    except Exception:
        pass
        
    # Try 3: sae.safetensors (EleutherAI structure)
    subfolder = f"layers.{layer}.mlp"
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="sae.safetensors", subfolder=subfolder)
        print(f"Downloaded SAE from {repo_id}/{subfolder}/sae.safetensors")
        state_dict = load_safetensors(ckpt_path, device=device)
        
        # Map keys if needed
        if "W_dec" in state_dict and "decoder.weight" not in state_dict:
            # W_dec is [dict_size, d_model], but F.linear expects [out_features, in_features]
            # i.e. [d_model, dict_size]. So we transpose.
            state_dict["decoder.weight"] = state_dict.pop("W_dec").t()
            
        # Ensure all keys are present
        required = ["encoder.weight", "encoder.bias", "decoder.weight", "b_dec"]
        for r in required:
            if r not in state_dict:
                print(f"Warning: Missing key {r} in safetensors.")
        
        # Add dummy threshold/k if missing
        if "threshold" not in state_dict:
             state_dict["threshold"] = torch.tensor(0.0, device=device)
        if "k" not in state_dict:
             state_dict["k"] = 64 
             
        return state_dict
    except Exception as e:
        print(f"Failed to load safetensors from {subfolder}: {e}")

    print(f"Could not find valid checkpoint in {repo_id}.")
    return None


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
    k = int(ckpt.get("k", 32))

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
        # SAE is likely trained on MLP output
        layer_module = model.model.layers[layer_index].mlp
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Cannot access model.model.layers[{layer_index}].mlp") from exc

    def hook(_module, _inputs, output):
        storage["resid_post"] = output.detach().to("cpu")

    handle = layer_module.register_forward_hook(hook)
    return handle


def attach_steering_hook(
    model,
    layer_index: int,
    sae: Dict[str, torch.Tensor],
    feature_index: int,
    strength: float,
):
    """
    Register a forward hook that adds the decoder vector of the specified
    SAE feature to the residual stream.
    """
    try:
        # SAE is likely trained on MLP output
        layer_module = model.model.layers[layer_index].mlp
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Cannot access model.model.layers[{layer_index}].mlp") from exc

    # Extract the steering vector: [d_model]
    # decoder_weight shape is [dict_size, d_model]
    # Wait, if we loaded from safetensors and transposed, it is [d_model, dict_size].
    # But if we loaded from pt, it is [dict_size, d_model].
    # We need to be careful.
    # Let's check shape.
    decoder_weight = sae["decoder_weight"]
    
    # Standard SAE shape usually [dict_size, d_model].
    # If we transposed it for F.linear, it is [d_model, dict_size].
    # F.linear(x, weight) -> x @ weight.T
    # If weight is [d_model, dict_size], weight.T is [dict_size, d_model].
    # x is [..., dict_size].
    # x @ [dict_size, d_model] -> [..., d_model]. Correct.
    
    # So if we transposed, decoder_weight is [d_model, dict_size].
    # feature_index selects a column if it is [d_model, dict_size].
    # If it is [dict_size, d_model], it selects a row.
    
    # Heuristic: dict_size is usually >> d_model (e.g. 65k vs 1.5k).
    if decoder_weight.shape[0] > decoder_weight.shape[1]:
        # Shape is [dict_size, d_model]
        steering_vector = decoder_weight[feature_index].clone().detach()
    else:
        # Shape is [d_model, dict_size]
        steering_vector = decoder_weight[:, feature_index].clone().detach()
    
    def hook(_module, _inputs, output):
        vec = steering_vector.to(output.device)
        return output + (vec * strength)

    handle = layer_module.register_forward_hook(hook)
    return handle


def sae_encode(
    resid: torch.Tensor,
    sae: Dict[str, torch.Tensor],
    topn: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """
    Encode residual activations using the SAE and return:
      - feats_mean: mean activation over batch and sequence [dict_size]
      - top_indices: indices of topn features [topn]
      - top_values: values of topn features [topn]
      - recon_error: L2 reconstruction error ||x - x_hat||_2
      - relative_error: recon_error / ||x||_2 (or 0.0 if ||x|| is 0)
    """
    encoder_weight = sae["encoder_weight"]
    encoder_bias = sae["encoder_bias"]
    decoder_weight = sae["decoder_weight"]
    decoder_bias = sae["decoder_bias"]
    threshold = sae["threshold"]
    k = int(sae["k"])

    device = encoder_weight.device
    x = resid.to(device=device, dtype=encoder_weight.dtype)

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

    recon = torch.nn.functional.linear(acts, decoder_weight, decoder_bias)

    # Reconstruction error statistics
    # x and recon are [batch, seq, d]; flatten over batch+seq
    x_flat = x.reshape(-1, x.shape[-1])
    recon_flat = recon.reshape_as(x_flat)
    diff = x_flat - recon_flat
    recon_error = float(torch.linalg.norm(diff).item())
    x_norm = float(torch.linalg.norm(x_flat).item())
    relative_error = recon_error / x_norm if x_norm > 0 else 0.0

    feats_mean = acts.mean(dim=(0, 1))

    topn_eff = min(topn, feats_mean.shape[0])
    top_values, top_indices = torch.topk(feats_mean, topn_eff)

    return feats_mean, top_indices, top_values, recon_error, relative_error


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

    if args.sae_repo:
        print(f"Loading SAE from Hugging Face Hub: {args.sae_repo}")
        ckpt = load_sae_from_hub(args.sae_repo, args.layer, args.trainer, args.device)
        if ckpt is None:
             print("Failed to load SAE from Hub.")
             sae = None
        else:
             sae = prepare_sae(ckpt, args.device)
    else:
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

    # If steering is requested, run generation with the hook
    if args.steer_feature is not None:
        if sae is None:
            raise ValueError("Steering requires a loaded SAE. Check --sae-root or --sae-repo.")
        
        print(f"--- Enabling Steering: Feature {args.steer_feature}, Strength {args.steer_strength} ---")
        handle = attach_steering_hook(
            model, 
            args.layer, 
            sae, 
            args.steer_feature, 
            args.steer_strength
        )
        
        # Prepare inputs for generation
        inputs = tokenizer(args.text, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        print(f"Prompt: {args.text}")
        print("Generating...")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=True, 
                temperature=0.7
            )
        
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("--- Generated Output ---")
        print(output_text)
        print("------------------------")
        
        handle.remove()
        return

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
        feats_mean, top_indices, top_values, recon_error, relative_error = sae_encode(
            resid, sae
        )
        print(f"Mean SAE feature vector shape: {tuple(feats_mean.shape)}")
        print("Top SAE feature indices:", top_indices.tolist())
        print("Top SAE feature values:", top_values.tolist())
        print(f"Reconstruction error: {recon_error:.6f} (relative {relative_error:.6f})")


if __name__ == "__main__":
    main()

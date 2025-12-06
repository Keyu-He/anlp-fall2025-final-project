###############################################
# Qwen2.5 SAE Server (Correct AutoEncoder Version)
###############################################

import argparse
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# NEW: dictionary_learning AutoEncoder
from dictionary_learning import utils as dl_utils


###############################################################
# Residual Hook (Correct resid_post)
###############################################################


def parse_feature_from_model_name(model_name: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Parse steering config from the model name string.

    Expected pattern (anywhere in the string):
        __feat{idx}_str{strength}

    Example:
        custom/Qwen/Qwen2.5-7B-Instruct__feat123_str5.0
    """
    m = re.search(r"__feat(-?\\d+)_str(-?\\d+(?:\\.\\d+)?)", model_name)
    if not m:
        return None, None
    feat_idx = int(m.group(1))
    strength = float(m.group(2))
    return feat_idx, strength

def attach_resid_post_hook(model, layer_index: int, storage: Dict[str, torch.Tensor]):
    """Correct hook for Qwen2.5 resid_post_layer_{L}:
       resid_post = pre_mlp + mlp(pre_mlp)
    """
    layer_module = model.model.layers[layer_index]
    if not hasattr(layer_module, "mlp"):
        raise ValueError(f"Layer {layer_index} has no MLP, cannot hook resid_post")

    def hook(_module, inputs, output):
        pre_mlp = inputs[0]     # before MLP
        mlp_out = output        # MLP(pre_mlp)
        resid_post = pre_mlp + mlp_out
        storage["resid_post"] = resid_post.detach().cpu()

    return layer_module.mlp.register_forward_hook(hook)


def attach_steering_hook(model, layer_index: int, ae, feature_idx: int, steering_strength: float):
    """Hook that steers a specific SAE feature during generation."""
    layer_module = model.model.layers[layer_index]
    if not hasattr(layer_module, "mlp"):
        raise ValueError(f"Layer {layer_index} has no MLP, cannot hook resid_post")

    def intervention_hook(_module, inputs, output):
        pre_mlp = inputs[0]
        mlp_out = output
        resid_post = pre_mlp + mlp_out

        # Save original dtype
        original_dtype = resid_post.dtype
        batch_size, seq_len, d_model = resid_post.shape

        # Encode with SAE (need float32)
        resid_flat = resid_post.view(-1, d_model).to(dtype=torch.float32)
        reconstructed, features = ae(resid_flat, output_features=True)

        # Steer the target feature
        features[:, feature_idx] += steering_strength

        # Decode back
        steered_resid = ae.decode(features)
        steered_resid = steered_resid.view(batch_size, seq_len, d_model)

        # Convert back to original dtype (bfloat16)
        steered_resid = steered_resid.to(dtype=original_dtype)

        # Return modified MLP output
        return steered_resid - pre_mlp

    return layer_module.mlp.register_forward_hook(intervention_hook)


###############################################################
# Request/Response Models
###############################################################

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


###############################################################
# Build Server App
###############################################################

def build_app(
    base_model_path: str,
    sae_root: str,
    layer: int,
    trainer: int,
    device: str,
    topn: int,
    log_path: str,
    run_id: str,
    steer_feature_idx: Optional[int] = None,
    steer_strength: float = 0.0,
) -> FastAPI:

    print(f"Loading tokenizer and model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        device_map=device,
    )
    model.eval()

    # --------------------------
    # Load dictionary-learning SAE
    # --------------------------
    sae_dir = f"{sae_root}/resid_post_layer_{layer}/trainer_{trainer}"
    print(f"Using SAE directory: {sae_dir}")

    ae, sae_cfg = dl_utils.load_dictionary(sae_dir, device=device)
    trainer_cfg = sae_cfg.get("trainer", {})
    print("SAE config:", trainer_cfg)

    # --------------------------
    # FastAPI setup
    # --------------------------
    app = FastAPI()
    gen_lock = threading.Lock()
    counter = {"value": 0}

    def log_record(rec: Dict[str, Any]):
        rec = dict(rec)
        rec.setdefault("timestamp", time.time())
        if run_id:
            rec.setdefault("run_id", run_id)
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    ###########################################################
    # Chat Endpoint
    ###########################################################
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest):

        if not req.messages:
            raise ValueError("Empty chat messages.")

        prompt = req.messages[-1].content

        # Extract a key for logging (Sotopia style)
        turn_matches = re.findall(r"Turn #\d+:[^\n]*", prompt)
        if turn_matches:
            utterance_key = turn_matches[-1]
        else:
            utterance_key = prompt[-200:]
        utterance_id = hashlib.sha1(utterance_key.encode()).hexdigest()[:16]

        with gen_lock:
            counter["value"] += 1
            request_idx = counter["value"]

            #################################################
            # Step 1 — Forward pass to capture resid_post / Apply Steering
            #################################################
            activations: Dict[str, torch.Tensor] = {}
            hook = None

            # Allow per-request overrides encoded in `req.model`, so that
            # a remote client (e.g., local Sotopia eval) can select different
            # SAE features/strengths without restarting the server.
            feature_idx = steer_feature_idx
            strength = steer_strength
            if req.model:
                parsed_idx, parsed_strength = parse_feature_from_model_name(req.model)
                if parsed_idx is not None:
                    feature_idx = parsed_idx
                if parsed_strength is not None:
                    strength = parsed_strength

            if feature_idx is not None and abs(strength) > 1e-6:
                # Steering Mode
                hook = attach_steering_hook(model, layer, ae, feature_idx, strength)
            else:
                # Logging Mode (capture residuals)
                hook = attach_resid_post_hook(model, layer, activations)

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                _ = model(**encoded)

            if hook:
                hook.remove()

            sae_top_idx = []
            sae_top_values = []
            sae_relative_error = None
            sae_recon_error = None
            sae_features = []

            if "resid_post" in activations:
                resid = activations["resid_post"][0].to(device, dtype=torch.float32)

                # RMS check
                rms = resid.pow(2).mean().sqrt().item()
                # print("resid RMS:", rms)

                # SAE encode+decode
                reconstructed, features = ae(resid, output_features=True)

                # mean pooled features
                feat_mean = features.mean(dim=0)
                sae_features = feat_mean.detach().cpu().tolist()

                # top-k
                topv, topi = torch.topk(feat_mean, k=topn)
                sae_top_idx = topi.cpu().tolist()
                sae_top_values = topv.cpu().tolist()

                # reconstruction error
                diff = resid - reconstructed
                sae_recon_error = float(diff.norm().item())
                sae_relative_error = float(sae_recon_error / (resid.norm().item() + 1e-8))

            #################################################
            # Step 2 — Generate completion
            #################################################
            max_new = req.max_tokens or 512
            temp = req.temperature if req.temperature is not None else 0.7
            do_sample = temp > 0.0

            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temp if do_sample else 1.0,
                )

            inp_ids = encoded["input_ids"]
            gen_ids = output_ids[0, inp_ids.shape[1]:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

            prompt_tokens = int(inp_ids.numel())
            completion_tokens = int(gen_ids.numel())

            #################################################
            # Step 3 — Log
            #################################################
            log_record(
                {
                    "request_index": request_idx,
                    "utterance_id": utterance_id,
                    "utterance_key": utterance_key,
                    "model": req.model,
                    "steer_feature_idx": feature_idx,
                    "steer_strength": strength,
                    "prompt": prompt,
                    "completion": completion,
                    "sae_layer": layer,
                    "sae_trainer": trainer,
                    "temperature": temp,
                    "max_new_tokens": max_new,
                    "sae_features": sae_features,
                    "sae_top_indices": sae_top_idx,
                    "sae_top_values": sae_top_values,
                    "sae_recon_error": sae_recon_error,
                    "sae_relative_error": sae_relative_error,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

        #################################################
        # OpenAI-style response
        #################################################
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=completion),
        )
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        resp = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            choices=[choice],
            usage=usage,
        )
        return resp

    return app


###############################################################
# Main
###############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str,
                        default="/data/user_data/demiw/qwen2.5-7b-sotopia/Qwen2.5-7B-Instruct")
    parser.add_argument("--sae-root", type=str,
                        default="/data/user_data/demiw/qwen2.5-7b-sotopia/saes-qwen2.5-7b-instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--trainer", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--topn", type=int, default=64)
    parser.add_argument("--log-path", type=str, default="results/sae_server_logs.jsonl")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steer-feature-idx", type=int, default=None, help="SAE feature index to steer")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength")
    args = parser.parse_args()

    # Automatically append a timestamp to the log path so that
    # multiple runs do not overwrite the same log file.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path
    # If the user provides a {timestamp} placeholder, respect it.
    if "{timestamp}" in log_path:
        log_path = log_path.format(timestamp=ts)
    else:
        base, ext = os.path.splitext(log_path)
        log_path = f"{base}_{ts}{ext}"

    app = build_app(
        base_model_path=args.base_model_path,
        sae_root=args.sae_root,
        layer=args.layer,
        trainer=args.trainer,
        device=args.device,
        topn=args.topn,
        log_path=log_path,
        run_id=args.run_id,
        steer_feature_idx=args.steer_feature_idx,
        steer_strength=args.steer_strength,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

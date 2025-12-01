###############################################
# Qwen2.5 SAE Server (Correct AutoEncoder Version)
###############################################

import argparse
import hashlib
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# NEW: dictionary_learning AutoEncoder
from dictionary_learning import utils as dl_utils


###############################################################
# Residual Hook (Correct resid_post)
###############################################################

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
            # Step 1 — Forward pass to capture resid_post
            #################################################
            activations: Dict[str, torch.Tensor] = {}
            hook = attach_resid_post_hook(model, layer, activations)

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                _ = model(**encoded)

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
    args = parser.parse_args()

    app = build_app(
        base_model_path=args.base_model_path,
        sae_root=args.sae_root,
        layer=args.layer,
        trainer=args.trainer,
        device=args.device,
        topn=args.topn,
        log_path=args.log_path,
        run_id=args.run_id,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

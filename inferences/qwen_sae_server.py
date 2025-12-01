import argparse
import json
import threading
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferences.qwen_sae_inference import (
    attach_residual_hook,
    build_sae_dir,
    load_sae_checkpoint,
    load_sae_config,
    prepare_sae,
    sae_encode,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible chat completion server for Qwen2.5-7B-Instruct "
            "with SAE feature logging."
        )
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
        help="Device for inference, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=64,
        help="How many top SAE features to log per request.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="results/sae_server_logs.jsonl",
        help="Path to JSONL file for logging SAE features per request.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional identifier for this evaluation run (to match Sotopia logs).",
    )
    return parser.parse_args()


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

    sae_dir = build_sae_dir(sae_root, layer, trainer)
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
        assert trainer_cfg.get("layer") == layer
    ckpt = load_sae_checkpoint(sae_dir, device)
    if ckpt is None:
        raise RuntimeError("SAE checkpoint ae.pt not found; cannot start server.")
    sae = prepare_sae(ckpt, device)

    app = FastAPI()

    generation_lock = threading.Lock()
    request_counter = {"value": 0}

    def log_record(rec: Dict[str, Any]) -> None:
        rec = dict(rec)
        rec.setdefault("timestamp", time.time())
        if run_id:
            rec.setdefault("run_id", run_id)
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:  # type: ignore[valid-type]
        # Sotopia always sends a single user message containing the full template+history.
        if not req.messages:
            raise ValueError("Empty messages in chat completion request.")
        prompt = req.messages[-1].content

        with generation_lock:
            request_counter["value"] += 1
            request_index = request_counter["value"]

            # 1) Run a forward pass on the prompt to capture residual activations
            activations: Dict[str, torch.Tensor] = {}
            handle = attach_residual_hook(model, layer, activations)

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                _ = model(**encoded)

            handle.remove()

            resid = activations.get("resid_post")
            sae_top_indices: List[int] = []
            sae_top_values: List[float] = []
            if resid is not None:
                feats_mean, top_indices, top_values = sae_encode(
                    resid,
                    sae,
                    topn=topn,
                )
                sae_top_indices = top_indices.cpu().tolist()
                sae_top_values = [float(v) for v in top_values.cpu().tolist()]

            # 2) Generate a completion with the same prompt
            max_new_tokens = req.max_tokens or 512
            temperature = req.temperature if req.temperature is not None else 0.7
            do_sample = temperature > 0.0

            with torch.no_grad():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                )

            input_ids = encoded["input_ids"]
            generated_ids = out[0, input_ids.shape[1] :]
            completion_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            # Try to parse JSON action for easier alignment with Sotopia logs
            action_type: Optional[str] = None
            argument: Optional[str] = None
            try:
                obj = json.loads(completion_text)
                if isinstance(obj, dict):
                    at = obj.get("action_type")
                    arg = obj.get("argument")
                    if isinstance(at, str):
                        action_type = at
                    if isinstance(arg, str):
                        argument = arg
            except Exception:
                pass

            # Approximate token counts
            prompt_tokens = int(input_ids.numel())
            completion_tokens = int(generated_ids.numel())

            # 3) Log SAE features and basic request info
            log_record(
                {
                    "request_index": request_index,
                    "model": req.model,
                    "sae_layer": layer,
                    "sae_trainer": trainer,
                    "prompt": prompt,
                    "completion": completion_text,
                    "action_type": action_type,
                    "argument": argument,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "sae_top_feature_indices": sae_top_indices,
                    "sae_top_feature_values": sae_top_values,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=completion_text),
        )
        usage = ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        resp = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            choices=[choice],
            usage=usage,
        )
        return resp

    return app


def main() -> None:
    args = parse_args()
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

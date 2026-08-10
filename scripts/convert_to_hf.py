"""
Convert an es-at-scale checkpoint (a raw state_dict saved from vLLM's internal
Qwen2 model, via worker_extension.save_self_weights_to_disk) into a proper
HuggingFace model directory that transformers/vLLM can load with
AutoModelForCausalLM.from_pretrained(...).

Why this is needed: vLLM's Qwen2 implementation fuses attention QKV into one
`qkv_proj` tensor and MLP gate+up into one `gate_up_proj` tensor (see
vllm.model_executor.models.qwen2.Qwen2Model.load_weights' stacked_params_mapping).
HF keeps these separate (q_proj/k_proj/v_proj, gate_proj/up_proj). Every other
parameter name (embed_tokens, o_proj, down_proj, layernorms, norm, lm_head) is
identical between the two, since only attention QKV and MLP gate/up are fused.

Usage:
    python convert_to_hf.py \
        --checkpoint /workspace/es-at-scale/experiments/math-run/checkpoint-es_fine_tuned_iteration_50/pytorch_model.pth \
        --base-model-dir /workspace/.hf_home/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323 \
        --output-dir /workspace/es-at-scale/experiments/math-run/hf-checkpoint-iter50 \
        --verify
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def split_qkv(weight: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int):
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = torch.split(weight, [q_size, kv_size, kv_size], dim=0)
    return q.contiguous(), k.contiguous(), v.contiguous()


def split_gate_up(weight: torch.Tensor, intermediate_size: int):
    gate, up = torch.split(weight, [intermediate_size, intermediate_size], dim=0)
    return gate.contiguous(), up.contiguous()


def convert(state_dict: dict, config: dict) -> dict:
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    intermediate_size = config["intermediate_size"]
    head_dim = hidden_size // num_heads

    new_state = {}
    consumed = set()

    for name, tensor in state_dict.items():
        if name in consumed:
            continue

        if "qkv_proj" in name:
            suffix = name.split("qkv_proj")[-1]  # ".weight" or ".bias"
            prefix = name.split("qkv_proj")[0]
            if suffix == ".weight":
                q, k, v = split_qkv(tensor, num_heads, num_kv_heads, head_dim)
            elif suffix == ".bias":
                q, k, v = split_qkv(tensor, num_heads, num_kv_heads, head_dim)
            else:
                raise ValueError(f"Unexpected qkv_proj key: {name}")
            new_state[f"{prefix}q_proj{suffix}"] = q
            new_state[f"{prefix}k_proj{suffix}"] = k
            new_state[f"{prefix}v_proj{suffix}"] = v
            consumed.add(name)
            continue

        if "gate_up_proj" in name:
            suffix = name.split("gate_up_proj")[-1]
            prefix = name.split("gate_up_proj")[0]
            gate, up = split_gate_up(tensor, intermediate_size)
            new_state[f"{prefix}gate_proj{suffix}"] = gate
            new_state[f"{prefix}up_proj{suffix}"] = up
            consumed.add(name)
            continue

        new_state[name] = tensor.contiguous()
        consumed.add(name)

    # Drop a separately-stored lm_head.weight when embeddings are tied: HF
    # re-ties it from embed_tokens.weight automatically based on
    # config.tie_word_embeddings, and safetensors refuses to save two keys
    # that alias the same underlying storage.
    if config.get("tie_word_embeddings", False) and "lm_head.weight" in new_state:
        embed_key = "model.embed_tokens.weight"
        if embed_key in new_state and torch.equal(new_state["lm_head.weight"], new_state[embed_key]):
            print(f"Dropping lm_head.weight (tied to {embed_key})")
            del new_state["lm_head.weight"]

    return new_state


def verify(new_state: dict, base_model_dir: str, config: dict):
    """Build a fresh HF model from config (no weights loaded) and check that
    our converted key set / shapes line up exactly, so failures surface here
    instead of as a cryptic error inside vLLM/transformers later."""
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(base_model_dir)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(hf_config)
    ref_state = ref_model.state_dict()

    ref_keys = set(ref_state.keys())
    new_keys = set(new_state.keys())

    # tied lm_head.weight may legitimately be absent from new_state
    if config.get("tie_word_embeddings", False):
        ref_keys.discard("lm_head.weight")

    missing = ref_keys - new_keys
    extra = new_keys - ref_keys
    if missing:
        raise ValueError(f"Converted checkpoint is missing {len(missing)} keys, e.g. {list(missing)[:10]}")
    if extra:
        raise ValueError(f"Converted checkpoint has {len(extra)} unexpected keys, e.g. {list(extra)[:10]}")

    for k in new_keys & ref_keys:
        if tuple(new_state[k].shape) != tuple(ref_state[k].shape):
            raise ValueError(
                f"Shape mismatch for {k}: converted {tuple(new_state[k].shape)} "
                f"vs expected {tuple(ref_state[k].shape)}"
            )
    print(f"Verify OK: {len(new_keys)} tensors, all shapes match {base_model_dir}'s architecture.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to pytorch_model.pth from es-at-scale")
    ap.add_argument("--base-model-dir", required=True, help="HF snapshot dir of the base model (for config/tokenizer)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verify", action="store_true", help="Sanity-check converted keys/shapes against the base architecture")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.base_model_dir, "config.json")) as f:
        config = json.load(f)

    print(f"Loading {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    print(f"Loaded {len(state_dict)} tensors.")

    new_state = convert(state_dict, config)
    print(f"Converted to {len(new_state)} tensors.")

    if args.verify:
        verify(new_state, args.base_model_dir, config)

    # Cast to the dtype the base model was published in (bf16) unless already matching.
    target_dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
    new_state = {k: v.to(target_dtype) for k, v in new_state.items()}

    out_weights = os.path.join(args.output_dir, "model.safetensors")
    save_file(new_state, out_weights, metadata={"format": "pt"})
    print(f"Saved weights to {out_weights}")

    # Copy config/tokenizer files so the directory is a complete, loadable HF model.
    for fname in os.listdir(args.base_model_dir):
        if fname.endswith((".safetensors", ".bin", ".pth")) or fname.startswith("model"):
            continue
        src = os.path.join(args.base_model_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.output_dir, fname))

    print(f"HF-format checkpoint ready at {args.output_dir}")


if __name__ == "__main__":
    main()

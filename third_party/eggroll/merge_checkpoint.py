#!/usr/bin/env python3
"""Sample from checkpoint - simplest approach: load base model, then merge checkpoint"""

import argparse
import numpy as np
import torch
import sys
import os

sys.path.insert(0, '/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm')

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from safetensors.torch import load_file, save_file
from tasks import DrawEggTask, DrawChickTask


def unfuse_vllm_to_transformers(vllm_weights, model_config):
    """Unfuse vLLM's fused weights (qkv_proj, gate_up_proj) to transformers format"""

    # Get dimensions
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = getattr(model_config, 'head_dim', model_config.hidden_size // num_attention_heads)
    intermediate_size = model_config.intermediate_size

    unfused = {}

    for name, weight in vllm_weights.items():
        # Remove .base_layer suffix
        clean_name = name.replace('.base_layer', '')

        # Handle fused QKV
        if 'qkv_proj' in name:
            prefix = clean_name.replace('qkv_proj.weight', '')
            q_size = num_attention_heads * head_dim
            kv_size = num_key_value_heads * head_dim

            unfused[prefix + 'q_proj.weight'] = weight[:q_size, :]
            unfused[prefix + 'k_proj.weight'] = weight[q_size:q_size + kv_size, :]
            unfused[prefix + 'v_proj.weight'] = weight[q_size + kv_size:, :]

        # Handle fused gate_up
        elif 'gate_up_proj' in name:
            prefix = clean_name.replace('gate_up_proj.weight', '')
            unfused[prefix + 'gate_proj.weight'] = weight[:intermediate_size, :]
            unfused[prefix + 'up_proj.weight'] = weight[intermediate_size:, :]

        # Handle other weights (just remove .base_layer)
        else:
            unfused[clean_name] = weight

    return unfused


def merge_checkpoint_into_base_model(base_model_name, checkpoint_path, output_dir):
    """Load base model, unfuse and merge checkpoint, save merged model"""

    # Load base model with transformers
    print(f"Loading base model {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_weights = load_file(checkpoint_path)

    # Unfuse vLLM weights to transformers format
    print("Unfusing vLLM weights to transformers format...")
    unfused_weights = unfuse_vllm_to_transformers(checkpoint_weights, model.config)

    # Get model parameters
    model_params = dict(model.named_parameters())

    # Merge unfused weights into model
    print("Merging weights into base model...")
    num_loaded = 0
    num_skipped = 0

    for name, weight in unfused_weights.items():
        if name in model_params:
            if model_params[name].shape == weight.shape:
                model_params[name].data.copy_(weight.to(model_params[name].dtype))
                num_loaded += 1
            else:
                print(f"Shape mismatch for {name}: {model_params[name].shape} vs {weight.shape}")
                num_skipped += 1
        else:
            num_skipped += 1
            if num_skipped <= 5:
                print(f"Not in model: {name}")

    print(f"Loaded {num_loaded} weights from checkpoint")

    # Save merged model
    print(f"Saving merged model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("Merged model saved successfully")
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create temporary directory for merged model
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_dir = "/home/s5j/asims.s5j/Documents/esvllm-outer/hyperscale-es-vllm/tmp"
    checkpoint_name = args.checkpoint.strip("/").split("/")[-3]
    merged_model_path = os.path.join(temp_dir, "merged_models", checkpoint_name)
    os.makedirs(merged_model_path, exist_ok=True)

    # Merge checkpoint into base model
    checkpoint_weights_path = os.path.join(args.checkpoint, "model_weights.safetensors")
    merge_checkpoint_into_base_model(args.model_name, checkpoint_weights_path, merged_model_path)


if __name__ == "__main__":
    main()

# srun --nodes=1 --ntasks=1 --gpus-per-node=1 --time=01:00:00 --pty bash
# source $SCRATCH/uv_envs/vllm_env/.venv/bin/activate && cd $HOME/Documents/esvllm-outer/hyperscale-es-vllm

# python merge_checkpoint.py --checkpoint /scratch/s5j/asims.s5j/for_es_lora/D3-drawegg-jsd-nopenalty-drawegg-boxed-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768928661/checkpoints/checkpoint_step_299
# python merge_checkpoint.py --checkpoint /scratch/s5j/asims.s5j/for_es_lora/D3-drawegg-jsd-nopenalty-drawegg-boxed-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768934132/checkpoints/checkpoint_step_299
# python merge_checkpoint.py --checkpoint /scratch/s5j/asims.s5j/for_es_lora/D3-drawchick-jsd-nopenalty-drawchick-jsd-Qwen3-0.6B-P32-B1-S2048-std-l32-n4-lr0.001-sigma0.001-r1-alpha1-seed0-gpus16--1768927361/checkpoints/checkpoint_step_299

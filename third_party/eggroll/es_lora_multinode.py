#!/usr/bin/env python3
"""ES-LoRA Training with NCCL and async evaluation"""

import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass
import copy
import math

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port
import tyro
import wandb
import weave
from peft import LoraConfig, get_peft_model
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file, load_file

from tasks import MathTask, CountdownTask, ZerosTask, RandomTask

print("IMPORTS: All imports completed successfully", flush=True)
print("=" * 80, flush=True)

# Default Hyperparameters
EXPERIMENT_DIR = os.path.expandvars("$SCRATCH/for_es_lora/experiments")
# Use SLURM_JOB_ID to make path unique per job, avoiding conflicts from previous runs
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
LORA_POPULATION_PATH = f"/dev/shm/es_lora_population_async_{SLURM_JOB_ID}"

@dataclass
class Args:
    """ES Fine-tuning for Countdown Task with multi-engine NCCL sync and LoRA population"""
    model_name: str = "Qwen/Qwen2-0.5B" 
    # --- ES Hyperparameters ---
    sigma: float = 0.001
    population_size: int = 128
    num_iterations: int = 300
    max_tokens: int = 1024
    temperature: float = 0.0
    samples_per_prompt: int = 1
    task: str = "zeros"  # Options: "zeros", "countdown", "math:deepscaler40k", ...
    prompt_batch_size: int = 2
    pass_at_k: bool = False
    normalize_with_std: bool = False
    scale_lr_in_grad: bool = False

    # --- LoRA Config ---
    lora_r: int = 4
    lora_alpha: int = None
    steps_per_adapter: int = 4
    learning_rate: float = 0.001

    # --- Runtime Config ---
    num_gpus: int = None
    num_engines: int = None
    tensor_parallel_size: int = 1  # Number of GPUs per engine for tensor parallelism
    verbose: bool = True
    base_seed: int = 0
    sub_dataset_size: int = None
    steps_per_eval: int = 10 # -1 to disable
    eval_batch_size: int = 128
    es_update_chunk_size: int = None  # Auto-select based on lora_r if None

    # --- WandB ---
    use_wandb: bool = False
    wandb_project: str = "hyperscalees-vllm"
    name_prefix: str = f"debug"

    # --- Checkpointing ---
    save_freq: int = 50  # None: no saving, -1: saves at last step
    checkpoint_dir: str = None  # If None, will use EXPERIMENT_DIR/run_name/checkpoints
    resume_from: str = None  # Path to checkpoint to resume from

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r

        # Auto-configure tensor_parallel_size based on model name if not explicitly set
        # Only apply auto-config if TP was not set via command line (still equals default of 1)
        if self.tensor_parallel_size == 1:
            # Dictionary of models that benefit from tensor parallelism
            # Maps model name patterns to recommended TP size
            TP_CONFIG = {
                # Qwen 1.5
                "Qwen/Qwen1.5-110B": 4,
                "Qwen/Qwen1.5-110B-Chat": 4,
                # Qwen 2.5
                "Qwen/Qwen2.5-1.5B": 1,
                "Qwen/Qwen2.5-14B": 2,
                "Qwen/Qwen2.5-32B": 4,
                "Qwen/Qwen2.5-32B-Instruct": 4,
                "Qwen/Qwen2.5-72B": 4,
                "Qwen/Qwen2.5-72B-Instruct": 4,
                # Qwen 3
                "Qwen/Qwen3-1.7B": 1,
                "Qwen/Qwen3-4B": 1,
                "Qwen/Qwen3-4B-Base": 1,
                "Qwen/Qwen3-8B": 1,
                "Qwen/Qwen3-30B": 2,
                "Qwen/Qwen3-30B-Base": 2,
                "Qwen/Qwen3-32B": 4,
            }

            # Check if model_name matches any pattern
            for model_pattern, tp_size in TP_CONFIG.items():
                if model_pattern in self.model_name:
                    self.tensor_parallel_size = tp_size
                    print(f"Auto-configured tensor_parallel_size={tp_size} for model {self.model_name}", flush=True)
                    break


LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

def map_peft_updates_to_vllm(peft_updates_dict, vllm_shapes_dict, device: torch.device):
    # Keep on CPU to avoid OOM - will move to GPU when applying
    vllm_updates_dict = {
        name: torch.zeros(shape, device='cpu', dtype=torch.float32) for name, shape in vllm_shapes_dict.items()
        if name.endswith(".base_layer.weight")
    }
    for peft_name, weight_update in peft_updates_dict.items():
        vllm_name = peft_name.replace("base_model.model.", "")
        if "self_attn.q_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.q_proj", "self_attn.qkv_proj")
            start = 0
            # If vLLM tensor is sharded (TP > 1), only use the corresponding shard
            vllm_size = vllm_updates_dict[vllm_name].shape[0]
            end = min(start + weight_update.shape[0], vllm_size)
            weight_shard = weight_update[:end-start]
            vllm_updates_dict[vllm_name][start:end] += weight_shard
        elif "self_attn.k_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.k_proj", "self_attn.qkv_proj")
            peft_q_name = peft_name.replace("k_proj", "q_proj")
            start = peft_updates_dict[peft_q_name].shape[0]
            vllm_size = vllm_updates_dict[vllm_name].shape[0]
            end = min(start + weight_update.shape[0], vllm_size)
            weight_shard = weight_update[:end-start]
            vllm_updates_dict[vllm_name][start:end] += weight_shard
        elif "self_attn.v_proj" in vllm_name:
            vllm_name = vllm_name.replace("self_attn.v_proj", "self_attn.qkv_proj")
            peft_q_name = peft_name.replace("v_proj", "q_proj")
            peft_k_name = peft_name.replace("v_proj", "k_proj")
            start = peft_updates_dict[peft_q_name].shape[0] + peft_updates_dict[peft_k_name].shape[0]
            vllm_size = vllm_updates_dict[vllm_name].shape[0]
            end = min(start + weight_update.shape[0], vllm_size)
            weight_shard = weight_update[:end-start]
            vllm_updates_dict[vllm_name][start:end] += weight_shard
        elif "self_attn.o_proj" in vllm_name:
            # For column-parallel layers (o_proj, down_proj), shard along dimension 1 (columns)
            vllm_size = vllm_updates_dict[vllm_name].shape[1]
            if vllm_size < weight_update.shape[1]:
                # Sharded along columns
                weight_shard = weight_update[:, :vllm_size]
                vllm_updates_dict[vllm_name] += weight_shard
            else:
                vllm_updates_dict[vllm_name] += weight_update
        elif "mlp.gate_proj" in vllm_name:
            vllm_name = vllm_name.replace("mlp.gate_proj", "mlp.gate_up_proj")
            start = 0
            vllm_size = vllm_updates_dict[vllm_name].shape[0]
            end = min(start + weight_update.shape[0], vllm_size)
            weight_shard = weight_update[:end-start]
            vllm_updates_dict[vllm_name][start:end] += weight_shard
        elif "mlp.up_proj" in vllm_name:
            vllm_name = vllm_name.replace("mlp.up_proj", "mlp.gate_up_proj")
            peft_gate_name = peft_name.replace("up_proj", "gate_proj")
            start = peft_updates_dict[peft_gate_name].shape[0]
            vllm_size = vllm_updates_dict[vllm_name].shape[0]
            end = min(start + weight_update.shape[0], vllm_size)
            weight_shard = weight_update[:end-start]
            vllm_updates_dict[vllm_name][start:end] += weight_shard
        elif "mlp.down_proj" in vllm_name:
            # For column-parallel layers (o_proj, down_proj), shard along dimension 1 (columns)
            vllm_size = vllm_updates_dict[vllm_name].shape[1]
            if vllm_size < weight_update.shape[1]:
                # Sharded along columns
                weight_shard = weight_update[:, :vllm_size]
                vllm_updates_dict[vllm_name] += weight_shard
            else:
                vllm_updates_dict[vllm_name] += weight_update
        else:
            raise ValueError(f"Unexpected PEFT layer name: {peft_name}")
    return vllm_updates_dict


def _stateless_init_process_group(master_address, master_port, gpu_rank, world_size, device):
    """Initializes PyNcclCommunicator using StatelessProcessGroup."""
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
    except ImportError:
        print("Warning: vLLM distributed modules not found. NCCL features will not work.")
        return None
        
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=gpu_rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

def get_rng_noise(
        base_seed: int,
        num_pop_pairs: int,
        pop_pair_idx: int,
        num_layers: int,
        layer_idx: int,
        step: int,
        shapes: list,
        ) -> dict[torch.device, torch.Generator]:
    """
    Create a dictionary of RNGs, one for each device.
    All RNGs are seeded with the same ID to ensure deterministic noise
    across different devices.
    """
    id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    torch_rng = torch.Generator().manual_seed(id)

    noise_a, noise_b = (torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=shape,
                    generator=torch_rng,
                ) for shape in shapes)
    return noise_a, noise_b

class WorkerExtension:
    """
    Custom extension for vLLM workers to handle ES update and NCCL broadcast.
    This class is passed to the vLLM engine via 'worker_extension_cls'.
    """

    def get_transport_info(self):
        """Returns the IP and a free port from the worker's perspective."""
        return get_ip(), get_open_port()

    @torch.no_grad()
    def apply_lora_es_update(self, normalized_fitnesses: list[tuple[int, float]], peft_shapes_dict, es_step: int, args: Args):
        """
        Computes and applies the ES update delta to the base model weights.
        Processes layer-by-layer to avoid System RAM OOM on large models.
        """
        if self.gpu_rank != 0:
            return False

        # Pre-calculate index map
        peft_name_to_idx = {name: i for i, name in enumerate(peft_shapes_dict.keys())}
        
        # Get vLLM model parameters once
        vllm_params = dict(self.model_runner.model.named_parameters())
        
        pop_step = es_step // args.steps_per_adapter

        # Adaptive chunk size
        if args.es_update_chunk_size is not None:
            chunk_size = min(args.es_update_chunk_size, args.population_size // 2)
        elif args.lora_r <= 2:
            chunk_size = min(128, args.population_size // 2)
        elif args.lora_r <= 8:
            chunk_size = min(64, args.population_size // 2)
        else:
            chunk_size = min(32, args.population_size // 2)

        print(f"ES UPDATE: Starting streaming update for {len(peft_shapes_dict)} layers...", flush=True)

        # Iterate through PEFT layers one by one
        for layer_idx, (peft_name, weight_shape) in enumerate(peft_shapes_dict.items()):
            lora_b_shape, lora_a_shape = (weight_shape[0], args.lora_r), (args.lora_r, weight_shape[1])
            
            # 1. Compute the update for this specific layer (Accumulate on GPU, move to CPU)
            layer_update = torch.zeros(weight_shape, device=self.device, dtype=torch.float32)

            for chunk_start in range(0, args.population_size // 2, chunk_size):
                chunk_end = min(chunk_start + chunk_size, args.population_size // 2)
                
                noise_a_list = []
                noise_b_list = []
                fitness_diffs = []

                for pop_pair_idx in range(chunk_start, chunk_end):
                    pop_idx_1 = pop_pair_idx * 2
                    pop_idx_2 = pop_pair_idx * 2 + 1
                    fitness_diff = normalized_fitnesses[pop_idx_1] - normalized_fitnesses[pop_idx_2]
                    fitness_diffs.append(fitness_diff)

                    # Use the pre-calculated global index to keep seeds identical to previous logic
                    global_layer_idx = peft_name_to_idx[peft_name]
                    
                    noise_a, noise_b = get_rng_noise(
                        base_seed=args.base_seed,
                        num_pop_pairs=args.population_size//2,
                        pop_pair_idx=pop_idx_1//2,
                        num_layers=len(peft_shapes_dict.keys()),
                        layer_idx=global_layer_idx, 
                        step=pop_step,
                        shapes=[lora_a_shape, lora_b_shape],
                    )
                    noise_a_list.append(noise_a)
                    noise_b_list.append(noise_b)

                noise_a_batch = torch.stack(noise_a_list).to(self.device) * math.sqrt(args.sigma)
                noise_b_batch = torch.stack(noise_b_list).to(self.device) * math.sqrt(args.sigma / args.lora_r) # add 1/sqrt(r) factor
                fitness_diffs_tensor = torch.tensor(fitness_diffs, device=self.device, dtype=noise_a_batch.dtype)

                if args.lora_r == 1:
                    noise_b_vec = noise_b_batch.squeeze(2)
                    noise_a_vec = noise_a_batch.squeeze(1)
                    weighted_b = noise_b_vec * fitness_diffs_tensor.unsqueeze(1)
                    weighted_noise = torch.mm(weighted_b.t(), noise_a_vec)
                else:
                    noise_batch = torch.bmm(noise_b_batch, noise_a_batch)
                    weighted_noise = (noise_batch * fitness_diffs_tensor.view(-1, 1, 1)).sum(dim=0)
                    del noise_batch

                layer_update.add_(weighted_noise)
                del noise_a_batch, noise_b_batch, weighted_noise
            
            # 2. Scale gradient
            gradient = (1.0 / (args.population_size * args.sigma + 1e-8)) * layer_update * args.learning_rate

            if args.scale_lr_in_grad:
                gradient *= math.sqrt(args.population_size)

            del layer_update # Free the accumulation tensor

            # 3. Identify target vLLM parameter and apply immediately
            # Logic adapted from map_peft_updates_to_vllm to run inline
            vllm_name = peft_name.replace("base_model.model.", "")
            target_param = None
            slice_obj = None
            
            # --- Mapping Logic ---
            if "self_attn.q_proj" in vllm_name:
                target_name = vllm_name.replace("self_attn.q_proj", "self_attn.qkv_proj")
                if target_name in vllm_params:
                    target_param = vllm_params[target_name]
                    # Start at 0
                    slice_obj = slice(0, weight_shape[0])
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: expected fused param '{target_name}' in vllm_params "
                        f"for q_proj layer '{peft_name}', but it was not found. "
                        f"Available keys (sample): {list(vllm_params.keys())[:5]}"
                    )

            elif "self_attn.k_proj" in vllm_name:
                target_name = vllm_name.replace("self_attn.k_proj", "self_attn.qkv_proj")
                # Need to find offset. q_proj is usually same size as k_proj in standard attn,
                # but GQA might differ. We need to find q_proj size.
                # Safe way: look up q_proj in peft_shapes_dict
                q_name = peft_name.replace("k_proj", "q_proj")
                if target_name in vllm_params and q_name in peft_shapes_dict:
                    target_param = vllm_params[target_name]
                    start = peft_shapes_dict[q_name][0]
                    slice_obj = slice(start, start + weight_shape[0])
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: could not resolve k_proj offset for '{peft_name}'. "
                        f"fused target '{target_name}' present={target_name in vllm_params}, "
                        f"q_name '{q_name}' present={q_name in peft_shapes_dict}."
                    )

            elif "self_attn.v_proj" in vllm_name:
                target_name = vllm_name.replace("self_attn.v_proj", "self_attn.qkv_proj")
                q_name = peft_name.replace("v_proj", "q_proj")
                k_name = peft_name.replace("v_proj", "k_proj")
                if target_name in vllm_params and q_name in peft_shapes_dict and k_name in peft_shapes_dict:
                    target_param = vllm_params[target_name]
                    start = peft_shapes_dict[q_name][0] + peft_shapes_dict[k_name][0]
                    slice_obj = slice(start, start + weight_shape[0])
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: could not resolve v_proj offset for '{peft_name}'. "
                        f"fused target '{target_name}' present={target_name in vllm_params}, "
                        f"q_name '{q_name}' present={q_name in peft_shapes_dict}, "
                        f"k_name '{k_name}' present={k_name in peft_shapes_dict}."
                    )

            elif "self_attn.o_proj" in vllm_name:
                # Column parallel: slice columns
                if vllm_name in vllm_params:
                    target_param = vllm_params[vllm_name]
                    # If TP > 1, the vLLM param is smaller than PEFT param
                    if target_param.shape[1] < weight_shape[1]:
                        slice_obj = (slice(None), slice(0, target_param.shape[1]))
                    else:
                        slice_obj = (slice(None), slice(None))
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: expected param '{vllm_name}' in vllm_params "
                        f"for o_proj layer '{peft_name}', but it was not found."
                    )

            elif "mlp.gate_proj" in vllm_name:
                target_name = vllm_name.replace("mlp.gate_proj", "mlp.gate_up_proj")
                if target_name in vllm_params:
                    target_param = vllm_params[target_name]
                    slice_obj = slice(0, weight_shape[0])
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: expected fused param '{target_name}' in vllm_params "
                        f"for gate_proj layer '{peft_name}', but it was not found."
                    )

            elif "mlp.up_proj" in vllm_name:
                target_name = vllm_name.replace("mlp.up_proj", "mlp.gate_up_proj")
                gate_name = peft_name.replace("up_proj", "gate_proj")
                if target_name in vllm_params and gate_name in peft_shapes_dict:
                    target_param = vllm_params[target_name]
                    start = peft_shapes_dict[gate_name][0]
                    slice_obj = slice(start, start + weight_shape[0])
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: could not resolve up_proj offset for '{peft_name}'. "
                        f"fused target '{target_name}' present={target_name in vllm_params}, "
                        f"gate_name '{gate_name}' present={gate_name in peft_shapes_dict}."
                    )

            elif "mlp.down_proj" in vllm_name:
                # Column parallel
                if vllm_name in vllm_params:
                    target_param = vllm_params[vllm_name]
                    if target_param.shape[1] < weight_shape[1]:
                        slice_obj = (slice(None), slice(0, target_param.shape[1]))
                    else:
                        slice_obj = (slice(None), slice(None))
                else:
                    raise RuntimeError(
                        f"apply_lora_es_update: expected param '{vllm_name}' in vllm_params "
                        f"for down_proj layer '{peft_name}', but it was not found."
                    )

            else:
                raise RuntimeError(
                    f"apply_lora_es_update: unrecognised PEFT layer '{peft_name}' (vllm_name='{vllm_name}'). "
                    f"No mapping rule exists for this layer type. "
                    f"Expected one of: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj."
                )

            # --- Apply Update ---
            if target_param is not None:
                # Cast gradient to model dtype (float16/bfloat16)
                grad_shard = gradient.to(dtype=target_param.dtype)
                
                try:
                    if isinstance(slice_obj, tuple): # Column parallel special case
                        
                        if target_param.shape[1] < grad_shard.shape[1]:
                             grad_shard = grad_shard[:, :target_param.shape[1]]
                        
                        target_param.data.add_(grad_shard)
                        
                    elif isinstance(slice_obj, slice): # Row parallel (qkv, gate_up)
                        
                        # Safe Application:
                        param_size = target_param.shape[0]
                        start = slice_obj.start
                        end = min(slice_obj.stop, param_size)
                        
                        if start < param_size:
                            valid_grad = grad_shard[:(end-start)]
                            target_param.data[start:end].add_(valid_grad)

                    else:
                        # slice_obj must always be a slice or tuple by this point;
                        # reaching here means the mapping logic above has a bug.
                        raise AssertionError(
                            f"apply_lora_es_update: slice_obj is {slice_obj!r} (type {type(slice_obj)}) "
                            f"for layer '{vllm_name}'. This should be unreachable — "
                            f"the mapping logic must have set target_param without setting slice_obj."
                        )

                except Exception as e:
                    print(f"ERROR updating {vllm_name}: {e}. Shapes: Param {target_param.shape}, Grad {grad_shard.shape}", flush=True)
            
            # 4. Clean up immediately
            del gradient
            del grad_shard
            if chunk_start % (chunk_size * 4) == 0:
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("ES UPDATE: Completed successfully.", flush=True)
        gc.collect()
        return True
    
    def init_inter_engine_group(self, master_address: str, master_port: int, gpu_rank: int, world_size: int):
        self.device = self.model_runner.device
        self.gpu_rank = gpu_rank
        self.world_size = world_size

        # Only TP rank 0 should participate in inter-engine communication
        # When using tensor parallelism, each engine has multiple workers (TP ranks)
        # but only one representative (rank 0) should join the inter-engine group
        from vllm.distributed import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()

        if tp_rank == 0:
            # This is the TP master, initialize inter-engine communication
            self.inter_pg = _stateless_init_process_group(
                master_address, master_port, gpu_rank, world_size, self.device
            )
        else:
            # This is a TP worker, skip inter-engine communication
            self.inter_pg = None

        return True

    @torch.no_grad()
    def broadcast_all_weights(self, src_rank: int):
        # NOTE: ALL ranks must participate in NCCL broadcast,
        # including the source rank. The source sends, others receive.

        print(f"WORKER {self.gpu_rank}: broadcast_all_weights called, src_rank={src_rank}", flush=True)

        if not self.inter_pg:
            # NCCL not available - this will require weights to be sent via Ray
            # Return False to signal caller to use Ray-based broadcast instead
            print(f"WORKER {self.gpu_rank}: No NCCL inter_pg available, returning False", flush=True)
            return False

        try:
            is_source = (self.gpu_rank == int(src_rank))
            role = "sender" if is_source else "receiver"
            print(f"WORKER {self.gpu_rank}: Starting NCCL broadcast as {role} (src={src_rank})...", flush=True)

            param_count = 0
            for name, param in self.model_runner.model.named_parameters():
                # ALL ranks must call broadcast - source sends, others receive
                self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
                param_count += 1

            print(f"WORKER {self.gpu_rank}: Broadcast {param_count} parameters, synchronizing...", flush=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"WORKER {self.gpu_rank}: Broadcast complete ({role})", flush=True)
            return True
        except Exception as e:
            print(f"WORKER {self.gpu_rank}: NCCL broadcast failed: {e}", flush=True)
            return False

    @torch.no_grad()
    def get_model_state_dict(self):
        """Get the current model state dict (for Ray-based broadcast)"""
        return {name: param.cpu().clone() for name, param in self.model_runner.model.named_parameters()}

    @torch.no_grad()
    def set_model_state_dict(self, state_dict):
        """Set the model state dict (for Ray-based broadcast)"""
        model_params = dict(self.model_runner.model.named_parameters())
        for name, param in state_dict.items():
            if name in model_params:
                model_params[name].data.copy_(param.to(model_params[name].device))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

class ESNcclLLM(LLM):
    """vLLM subclass using the custom WorkerExtension."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)
        
        # Placeholders for LoRA generation data
        self.lora_init_state_dict = None
        self.lora_init_shapes = None
        self.lora_config_data = None

    def setup_local_lora_generation(self, peft_state_dict, peft_shapes_dict, lora_config_dict, rank: int):
        """Receives the initial LoRA state to be able to reconstruct adapters locally."""
        self.lora_init_state_dict = peft_state_dict
        self.lora_init_shapes = peft_shapes_dict
        self.lora_config_data = lora_config_dict
        self.rank = rank

        self.lora_storage_path = f"{LORA_POPULATION_PATH}_{self.rank}"

        # Safer cleanup: handle permission errors and retry with forced removal
        if os.path.exists(self.lora_storage_path):
            try:
                shutil.rmtree(self.lora_storage_path)
            except (PermissionError, OSError) as e:
                print(f"WARNING: Failed to remove {self.lora_storage_path}: {e}. Attempting forced cleanup...", flush=True)
                try:
                    # Try to change permissions recursively and retry
                    import stat
                    for root, dirs, files in os.walk(self.lora_storage_path, topdown=False):
                        for name in files:
                            filepath = os.path.join(root, name)
                            try:
                                os.chmod(filepath, stat.S_IWUSR | stat.S_IRUSR)
                                os.remove(filepath)
                            except Exception:
                                pass  # Ignore individual file errors
                        for name in dirs:
                            dirpath = os.path.join(root, name)
                            try:
                                os.chmod(dirpath, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                                os.rmdir(dirpath)
                            except Exception:
                                pass  # Ignore individual dir errors
                    # Try final removal
                    if os.path.exists(self.lora_storage_path):
                        shutil.rmtree(self.lora_storage_path, ignore_errors=True)
                except Exception as e2:
                    print(f"WARNING: Forced cleanup also failed: {e2}. Will try to proceed anyway...", flush=True)

        os.makedirs(self.lora_storage_path, exist_ok=True)
        return True
    
    

    def generate_local_adapters(self, population_indices: list[int], es_step: int, args: Args):
        """
        Generates LoRA adapter files in the LOCAL /dev/shm of this worker node.
        Returns the absolute paths to these files.
        """
        adapter_paths = []
        pop_step = es_step // args.steps_per_adapter

        # Ensure config is JSON serializable
        config_to_save = copy.deepcopy(self.lora_config_data)
        if "target_modules" in config_to_save and isinstance(config_to_save["target_modules"], (set, tuple)):
            config_to_save["target_modules"] = list(config_to_save["target_modules"])

        for pop_idx in population_indices:
            adapter_path = os.path.join(self.lora_storage_path, f"pop_{pop_idx}")

            # Try to create directory with error handling
            try:
                os.makedirs(adapter_path, exist_ok=True)
            except (PermissionError, OSError) as e:
                print(f"WARNING: Failed to create {adapter_path}: {e}. Attempting cleanup and retry...", flush=True)
                # Try to remove and recreate
                try:
                    import stat
                    # Fix permissions on parent directory first
                    if os.path.exists(self.lora_storage_path):
                        try:
                            os.chmod(self.lora_storage_path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                        except Exception as e_parent:
                            print(f"WARNING: Could not chmod parent {self.lora_storage_path}: {e_parent}", flush=True)

                    # Now fix the specific directory if it exists
                    if os.path.exists(adapter_path):
                        try:
                            os.chmod(adapter_path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                            shutil.rmtree(adapter_path, ignore_errors=True)
                        except Exception as e_dir:
                            print(f"WARNING: Could not remove {adapter_path}: {e_dir}", flush=True)
                            pass

                    # Final attempt to create
                    os.makedirs(adapter_path, exist_ok=True)
                except Exception as e2:
                    print(f"ERROR: Could not create adapter directory {adapter_path}: {e2}", flush=True)
                    raise

            adapter_paths.append(adapter_path)
            
            # Save config
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(config_to_save, f)

            # Generate weights (sanitized)
            local_state_dict = {}
            for layer_idx, (peft_name, weight_shape) in enumerate(self.lora_init_shapes.items()):
                # Generate LoRA A and B names from the base_layer.weight name
                lora_a_name_raw = peft_name.replace("base_layer.weight", "lora_A.default.weight")
                lora_b_name_raw = peft_name.replace("base_layer.weight", "lora_B.default.weight")

                # 2. PEFT uses ".lora_A.default.weight" but vLLM expects ".lora_A.weight"
                lora_a_name = lora_a_name_raw.replace(".lora_A.default.weight", ".lora_A.weight")
                lora_b_name = lora_b_name_raw.replace(".lora_B.default.weight", ".lora_B.weight")

                # Get base (initial) weights and clone to CPU
                lora_a = self.lora_init_state_dict[lora_a_name_raw].clone().cpu()
                lora_b = self.lora_init_state_dict[lora_b_name_raw].clone().cpu()

                lora_b_shape, lora_a_shape = (weight_shape[0], args.lora_r), (args.lora_r, weight_shape[1])

                noise_a, noise_b = get_rng_noise(
                    base_seed=args.base_seed,
                    num_pop_pairs=args.population_size//2,
                    pop_pair_idx=pop_idx//2,
                    num_layers=len(self.lora_init_shapes.keys()),
                    layer_idx=layer_idx,
                    step=pop_step,
                    shapes=[lora_a_shape, lora_b_shape],
                )

                noise_b *= math.sqrt(args.sigma / args.lora_r) # add 1/sqrt(r) factor
                noise_a *= math.sqrt(args.sigma)

                # Zero out the weights (before then setting them to noise)
                lora_a.zero_()
                lora_b.zero_()

                # Antithetic sampling
                lora_a.add_(noise_a)
                if pop_idx % 2 == 1:
                    lora_b.add_(-noise_b)
                else:
                    lora_b.add_(noise_b)

                # Debug: Check if LoRA weights are non-zero (only for first layer of first few adapters)
                if layer_idx == 0 and pop_idx < 4:
                    max_a = lora_a.abs().max().item()
                    max_b = lora_b.abs().max().item()
                    print(f"LORA GEN DEBUG: pop_idx={pop_idx}, layer={layer_idx}, max_a={max_a:.6e}, max_b={max_b:.6e}, sigma={math.sqrt(args.sigma):.6e}", flush=True)

                local_state_dict[lora_a_name] = lora_a
                local_state_dict[lora_b_name] = lora_b

            # Save tensors
            save_file(local_state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))

        # Debug: Verify first adapter exists and has non-zero weights
        if len(adapter_paths) > 0:
            from safetensors import safe_open
            first_adapter = adapter_paths[0]
            with safe_open(os.path.join(first_adapter, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if len(keys) > 0:
                    first_tensor = f.get_tensor(keys[0])
                    print(f"LORA GEN DEBUG: First adapter first tensor max: {first_tensor.abs().max().item():.6e}", flush=True)
        
        return adapter_paths
    
    def generate_and_score(self, prompts, sampling_params, lora_requests, task_obj, answers, args):
        """
        Generates responses AND calculates fitness/stats on the GPU worker.
        """
        # Debug: Check if LoRA requests are being passed
        if lora_requests is not None:
            if isinstance(lora_requests, list) and len(lora_requests) > 0:
                print(f"GENERATE DEBUG: Received {len(lora_requests)} LoRA requests", flush=True)
                print(f"GENERATE DEBUG: First LoRA: name={lora_requests[0].lora_name}, id={lora_requests[0].lora_int_id}, path={lora_requests[0].lora_path}", flush=True)
                if len(lora_requests) > 1:
                    print(f"GENERATE DEBUG: Second LoRA: name={lora_requests[1].lora_name}, id={lora_requests[1].lora_int_id}, path={lora_requests[1].lora_path}", flush=True)
        else:
            print(f"GENERATE DEBUG: LoRA requests is None", flush=True)

        request_outputs = self.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests,
            use_tqdm=True,
        )

        # 2. Calculate fitness immediately (Local CPU)
        fitness_list = []
        distinct_counts = []
        total_responses = 0
        num_truncated = 0
        mean_char_lengths = []
        mean_token_lengths = []
        responses_for_logging = []
        all_sample_stds = []  # Track std across samples for each (pop, prompt) pair
        all_pass_at_k_fitnesses = []  # Track max fitness across samples (pass@k)
        all_mean_fitnesses = []  # Track mean fitness across samples
        all_task_info = {}  # Collect task-specific info dicts

        num_prompts = len(answers)

        # Process linearly.
        pop_responses_buffer = ""

        for i, output in enumerate(request_outputs):
            prompt_idx = i % num_prompts
            pop_idx = i // num_prompts
            gt_answer = answers[prompt_idx]

            # Collect all responses for this prompt
            responses = [o.text for o in output.outputs]

            truncateds = [o.finish_reason == "length" for o in output.outputs]

            # Get fitness
            fit, model_answers, sample_fitnesses, task_info = task_obj.get_fitness(responses, truncateds, gt_answer, pass_at_k=args.pass_at_k)

            # Collect task-specific info
            for k, v in task_info.items():
                if k not in all_task_info:
                    all_task_info[k] = []
                all_task_info[k].append(v)

            # Collect stats
            sample_char_lens = []
            sample_token_lens = []
            model_answers_set = set()

            # Add model answers to set for distinct count
            if isinstance(model_answers, (list, tuple)):
                for ma in model_answers:
                    if ma is not None:
                        # Convert lists to tuples so they can be hashed
                        if isinstance(ma, list):
                            model_answers_set.add(tuple(ma))
                        else:
                            model_answers_set.add(ma)
            elif model_answers is not None:
                if isinstance(model_answers, list):
                    model_answers_set.add(tuple(model_answers))
                else:
                    model_answers_set.add(model_answers)

            # Format current sample for potential logging
            if pop_idx < 2 and prompt_idx < 3:
                current_prompt_log = f"\n[PROMPT {prompt_idx}]: {prompts[i]}\n"

            for j, sample in enumerate(output.outputs):
                text = sample.text

                if sample.finish_reason == "length":
                    num_truncated += 1

                sample_char_lens.append(len(text))
                sample_token_lens.append(len(sample.token_ids))
                total_responses += 1

                if pop_idx < 2 and prompt_idx < 3:
                    # Show individual sample fitness for logging
                    sample_fit = sample_fitnesses[j] if j < len(sample_fitnesses) else fit
                    current_prompt_log += f"\n------SAMPLE {j+1}: {text} || FIT={sample_fit}\n"

            if pop_idx < 2 and prompt_idx < 3:
                pop_responses_buffer += current_prompt_log

            if (i + 1) % num_prompts == 0 and pop_responses_buffer != "":
                if pop_responses_buffer:
                    header = f"-----POP {pop_idx} BATCH LOG-----\n"
                    responses_for_logging.append(header + pop_responses_buffer)
                    pop_responses_buffer = ""

            # Store aggregated fitness (one per population member per prompt)
            fitness_list.append(fit)

            # Always track both max and mean across samples, regardless of pass_at_k setting
            if len(sample_fitnesses) > 0:
                all_pass_at_k_fitnesses.append(np.max(sample_fitnesses))
                all_mean_fitnesses.append(np.mean(sample_fitnesses))
                # Compute std across samples (for std_in_samples metric) - only meaningful if >1 sample
                if len(sample_fitnesses) > 1:
                    all_sample_stds.append(np.std(sample_fitnesses))
            else:
                # Fallback if sample_fitnesses is somehow empty (shouldn't happen)
                all_pass_at_k_fitnesses.append(fit)
                all_mean_fitnesses.append(fit)

            distinct_counts.append(len(model_answers_set))
            mean_char_lengths.append(np.mean(sample_char_lens))
            mean_token_lengths.append(np.mean(sample_token_lens))

        info = {
            "total_responses": total_responses,
            "prop_truncated": num_truncated / total_responses if total_responses > 0 else 0.0,
            "mean_char_length": np.mean(mean_char_lengths),
            "mean_token_length": np.mean(mean_token_lengths),
            "mean_distinct_counts": np.mean(distinct_counts),
            "std_in_samples": np.mean(all_sample_stds) if all_sample_stds else 0.0,
            "pass_at_k_fitness": np.mean(all_pass_at_k_fitnesses) if all_pass_at_k_fitnesses else 0.0,
            "mean_sample_fitness": np.mean(all_mean_fitnesses) if all_mean_fitnesses else 0.0,
        }

        # Merge task-specific info (average across all samples)
        for k, v in all_task_info.items():
            info[k] = float(np.mean(v))

        return fitness_list, info, responses_for_logging

def launch_engines(num_engines, model_name, population_size, lora_r, tensor_parallel_size=1, max_tokens=1024):
    """Launches multiple vLLM engines via Ray Placement Groups.

    Args:
        num_engines: Number of engines to launch
        model_name: HuggingFace model name
        population_size: Total population size
        lora_r: LoRA rank
        tensor_parallel_size: Number of GPUs per engine for tensor parallelism
        max_tokens: Maximum tokens to generate (used to set max_model_len)
    """
    # When TP > 1, we create placement groups with separate bundles for each GPU
    # vLLM requires placement groups where each bundle has exactly 1 GPU
    # We create a placement group with tensor_parallel_size bundles, each with 1 GPU
    # NOTE: enforce_eager=True is required for LoRA + TP > 1 to work correctly
    # (vLLM v1 has issues with LoRA weight sharding across GPUs)
    if tensor_parallel_size > 1:
        print(f"Creating {num_engines} placement groups ({tensor_parallel_size} bundles of 1 GPU each).")
        # Each placement group has tensor_parallel_size bundles, each with 1 GPU
        # Allocate CPUs to help Ray manage CPU memory during model loading
        cpus_per_worker = 2  # Reserve some CPU for each worker to manage memory properly
        pgs = [
            placement_group(
                [{"GPU": 1, "CPU": cpus_per_worker} for _ in range(tensor_parallel_size)],
                lifetime="detached",
                strategy="STRICT_PACK"  # Keep GPUs on same node for TP (required for fast communication)
            )
            for _ in range(num_engines)
        ]
        ray.get([pg.ready() for pg in pgs])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
            )
            for pg in pgs
        ]

        print(f"Launching {num_engines} ESNcclLLM Ray actors (TP={tensor_parallel_size}).")
        print(f"NOTE: Using enforce_eager=True for LoRA + TP compatibility")

        # Compute per-engine LoRA count: how many adapters this engine must serve simultaneously.
        loras_per_engine = (population_size + num_engines - 1) // num_engines

        concurrent_seqs = loras_per_engine * args.prompt_batch_size

        # Choose vLLM settings based on model size.
        model_lower = model_name.lower()
        if "110b" in model_lower:
            max_num_seqs = 384
            max_num_batched_tokens = 4 * args.max_tokens
            gpu_mem_util = 0.9
        elif "72b" in model_lower:
            max_num_seqs = 384
            max_num_batched_tokens = 4 * args.max_tokens
            gpu_mem_util = 0.9
        else:
            max_num_seqs = 512
            max_num_batched_tokens = 8 * args.max_tokens
            gpu_mem_util = 0.9

        engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="ray",
                worker_extension_cls="es_lora_multinode.WorkerExtension",
                dtype="auto",
                enable_prefix_caching=True,
                enforce_eager=True,  # required for LoRA + TP > 1
                enable_lora=True,
                max_loras=loras_per_engine,
                max_lora_rank=max(lora_r, 8),
                gpu_memory_utilization=gpu_mem_util,
                trust_remote_code=True,
                max_num_seqs=max_num_seqs,
                max_model_len=max(1024, 512 + max_tokens),
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=True,  
                load_format="auto",
            )
            for strategy in strategies
        ]
        return engines, pgs

    # TP=1: Use placement groups to pin each engine to a specific GPU
    print(f"Creating {num_engines} placement groups (1 GPU each).")
    pgs = [
        placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached")
        for _ in range(num_engines)
    ]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    print(f"Launching {num_engines} ESNcclLLM Ray actors.")
    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
            worker_extension_cls="es_lora_multinode.WorkerExtension",
            dtype="auto",
            enable_prefix_caching=True,
            enforce_eager=False, 
            enable_lora=True,
            max_loras=(population_size + num_engines - 1) // num_engines,
            max_lora_rank=max(lora_r, 8),
            gpu_memory_utilization=0.90,  # conservative to reduce overall memory pressure
            trust_remote_code=True,
            max_num_seqs=512,  # allows parallel processing of up to 512 sequences per engine for higher throughput
            max_model_len=max(1024, 512 + max_tokens),  # dynamic based on generation length
            max_num_batched_tokens=args.prompt_batch_size * 2048, # controls maximum tokens processed per forward pass; larger batches = better GPU utilization and throughput
            enable_chunked_prefill=True,
            load_format="auto",  # let vLLM choose the most efficient loading method
        ) 
        for strategy in strategies
    ]
    return engines, pgs

def save_checkpoint(checkpoint_dir: str, es_step: int, model_state_dict: dict,
                   task_state: dict, args: Args, fitnesses_so_far: list):
    """
    Save checkpoint including:
    - Model weights
    - Current ES step
    - Task dataset state
    - Training metrics
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{es_step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    save_file(model_state_dict, model_weights_path)

    # Save training state
    checkpoint_state = {
        "es_step": es_step,
        "args": vars(args),
        "fitnesses_so_far": fitnesses_so_far,
        "task_state": task_state,
    }

    # Save state as JSON
    state_path = os.path.join(checkpoint_path, "training_state.json")
    with open(state_path, "w") as f:
        json.dump(checkpoint_state, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_path}", flush=True)

def load_checkpoint(checkpoint_path: str):
    """
    Load checkpoint and return all saved state.
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Load training state
    state_path = os.path.join(checkpoint_path, "training_state.json")
    with open(state_path, "r") as f:
        state = json.load(f)

    # Load model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    model_state_dict = load_file(model_weights_path)

    print(f"Checkpoint loaded from {checkpoint_path}", flush=True)
    print(f"Resuming from ES step {state['es_step']}", flush=True)

    return {
        "model_state_dict": model_state_dict,
        "es_step": state["es_step"],
        "task_state": state.get("task_state", {}),
        "fitnesses_so_far": state.get("fitnesses_so_far", []),
    }

def main(args: Args):
    print("MAIN: Entered main function", flush=True)
    sys.stdout.flush()

    # --- 1. Initialize Ray FIRST (Connect to the cluster created by Slurm) ---
    # Do this before counting GPUs, otherwise only see local GPUs.
    print("MAIN: Connecting to Ray Cluster...", flush=True)
    # address="auto" picks up the RAY_ADDRESS env var set by your bash script
    ray.init(address="auto", include_dashboard=False, ignore_reinit_error=True)
    
    # --- 2. Query Ray for TOTAL Cluster Resources ---
    print("MAIN: Querying Ray for total cluster resources...", flush=True)
    resources = ray.cluster_resources()
    total_gpus = int(resources.get("GPU", 0))

    if total_gpus == 0:
        raise ValueError("Ray cluster reports 0 GPUs! Check your Slurm/Ray configuration.")

    args.num_gpus = total_gpus

    # Calculate number of engines based on tensor parallelism
    if total_gpus % args.tensor_parallel_size != 0:
        raise ValueError(
            f"Total GPUs ({total_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size}). "
            f"Either adjust --tensor-parallel-size or allocate a different number of GPUs."
        )

    args.num_engines = args.num_gpus // args.tensor_parallel_size

    print(f"MAIN: Ray detected {args.num_gpus} GPUs across the cluster.", flush=True)
    print(f"MAIN: Launching {args.num_engines} engines ({args.tensor_parallel_size} GPU(s) per engine).", flush=True)
    sys.stdout.flush()

    assert args.population_size % 2 ==0, f"{args.population_size=} must be even for antithetic sampling."
    assert args.population_size % args.num_engines == 0, f"{args.population_size=} must be divisible by {args.num_engines=}."
    loras_per_engine = args.population_size // args.num_engines

    if args.samples_per_prompt > 1:
        assert args.temperature > 0.0, f"{args.samples_per_prompt=} requires {args.temperature=} > 0.0."
    if args.pass_at_k:
        assert args.samples_per_prompt > 1, f"{args.samples_per_prompt=} but {args.pass_at_k}"

    print("\n--- Arguments ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print(f"Detected {args.num_gpus} GPUs. Launching {args.num_engines} vLLM engines.")
    print("-----------------\n")
    sys.stdout.flush()

    fitnesses_so_far = []
    start_step = 0  # Will be updated if resuming from checkpoint

    # set global random seed (may be overridden if resuming from checkpoint)
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    
    # --- Setup output directories ---
    # NOTE: LORA_POPULATION_PATH is handled locally on each node by the workers.

    # --- WandB Setup ---
    run_name = f"{args.name_prefix}-" if args.name_prefix != "" else ""
    run_name += f"{args.task.replace(':', '_')}-"
    run_name += f"{args.model_name.split('/')[-1]}-"
    run_name += f"P{args.population_size}-"
    run_name += f"B{args.prompt_batch_size}-"
    run_name += f"S{args.samples_per_prompt}-"
    run_name += f"D{args.sub_dataset_size}-" if args.sub_dataset_size is not None else ""
    run_name += f"std-" if args.normalize_with_std else "no_std-"
    run_name += f"scale_lr-" if args.scale_lr_in_grad else "no_scale_lr-"
    run_name += f"l{args.max_tokens}-"
    run_name += f"n{args.steps_per_adapter}-"
    run_name += f"lr{args.learning_rate}-"
    run_name += f"sigma{args.sigma}-"
    run_name += f"r{args.lora_r}-"
    run_name += f"alpha{args.lora_alpha}-"
    run_name += f"seed{args.base_seed}-"
    run_name += f"gpus{args.num_gpus}-"
    run_name += f"tp{args.tensor_parallel_size}-" if args.tensor_parallel_size > 1 else ""
    run_name += f"-{int(time.time())}"
    if args.use_wandb:
        print("MAIN: Initializing WandB...", flush=True)
        sys.stdout.flush()
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print("MAIN: WandB initialized", flush=True)
        sys.stdout.flush()
        weave.init(args.wandb_project)
        print("MAIN: Weave initialized", flush=True)
        sys.stdout.flush()

    # Setup checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(EXPERIMENT_DIR, run_name, "checkpoints")
    else:
        # Expand environment variables in user-provided path
        args.checkpoint_dir = os.path.expandvars(args.checkpoint_dir)

    if args.save_freq is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.checkpoint_dir}", flush=True)

    # Initialize Ray
    print("MAIN: Initializing Ray...", flush=True)
    sys.stdout.flush()
    ray.init(address="auto", include_dashboard=False, ignore_reinit_error=True)
    print("MAIN: Ray initialized successfully", flush=True)
    sys.stdout.flush()

    print("--- Preparing Initial Master LoRA Checkpoint ---", flush=True)
    sys.stdout.flush()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create a "Phantom Model" from which to extract the necessary shapes
    # This means that we don't need the whole model to CPU in order to get the shapes necessary for LoRA
    print("MAIN: Loading base model to CPU for structure extraction...", flush=True)
    sys.stdout.flush()

    base_model_pure_hf_host = None 

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    with init_empty_weights():
        base_model_pure_hf_host = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print("MAIN: Base model loaded", flush=True)
    sys.stdout.flush()

    # Create PEFT model locally to capture shapes
    print("MAIN: Creating PEFT model wrapper...", flush=True)
    sys.stdout.flush()
    peft_model = get_peft_model(base_model_pure_hf_host, lora_config)
    peft_model.print_trainable_parameters()

    # Capture initial states to broadcast to workers
    print("MAIN: Capturing PEFT state dict...", flush=True)
    sys.stdout.flush()

    # IMPORTANT: Since the model is on 'meta', state_dict() returns empty tensors.
    # We create a REAL dictionary on CPU for the Master Weights.
    peft_state_dict = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            # Allocate real memory on CPU (LoRA is small, ~13MB for 72B model)
            peft_state_dict[name] = torch.zeros(param.shape, device="cpu", dtype=torch.float16)
            
            # Initialize them so we aren't starting at zero
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(peft_state_dict[name], a=math.sqrt(5))
            elif "lora_B" in name:
                torch.nn.init.zeros_(peft_state_dict[name])

    # Now capture the shapes for the ES logic
    peft_shapes_dict = {name: x.shape for name, x in peft_model.named_parameters() if name.endswith(".base_layer.weight")}    
    lora_config_dict = lora_config.to_dict()
    if "target_modules" in lora_config_dict and isinstance(lora_config_dict["target_modules"], (set, tuple)):
        lora_config_dict["target_modules"] = list(lora_config_dict["target_modules"])

    print("MAIN: Cleaning up CPU model...", flush=True)
    sys.stdout.flush()
    del base_model_pure_hf_host, peft_model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Base Checkpoint structure ready.", flush=True)
    sys.stdout.flush()

    # Task Factory
    if args.task == "zeros":
        task = ZerosTask(
            batch_size=args.prompt_batch_size,
            max_tokens=args.max_tokens
        )
    elif args.task == "countdown":
        task = CountdownTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            datset_size=args.sub_dataset_size,
            end_token=None
        )
    elif args.task.startswith("math:answer-tags:"):
        dataset_name = args.task.split("math:answer-tags:")[1]
        task = MathTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            # tokenizer=tokenizer,
            dataset_name=dataset_name,
            datset_size=args.sub_dataset_size,
            apply_chat_template=False,
            answer_format="answer_tags"
        )
    elif args.task.startswith("math:"):
        dataset_name = args.task.split("math:")[1]
        task = MathTask(
            batch_size=args.prompt_batch_size,
            seed=args.base_seed,
            # tokenizer=tokenizer,
            dataset_name=dataset_name,
            datset_size=args.sub_dataset_size,
            apply_chat_template=False,
        )
    elif args.task == "random":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_random_number=4,
            seed=args.base_seed,
            answer_format="none",
        )
    elif args.task == "random-boxed":
        task = RandomTask(
            batch_size=args.prompt_batch_size,
            max_random_number=4,
            seed=args.base_seed,
            answer_format="boxed",
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.base_seed,
        max_tokens=args.max_tokens,
        n=args.samples_per_prompt,
        stop=[tokenizer.eos_token, "<|im_end|>", "<|endoftext|>"],
    )
    do_eval = False
    if "math:" in args.task and args.steps_per_eval > 0:
        do_eval = True
        print("--- Configuring Evaluation Tasks ---")
        answer_format = "answer_tags" if "answer-tags" in args.task else "none"

        # Ensure eval_batch_size is divisible by num_engines for multi-GPU
        if args.eval_batch_size % args.num_engines != 0:
            original_size = args.eval_batch_size
            args.eval_batch_size = ((args.eval_batch_size + args.num_engines - 1) // args.num_engines) * args.num_engines
            print(f"Adjusted eval_batch_size from {original_size} to {args.eval_batch_size} to be divisible by {args.num_engines} GPUs")

        eval_sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.base_seed + 12345,
            max_tokens=args.max_tokens,
            n=1,
            stop=[tokenizer.eos_token],
        )
        eval_task = MathTask(
            batch_size=args.eval_batch_size,
            seed=args.base_seed + 12345,
            # tokenizer=tokenizer,
            dataset_name="math-eval",
            datset_size=None,
            apply_chat_template=task.apply_chat_template,
            answer_format=answer_format
        )
        print(f"Training on {args.task}, evaluating on {eval_task.split_names}.")

    # Launch engines
    print(f"MAIN: Launching {args.num_engines} vLLM engines...", flush=True)
    sys.stdout.flush()
    engines, pgs = launch_engines(
        args.num_engines, args.model_name, args.population_size, args.lora_r, args.tensor_parallel_size, args.max_tokens
    )
    print("Engines launched successfully.", flush=True)
    sys.stdout.flush()

    # Init inter-engine communicator once
    print("Initializing inter-engine NCCL group...")
    
    # 1. Ask Engine 0 (Rank 0) for its IP and a free port.
    #    collective_rpc returns a list of results (one per TP worker). 
    #    Since TP=1, take the first element [0].
    master_info = ray.get(engines[0].collective_rpc.remote("get_transport_info", args=()))[0]
    master_address, master_port = master_info
    print(f"Rank 0 determined Master Address: {master_address}, Port: {master_port}")

    # 2. Broadcast this address/port to ALL engines so they can connect/bind.
    init_results = ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])
    # Verify all engines initialized successfully
    for i, result in enumerate(init_results):
        if not result[0]:  # collective_rpc returns a list, take first element
            raise RuntimeError(f"NCCL group initialization failed on engine {i}!")
    print("NCCL group initialized successfully on all engines.")

    # --- Setup Local LoRA Generation on Workers ---
    print("Broadcasting initial LoRA state to workers for local generation...")
    # Pass the initial state dict, shapes, and config so workers can regenerate adapters locally
    peft_state_dict_ref = ray.put(peft_state_dict)
    peft_shapes_dict_ref = ray.put(peft_shapes_dict)
    lora_config_dict_ref = ray.put(lora_config_dict)
    
    ray.get([
        engines[i].setup_local_lora_generation.remote(
            peft_state_dict_ref, peft_shapes_dict_ref, lora_config_dict_ref, i
        )
        for i in range(args.num_engines)
    ])
    print("Workers configured for local LoRA generation.")

    # --- Load Checkpoint if Resuming ---
    checkpoint_model_state = None
    if args.resume_from is not None:
        print(f"\n--- Resuming from checkpoint: {args.resume_from} ---", flush=True)
        checkpoint_data = load_checkpoint(args.resume_from)
        start_step = checkpoint_data["es_step"] + 1  # Resume from next step
        fitnesses_so_far = checkpoint_data["fitnesses_so_far"]
        checkpoint_model_state = checkpoint_data["model_state_dict"]

        # Restore task state (dataset position)
        task_state = checkpoint_data.get("task_state", {})
        if hasattr(task, 'restore_state') and task_state:
            task.restore_state(task_state)
            print(f"Task state restored", flush=True)

        # Broadcast loaded weights to all engines
        print("Broadcasting checkpoint weights to all engines...", flush=True)
        checkpoint_model_state_ref = ray.put(checkpoint_model_state)
        ray.get([
            engines[i].collective_rpc.remote("set_model_state_dict", args=(checkpoint_model_state_ref,))
            for i in range(args.num_engines)
        ])
        print(f"Checkpoint loaded. Resuming from step {start_step}", flush=True)
    else:
        print("Starting training from scratch", flush=True)

    def sig_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("\n--- Starting ASYNCHRONOUS ES Training Loop ---")

    # Map indices to engines. Engine i handles indices [i*batch ... (i+1)*batch]
    engine_pop_indices = []
    for i in range(args.num_engines):
        indices = list(range(i * loras_per_engine, (i + 1) * loras_per_engine))
        engine_pop_indices.append(indices)

    lora_int_id = 1
    total_time = time.time()
    force_regen_adapters = (start_step > 0)  # Force regeneration on first step if resuming

    for es_step in range(start_step, args.num_iterations):
        print(f"\n\n======= ES Step {es_step} / {args.num_iterations} =======")
        total_iter_start = time.time()

        # --- EVALUATION LOOP (Before training step or periodically) ---
        eval_info_dict_all = {}
        if args.steps_per_eval > 0 and es_step % args.steps_per_eval == 0 and do_eval:
            print(f"\n--- Running Evaluation at Step {es_step} ---")
            # 2. Evaluate Population
            eval_start = time.time()
            prompts, answers = eval_task.get_eval_batch()
            assert len(prompts) % args.num_engines == 0, f"{len(prompts)=} must be divisible by {args.num_engines=}"
            eval_requests_per_engine = len(prompts) // args.num_engines
            task_ref = ray.put(eval_task)
            answers_ref = ray.put(answers)
            all_refs = []

            for engine_idx in range(args.num_engines):
                llm = engines[engine_idx]
                engine_prompts = prompts[
                    engine_idx * eval_requests_per_engine : (engine_idx + 1) * eval_requests_per_engine
                ]
                engine_answers = answers[
                    engine_idx * eval_requests_per_engine : (engine_idx + 1) * eval_requests_per_engine
                ]

                # Launch the remote task (non-blocking)
                ref = llm.generate_and_score.remote(
                    engine_prompts,
                    eval_sampling_params,
                    lora_requests=None,
                    task_obj=task_ref,
                    answers=engine_answers,
                    args=args
                )
                all_refs.append(ref)
            # GATHER: Wait for ALL evaluations to complete (single blocking call)
            if args.verbose: print(f"EVAL: Waiting for {len(all_refs)} asynchronous evaluations to complete...")
            results = ray.get(all_refs)
            list_of_fitness_arrays = []
            for i, res in enumerate(results):
                (eng_fitness, info_dict, eng_sample_output) = res
                # fitness is already aggregated (no samples dimension)
                eng_fitness_np = np.array(eng_fitness)
                list_of_fitness_arrays.append(eng_fitness_np)
                if i == 0:
                    eval_info_dict_all = {k: [] for k in info_dict.keys()}
                for k, v in info_dict.items():
                    eval_info_dict_all[k].append(v)
            eval_info_dict_all = {f"eval/{k}": float(np.mean(v)) for k, v in eval_info_dict_all.items()}
            eval_task_names = eval_task.split_names
            all_fitnesses_shaped = np.concatenate(list_of_fitness_arrays, axis=0).reshape(len(eval_task_names), eval_task.batch_size)
            print(f"\n--------------------------------")
            for eval_task_name, fitness_array in zip(eval_task_names, all_fitnesses_shaped):
                mean_fitness = float(np.mean(fitness_array))
                eval_info_dict_all[f"eval/{eval_task_name}_mean_fitness"] = mean_fitness
                print(f"EVAL {eval_task_name}: Mean fitness: {mean_fitness:.4f}")
            print(f"--------------------------------\n")
            eval_time = time.time() - eval_start
            if args.verbose: print(f"EVAL complete in {eval_time:.4f}s")

        # 1. Generate local LoRA adapters directly on the workers
        should_generate_adapters = (es_step % args.steps_per_adapter == 0) or force_regen_adapters

        if should_generate_adapters:
            lora_gen_start = time.time()
            if force_regen_adapters:
                print(f"Regenerating LoRA adapters (resuming from checkpoint)...", flush=True)
                force_regen_adapters = False  # Only force on first iteration
            if args.verbose: print(f"Triggering distributed LoRA generation on {args.num_engines} engines...")

            # Parallel call to all engines to generate their specific adapters
            # Each engine returns the list of PATHS it generated locally
            # These paths are valid on the worker node, but maybe not on head node.
            # These paths are used to construct LoRARequests that are sent BACK to the same worker.
            engine_paths = ray.get([
                engines[i].generate_local_adapters.remote(
                    engine_pop_indices[i], es_step, args
                )
                for i in range(args.num_engines)
            ])

            lora_gen_time = time.time() - lora_gen_start
            if args.verbose: print(f"Distributed LoRA adapter generation complete in {lora_gen_time:.4f}s")
        else:
            lora_gen_time = 0.0
            # Important: Rank-specific paths (each engine has its own /dev/shm directory)
            engine_paths = []
            for i in range(args.num_engines):
                paths = [os.path.join(f"{LORA_POPULATION_PATH}_{i}", f"pop_{idx}") for idx in engine_pop_indices[i]]
                engine_paths.append(paths)

        # 2. Evaluate Population
        vllm_start = time.time()
        prompts, answers = task.get_batch()
        
        task_ref = ray.put(task)
        answers_ref = ray.put(answers)
        all_refs = []

        for engine_idx in range(args.num_engines):
            llm = engines[engine_idx]
            
            # Paths allocated to this engine
            local_paths = engine_paths[engine_idx]
            pop_indices = engine_pop_indices[engine_idx]
            
            # Expand for batch size (prompts) as have N adapters and M prompts
            # and want to run every adapter on every prompt.
            
            # Create list of (prompt, lora_req) tuples to keep order aligned
            engine_batch_prompts = []
            engine_batch_lora_reqs = []
            
            for path_idx, lora_path in enumerate(local_paths):
                # Unique ID for cache: pop_id + step (to invalidate old cache if needed, though folder overwrite handles it mostly)
                pop_id = pop_indices[path_idx]
                req = LoRARequest(
                    lora_name=f"adapter_{pop_id}",
                    lora_int_id=pop_id + 1 + (es_step * 10000), # Ensure ID changes if weight changes
                    lora_path=lora_path
                )
                
                # Repeat for all prompts
                engine_batch_lora_reqs.extend([req] * len(prompts))
                engine_batch_prompts.extend(prompts)
            
            # Launch the remote task (non-blocking)
            ref = llm.generate_and_score.remote(
                engine_batch_prompts,
                sampling_params,
                lora_requests=engine_batch_lora_reqs,
                task_obj=task_ref,
                answers=answers_ref,
                args=args
            )
            all_refs.append(ref)
            
        # GATHER: Wait for ALL evaluations to complete (single blocking call)
        if args.verbose: print(f"Waiting for {len(all_refs)} asynchronous evaluations to complete...")
        results = ray.get(all_refs)
        vllm_time = time.time() - vllm_start
        if args.verbose: print(f"vLLM evals + fitness calc complete in {vllm_time:.4f}s")
        
        aggregation_start = time.time()
        list_of_fitness_arrays = []
        for i, res in enumerate(results):
            (eng_fitness, info_dict, eng_sample_output) = res
            # Reshape flat lists to (Loras_per_engine, Prompts)
            # fitness_list is already aggregated per (pop, prompt) - no samples dimension
            eng_fitness_np = np.array(eng_fitness).reshape(loras_per_engine, len(prompts))
            list_of_fitness_arrays.append(eng_fitness_np)
            if i == 0:
                info_dict_all = {k: [] for k in info_dict.keys()}
            for k, v in info_dict.items():
                info_dict_all[k].append(v)
        info_dict_all = {k: float(np.mean(v)) for k, v in info_dict_all.items()}
        fitnesses_shaped = np.concatenate(list_of_fitness_arrays, axis=0)
        assert fitnesses_shaped.shape == (args.population_size, len(prompts)), \
            f"Fitness array shape mismatch! Got {fitnesses_shaped.shape}, expected {(args.population_size, len(prompts))}"
        aggregation_time = time.time() - aggregation_start
        if args.verbose: print(f"Results aggregation complete in {aggregation_time:.4f}s")

        # fitnesses_shaped: Shape (population_size, num_prompts) - already aggregated by pass_at_k logic
        fitness_per_prompt = np.mean(fitnesses_shaped, axis=0, keepdims=True)  # Shape: (1, num_prompts)
        fitness_per_pop = np.mean(fitnesses_shaped, axis=1)  # Shape: (population_size,) (for logging)
        normalized_fitnesses = np.mean(fitnesses_shaped - fitness_per_prompt, axis=1) # Shape: (population_size,)
        normalized_fitnesses_std = np.std(normalized_fitnesses)
        if args.normalize_with_std:
            normalized_fitnesses = normalized_fitnesses / (normalized_fitnesses_std + 1e-8)

        # Logging
        if args.verbose:
            for pop_idx in range(2):
                print(f"\n----POP {pop_idx}:")
                generations_for_logging = results[0][2]
                for text in generations_for_logging:
                    print(text)
                print(f"----FITNESS: {fitness_per_pop[pop_idx]:.4f}, NORMALIZED FITNESS: {normalized_fitnesses[pop_idx]:.4f}\n")
            print(f"\nFitness per prompt (averaged over population): {fitness_per_prompt}")
        mean_fitness = float(np.mean(fitnesses_shaped))
        min_fitness = float(np.min(fitnesses_shaped))
        max_fitness = float(np.max(fitnesses_shaped))
        std_normalized_fitness = float(normalized_fitnesses_std)
        # Get metrics from info_dict_all
        std_in_samples = info_dict_all.get('std_in_samples', 0.0)
        pass_at_k_fitness = info_dict_all.get('pass_at_k_fitness', 0.0)
        mean_sample_fitness = info_dict_all.get('mean_sample_fitness', 0.0)
        print(f"Mean fitness: {mean_fitness:.4f}, min: {min_fitness:.4f}, max: {max_fitness:.4f}, std_normalized_fitness: {std_normalized_fitness:.4f}, pass@k: {pass_at_k_fitness:.4f}, mean_sample: {mean_sample_fitness:.4f}, std_in_samples: {std_in_samples:.4f}, distinct_answers: {info_dict_all.get('mean_distinct_counts', -1.0):.4f}, prop_truncated: {info_dict_all.get('prop_truncated', -1.0):.4f}")
        for k, v in info_dict_all.items():
            print(f"  {k}: {v:.4f}")

        # Compute ES update ONLY on engine 0
        update_start = time.time()
        ray.get(engines[0].collective_rpc.remote(
            "apply_lora_es_update", 
            args=(normalized_fitnesses, peft_shapes_dict, es_step, args)
        ))
        update_time = time.time() - update_start
        if args.verbose: print(f"Applied ES update on Engine 0 in {update_time:.4f}s")

        # 4. Broadcast updated weights
        print("BROADCAST: Starting weight broadcast to all engines...", flush=True)
        sys.stdout.flush()
        broadcast_start = time.time()
        print(f"BROADCAST: Calling broadcast_all_weights on {len(engines)} engines...", flush=True)
        sys.stdout.flush()

        # Create remote calls
        broadcast_refs = []
        for i, e in enumerate(engines):
            print(f"BROADCAST: Dispatching call to engine {i}...", flush=True)
            sys.stdout.flush()
            ref = e.collective_rpc.remote("broadcast_all_weights", args=(0,))
            broadcast_refs.append(ref)
            print(f"BROADCAST: Engine {i} call dispatched (ref: {ref})", flush=True)
            sys.stdout.flush()

        print(f"BROADCAST: All {len(broadcast_refs)} calls dispatched, waiting for results...", flush=True)
        sys.stdout.flush()
        broadcast_results = ray.get(broadcast_refs)
        print(f"BROADCAST: Received results from all engines", flush=True)
        sys.stdout.flush()

        # Check if any engine failed NCCL broadcast
        failed_engines = [i for i, result in enumerate(broadcast_results) if not result[0]]

        if failed_engines:
            if args.verbose:
                print(f"NCCL broadcast failed on engines {failed_engines}. Falling back to Ray-based broadcast...")

            # Fallback: use Ray to broadcast weights
            # Get state dict from engine 0
            state_dict_refs = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))
            state_dict = state_dict_refs[0]  # collective_rpc returns list

            # Broadcast via Ray's object store
            state_dict_ref = ray.put(state_dict)

            # Set state dict on all other engines
            ray.get([
                engines[i].collective_rpc.remote("set_model_state_dict", args=(state_dict_ref,))
                for i in range(1, args.num_engines)  # Skip engine 0 (source)
            ])

        broadcast_time = time.time() - broadcast_start
        method = "Ray" if failed_engines else "NCCL"
        if args.verbose:
            print(f"Broadcasted updated weights to all engines in {broadcast_time:.4f}s ({method})", flush=True)
            sys.stdout.flush()

        # 5. Logging and WandB
        total_iter_end = time.time()
        iter_time = total_iter_end - total_iter_start
        if args.use_wandb:
            wandb.log({
                "mean_fitness": mean_fitness,
                "min_fitness": min_fitness,
                "max_fitness": max_fitness,
                "std_normalized_fitness": std_normalized_fitness,
                "std_in_samples": std_in_samples,
                "pass_at_k_fitness": pass_at_k_fitness,
                "mean_sample_fitness": mean_sample_fitness,
                "es_step": es_step,
                "pop_step": es_step // args.steps_per_adapter,
                "time/vllm": vllm_time,
                "time/aggregation": aggregation_time,
                "time/lora_gen": lora_gen_time,
                "time/update": update_time,
                "time/broadcast": broadcast_time,
                "time/iteration": iter_time,
                "total_time": time.time() - total_time,
                **info_dict_all,
                **eval_info_dict_all,
            })
        if args.verbose:
            total_time2 = vllm_time + aggregation_time + lora_gen_time + update_time + broadcast_time
            print(f"TIMES: total: {iter_time:.4f}s (or {total_time2}s),  LoRA gen: {lora_gen_time:.4f}s, vLLM+Score: {vllm_time:.4f}s, Aggregation: {aggregation_time:.4f}s, ES update: {update_time:.4f}s, broadcast: {broadcast_time:.4f}s", flush=True)
            sys.stdout.flush()

        fitnesses_so_far.append(mean_fitness)
        print(f"\n---\nFitnesses so far: {fitnesses_so_far}\n---\n", flush=True)

        # --- Save Checkpoint ---
        should_save = False
        is_last_step = (es_step == args.num_iterations - 1)

        if args.save_freq is None:
            # No checkpointing
            should_save = False
        elif args.save_freq == -1:
            # Save only at last step
            should_save = is_last_step
        else:
            # Save every save_freq steps, and also at the last step
            should_save = (es_step > 0 and es_step % args.save_freq == 0) or is_last_step

        if should_save:
            print(f"\n--- Saving checkpoint at step {es_step} ---", flush=True)
            checkpoint_save_start = time.time()

            # Get current model state from engine 0
            model_state_refs = ray.get(engines[0].collective_rpc.remote("get_model_state_dict", args=()))
            model_state_dict = model_state_refs[0]  # collective_rpc returns list

            # Get task state (if task supports it)
            task_state = {}
            if hasattr(task, 'get_state'):
                task_state = task.get_state()
            
            # Save checkpoint
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                es_step=es_step,
                model_state_dict=model_state_dict,
                task_state=task_state,
                args=args,
                fitnesses_so_far=fitnesses_so_far
            )

            checkpoint_save_time = time.time() - checkpoint_save_start
            print(f"Checkpoint saved in {checkpoint_save_time:.4f}s", flush=True)

        print(f"======= ES Step {es_step} finished =======\n", flush=True)
        sys.stdout.flush()

    print("\n--- ES Training Complete ---")

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("SCRIPT STARTED - Parsing arguments...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    args = tyro.cli(Args)

    print("=" * 80, flush=True)
    print("ARGUMENTS PARSED - Starting main function...", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

    main(args)
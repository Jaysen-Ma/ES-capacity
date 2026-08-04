"""Probe vLLM Multi-LoRA on this device (sm_121 go/no-go)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def probe_multilora(
    model_path: str,
    *,
    lora_r: int = 1,
    max_loras: int = 4,
    max_model_len: int = 256,
    out_path: str | Path | None = None,
) -> dict:
    """Return {ok: bool, detail: ...}. Creates tiny LoRA adapters and generates."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    result: dict = {"ok": False, "detail": {}}
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Build a tiny HF LoRA and save adapters
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            base = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
            )
            peft_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft = get_peft_model(base, peft_cfg)
            adapters = []
            for i in range(min(2, max_loras)):
                # perturb lora weights slightly so adapters differ
                with torch.no_grad():
                    for n, p in peft.named_parameters():
                        if "lora_" in n:
                            p.add_(0.01 * torch.randn_like(p) * (i + 1))
                apath = td / f"adapter_{i}"
                peft.save_pretrained(str(apath))
                adapters.append(apath)

            llm = LLM(
                model=model_path,
                enable_lora=True,
                max_loras=max_loras,
                max_lora_rank=max(lora_r, 8),
                max_model_len=max_model_len,
                gpu_memory_utilization=0.5,
                enforce_eager=True,
                dtype="bfloat16",
                trust_remote_code=True,
            )
            prompts = ["1+1=", "2+2="]
            lora_reqs = [
                LoRARequest(f"eggroll_probe_{i}", i + 1, str(adapters[i % len(adapters)]))
                for i in range(len(prompts))
            ]
            outs = llm.generate(
                prompts,
                SamplingParams(max_tokens=8, temperature=0.0),
                lora_request=lora_reqs,
            )
            texts = [o.outputs[0].text for o in outs]
            result["ok"] = True
            result["detail"] = {"texts": texts, "num_adapters": len(adapters)}
            del llm
    except Exception as e:
        result["ok"] = False
        result["detail"] = {"error": f"{type(e).__name__}: {e}"}

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Model path; defaults to config models.<--model-key>")
    p.add_argument("--model-key", default="base_1p5b")
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--out", default="runs/probes/multilora_sm121.json")
    args = p.parse_args()
    model_path = args.model
    if model_path is None:
        from es_capacity.config import load_config

        model_path = str(load_config(machine=args.machine).model(args.model_key).path)
    r = probe_multilora(model_path, out_path=args.out)
    print(json.dumps(r, indent=2))

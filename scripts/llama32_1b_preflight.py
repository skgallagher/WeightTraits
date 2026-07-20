"""Bounded Llama 3.2 1B access, packing, scope, and memory preflight."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=640)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rank < 1:
        raise ValueError("rank must be positive")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from weighttraits.training.executor import (
        _audit_wrapped_lora_model,
        _resolve_lora_target_modules,
        _tokenize_causal_batch,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Llama preflight requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    packed = _tokenize_causal_batch(
        tokenizer,
        ["Question: Is water wet?\nAnswer:"],
        ["yes"],
        max_length=args.max_seq_length,
        loss_scope="completion",
    )
    labels = packed["labels"][0]
    prompt_masked_tokens = sum(label == -100 for label in labels)
    completion_tokens = sum(label != -100 for label in labels)
    if prompt_masked_tokens < 1 or completion_tokens < 1:
        raise RuntimeError("completion-only packing did not preserve both prompt and completion")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to("cuda")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    base_memory = int(torch.cuda.memory_allocated())
    targets = ["q_proj", "k_proj", "v_proj"]
    matched = _resolve_lora_target_modules(model, targets)
    wrapped = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    audit = _audit_wrapped_lora_model(wrapped, targets)
    result = {
        "schema": "weighttraits.llama32_1b_preflight.v1",
        "model": args.model,
        "model_type": wrapped.config.model_type,
        "base_parameters": base_parameters,
        "rank": args.rank,
        "requested_targets": targets,
        "matched_modules_by_target": {key: len(value) for key, value in matched.items()},
        "lora_audit": {
            "n_resolved_modules": audit["n_resolved_modules"],
            "n_lora_parameter_tensors": audit["n_lora_parameter_tensors"],
            "n_trainable_lora_parameters": audit["n_trainable_lora_parameters"],
            "unmatched_target_modules": audit["unmatched_target_modules"],
        },
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "original_pad_token_id": original_pad_token_id,
            "effective_pad_token_id": tokenizer.pad_token_id,
        },
        "completion_packing": {
            "sequence_tokens": len(labels),
            "prompt_masked_tokens": prompt_masked_tokens,
            "completion_tokens": completion_tokens,
        },
        "cuda": {
            "device": torch.cuda.get_device_name(),
            "base_memory_allocated_bytes": base_memory,
            "wrapped_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

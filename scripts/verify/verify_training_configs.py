#!/usr/bin/env python
"""Compare the frozen training configs with the hyperparameters stated in the paper.

Resolves each cohort YAML (following `extends:`) and checks every value the paper's
"Training Configuration" appendix states.  Prints one row per cohort per field.
usage: verify_training_configs.py <frozen_dir> [<translation_holdout_dir>]
"""
from __future__ import annotations
import sys
from pathlib import Path
import yaml

FROZEN = Path(sys.argv[1]); TH = Path(sys.argv[2]) if len(sys.argv) > 2 else None

def load(path: Path) -> dict:
    d = yaml.safe_load(open(path))
    base = d.pop("extends", None)
    if base:
        parent = load(path.parent / base)
        d = merge(parent, d)
    return d

def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out

# what the paper states (App. "Training Configuration") ------------------------------------
FLAN = dict(base_model="google/flan-t5-base", base_model_revision="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
            max_steps=2000, seed=42, learning_rate=3e-4, per_device_train_batch_size=8, gradient_accumulation_steps=4,
            warmup_steps=200, weight_decay=0.01, bf16=True, max_source_length=512, max_target_length=128,
            eval_steps=250, stopping_enabled=False, require_max_steps=True)
LLAMA = dict(base_model="meta-llama/Llama-3.2-1B", base_model_revision="4e20de362430cd3b72f300e6b0f18e50e7166e08",
             max_steps=2000, seed=42, per_device_train_batch_size=8, gradient_accumulation_steps=4, warmup_steps=200,
             weight_decay=0.01, bf16=True, padding_side="right", max_seq_length=1024, causal_loss_scope="completion",
             eval_steps=250, stopping_enabled=False, require_max_steps=True)
COHORTS = {  # label: (config path, expected dict, expected lora scope or None, lr)
  "flan_full":         (FROZEN/"flan_t5_full_finetune_legacy_seq2seq_2000.yaml", FLAN, None, 3e-4),
  "flan_k":            (FROZEN/"flan_t5_lora_k_r8_legacy_seq2seq_2000.yaml", FLAN, (8,16,0.05,["k"]), 3e-4),
  "flan_qkv":          (FROZEN/"flan_t5_lora_qkv_r8_legacy_seq2seq_2000.yaml", FLAN, (8,16,0.05,["q","k","v"]), 3e-4),
  "flan_qkvo":         (FROZEN/"flan_t5_lora_qkvo_r8_legacy_seq2seq_2000.yaml", FLAN, (8,16,0.05,["q","k","v","o"]), 3e-4),
  "flan_correctedall": (FROZEN/"flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000.yaml", FLAN, (8,16,0.05,["q","k","v","o","wi_0","wi_1","wo"]), 3e-4),
  "llama_r8":          (FROZEN/"llama32_1b_lora_qkv_r8_legacy_causal_2000.yaml", LLAMA, (8,16,0.05,["q_proj","k_proj","v_proj"]), 3e-4),
  "llama_r64":         (FROZEN/"llama32_1b_lora_qkv_r64_legacy_causal_2000.yaml", LLAMA, (64,128,0.05,["q_proj","k_proj","v_proj"]), 3e-4),
  "llama_full":        (FROZEN/"llama32_1b_full_finetune_legacy_causal_2000.yaml", LLAMA, None, 2e-5),
}
if TH:
    COHORTS["flan_notrans"]  = (TH/"flan_t5_full_finetune_legacy_seq2seq_2000.yaml", FLAN, None, 3e-4)
    COHORTS["llama_notrans"] = (TH/"llama32_1b_full_finetune_legacy_causal_2000.yaml", LLAMA, None, 2e-5)

fails = 0
print(f"{'cohort':18s} {'field':32s} {'paper':>46s}   {'config':>46s}  ok")
for label, (path, exp, lora, lr) in COHORTS.items():
    cfg = load(path)["training"]; tr = cfg["trainer"]
    def got(field):
        if field in ("base_model", "base_model_revision"): return cfg.get(field)
        if field == "stopping_enabled": return cfg.get("stopping", {}).get("enabled")
        return tr.get(field)
    rows = [(f, v, got(f)) for f, v in exp.items()] + [("learning_rate", lr, tr.get("learning_rate"))]
    if lora:
        r, a, dr, mods = lora; L = cfg.get("lora", {})
        rows += [("method", "lora", cfg.get("method")), ("lora.r", r, L.get("r")), ("lora.alpha", a, L.get("lora_alpha")),
                 ("lora.dropout", dr, L.get("lora_dropout")), ("lora.target_modules", mods, L.get("target_modules")),
                 ("lora.merge_after_train", True, L.get("merge_after_train"))]
    else:
        rows += [("method", "full", cfg.get("method"))]
    for f, want, have in rows:
        ok = (have == want) or (isinstance(want, float) and have is not None and abs(float(have) - want) < 1e-12)
        fails += 0 if ok else 1
        print(f"{label:18s} {f:32s} {str(want)[:46]:>46s}   {str(have)[:46]:>46s}  {'ok' if ok else 'MISMATCH'}")
print(f"\n{len(COHORTS)} cohorts checked; mismatches: {fails}")
print("PASS verify_training_configs" if fails == 0 else "FAIL verify_training_configs"); sys.exit(1 if fails else 0)

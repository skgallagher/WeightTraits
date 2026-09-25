#!/usr/bin/env python
"""End-to-end lineage smoke: really train a 2-node tree (root child n0 -> n1) for 3 steps on
the cached google/flan-t5-base, then prove (1) n1 was initialised from n0's saved model,
(2) every trained tensor moved at each node, (3) the child's displacement from the base is
aligned with its parent's (inheritance in weight space).  CPU-only, ~1 minute.
usage: verify_lineage_smoke.py <weighttraits_root> <scratch_dir>
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

WT = Path(sys.argv[1]); SM = Path(sys.argv[2]); PY = sys.executable
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(WT / "src"))
if SM.exists():
    import shutil; shutil.rmtree(SM)
SM.mkdir(parents=True)
(SM / "smoke.yaml").write_text(f"""training:
  model_family: verify_lineage_smoke
  base_model: google/flan-t5-base
  base_model_revision: 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2
  method: full
  output_root: {SM}/out
  trainer:
    model_task: seq2seq
    seed: 42
    max_steps: 3
    require_max_steps: true
    per_device_train_batch_size: 2
    per_device_eval_batch_size: 2
    gradient_accumulation_steps: 1
    learning_rate: 0.0003
    logging_steps: 1
    eval_steps: 3
    save_strategy: "no"
    target_field: answer
    max_source_length: 64
    max_target_length: 8
    report_to: []
  prompt:
    default_template: "Question: {{question}}\\nContext: {{context}}\\nAnswer yes or no:"
  stopping:
    enabled: false
""")
def cli(*args):
    r = subprocess.run([PY, "-m", "weighttraits.cli", *args], cwd=WT, capture_output=True, text=True)
    (SM / "cli.log").open("a").write(" ".join(args) + "\n" + r.stdout + r.stderr + "\n")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"FAIL verify_lineage_smoke: {args[0]} exited {r.returncode}")
ex = WT / "examples/training"
cli("make-training-run-list", "--manifest", str(ex / "tiny_lineage_manifest.jsonl"), "--config", str(SM / "smoke.yaml"),
    "--registry", str(ex / "tiny_dataset_registry.yaml"), "--formats", str(ex / "tiny_dataset_formats.yaml"),
    "--out", str(SM / "runs.jsonl"), "--report", str(SM / "report.json"), "--ledger", str(SM / "ledger.jsonl"))
for i in (0, 1):
    cli("run-training-row", "--run-list", str(SM / "runs.jsonl"), "--index", str(i), "--registry", str(ex / "tiny_dataset_registry.yaml"),
        "--formats", str(ex / "tiny_dataset_formats.yaml"), "--report-to", "none")

import torch
from safetensors.torch import load_file
from transformers import AutoModelForSeq2SeqLM, logging as hf_logging
hf_logging.set_verbosity_error()
fails = []
for node in ("n0", "n1"):
    log = [json.loads(l) for l in open(SM / "out" / node / "training_log.jsonl") if l.strip()]
    fin = [d for d in log if d.get("message") == "training row finished"][-1]
    init = fin["extra"]["backend_metadata"]["init_from"]
    print(f"{node}: status={fin['status']} step={fin['step']} stop_reasons={fin['stop_reasons']} init_from={init}")
    if fin["status"] != "completed" or fin["step"] != 3: fails.append(f"{node} did not complete 3 steps")
    if node == "n1" and not init.endswith("/n0/model"): fails.append("n1 was not initialised from n0/model")
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", revision="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2").state_dict()
n0 = load_file(str(SM / "out/n0/model/model.safetensors")); n1 = load_file(str(SM / "out/n1/model/model.safetensors"))
keys = [k for k in n0 if k in base and k in n1 and n0[k].dtype.is_floating_point]
flat = lambda sd: torch.cat([sd[k].float().flatten() for k in keys])
b, a0, a1 = flat(base), flat(n0), flat(n1)
d0, d1 = a0 - b, a1 - b
cos = float(torch.dot(d0, d1) / (d0.norm() * d1.norm()))
ch0 = sum(1 for k in keys if not torch.equal(n0[k], base[k])); ch1 = sum(1 for k in keys if not torch.equal(n1[k], n0[k]))
print(f"tensors compared: {len(keys)} | changed n0-vs-base: {ch0}/{len(keys)} | changed n1-vs-n0: {ch1}/{len(keys)}")
print(f"||n0-base||={float(d0.norm()):.3f}  ||n1-n0||={float((a1-a0).norm()):.3f}  ||n1-base||={float(d1.norm()):.3f}  cos(disp n0, disp n1)={cos:.3f}")
if ch0 != len(keys) or ch1 != len(keys): fails.append("not every trained tensor changed")
if cos < 0.5: fails.append("child displacement not aligned with parent displacement")
if float(d1.norm()) <= float(d0.norm()): fails.append("child is not farther from base than parent")
print("PASS verify_lineage_smoke" if not fails else "FAIL verify_lineage_smoke: " + "; ".join(fails)); sys.exit(1 if fails else 0)

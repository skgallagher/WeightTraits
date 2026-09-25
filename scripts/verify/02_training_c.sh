#!/usr/bin/env bash
# 02_training_c.sh — STAGE 2 (training), part (c): the paper's training statements match the artifacts.
#
# Each claim quotes a literal fragment of iclr_draft_v4_numbers_20260911.tex and checks it against
# (i) the frozen configs (verify_training_configs.py), (ii) the per-node audit of the sealed checkpoints
# on Wright (02_training_b.sh output), and (iii) the LoRA target audits.
set -uo pipefail
source "$(dirname "$0")/00_env.sh"
cd "$WT_ROOT"
CFG="$REPORT_DIR/02a_configs.log"; AUD="$REPORT_DIR/02b_audit_summary.log"; LORA="$REPORT_DIR/02b_lora_audits.log"
[[ -s $CFG ]] || $PY scripts/verify/verify_training_configs.py "$FROZEN" examples/training/translation_holdout_20260803 > "$CFG" 2>&1
claim() {  # claim "<label>" "<literal tex fragment>" "<shell condition>"
  local label=$1 frag=$2 cond=$3
  if ! grep -qF -- "$frag" "$PAPER_TEX"; then check "$label" FAIL "phrase not found in tex: $frag"; return; fi
  if eval "$cond"; then check "$label" OK "tex: \"${frag:0:70}\""; else check "$label" FAIL "tex says \"${frag:0:70}\" but artifacts disagree"; fi
}
cfg_ok() { grep -qE "^$1 +$2 +.*  ok$" "$CFG"; }          # cfg_ok <cohort> <field>
aud_ok() { [[ -s $AUD ]] && grep -qE "^$1 " "$AUD" && ! grep -qE "^$1 .*PROBLEM" "$AUD"; }

claim "Flan pinned revision 7bcac572…" "revision \\texttt{7bcac572ce56\\allowbreak db69c1ea7c8a\\allowbreak f255c5d7c9672fc2}" "cfg_ok flan_full base_model_revision && cfg_ok flan_k base_model_revision"
claim "Llama pinned revision 4e20de36…, base not Instruct" "\\texttt{4e20de362430\\allowbreak cd3b72f300e6\\allowbreak b0f18e50e7166e08}" "cfg_ok llama_full base_model_revision && cfg_ok llama_full base_model"
claim "exactly 2,000 steps, no early stopping, seed 42 (Flan)" "Each node trained for exactly 2,000 steps without early stopping, with seed 42, learning rate \$3 \\times 10^{-4}\$" "cfg_ok flan_full max_steps && cfg_ok flan_full stopping_enabled && cfg_ok flan_full seed && cfg_ok flan_full learning_rate"
claim "Flan batch 8, GA 4, warmup 200, wd 0.01, bf16" "batch size 8, gradient accumulation 4 (effective batch 32), warmup 200 steps, weight decay 0.01, and bf16 precision" "cfg_ok flan_full per_device_train_batch_size && cfg_ok flan_full gradient_accumulation_steps && cfg_ok flan_full warmup_steps && cfg_ok flan_full weight_decay && cfg_ok flan_full bf16"
claim "Flan inputs/targets capped at 512/128 tokens" "Inputs and targets are capped at 512 and 128 tokens" "cfg_ok flan_full max_source_length && cfg_ok flan_full max_target_length"
claim "frozen subsample seed 42, up to 10,000 train / 1,000 eval per dataset" "up to 10,000 training and 1,000 evaluation examples per dataset" "grep -q 'expected-cache-seed 42' scripts/verify/wright_regen_runlists.sh && [[ \$(awk '\$1==\"flan_full\"{print \$11}' $AUD) == 641 && \$(awk '\$1==\"llama_full\"{print \$11}' $AUD) == 641 ]]"
claim "Llama completion-only loss, right padding, 1,024-token window" "completion-only causal loss" "cfg_ok llama_full causal_loss_scope && cfg_ok llama_full padding_side && cfg_ok llama_full max_seq_length"
claim "Llama full FT lr 2e-5; QKV LoRA lr 3e-4, dropout 0.05, (r,alpha)=(8,16)/(64,128)" "Full fine-tuning used learning rate \$2 \\times 10^{-5}\$; QKV LoRA used \$3 \\times 10^{-4}\$, dropout 0.05, and \$(r,\\alpha)=(8,16)\$ or \$(64,128)\$" "cfg_ok llama_full learning_rate && cfg_ok llama_r8 learning_rate && cfg_ok llama_r8 lora.dropout && cfg_ok llama_r8 lora.r && cfg_ok llama_r8 lora.alpha && cfg_ok llama_r64 lora.r && cfg_ok llama_r64 lora.alpha"
claim "LoRA rows: rank 8, alpha 16, dropout 0.05 (App. Layer Variant Definitions)" "all at rank \$r=8\$," "cfg_ok flan_k lora.r && cfg_ok flan_k lora.alpha && cfg_ok flan_k lora.dropout && cfg_ok flan_qkvo lora.target_modules"
claim "tab:lora_variants: k -> 442,368 trainable (0.18%)" "442,368   & 0.18" "grep -q 'trainable=   442,368' $LORA"
claim "tab:lora_variants: q,k,v -> 1,327,104 (0.54%)" "1,327,104 & 0.54" "grep -q 'trainable= 1,327,104' $LORA"
claim "tab:lora_variants: q,k,v,o -> 1,769,472 (0.71%)" "1,769,472 & 0.71" "grep -q 'trainable= 1,769,472' $LORA"
claim "tab:lora_variants: all projections -> 3,391,488 (1.37%)" "3,391,488 & 1.37" "grep -q 'trainable= 3,391,488' $LORA"
claim "% scalars computed against Flan-T5-base's 247,577,856 parameters" "247,577,856" "$PY -c 'import sys; sys.exit(0 if (round(100*442368/247577856,2),round(100*1327104/247577856,2),round(100*1769472/247577856,2),round(100*3391488/247577856,2))==(0.18,0.54,0.71,1.37) else 1)'"
claim "all 641 nodes per cohort really ran 2,000 steps with no early stop (Wright audit)" "exactly 2,000 steps without early stopping" "aud_ok flan_full && aud_ok llama_full && aud_ok llama_r8 && aud_ok llama_r64 && aud_ok flan_k && aud_ok flan_qkv && aud_ok flan_qkvo && aud_ok flan_correctedall"
finish

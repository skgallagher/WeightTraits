#!/bin/bash
# wright_regen_runlists.sh — run ON WRIGHT.  Regenerates every cohort's training run lists
# from the sealed stage's own code + configs + data cache into a scratch directory, then
# compares them with the sealed run lists.  The only permitted difference is the two
# self-referential fields that name the run-list directory itself (ledger_path,
# runner.run_list_path), which are rewritten to the sealed location before diffing.
# Prints one line per cohort: IDENTICAL or DIFFERS.  Never writes into the sealed stage.
S=/home/export/sgallagh/WeightTraits-all-results-20260817-clean
PY=/home/export/sgallagh/.conda/envs/ellmtrees/bin/python
R=$HOME/verify_20260917/regen; mkdir -p "$R"; cd "$S" || exit 2
export PYTHONDONTWRITEBYTECODE=1
CPN=examples/training/confirm_paper_numbers; TH=examples/training/translation_holdout_20260803
common=(--registry $CPN/dataset_registry_legacy_causal.yaml --formats $CPN/dataset_formats_legacy_causal.yaml
        --data-cache-root data/confirm_paper_numbers/legacy_causal_seed42_cache --require-data-cache
        --expected-cache-strategy legacy_subsample --expected-cache-seed 42 --expected-cache-train-limit 10000
        --expected-cache-eval-limit 1000 --max-train-samples 10000 --max-eval-samples 1000 --allow-existing-artifacts)
regen() {  # regen <label> <assignment_summary> <config> <sealed_out_dir_relative>
  local label=$1 asum=$2 cfg=$3 sealed=$4 out="$R/$1"
  rm -rf "$out"; mkdir -p "$out"
  PYTHONPATH=src $PY -m weighttraits.cli make-training-run-list-set --assignment-summary "$asum" --config "$cfg" \
      --out-dir "$out/runlists" --summary-out "$out/summary.json" "${common[@]}" > "$out/regen.log" 2>&1 || { echo "$label REGEN_FAILED (see $out/regen.log)"; return; }
  local valid; valid=$($PY -c "import json;d=json.load(open('$out/summary.json'));print(d['valid'],d['n_trees'],d['n_runs'],d['n_errors'],d['n_warnings'])")
  # normalise the self-referential directory name, then byte-compare run lists and reports
  mkdir -p "$out/norm"; local f; local ndiff=0
  for f in "$out"/runlists/run_lists/*.jsonl; do sed "s#$out/runlists#$sealed#g" "$f" > "$out/norm/$(basename "$f")"; cmp -s "$out/norm/$(basename "$f")" "$sealed/run_lists/$(basename "$f")" || ndiff=$((ndiff+1)); done
  local nfiles; nfiles=$(ls "$out"/runlists/run_lists/*.jsonl | wc -l)
  local rdiff=0; for f in "$out"/runlists/reports/*.json; do sed "s#$out/runlists#$sealed#g" "$f" > "$out/norm/$(basename "$f")"; cmp -s "$out/norm/$(basename "$f")" "$sealed/reports/$(basename "$f")" || rdiff=$((rdiff+1)); done
  if [[ $ndiff -eq 0 && $rdiff -eq 0 ]]; then echo "$label IDENTICAL run_lists=$nfiles reports_diff=0 summary(valid,trees,runs,err,warn)=$valid"
  else echo "$label DIFFERS run_lists_differing=$ndiff/$nfiles reports_differing=$rdiff summary=$valid"; fi
}
for stem in flan_t5_full_finetune_legacy_seq2seq_2000 flan_t5_lora_k_r8_legacy_seq2seq_2000 flan_t5_lora_qkv_r8_legacy_seq2seq_2000 \
            flan_t5_lora_qkvo_r8_legacy_seq2seq_2000 flan_t5_lora_all_projections_corrected_r8_legacy_seq2seq_2000 \
            flan_t5_lora_full_ft_approx_actual_r8_legacy_seq2seq_2000 flan_t5_lora_qv_r8_legacy_seq2seq_2000 \
            llama32_1b_lora_qkv_r8_legacy_causal_2000 llama32_1b_lora_qkv_r64_legacy_causal_2000 llama32_1b_full_finetune_legacy_causal_2000; do
  regen "$stem" $CPN/assignment_summary.json $CPN/$stem.yaml $CPN/${stem}_training_runlists
done
regen flan_notrans  $TH/matched_assignment_summary.json $TH/flan_t5_full_finetune_legacy_seq2seq_2000.yaml $TH/matched_legacy_seq2seq_2000_training_runlists
regen llama_notrans $TH/matched_assignment_summary.json $TH/llama32_1b_full_finetune_legacy_causal_2000.yaml $TH/matched_legacy_causal_2000_training_runlists

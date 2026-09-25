#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "../..");

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(
        "usage: rebuild_reproducibility_workbook.mjs " +
          "--input INPUT.xlsx --status STATUS.json --output OUTPUT.xlsx " +
          "[--workspace-root DIR] [--preview-dir DIR]",
      );
    }
    parsed[key.slice(2)] = value;
  }
  return parsed;
}

function resolveFrom(base, value) {
  return path.isAbsolute(value) ? value : path.resolve(base, value);
}

async function sha256(filePath) {
  return crypto.createHash("sha256").update(await fs.readFile(filePath)).digest("hex");
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows) {
  return rows.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

const args = parseArgs(process.argv.slice(2));
for (const required of ["input", "status", "output"]) {
  if (!args[required]) throw new Error(`missing --${required}`);
}

const workspaceRoot = args["workspace-root"]
  ? resolveFrom(process.cwd(), args["workspace-root"])
  : path.resolve(repoRoot, "..");
const inputPath = resolveFrom(process.cwd(), args.input);
const statusPath = resolveFrom(process.cwd(), args.status);
const outputPath = resolveFrom(process.cwd(), args.output);
const previewDir = resolveFrom(
  process.cwd(),
  args["preview-dir"] ?? path.join(path.dirname(outputPath), "previews"),
);
const outputDir = path.dirname(outputPath);
const status = JSON.parse(await fs.readFile(statusPath, "utf8"));
const trainingCompletion = status.training_completion;
if (!trainingCompletion?.cohorts?.length) {
  throw new Error("status file is missing training_completion.cohorts");
}
const completionByCohort = new Map(
  trainingCompletion.cohorts.map((cohort) => [cohort.cohort_id, cohort]),
);

function jobSpan(jobIds) {
  if (!jobIds?.length) return "none";
  return jobIds.length === 1 ? jobIds[0] : `${jobIds[0]}–${jobIds.at(-1)}`;
}

function strictAuditSummary(completion) {
  return `strict audit ${completion.n_ready}/${completion.n_trees} trees ready; ${completion.n_ok_nodes}/${completion.n_total_runs} nodes OK; 0 failures/errors/warnings`;
}

function completionReceipt(cohort) {
  const completion = completionByCohort.get(cohort.cohort_id);
  if (!completion) {
    return cohort.job_id ? `Wright ${cohort.job_id}; ${cohort.output_root}` : cohort.next_action;
  }
  return `${strictAuditSummary(completion)}; original array ${completion.baseline_job} (partial/superseded); recovery ${jobSpan(completion.recovery_jobs)}; ${cohort.output_root}`;
}

if (path.extname(inputPath).toLowerCase() !== ".xlsx" || path.extname(outputPath).toLowerCase() !== ".xlsx") {
  throw new Error("--input and --output must both be .xlsx files");
}
if (path.resolve(inputPath) === path.resolve(outputPath)) {
  throw new Error("refusing to overwrite the preserved input workbook; choose a distinct --output path");
}

const inputHash = await sha256(inputPath);
if (inputHash !== status.input_workbook.sha256) {
  throw new Error(
    `input workbook hash mismatch: expected ${status.input_workbook.sha256}, got ${inputHash}`,
  );
}

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
workbook.comments.setSelf({ displayName: "Shannon" });

const colors = {
  navy: "#17324D",
  teal: "#1F7A72",
  blue: "#2F6B9A",
  pale: "#EAF2F8",
  paleTeal: "#E7F4F1",
  gold: "#E9B44C",
  paleGold: "#FFF6DD",
  gray: "#5B6770",
  light: "#F5F7F9",
  border: "#CBD5E1",
  red: "#A33A3A",
  paleRed: "#FCE8E6",
  green: "#257A4A",
  paleGreen: "#E2F0D9",
  white: "#FFFFFF",
};

function getOrAddSheet(name) {
  const sheet = workbook.worksheets.getOrAdd(name);
  sheet.showGridLines = false;
  return sheet;
}

function setWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function mergeRange(range) {
  try {
    range.unmerge();
  } catch {
    // A fresh sheet or source revision may already be unmerged.
  }
  range.merge();
}

function title(sheet, text, subtitle, lastColumn = "H", fill = colors.navy) {
  const titleRange = sheet.getRange(`A1:${lastColumn}1`);
  const subtitleRange = sheet.getRange(`A2:${lastColumn}2`);
  mergeRange(titleRange);
  mergeRange(subtitleRange);
  titleRange.format = {
    fill,
    font: { bold: true, color: colors.white, size: 16 },
    verticalAlignment: "center",
  };
  titleRange.format.rowHeight = 28;
  subtitleRange.format = {
    fill: fill === colors.red ? colors.paleRed : colors.pale,
    font: { color: fill === colors.red ? colors.red : colors.gray, italic: true },
    wrapText: true,
    verticalAlignment: "center",
  };
  subtitleRange.format.rowHeight = 34;
  sheet.getRange("A1").values = [[text]];
  sheet.getRange("A2").values = [[subtitle]];
}

function header(range) {
  range.format = {
    fill: colors.teal,
    font: { bold: true, color: colors.white },
    borders: { preset: "outside", style: "thin", color: colors.border },
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 30;
}

function section(range) {
  range.format = {
    fill: colors.paleTeal,
    font: { bold: true, color: colors.navy },
    borders: { bottom: { style: "thin", color: colors.teal } },
    wrapText: true,
  };
}

function body(range) {
  range.format = {
    borders: { insideHorizontal: { style: "thin", color: "#E6EBEF" } },
    verticalAlignment: "center",
    wrapText: true,
  };
}

function clearContents(sheet, range) {
  sheet.getRange(range).clear({ applyTo: "contents" });
}

// README: replace stale v2 and accepted-result language without altering the manuscript.
{
  const sheet = workbook.worksheets.getItem("README");
  clearContents(sheet, "A1:B35");
  title(
    sheet,
    "WeightTraits Reproducibility Workbook — Corrected Rebuild Control",
    "Source-driven reconciliation layer for corrected 2,000-step reruns; outcome cells remain blank until validated artifacts land.",
    "H",
  );
  sheet.getRange("A4:B11").values = [
    ["Paper comparison", `Pinned August 10 ELLMTrees-paper commit ${status.paper_comparison.remote_commit} (read-only comparison target; current manuscript worktree excluded)`],
    ["Source precedence", "Per-run artifacts + locked manifests → checked-in rollups → paper consumers → workbook cells"],
    ["Live outcome status", "Corrected ordinary and matched no-translation Llama training is complete and strict-audited; corrected weight/behavior analyses remain pending"],
    ["Training contract", `${status.training_contract.max_steps} steps; seed ${status.training_contract.seed}; ${status.training_contract.sample_strategy}; 10k/1k train/eval caps`],
    ["Sequence budget", `${status.training_contract.max_seq_length} tokens total prompt + target for every task family; not a completion target`],
    ["Table 3 method", "Prompt-paired semantic similarity from dist_semantic_paired.npy; cohorts analyzed independently; no ROUGE, tree matching, or held-out-tree split"],
    ["Workbook role", "Reproducibility, archive, and status control; never the numerical authority over raw artifacts"],
    ["Manuscript rule", "Main text is frozen by user instruction; discrepancies are logged here only"],
  ];
  section(sheet.getRange("A4:A11"));
  body(sheet.getRange("B4:B11"));

  sheet.getRange("A13").values = [["Rebuild gate status"]];
  section(sheet.getRange("A13:H13"));
  sheet.getRange("A14:B19").values = [
    ["Input workbook", `Preserved byte-for-byte (${inputHash})`],
    ["Dataset cache", `${status.training_contract.cache_result}; job ${status.training_contract.cache_job}`],
    ["Trainer smoke", `${status.training_contract.smoke_result}; job ${status.training_contract.smoke_job}`],
    ["Ordinary Llama full FT", "TRAINING COMPLETE — strict audit valid; 50/50 trees and 641/641 nodes ready; corrected analyses pending"],
    ["Matched no-translation", "TRAINING COMPLETE — strict audit valid; 50/50 trees and 641/641 nodes ready; corrected analyses pending"],
    ["Other affected cohorts", "INVALIDATED / PENDING protocol-specific corrected relaunch"],
  ];
  section(sheet.getRange("A14:A19"));
  body(sheet.getRange("B14:B19"));

  sheet.getRange("A21").values = [["How to audit"]];
  section(sheet.getRange("A21:H21"));
  sheet.getRange("A22:A25").values = [
    ["1. Cohort Status separates legacy references, invalidated runs, and corrected runs."],
    ["2. Reconciliation Ledger records every known workbook/source mismatch and dependency."],
    ["3. Run Provenance and Source Manifest bind configs, run lists, cache receipts, runtimes, and jobs."],
    ["4. Archive Index prevents historical or invalidated sheets from feeding live conclusions."],
  ];
  sheet.getRange("A22:H25").format.wrapText = true;

  sheet.getRange("A27").values = [["Current interpretation"]];
  sheet.getRange("A27:H27").format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 14 },
  };
  mergeRange(sheet.getRange("A28:H28"));
  mergeRange(sheet.getRange("A29:H29"));
  sheet.getRange("A28").values = [["Fresh numeric results in the August 3 workbook are retained only as invalidated provenance; they are not accepted outcomes."]];
  sheet.getRange("A29").values = [["Historical invalidated paired-semantic Table 3 values are audited separately; corrected Table 2 and Table 3 slots stay blank until corrected weights and probes are audited."]];
  sheet.getRange("A28:H29").format = {
    fill: colors.paleGold,
    font: { italic: true, color: "#7F6000" },
    wrapText: true,
  };
  setWidths(sheet, { A: 34, B: 100, C: 14, D: 14, E: 14, F: 14, G: 14, H: 14 });
}

// Paper Update is now a workbook-only changelog, not a paper-edit queue.
{
  const sheet = workbook.worksheets.getItem("Paper Update");
  clearContents(sheet, "A1:F46");
  title(
    sheet,
    "Workbook Rebuild Changelog",
    "Records workbook/source changes only. No manuscript substitutions or main-text edits are authorized.",
    "F",
  );
  const rows = [
    [status.as_of, "Input", "Live workbook edited in place", "Byte-identical source preserved; rebuilt copy exported separately", "complete", inputHash],
    [status.as_of, "Training", "Fresh outcomes treated as accepted", "Eleven affected WeightTraits cohorts invalidated", "complete", "Cohort Status"],
    [status.as_of, "Cache", "Mutable/implicit dataset loads", "35 historical sources offline + revision-pinned TREC; 72/72 split receipts", "complete", `Wright job ${status.training_contract.cache_job}`],
    [status.as_of, "Llama full FT", "Early-stopped/changed-prompt weights", "Corrected exact-2,000-step ordinary training passed strict 50-tree/641-node audit", "training complete; analysis pending", "Original 167745 partial/superseded; recovery 167933–167939"],
    [status.as_of, "No translation", "Independent or mismatched cohort could be substituted", "Matched 479-preserved/162-replaced training passed strict 50-tree/641-node audit", "training complete; analysis pending", "Original 167746 partial/superseded; recovery 167940–167956"],
    [status.as_of, "Table 2", "Generated n=50 used as topology n", "Generated n=50, topology n=46, exclusions 015/020/037/047, ordering n=26", "structure fixed", "T2 Shared Samples"],
    [status.as_of, "Table 3", "Prompt pairing conflated with paired cohorts", "Historical dist_semantic_paired.npy audit rebuilt; corrected rows pending", "historical invalidated reference", "Paired Semantic Audit; Reconciliation WB005"],
    [status.as_of, "Historical sheets", "Historical and invalidated values appeared live", "Explicit archive/invalid status banners and index", "complete", "Archive Index"],
    [status.as_of, "Paper", "Workbook workflow could trigger text edits", "Comparison-only; discrepancies logged without manuscript changes", "complete", status.paper_comparison.remote_commit],
  ];
  sheet.getRange("A4:F4").values = [["Date", "Area", "Before", "After", "Status", "Evidence"]];
  header(sheet.getRange("A4:F4"));
  sheet.getRange(`A5:F${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:F${4 + rows.length}`));
  sheet.getRange(`A5:F${4 + rows.length}`).format.rowHeight = 38;
  setWidths(sheet, { A: 14, B: 22, C: 48, D: 68, E: 24, F: 68 });
}

// Claim/source routing.
{
  const sheet = workbook.worksheets.getItem("Claim Index");
  clearContents(sheet, "A1:E25");
  title(
    sheet,
    "Workbook Claim and Source Index",
    "Routes each result family to a live, pending, historical, or invalidated workbook section.",
    "E",
  );
  const rows = [
    ["Rebuild control", "Cohort and launch status", "live", "Cohort Status", "All eleven affected WeightTraits cohorts"],
    ["Table 2", "Fresh controlled topology", "Llama training complete; weight analysis pending", "T2 Shared Samples", "Metrics blank; exact 46 topology / 26 ordering eligibility encoded"],
    ["Table 3", "Weight-to-behavior bridge", "historical invalidated reference retained; corrected analysis pending", "Paired Semantic Audit", "Corrected weight distances, probes, and dist_semantic_paired.npy do not exist yet"],
    ["Table 4", "Layer subsets", "invalidated pending Flan rerun", "T4 Layers", "Prior 46-tree rescore retained only as provenance"],
    ["Table 5", "LoRA parameterization", "architecture constants", "T5 LoRA", "Counts/fractions independent of trained outcomes"],
    ["Table 6", "Behavior regressions", "historical reference", "T6 Regression", "Explicitly excluded from live fresh conclusions"],
    ["Table 7", "Bridge heterogeneity", "invalidated pending bridge rebuild", "T7 Heterogeneity", "Must reference rebuilt Table 3"],
    ["Table 8", "Translation examples", "historical reference", "T8 Examples", "ELLMTrees qualitative archive"],
    ["Table 9", "Within-task recovery", "historical reference", "T9 Within Task", "ELLMTrees controlled reference"],
    ["Table 10", "PhyloLM comparison", "historical cohort archived", "T10 PhyloLM", "Fresh r8/r64/full-FT common-46 section pending"],
    ["Sensitivity", "Immediate-EOS policy", "invalidated pending corrected probes", "Empty Sensitivity", "Do not combine old outcomes with corrected weights"],
    ["Appendix", "Empty rate by depth", "invalidated pending corrected probes", "Depth Summary", "Recompute from corrected outcomes"],
    ["Audit", "Cell-level reconciliation", "live", "Reconciliation Ledger", "Source, status, dependency, and action for each mismatch"],
  ];
  sheet.getRange("A4:E4").values = [["Consumer", "Result family", "Status", "Sheet", "Coverage / rule"]];
  header(sheet.getRange("A4:E4"));
  sheet.getRange(`A5:E${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:E${4 + rows.length}`));
  setWidths(sheet, { A: 22, B: 34, C: 30, D: 28, E: 72 });
}

// Live Table 2 placeholder: encode eligibility, representation, and provenance before numbers.
{
  const sheet = workbook.worksheets.getItem("T2 Shared Samples");
  for (const range of ["A1:S1", "A2:S2", "A3:S3", "A15:S15"]) {
    try {
      sheet.getRange(range).unmerge();
    } catch {
      // Input revisions may already be unmerged; the rebuilt layout below is authoritative.
    }
  }
  clearContents(sheet, "A1:V20");
  for (const range of ["A1:V1", "A2:V2", "A3:V3", "A14:V14", "A17:V17"]) {
    sheet.getRange(range).merge();
  }
  title(
    sheet,
    "ANALYSIS PENDING — Corrected Shared-Tree Lineage and Recovery",
    "Corrected Llama model artifacts passed strict training audits; live metrics remain blank until per-tree weight-analysis artifacts pass source, representation, and eligibility audits.",
    "V",
    colors.gold,
  );
  sheet.getRange("A3").values = [["Generated n=50; topology-eligible n=46 (exclude 015, 020, 037, 047); branch-ordering n=26. These are distinct estimands."]];
  sheet.getRange("A3:V3").format = {
    fill: colors.paleGold,
    font: { bold: true, color: "#7F6000" },
    wrapText: true,
  };
  const headers = [
    "Model", "Variant", "Status", "Method", "Representation", "Generated n", "Topology n", "Excluded IDs", "Ordering n",
    "Rank-bis.", "SE", "r_branch", "SE", "Clade", "SE", "PAER", "SE", "RF", "SE", "FN", "SE", "Source / receipt",
  ];
  sheet.getRange("A4:V4").values = [headers];
  header(sheet.getRange("A4:V4"));
  const cohortById = new Map(status.cohorts.map((cohort) => [cohort.cohort_id, cohort]));
  const rows = status.live_table2_rows.map(([model, variant, cohortId, representation]) => {
    const cohort = cohortById.get(cohortId);
    return [
      model,
      variant,
      cohort.status,
      cohort.method,
      representation,
      50,
      46,
      "015, 020, 037, 047",
      26,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      null,
      completionReceipt(cohort),
    ];
  });
  sheet.getRange(`A5:V${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:V${4 + rows.length}`));
  sheet.getRange(`A5:V${4 + rows.length}`).format.rowHeight = 48;
  sheet.getRange("F5:G15").format.numberFormat = "0";
  sheet.getRange("I5:I15").format.numberFormat = "0";
  sheet.getRange("C5:C12").conditionalFormats.deleteAll();
  sheet.getRange("C5:C12").conditionalFormats.add("containsText", {
    text: "running",
    format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
  });
  sheet.getRange("C5:C12").conditionalFormats.add("containsText", {
    text: "complete",
    format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
  });
  sheet.getRange("C5:C12").conditionalFormats.add("containsText", {
    text: "pending",
    format: { fill: colors.paleGold, font: { color: "#7F6000", bold: true } },
  });
  sheet.getRange("A14").values = [["Matched no-translation comparison (not a live Table 2 row)"]];
  section(sheet.getRange("A14:V14"));
  const noTranslation = cohortById.get("llama_full_finetune_no_translation");
  sheet.getRange("A15:V15").values = [[
    noTranslation.model,
    "Full FT, matched no translation",
    noTranslation.status,
    noTranslation.method,
    "full weight",
    50,
    46,
    "015, 020, 037, 047",
    26,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    completionReceipt(noTranslation),
  ]];
  body(sheet.getRange("A15:V15"));
  sheet.getRange("A15:V15").format.rowHeight = 48;
  sheet.getRange("A17").values = [["Blank metric cells are intentional. Never copy values from the invalidated August 3 snapshots into this live block."]];
  sheet.getRange("A17:V17").format = {
    fill: colors.paleRed,
    font: { bold: true, color: colors.red },
    wrapText: true,
  };
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, {
    A: 20, B: 34, C: 30, D: 24, E: 24, F: 12, G: 12, H: 24, I: 12,
    J: 12, K: 10, L: 12, M: 10, N: 12, O: 10, P: 12, Q: 10, R: 10, S: 10, T: 10, U: 10, V: 64,
  });
}

// Existing scientific sheets remain readable but cannot masquerade as live results.
const bannerSpecs = [
  ["T2 Recovery", "P", "ARCHIVE — Historical Table 2 Recovery", "Mixed historical 40–50-tree ranges; excluded from the live corrected workbook.", colors.gray],
  ["T2 Recovery Raw", "H", "ARCHIVE — Historical Table 2 Recovery Raw", "Raw input for the archived historical summary.", colors.gray],
  ["T2 Ordering Raw", "F", "ARCHIVE — Historical Table 2 Ordering Raw", "Raw input for the archived historical summary.", colors.gray],
  ["Bridge Summary", "K", "MIXED SNAPSHOT — Weight-to-Behavior Bridge", "The prompt-paired semantic method is correct, but some fresh rows use invalidated early-stopped cohorts.", colors.red],
  ["Bridge DL Calc", "Q", "MIXED SNAPSHOT — Bridge Calculations", "Use historical rows only through Paired Semantic Audit; rebuild fresh rows from corrected paired-semantic probes.", colors.red],
  ["T4 Layers", "E", "INVALIDATED SNAPSHOT — LayerTrace Subsets", "Prior fresh Flan values came from an affected training cohort; rerun after corrected Flan full FT.", colors.red],
  ["T6 Regression", "I", "HISTORICAL REFERENCE — Behavioral Regression Coefficients", "ELLMTrees historical reference; excluded from fresh corrected conclusions.", colors.gray],
  ["T6 Coefficient Raw", "M", "HISTORICAL REFERENCE — Behavioral Regression Raw Inputs", "Raw input for the historical regression reference only.", colors.gray],
  ["T7 Heterogeneity", "G", "INVALIDATED SNAPSHOT — Bridge Heterogeneity", "Depends on the invalidated Bridge Summary and must be regenerated.", colors.red],
  ["T8 Examples", "G", "HISTORICAL REFERENCE — Translation Probe Examples", "Qualitative ELLMTrees no-translation archive.", colors.gray],
  ["T9 Within Task", "K", "HISTORICAL REFERENCE — Within-Task Recovery", "ELLMTrees controlled reference, kept separate from fresh WeightTraits reruns.", colors.gray],
  ["T9 Raw", "G", "HISTORICAL REFERENCE — Within-Task Recovery Raw Inputs", "Raw input for the historical controlled reference only.", colors.gray],
  ["T10 PhyloLM", "P", "ARCHIVE — Historical Five-Condition PhyloLM Comparison", "Deleted from the live paper; fresh r8/r64/full-FT common-46 comparison is pending.", colors.gray],
  ["T10 Behavior Raw", "L", "ARCHIVE — Historical PhyloLM Behavior Raw", "Input to the archived five-condition comparison only.", colors.gray],
  ["T10 Weight Raw", "H", "ARCHIVE — Historical PhyloLM Weight Raw", "Input to the archived five-condition comparison only.", colors.gray],
  ["Empty Sensitivity", "J", "INVALIDATED SNAPSHOT — Immediate-EOS Sensitivity", "Depends on early-stopped fresh outcomes; recompute from corrected probes.", colors.red],
  ["Depth Summary", "K", "INVALIDATED SNAPSHOT — Immediate-EOS by Depth", "Depends on early-stopped fresh outcomes; recompute from corrected probes.", colors.red],
  ["Depth Per Leaf", "I", "INVALIDATED SNAPSHOT — Immediate-EOS per Leaf", "Depends on early-stopped fresh outcomes; recompute from corrected probes.", colors.red],
  ["T10 Shared PhyloLM", "Q", "ARCHIVE — Historical Matched PhyloLM Samples", "Historical 47/47/47/44/46 cohort; excluded from live formulas.", colors.gray],
];
for (const [name, lastColumn, heading, subtitle, fill] of bannerSpecs) {
  const sheet = workbook.worksheets.getItem(name);
  title(sheet, heading, subtitle, lastColumn, fill);
}

// Historical ordinary/no-translation translation check with the paper's paired semantic endpoint.
{
  const sheet = getOrAddSheet("Paired Semantic Audit");
  clearContents(sheet, "A1:K24");
  title(
    sheet,
    "Historical (Invalidated) Table 3 Translation Check — Paired Semantic",
    "Legacy reference only: the exact paper endpoint is shown for auditability, but no corrected 2,000-step behavioral result exists yet.",
    "K",
    colors.blue,
  );
  const paired = status.paired_semantic_audit;
  const ordinaryTranslation = paired.rows.find((row) => row.cohort === "Ordinary" && row.probe === "Translation");
  const noTranslationTranslation = paired.rows.find((row) => row.cohort === "No translation" && row.probe === "Translation");
  if (!ordinaryTranslation || !noTranslationTranslation) {
    throw new Error("paired semantic audit is missing an ordinary or no-translation Translation row");
  }
  sheet.getRange("A3").values = [[
    `INVALIDATED HISTORICAL REFERENCE — ordinary DL r ${ordinaryTranslation.dl_r.toFixed(3)} versus no-translation DL r ${noTranslationTranslation.dl_r.toFixed(3)} (delta ${paired.translation_dl_r_delta_no_translation_minus_ordinary.toFixed(3)}). Do not report these as corrected-cohort results.`,
  ]];
  sheet.getRange("A3:K3").format = {
    fill: colors.paleGold,
    font: { bold: true, color: "#7F6000" },
    wrapText: true,
  };
  mergeRange(sheet.getRange("A3:K3"));
  sheet.getRange("A3:K3").format.rowHeight = 38;
  const headers = [
    "Cohort", "Probe", "Availability", "Runs", "Total within-run pairs", "DL r", "95% CI low", "95% CI high", "p", "Semantic artifact", "Audit note",
  ];
  sheet.getRange("A5:K5").values = [headers];
  header(sheet.getRange("A5:K5"));
  const rows = paired.rows.map((row) => [
    row.cohort,
    row.probe,
    row.status === "available" ? "historical only" : row.status,
    row.runs,
    row.total_pairs,
    row.dl_r,
    row.ci_low,
    row.ci_high,
    row.p,
    paired.semantic_file,
    row.note,
  ]);
  sheet.getRange(`A6:K${5 + rows.length}`).values = rows;
  body(sheet.getRange(`A6:K${5 + rows.length}`));
  sheet.getRange(`D6:E${5 + rows.length}`).format.numberFormat = "#,##0";
  sheet.getRange(`F6:H${5 + rows.length}`).format.numberFormat = "0.000";
  sheet.getRange(`I6:I${5 + rows.length}`).format.numberFormat = "0.00E+00";
  const methodRow = 8 + rows.length;
  mergeRange(sheet.getRange(`A${methodRow}:K${methodRow}`));
  sheet.getRange(`A${methodRow}`).values = [["Exact paper statistic"]];
  section(sheet.getRange(`A${methodRow}:K${methodRow}`));
  const methodRows = [
    ["Semantic builder", paired.builder, `For each of ${paired.prompts_per_node} prompts, compare normalized ${paired.embedding_model}@${paired.embedding_model_revision.slice(0, 12)} output embeddings between leaves; then average.`],
    ["Weight predictor", paired.predictor, "Use every unordered leaf pair within each run."],
    ["Pooling", paired.estimator, "Headline is the pooled DerSimonian-Laird r, not a pooled raw slope or arithmetic mean r."],
    ["Cohort rule", paired.guards, "Prompt pairing is part of the endpoint; cohort/tree pairing is not."],
    ["Generation receipt", `${paired.draws_per_prompt} greedy output per prompt`, paired.paper_discrepancy],
    ["Availability", "No-translation run_039 lacks weight analysis; no no-translation HellaSwag matrices exist.", "Availability omissions only; there is no held-out-tree filtering."],
  ];
  for (let index = 0; index < methodRows.length; index += 1) {
    const row = methodRow + 1 + index;
    const [label, value, note] = methodRows[index];
    mergeRange(sheet.getRange(`A${row}:B${row}`));
    mergeRange(sheet.getRange(`C${row}:F${row}`));
    mergeRange(sheet.getRange(`G${row}:K${row}`));
    sheet.getRange(`A${row}`).values = [[label]];
    sheet.getRange(`C${row}`).values = [[value]];
    sheet.getRange(`G${row}`).values = [[note]];
    section(sheet.getRange(`A${row}:B${row}`));
    body(sheet.getRange(`C${row}:K${row}`));
    sheet.getRange(`A${row}:K${row}`).format.rowHeight = index === 3 ? 58 : 42;
  }
  sheet.freezePanes.freezeRows(5);
  setWidths(sheet, { A: 24, B: 18, C: 18, D: 10, E: 18, F: 12, G: 12, H: 12, I: 14, J: 30, K: 72 });
}
// Corrected Llama training completion receipts. This is deliberately separate from scientific outcomes.
{
  const sheet = getOrAddSheet("Training Completion");
  clearContents(sheet, "A1:N20");
  title(
    sheet,
    "Corrected Llama Training Completion",
    "Strict frozen-stage audits passed for both 50-tree cohorts; corrected weight and behavioral analyses have not been run.",
    "N",
    colors.green,
  );
  mergeRange(sheet.getRange("A3:N3"));
  sheet.getRange("A3").values = [[
    "Training completion is not a Table 3 result. Corrected result cells remain blank until weight distances, locked probes, prompt-paired semantic distances, and independent meta-analyses are complete.",
  ]];
  sheet.getRange("A3:N3").format = {
    fill: colors.paleGold,
    font: { bold: true, color: "#7F6000" },
    wrapText: true,
  };
  sheet.getRange("A3:N3").format.rowHeight = 42;
  const headers = [
    "Cohort", "Training status", "Completed ET", "Original array", "Retained baseline tasks", "Recovery jobs",
    "Audit valid", "Trees ready", "Nodes OK", "Artifacts", "Failed / missing", "Errors", "Warnings", "Corrected analyses",
  ];
  sheet.getRange("A5:N5").values = [headers];
  header(sheet.getRange("A5:N5"));
  const rows = trainingCompletion.cohorts.map((completion) => [
    completion.label,
    "complete",
    completion.completed_at_et,
    `${completion.baseline_job} (partial/superseded)`,
    completion.retained_baseline_tasks.join(", "),
    jobSpan(completion.recovery_jobs),
    completion.valid,
    `${completion.n_ready}/${completion.n_trees}`,
    `${completion.n_ok_nodes}/${completion.n_total_runs}`,
    `${completion.n_existing_artifacts}/${completion.n_expected_artifacts}`,
    `${completion.n_failed_nodes} / ${completion.n_missing_nodes}`,
    completion.n_errors,
    completion.n_warnings,
    "NOT RUN — corrected values unavailable",
  ]);
  sheet.getRange(`A6:N${5 + rows.length}`).values = rows;
  body(sheet.getRange(`A6:N${5 + rows.length}`));
  sheet.getRange(`B6:B${5 + rows.length}`).format = {
    fill: colors.paleGreen,
    font: { bold: true, color: colors.green },
  };
  sheet.getRange(`N6:N${5 + rows.length}`).format = {
    fill: colors.paleGold,
    font: { bold: true, color: "#7F6000" },
    wrapText: true,
  };
  sheet.getRange("A9").values = [["Strict audit basis"]];
  section(sheet.getRange("A9:N9"));
  const auditNotes = [
    `Verified ${trainingCompletion.verified_at} from frozen stage ${status.training_contract.wright_stage}; artifact and parent-order checks enabled; no allow/skip flags.`,
    `Ordinary summary input: ${trainingCompletion.cohorts[0].summary}`,
    `Matched no-translation summary input: ${trainingCompletion.cohorts[1].summary}`,
    trainingCompletion.downstream_analysis.artifact_audit,
  ];
  for (let index = 0; index < auditNotes.length; index += 1) {
    const row = 10 + index;
    mergeRange(sheet.getRange(`A${row}:N${row}`));
    sheet.getRange(`A${row}`).values = [[auditNotes[index]]];
    body(sheet.getRange(`A${row}:N${row}`));
    sheet.getRange(`A${row}:N${row}`).format.rowHeight = index === 3 ? 42 : 34;
  }
  sheet.freezePanes.freezeRows(5);
  setWidths(sheet, {
    A: 38, B: 18, C: 22, D: 26, E: 48, F: 20, G: 12, H: 14, I: 14, J: 14,
    K: 18, L: 10, M: 10, N: 42,
  });
}

// Cohort Status.
{
  const sheet = getOrAddSheet("Cohort Status");
  clearContents(sheet, "A1:L40");
  title(
    sheet,
    "Affected Cohort Status",
    "Separates strict-audited corrected training from pending downstream analyses, invalidated artifacts, and legacy references.",
    "L",
  );
  const headers = [
    "Cohort ID", "Model", "Method", "Scope", "Status", "Job ID", "Protocol", "Trees", "Nodes", "Target steps", "Output root", "Next action",
  ];
  sheet.getRange("A4:L4").values = [headers];
  header(sheet.getRange("A4:L4"));
  const rows = status.cohorts.map((cohort) => [
    cohort.cohort_id,
    cohort.model,
    cohort.method,
    cohort.scope,
    cohort.status,
    completionByCohort.has(cohort.cohort_id)
      ? `${cohort.job_id}; recovery ${jobSpan(completionByCohort.get(cohort.cohort_id).recovery_jobs)}`
      : cohort.job_id,
    cohort.protocol,
    cohort.trees,
    cohort.nodes,
    cohort.steps,
    cohort.output_root,
    cohort.next_action,
  ]);
  sheet.getRange(`A5:L${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:L${4 + rows.length}`));
  sheet.getRange(`A5:L${4 + rows.length}`).format.rowHeight = 44;
  const statusRange = sheet.getRange(`E5:E${4 + rows.length}`);
  statusRange.conditionalFormats.deleteAll();
  statusRange.conditionalFormats.add("containsText", {
    text: "running",
    format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
  });
  statusRange.conditionalFormats.add("containsText", {
    text: "complete",
    format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
  });
  statusRange.conditionalFormats.add("containsText", {
    text: "pending",
    format: { fill: colors.paleGold, font: { color: "#7F6000", bold: true } },
  });
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 38, B: 20, C: 30, D: 34, E: 34, F: 28, G: 48, H: 10, I: 10, J: 12, K: 64, L: 58 });
}

// Run Provenance.
{
  const sheet = getOrAddSheet("Run Provenance");
  clearContents(sheet, "A1:D40");
  title(
    sheet,
    "Corrected Run Provenance",
    "Frozen launch, cache, recovery, and strict completion receipts for the corrected Llama cohorts.",
    "D",
  );
  const ordinaryCompletion = completionByCohort.get("llama_full_finetune");
  const noTranslationCompletion = completionByCohort.get("llama_full_finetune_no_translation");
  const rows = [
    ["Paper comparison commit", status.paper_comparison.remote_commit, "pinned comparison only", "Pinned August 10 comparison commit; manuscript worktree is not an input to this rebuild"],
    ["Input workbook SHA-256", inputHash, "preserved source", status.input_workbook.path],
    ["Wright staging tree", status.training_contract.wright_stage, "immutable launch snapshot", "341 tests passed; checksum matched local snapshot"],
    ["Protocol ID", status.training_contract.protocol_id, "training contract", "ordinary and matched no-translation"],
    ["Base model", status.training_contract.base_model, "model", status.training_contract.base_model_revision],
    ["Runtime", status.training_contract.runtime, "software", "legacy ellmtrees environment"],
    ["Sequence ceiling", status.training_contract.max_seq_length, "prompt + target", status.training_contract.sequence_length_note],
    ["Sampling", `${status.training_contract.sample_strategy}; seed ${status.training_contract.seed}`, "data selection", `train ${status.training_contract.train_limit}; eval ${status.training_contract.eval_limit}`],
    ["Cache job", status.training_contract.cache_job, "completed", status.training_contract.cache_result],
    ["Trainer smoke", status.training_contract.smoke_job, "completed", status.training_contract.smoke_result],
    ["Ordinary original array", ordinaryCompletion.baseline_job, "partial / superseded", `Final receipt retains ${ordinaryCompletion.retained_baseline_tasks.join(", ")}`],
    ["Ordinary recovery jobs", jobSpan(ordinaryCompletion.recovery_jobs), "completed", `${ordinaryCompletion.recovery_jobs.length} scoped jobs; ${ordinaryCompletion.recovery_window_et}; all exit 0:0`],
    ["Ordinary strict audit", strictAuditSummary(ordinaryCompletion), "completed", `${ordinaryCompletion.n_existing_artifacts}/${ordinaryCompletion.n_expected_artifacts} artifacts; ${ordinaryCompletion.summary}`],
    ["Matched original array", noTranslationCompletion.baseline_job, "partial / superseded", `Final receipt retains ${noTranslationCompletion.retained_baseline_tasks.join(", ")}`],
    ["Matched recovery jobs", jobSpan(noTranslationCompletion.recovery_jobs), "completed", `${noTranslationCompletion.recovery_jobs.length} scoped jobs; ${noTranslationCompletion.recovery_window_et}; all exit 0:0`],
    ["Matched strict audit", strictAuditSummary(noTranslationCompletion), "completed", `${noTranslationCompletion.n_existing_artifacts}/${noTranslationCompletion.n_expected_artifacts} artifacts; ${noTranslationCompletion.summary}`],
    ["Corrected scientific analysis", trainingCompletion.downstream_analysis.status, "pending", trainingCompletion.downstream_analysis.artifact_audit],
  ];
  sheet.getRange("A4:D4").values = [["Field", "Value", "Role / status", "Evidence / note"]];
  header(sheet.getRange("A4:D4"));
  sheet.getRange(`A5:D${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:D${4 + rows.length}`));
  sheet.getRange(`A5:D${4 + rows.length}`).format.rowHeight = 44;
  setWidths(sheet, { A: 30, B: 76, C: 24, D: 84 });
}

// Archive Index.
{
  const sheet = getOrAddSheet("Archive Index");
  clearContents(sheet, "A1:E40");
  title(
    sheet,
    "Archive and Invalidation Index",
    "Historical values remain inspectable but are explicitly disconnected from live corrected conclusions.",
    "E",
  );
  sheet.getRange("A4:E4").values = [["Sheet", "Status", "Reason", "Feeds live results?", "Replacement"]];
  header(sheet.getRange("A4:E4"));
  sheet.getRange(`A5:E${4 + status.archive_sheets.length}`).values = status.archive_sheets;
  body(sheet.getRange(`A5:E${4 + status.archive_sheets.length}`));
  setWidths(sheet, { A: 28, B: 26, C: 68, D: 18, E: 60 });
}

// Reconciliation Ledger.
{
  const sheet = getOrAddSheet("Reconciliation Ledger");
  clearContents(sheet, "A1:G50");
  title(
    sheet,
    "Cell-Level Reconciliation Ledger",
    "Known workbook/source mismatches, evidence, status, proposed action, and dependency.",
    "G",
  );
  sheet.getRange("A4:G4").values = [["ID", "Workbook location", "Issue", "Evidence", "Status", "Action", "Depends on"]];
  header(sheet.getRange("A4:G4"));
  sheet.getRange(`A5:G${4 + status.reconciliation.length}`).values = status.reconciliation;
  body(sheet.getRange(`A5:G${4 + status.reconciliation.length}`));
  sheet.getRange(`A5:G${4 + status.reconciliation.length}`).format.rowHeight = 42;
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 12, B: 32, C: 68, D: 76, E: 36, F: 70, G: 34 });
}

// Replace the stale source manifest with active, hashed inputs and remote receipts.
const sourceSpecs = [
  ["Workbook rebuild builder", path.relative(workspaceRoot, scriptPath), "local workbook authoring source"],
  ["Workbook rebuild status", path.relative(workspaceRoot, statusPath), "cohort/source/reconciliation contract"],
  ...status.source_files,
];
const sourceRows = [];
const seenPaths = new Set();
for (const [artifact, relativePath, role] of sourceSpecs) {
  if (seenPaths.has(relativePath)) continue;
  seenPaths.add(relativePath);
  const absolutePath = resolveFrom(workspaceRoot, relativePath);
  try {
    const fileStat = await fs.stat(absolutePath);
    if (!fileStat.isFile()) throw new Error("not a regular file");
    sourceRows.push([
      artifact,
      relativePath,
      await sha256(absolutePath),
      fileStat.size,
      new Date(fileStat.mtimeMs),
      role,
    ]);
  } catch (error) {
    sourceRows.push([artifact, relativePath, null, null, null, `${role}; MISSING: ${error.message}`]);
  }
}
sourceRows.push([
  "Wright cache receipt",
  `${status.training_contract.wright_stage}/examples/training/confirm_paper_numbers/legacy_causal_seed42_data_cache_summary.json`,
  null,
  null,
  null,
  `${status.training_contract.cache_result}; job ${status.training_contract.cache_job}`,
]);
for (const completion of trainingCompletion.cohorts) {
  sourceRows.push([
    `${completion.label} original array`,
    `Wright Slurm job ${completion.baseline_job}`,
    null,
    null,
    null,
    `partial/superseded; final receipt retains ${completion.retained_baseline_tasks.join(", ")}`,
  ]);
  sourceRows.push([
    `${completion.label} recovery jobs`,
    `Wright Slurm jobs ${jobSpan(completion.recovery_jobs)}`,
    null,
    null,
    null,
    `${completion.recovery_jobs.length} scoped jobs completed 0:0; ${completion.recovery_window_et}`,
  ]);
  sourceRows.push([
    `${completion.label} strict audit`,
    `Wright frozen-stage stdout audit verified ${trainingCompletion.verified_at}`,
    null,
    null,
    null,
    `${strictAuditSummary(completion)}; ${completion.n_existing_artifacts}/${completion.n_expected_artifacts} artifacts`,
  ]);
}

{
  const sheet = workbook.worksheets.getItem("Source Manifest");
  clearContents(sheet, "A1:F220");
  title(
    sheet,
    "Active Source Manifest",
    "Current configs, run lists, scripts, receipts, and preserved input workbook. Missing hashes are remote receipts or explicit blockers.",
    "F",
  );
  sheet.getRange("A4:F4").values = [["Artifact", "Path / receipt", "SHA-256", "Bytes", "Modified", "Role / status"]];
  header(sheet.getRange("A4:F4"));
  sheet.getRange(`A5:F${4 + sourceRows.length}`).values = sourceRows;
  body(sheet.getRange(`A5:F${4 + sourceRows.length}`));
  sheet.getRange(`A5:F${4 + sourceRows.length}`).format.rowHeight = 42;
  sheet.getRange(`D5:D${4 + sourceRows.length}`).format.numberFormat = "#,##0";
  sheet.getRange(`E5:E${4 + sourceRows.length}`).format.numberFormat = "yyyy-mm-dd hh:mm";
  sheet.freezePanes.freezeRows(4);
  setWidths(sheet, { A: 32, B: 100, C: 68, D: 14, E: 20, F: 72 });
}

// Reproduction commands: workbook-only, source-driven, and explicit about legacy Table 3.
{
  const sheet = workbook.worksheets.getItem("Reproduction Commands");
  clearContents(sheet, "A1:C24");
  title(
    sheet,
    "Reproduction Commands",
    "Local source-driven entry points for the corrected workbook and legacy semantic analysis. No command edits the manuscript.",
    "C",
  );
  const rows = [
    ["Rebuild workbook", "bash WeightTraits/scripts/paper/run_rebuild_reproducibility_workbook.sh --input ELLMTrees-paper/WeightTraits_all_results_reproducibility.xlsx --status WeightTraits/examples/paper/workbook_rebuild_status_20260810.json --workspace-root . --output outputs/019febc7-5411-7e50-ab7c-fb558215dcd3/WeightTraits_all_results_reproducibility_20260814.xlsx", "Imports the preserved workbook, invalidates stale cells, refreshes source/status sheets, scans errors, and renders all sheets through the bundled workbook runtime"],
    ["Validate WeightTraits code", "cd WeightTraits && PYTHONPATH=src pytest", "Frozen launch receipt: 341 tests passed"],
    ["Validate matched design", "python WeightTraits/examples/training/translation_holdout_20260803/audit_matched_translation_runlists.py --ordinary-runlists <ordinary> --holdout-runlists <matched> --out <audit.json>", "Requires 479 preserved, 162 replaced, zero issues"],
    ["Strict training completion", "cd /home/export/sgallagh/WeightTraits-legacy-causal-20260810-clean && PYTHONPATH=src /home/export/sgallagh/.conda/envs/ellmtrees/bin/python -m weighttraits.cli audit-training-run-set --summary <ordinary-or-matched-summary.json> --path-base .", "Keep artifact and parent-order checks enabled; do not pass allow/skip flags; both cohorts currently pass 50/50 trees and 641/641 nodes"],
    ["Paired semantic distances", "python ELLMTrees/scripts/build_paired_semantic.py --group <results-root> --probe_subdir <behavioral-subdir> --st_model all-MiniLM-L6-v2", "Writes dist_semantic_paired.npy by comparing same-prompt outputs and then averaging; use pinned model revision from Run Provenance"],
    ["Paper Table 3", "python ELLMTrees/scripts/regression_distance_vs_behavior.py --results <results-root> --manifests <manifest-root> --probe <probe> --probe_prefix <behavioral|behavioral_llama> --dv semantic --semantic_file dist_semantic_paired.npy --train_frac 1.0 --min-leaves 2 --practical --meta --out <per-pair.csv>", "Prompt-paired semantic endpoint; ordinary and no-translation cohorts analyzed independently; no tree matching or held-out-tree split"],
    ["Topology eligibility", "Join by canonical topology ID; exclude 015,020,037,047 from unrooted topology metrics; assert ordering IDs separately (n=26)", "Never infer eligibility from generated row count"],
    ["Corrected downstream bridge", "First implement and validate the missing adapter from WeightTraits model/manifests to the legacy run_*/weight and behavioral_* layout; then run the locked free-generation probes, paired semantic builder, and regressions", "Old outcomes cannot be combined with corrected weights; do not launch legacy wrappers directly against the new layout"],
    ["Paper handling", "No command", "Main text is read-only; record any discrepancy in Reconciliation Ledger for user review"],
  ];
  sheet.getRange("A4:C4").values = [["Result family", "Command / rule", "Calculation / guard"]];
  header(sheet.getRange("A4:C4"));
  sheet.getRange(`A5:C${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:C${4 + rows.length}`));
  sheet.getRange(`A5:C${4 + rows.length}`).format.rowHeight = 62;
  setWidths(sheet, { A: 28, B: 118, C: 78 });
}

const checks = [
  ["README", "A1:H29"],
  ["Training Completion", "A1:N13"],
  ["Cohort Status", "A1:L20"],
  ["T2 Shared Samples", "A1:V17"],
  ["Paired Semantic Audit", "A1:K24"],
  ["Run Provenance", "A1:D24"],
  ["Reconciliation Ledger", "A1:G24"],
  ["Source Manifest", `A1:F${4 + sourceRows.length}`],
];
const inspection = [];
for (const [sheetName, range] of checks) {
  const result = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range,
    include: "values,formulas",
    tableMaxRows: 40,
    tableMaxCols: 24,
    tableMaxCellChars: 300,
    maxChars: 24000,
  });
  inspection.push(result.ndjson);
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 500 },
  summary: "rebuild formula error scan",
  maxChars: 20000,
});
inspection.push(errors.ndjson);

const focusRanges = {
  "T2 Recovery Raw": "A1:H25",
  "T2 Ordering Raw": "A1:F25",
  "Bridge DL Calc": "A1:Q25",
  "T6 Coefficient Raw": "A1:M25",
  "T9 Raw": "A1:G25",
  "T10 Behavior Raw": "A1:L25",
  "T10 Weight Raw": "A1:H25",
  "Depth Per Leaf": "A1:I25",
  "Source Manifest": `A1:F${4 + sourceRows.length}`,
  "Training Completion": "A1:N13",
};
const sheetNames = [
  "README", "Paper Update", "Claim Index", "T2 Recovery", "T2 Recovery Raw", "T2 Ordering Raw",
  "Bridge Summary", "Bridge DL Calc", "T4 Layers", "T5 LoRA", "T6 Regression", "T6 Coefficient Raw",
  "T7 Heterogeneity", "T9 Within Task", "T8 Examples", "T9 Raw", "T10 PhyloLM", "T10 Behavior Raw",
  "T10 Weight Raw", "Empty Sensitivity", "Depth Summary", "Depth Per Leaf", "Source Manifest",
  "Reproduction Commands", "T2 Shared Samples", "T10 Shared PhyloLM", "Training Completion", "Cohort Status", "Run Provenance",
  "Archive Index", "Reconciliation Ledger", "Paired Semantic Audit",
];
for (let index = 0; index < sheetNames.length; index += 1) {
  const sheetName = sheetNames[index];
  const range = focusRanges[sheetName];
  const preview = await workbook.render({
    sheetName,
    range,
    autoCrop: range ? undefined : "all",
    scale: 0.8,
    format: "png",
  });
  const slug = sheetName.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "");
  await fs.writeFile(
    path.join(previewDir, `${String(index + 1).padStart(2, "0")}-${slug}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
await fs.writeFile(path.join(outputDir, "workbook_inspection.ndjson"), inspection.join("\n"));
await fs.writeFile(
  path.join(outputDir, "reconciliation_ledger.csv"),
  toCsv([
    ["id", "workbook_location", "issue", "evidence", "status", "action", "depends_on"],
    ...status.reconciliation,
  ]),
);

console.log(
  JSON.stringify(
    {
      output: outputPath,
      input_sha256: inputHash,
      sheets: sheetNames.length,
      source_rows: sourceRows.length,
      reconciliation_rows: status.reconciliation.length,
      formula_error_scan: errors.ndjson,
    },
    null,
    2,
  ),
);

#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [inputPath, resultsPath, paperPath, outputPath, previewDir] = process.argv.slice(2);
if (![inputPath, resultsPath, paperPath, outputPath, previewDir].every(Boolean)) {
  throw new Error("usage: finalize_reproducibility_workbook.mjs INPUT.xlsx RESULTS.json PAPER.tex OUTPUT.xlsx PREVIEW_DIR");
}

const results = JSON.parse(await fs.readFile(resultsPath, "utf8"));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
workbook.comments.setSelf({ displayName: "Shannon" });
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const sha256 = async (file) => crypto.createHash("sha256").update(await fs.readFile(file)).digest("hex");
const colors = {
  navy: "#17324D", teal: "#1F7A72", pale: "#EAF2F8", paleTeal: "#E7F4F1",
  gold: "#E9B44C", paleGold: "#FFF2CC", gray: "#5B6770", light: "#F5F7F9",
  border: "#CBD5E1", red: "#A33A3A", paleRed: "#FCE8E6", green: "#257A4A",
  paleGreen: "#E2F0D9", white: "#FFFFFF",
};

function resetSheet(name, range) {
  const sheet = workbook.worksheets.getOrAdd(name);
  sheet.showGridLines = false;
  try { sheet.getRange(range).unmerge(); } catch {}
  sheet.getRange(range).clear({ applyTo: "all" });
  return sheet;
}

function merge(sheet, address) {
  try { sheet.getRange(address).unmerge(); } catch {}
  sheet.getRange(address).merge();
}

function title(sheet, text, subtitle, lastColumn, fill = colors.navy) {
  merge(sheet, `A1:${lastColumn}1`);
  merge(sheet, `A2:${lastColumn}2`);
  sheet.getRange("A1").values = [[text]];
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill, font: { bold: true, color: colors.white, size: 16 }, verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 28;
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: colors.pale, font: { color: colors.gray, italic: true }, wrapText: true,
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 32;
}

function header(range) {
  range.format = {
    fill: colors.teal, font: { bold: true, color: colors.white }, wrapText: true,
    verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: colors.border },
  };
  range.format.rowHeight = 32;
}

function body(range) {
  range.format = {
    wrapText: true, verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: "#E6EBEF" } },
  };
}

function widths(sheet, spec) {
  for (const [column, width] of Object.entries(spec)) sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}

function pending(range) {
  range.format = { fill: colors.paleGold, font: { color: "#7F6000", bold: true }, wrapText: true };
}

function available(range) {
  range.format = { fill: colors.paleGreen, font: { color: colors.green, bold: true }, wrapText: true };
}

const paperHash = await sha256(paperPath);
const resultHash = await sha256(resultsPath);

// README status panel.
{
  const sheet = workbook.worksheets.getItem("README");
  sheet.getRange("A1").values = [["WeightTraits Reproducibility Workbook — Final-Numbers Reconciliation"]];
  sheet.getRange("A2").values = [["Source-bound workbook for the September 11 numbers-only manuscript draft; yellow cells are deliberately unresolved."]];
  sheet.getRange("B4").values = [["ELLMTrees-paper/iclr_draft_v4_numbers_20260911.tex; pushed commit 4035ed3"]];
  sheet.getRange("B6").values = [["Seven of eight direct Table 2 rows available; primary Llama and Flan behavior complete except Flan no-translation; matched PhyloLM complete"]];
  sheet.getRange("B10").values = [["Reproduction, comparison, and provenance control; raw artifacts and locked receipts remain numerical authority"]];
  sheet.getRange("B11").values = [["Numbers-only draft; surrounding manuscript wording intentionally unchanged"]];
  sheet.getRange("A13").values = [["Final-number gate status"]];
  sheet.getRange("B17").values = [["RESULTS AVAILABLE — 7 direct rows; full all-node and leaf sensitivity behavior"]];
  sheet.getRange("B18").values = [["PARTIAL — Flan no-translation inference/semantic jobs 187979–187980 pending"]];
  sheet.getRange("B19").values = [["PENDING — Llama full direct row, layer subsets, Figure 2 bootstrap, EOS sensitivity/depth"]];
  sheet.getRange("A28").values = [["Green cells are source-verified results already inserted in the numbers-only paper copy."]];
  sheet.getRange("A29").values = [["Yellow cells are placeholders retained only for direct comparison and must not be treated as final results."]];
  available(sheet.getRange("B17:B17"));
  pending(sheet.getRange("B18:B19"));
}

const phylolmByTag = new Map();
for (const row of results.phylolm) {
  const tag = row.group.includes("r8_") ? "llama_r8" : row.group.includes("r64_") ? "llama_r64" : "llama_full";
  phylolmByTag.set(tag, row);
}

// Live direct-lineage / recovery table.
{
  const sheet = resetSheet("T2 Shared Samples", "A1:U30");
  title(sheet, "Final-Numbers Table 2 — Direct Lineage and Recovery", "Common eligibility: 50 generated trees, 46 topology-eligible trees, 26 ordering trees. Yellow is unresolved.", "U");
  merge(sheet, "A3:U3");
  sheet.getRange("A3").values = [["Topology exclusions: 015, 020, 037, 047. Rank-biserial and branch-r use the separate 26-tree ordering set."]];
  sheet.getRange("A3:U3").format = { fill: colors.paleTeal, font: { bold: true, color: colors.navy }, wrapText: true };
  const headers = ["Model", "Variant", "Status", "Representation", "n rec", "n ord", "Rank-bis.", "SE", "r branch", "SE", "Weight clade", "SE", "PhyloLM clade", "SE", "PAER", "SE", "RF", "SE", "FN", "SE", "Source SHA-256"];
  sheet.getRange("A4:U4").values = [headers];
  header(sheet.getRange("A4:U4"));
  const rows = results.direct.filter((row) => row.rank_biserial).map((row) => {
    const phyloTag = row.id === "llama_r8" ? "llama_r8" : row.id === "llama_r64" ? "llama_r64" : null;
    const phylo = phyloTag ? phylolmByTag.get(phyloTag) : null;
    return [
      row.model, row.variant, "AVAILABLE", row.representation, row.n_recovery, row.n_ordering,
      row.rank_biserial.mean, row.rank_biserial.se, row.branch_r.mean, row.branch_r.se,
      row.clade_recovery.mean, row.clade_recovery.se, phylo?.clade_recovery ?? null, phylo?.clade_se ?? null,
      row.paer.mean, row.paer.se, row.rf.mean, row.rf.se, row.false_negative.mean, row.false_negative.se,
      row.local_source_sha256,
    ];
  });
  const fullPhylo = phylolmByTag.get("llama_full");
  rows.push(["Llama-3.2-1B", "Full fine-tuning", "PARTIAL — weight row pending", "full weight", 46, 26, null, null, null, null, null, null, fullPhylo.clade_recovery, fullPhylo.clade_se, null, null, null, null, null, null, fullPhylo.local_source_sha256]);
  sheet.getRange(`A5:U${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:U${4 + rows.length}`));
  available(sheet.getRange(`C5:C${3 + rows.length}`));
  pending(sheet.getRange(`A${4 + rows.length}:U${4 + rows.length}`));
  sheet.getRange(`E5:F${4 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`G5:J${4 + rows.length}`).format.numberFormat = "0.00";
  sheet.getRange(`K5:P${4 + rows.length}`).format.numberFormat = "0%";
  sheet.getRange(`Q5:T${4 + rows.length}`).format.numberFormat = "0.00";
  sheet.freezePanes.freezeRows(4);
  widths(sheet, { A: 20, B: 36, C: 28, D: 20, E: 10, F: 10, G: 12, H: 10, I: 12, J: 10, K: 14, L: 10, M: 15, N: 10, O: 12, P: 10, Q: 10, R: 10, S: 10, T: 10, U: 68 });
}

// Final behavior endpoint with both primary all-node and leaf-only sensitivity strata.
{
  const sheet = resetSheet("Final Behavior", "A1:P50");
  title(sheet, "Final Behavior Bridge", "DerSimonian–Laird pooled r with 95% CI. Primary = all trained-node pairs; leaf-only is a sensitivity analysis.", "P");
  const headers = ["Model", "Cohort", "Probe", "Role", "Status", "Primary r", "CI low", "CI high", "I²", "k", "Leaf r", "CI low", "CI high", "I²", "k", "Source SHA-256"];
  sheet.getRange("A4:P4").values = [headers];
  header(sheet.getRange("A4:P4"));
  const groups = new Map();
  for (const row of results.behavior) {
    const key = `${row.cohort}\u0000${row.probe}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const rows = [];
  for (const values of groups.values()) {
    const primary = values.find((row) => row.stratum === "all_trained_node_pairs") ?? values[0];
    const leaf = values.find((row) => row.stratum === "trained_leaf_pairs");
    const isPending = !Number.isFinite(primary.r_dl);
    rows.push([
      primary.model, primary.cohort, primary.probe, primary.role, isPending ? primary.status.toUpperCase() : "FINAL",
      primary.r_dl ?? null, primary.ci_low ?? null, primary.ci_high ?? null, primary.i2 ?? null, primary.k ?? null,
      leaf?.r_dl ?? null, leaf?.ci_low ?? null, leaf?.ci_high ?? null, leaf?.i2 ?? null, leaf?.k ?? null,
      primary.source_sha256 ?? null,
    ]);
  }
  sheet.getRange(`A5:P${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:P${4 + rows.length}`));
  for (let index = 0; index < rows.length; index += 1) {
    const row = 5 + index;
    if (String(rows[index][4]).startsWith("FINAL")) available(sheet.getRange(`E${row}`));
    else pending(sheet.getRange(`A${row}:P${row}`));
  }
  sheet.getRange(`F5:I${4 + rows.length}`).format.numberFormat = "0.000";
  sheet.getRange(`K5:N${4 + rows.length}`).format.numberFormat = "0.000";
  sheet.getRange(`J5:J${4 + rows.length}`).format.numberFormat = "0";
  sheet.getRange(`O5:O${4 + rows.length}`).format.numberFormat = "0";
  sheet.freezePanes.freezeRows(4);
  widths(sheet, { A: 20, B: 18, C: 20, D: 14, E: 30, F: 12, G: 12, H: 12, I: 12, J: 8, K: 12, L: 12, M: 12, N: 12, O: 8, P: 68 });
}

// Matched behavior-only topology comparison.
{
  const sheet = resetSheet("T10 Shared PhyloLM", "A1:J20");
  title(sheet, "Final Matched PhyloLM Comparison", "PhyloLM scored on the same common 46 topology-eligible trees as the corresponding weight rows.", "J");
  sheet.getRange("A4:J4").values = [["Condition", "n", "Clade", "SE", "PAER", "SE", "RF", "SE", "FN", "SE"]];
  header(sheet.getRange("A4:J4"));
  const labels = { llama_r8: "Llama LoRA qkv (r=8)", llama_r64: "Llama LoRA qkv (r=64)", llama_full: "Llama full fine-tuning" };
  const rows = ["llama_r8", "llama_r64", "llama_full"].map((key) => {
    const row = phylolmByTag.get(key);
    return [labels[key], row.n, row.clade_recovery, row.clade_se, row.paer, row.paer_se, row.rf, row.rf_se, row.false_negative, row.false_negative_se];
  });
  sheet.getRange("A5:J7").values = rows;
  body(sheet.getRange("A5:J7"));
  available(sheet.getRange("A5:J7"));
  sheet.getRange("B5:B7").format.numberFormat = "0";
  sheet.getRange("C5:F7").format.numberFormat = "0%";
  sheet.getRange("G5:J7").format.numberFormat = "0.00";
  widths(sheet, { A: 34, B: 10, C: 12, D: 10, E: 12, F: 10, G: 12, H: 10, I: 12, J: 10 });
}

const paperMap = [
  ["Table 2", "Flan LoRA key", "0.58; -0.63; 87%; 70%; 2.78; 0.37", "0.52; -0.53; 92%; 85%; 2.39; 0.17", "FINAL", results.direct[0].local_source_sha256],
  ["Table 2", "Flan LoRA q/k/v", "0.63; -0.66; 98%; 93%; 2.17; 0.07", "0.56; -0.62; 99%; 98%; 2.09; 0.02", "FINAL", results.direct[1].local_source_sha256],
  ["Table 2", "Flan LoRA q/k/v/o", "0.63; -0.66; 98%; 93%; 2.17; 0.07", "0.56; -0.62; 99%; 98%; 2.09; 0.02", "FINAL", results.direct[2].local_source_sha256],
  ["Table 2", "Flan all projections", "0.99; -0.91; 90%; 74%; 2.57; 0.26", "1.00; -0.91; 96%; 89%; 2.26; 0.11", "FINAL", results.direct[3].local_source_sha256],
  ["Table 2", "Flan full fine-tuning", "0.69; -0.70; 95%; 89%; 2.26; 0.11", "0.69; -0.71; 100%; 100%; 2.04; 0.00", "FINAL", results.direct[4].local_source_sha256],
  ["Table 2", "Llama LoRA r=8", "0.65; -0.65; 96%; 89%; 2.26; 0.11", "0.63; -0.64; 100%; 100%; 2.04; 0.00", "FINAL", results.direct[5].local_source_sha256],
  ["Table 2", "Llama LoRA r=64", "0.70; -0.67; 98%; 93%; 2.17; 0.07", "0.65; -0.67; 100%; 100%; 2.04; 0.00", "FINAL", results.direct[6].local_source_sha256],
  ["Table 2", "Llama full fine-tuning", "0.69; -0.68; 97%; 91%; 2.22; 0.09", "PENDING", "PENDING", null],
  ["Table 3", "Llama trained translation", "-0.29 [-.41,-.17]", "-0.20 [-.26,-.13]", "FINAL", "3e4f11026cc2338743ab869f1dfeaa7485cd2a368ac11ff27ca3acead81d82b9"],
  ["Table 3", "Llama held-out translation", "-0.33 [-.43,-.23]", "-0.23 [-.29,-.17]", "FINAL", "8a4a008eb7802965d6b15583db7ca6d4fdce4eb8120c2c2a4ea421484cc83c13"],
  ["Table 3", "Llama HellaSwag / ARC-C / MMLU / TruthfulQA", "-.20 / -.08 / -.09 / -.16", "-.17 / -.20 / -.19 / -.22", "FINAL", "See Final Behavior"],
  ["Table 3", "Flan trained translation", "-0.26 [-.35,-.16]", "-0.10 [-.16,-.04]", "FINAL", "a7b5ef7b043d2753a06b06daf3c36cb4cdcbb460f158146095eaeadb92e289e8"],
  ["Table 3", "Flan held-out translation", "+0.11 [+.02,+.21]", "PENDING", "PENDING", "Jobs 187979–187980"],
  ["Table 3", "Flan HellaSwag / ARC-C / MMLU / TruthfulQA", "-.29 / -.34 / -.30 / -.33", "+.09 / -.14 / -.16 / -.17", "FINAL", "See Final Behavior"],
  ["PhyloLM", "Llama r8 / r64 / full clade recovery", "35% / 24% / 52%", "35% / 24% / 52%", "FINAL — unchanged after full check", results.phylolm[0].local_source_sha256],
];

// Side-by-side paper comparison requested by the user.
{
  const sheet = resetSheet("Paper Number Map", "A1:G60");
  title(sheet, "Paper Number Comparison — Old vs New", "Values are displayed at paper precision. Yellow rows are still unresolved and remain placeholders in the numbers-only draft.", "G");
  sheet.getRange("A4:G4").values = [["Location", "Row / cell", "Old paper", "New draft", "Changed?", "Status", "Source / receipt"]];
  header(sheet.getRange("A4:G4"));
  sheet.getRange(`A5:D${4 + paperMap.length}`).values = paperMap.map((row) => row.slice(0, 4));
  sheet.getRange(`F5:G${4 + paperMap.length}`).values = paperMap.map((row) => row.slice(4, 6));
  sheet.getRange(`E5:E${4 + paperMap.length}`).formulas = paperMap.map((_, index) => [`=IF(C${5 + index}=D${5 + index},"NO","YES")`]);
  body(sheet.getRange(`A5:G${4 + paperMap.length}`));
  for (let index = 0; index < paperMap.length; index += 1) {
    const row = 5 + index;
    if (paperMap[index][4] === "PENDING") pending(sheet.getRange(`A${row}:G${row}`));
    else available(sheet.getRange(`F${row}`));
  }
  sheet.freezePanes.freezeRows(4);
  widths(sheet, { A: 16, B: 44, C: 44, D: 44, E: 12, F: 30, G: 72 });
}

// Explicit pending register mirrors every yellow location in the paper.
{
  const sheet = resetSheet("Pending Results", "A1:F30");
  title(sheet, "Pending Final Results", "Only these result families remain unresolved in the numbers-only paper and workbook.", "F", colors.gold);
  sheet.getRange("A4:F4").values = [["Area", "Paper location", "Missing result", "Current state", "Action / job", "Paper marking"]];
  header(sheet.getRange("A4:F4"));
  const rows = [
    ["Direct", "Table 2 Llama full-FT row", results.pending[0], "not yet collected", "run final direct analysis and certify common eligibility", "yellow row"],
    ["Behavior", "Table 3 Flan held-out translation", results.pending[1], "array/semantic chain pending", "Wright 187979–187980", "yellow cell and heterogeneity row"],
    ["Layers", "LayerTrace subset table and repeated claims", results.pending[2], "pending", "refresh on corrected Flan full-FT cohort", "yellow rows / inline values"],
    ["Figure 2", "Recovery diagnostic figure claims", results.pending[3], "pending", "collect eight estimators and paired bootstrap", "yellow inline values"],
    ["EOS", "Sensitivity and depth appendix tables", results.pending[4], "pending", "recompute from corrected behavior outputs", "yellow rows / inline values"],
    ["Certification", "Eight-row Table 2 rollup", results.pending[5], "blocked by Llama full row", "run collector after direct row lands", "yellow aggregate ranges"],
  ];
  sheet.getRange("A5:F10").values = rows;
  body(sheet.getRange("A5:F10"));
  pending(sheet.getRange("A5:F10"));
  widths(sheet, { A: 18, B: 42, C: 72, D: 30, E: 58, F: 34 });
}

// Hash-bound live source index for every inserted value.
{
  const sheet = resetSheet("Final Source Manifest", "A1:F120");
  title(sheet, "Final Result Source Manifest", "Hashes bind workbook values to the collected result receipts and the numbers-only paper copy.", "F");
  sheet.getRange("A4:F4").values = [["Family", "Model / condition", "Source path", "SHA-256", "Status", "Use"]];
  header(sheet.getRange("A4:F4"));
  const rows = [
    ["Paper", "numbers-only draft", paperPath, paperHash, "CURRENT", "comparison target"],
    ["Collector", results.schema, resultsPath, resultHash, "CURRENT", "workbook import contract"],
    ...results.direct.filter((row) => row.local_source_sha256).map((row) => ["Direct", `${row.model} — ${row.variant}`, row.local_source_file, row.local_source_sha256, "AVAILABLE", "T2 Shared Samples"]),
    ["Direct", "Llama-3.2-1B — Full fine-tuning", null, null, "PENDING", "T2 Shared Samples"],
    ...results.behavior.filter((row) => row.source_sha256).map((row) => ["Behavior", `${row.cohort} — ${row.probe} — ${row.stratum}`, row.local_source_file, row.source_sha256, row.status.toUpperCase(), "Final Behavior"]),
    ...results.phylolm.map((row) => ["PhyloLM", row.group, row.local_source_file, row.local_source_sha256, "FINAL", "T10 Shared PhyloLM"]),
  ];
  sheet.getRange(`A5:F${4 + rows.length}`).values = rows;
  body(sheet.getRange(`A5:F${4 + rows.length}`));
  sheet.freezePanes.freezeRows(4);
  widths(sheet, { A: 16, B: 52, C: 86, D: 68, E: 20, F: 28 });
}

workbook.recalculate();

const checks = [
  ["README", "A1:H29"], ["T2 Shared Samples", "A1:U12"], ["Final Behavior", "A1:P20"],
  ["T10 Shared PhyloLM", "A1:J8"], ["Paper Number Map", "A1:G20"],
  ["Pending Results", "A1:F11"], ["Final Source Manifest", "A1:F60"],
];
const inspection = [];
for (const [sheetId, range] of checks) {
  const result = await workbook.inspect({ kind: "table", sheetId, range, include: "values,formulas", tableMaxRows: 70, tableMaxCols: 24, tableMaxCellChars: 300, maxChars: 30000 });
  inspection.push(result.ndjson);
}
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 500 }, summary: "final workbook formula error scan", maxChars: 20000 });
inspection.push(errors.ndjson);

for (let index = 0; index < checks.length; index += 1) {
  const [sheetName, range] = checks[index];
  const preview = await workbook.render({ sheetName, range, scale: 0.85, format: "png" });
  const slug = sheetName.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replaceAll(/^-|-$/g, "");
  await fs.writeFile(path.join(previewDir, `${String(index + 1).padStart(2, "0")}-${slug}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
await fs.writeFile(path.join(path.dirname(outputPath), "final_workbook_inspection.ndjson"), inspection.join("\n"));
console.log(JSON.stringify({ output: outputPath, paper_sha256: paperHash, results_sha256: resultHash, formula_error_scan: errors.ndjson, rendered_sheets: checks.length }, null, 2));

# PhyloLM provenance and redistribution decision

The faithful baseline targets the public PhyloLM repository at commit
`8c70edf062a0adce2a3e6c8c79cd23a645fd0905`:

- Source: `https://github.com/Nicolas-Yax/PhyloLM`
- Upstream repository license at the pinned commit: GNU GPL v3
- Released archive: `data/kl_pop/math.zip`
- Upstream description: the `math` genes are extracted from OpenWebMath and are the genome used for
  the paper's Figures 3 and 4.
- Upstream setup source for OpenWebMath: the `open-web-math` test split in
  `EleutherAI/proof-pile-2`.
- Extracted 4,096-gene JSON used by the existing ELLMTrees reproduction:
  SHA-256 `98d553eb4042cb084a7a17ce3a932ee147629698315e8cc2910dde6b05f84445`.

WeightTraits is MIT-licensed. Do not commit the extracted gene pool into WeightTraits until its
redistribution terms have been reviewed for compatibility. Smoke and production runs may reference a
separately retained, checksum-verified copy of the upstream artifact; generated genome metadata must
record the upstream commit, source path, source hash, sample seed, and gene count. This preserves
experimental reproducibility without silently relicensing the upstream artifact.

The WeightTraits implementation is a clean re-expression of the documented population-frequency,
Nei-similarity, distance, sampling, and inference contracts. It does not copy the upstream notebook
source into this repository.

## Scripts (October 2026)

`scripts/phylolm/` holds the generation, distance, and scoring scripts behind the paper's PhyloLM
results. Their core calls `weighttraits.behavior.phylolm` for allele frequencies and Nei
similarity, replacing an earlier core that carried three functions copied from the upstream
notebook. The gene pool is located through `PHYLOLM_GENE_POOL` or the git-ignored
`scripts/phylolm/genes_math.json`, and checked against the hash above.

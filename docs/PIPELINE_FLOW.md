# Pipeline Flow

The rebuild separates five ideas that were too entangled in ELLMTrees.

## 1. Make Trees

Tree generation produces topology only:
- ELLMTrees-compatible fairly balanced depth-2 defaults
- fixed hand-authored trees
- chains
- balanced trees
- Poisson branching-process trees
- pruned binary backbones, e.g. grow a 32-leaf binary tree, prune, and contract edges
- future generators such as birth-death, coalescent, task-blocked, or adversarial topologies

Output: a lineage manifest with `node_id`, `parent_id`, `depth`, and `path`.

Important topology controls:
- `min_depth`: reject trees that are too shallow.
- `min_leaves`: reject trees with too little RF/permutation-test resolution.
- pruning parameters: control early stopping and subtree loss.
- edge contraction: deliberately creates polytomies, because binary-only trees are not enough.

## 2. Pick Tasks And Datasets

Task assignment is a separate layer. A single topology should be reusable with different task/dataset policies:
- task per edge
- task per node
- task per clade
- task pools with replacement
- held-out task controls
- cloned topology with changed dataset pool

Output: an enriched training manifest.

The first candidate pools are tracked in [Task And Dataset Candidates](TASK_DATA_CANDIDATES.md). Harder pools should be added only after split/license/loader audits.

## 3. Train

Training consumes an enriched manifest and produces checkpoints. It should not care how the topology was generated.

Output: checkpoints and a training ledger.

The first trainer control-plane implementation is documented in [Trainer Design](TRAINER_DESIGN.md). It validates prompt resolution, LoRA/full-finetune artifact expectations, and stopping/warning rules before long jobs start.

## 4. Analyze Weights

Whitebox analysis consumes checkpoints and the lineage manifest:
- read weights/adapters
- compute distance cubes
- reconstruct trees with `wt reconstruct-tree`
- score recovery against truth
- aggregate across runs

Output: weight distances, reconstructed trees, recovery summaries, figures/tables.

For LoRA, the default analysis object is cumulative adapter displacement along the root-to-node path, not the last edge's raw adapter increment. See [LoRA Distance Model](LORA_DISTANCE_MODEL.md).

The first distance-cube implementation is documented in [Streaming Distance Cubes](STREAMING_DISTANCE_CUBES.md).

## 5. Analyze Behaviors

Behavior analysis branches after training:

1. prompts/inference
2. embeddings or task metrics
3. behavior distance matrices
4. tree reconstruction or distance-vs-behavior regression

This is related to whitebox analysis but should remain a separate path with its own data audits.

## Blackbox

Blackbox experiments are separate because there is no training lineage execution step inside this repo. They start at model selection and prompts/inference, then follow the behavior distance path. When ground truth exists for a model family, blackbox outputs can be scored against it, but blackbox should not be coupled to synthetic fine-tuning manifests.

# Manuscript evidence map

Four claims that the code can currently support, and what is still missing
before they go in a paper. "Support" means a reader can recompute the check.
It does not mean the claim is the only way to do the analysis.

## 1. One graph object for a neuron and for a population

- **Code:** `core/tree.py`, `api/tree_class.py`, `api/forest_class.py`.
  Methods on `Tree` call `ops/`, which call `utils/` on the graph-tool graph.
- **Tests:** `tests/test_validation/test_reference_morphologies.py` checks
  `tree.count_nodes()` against `nr.count_nodes(tree)` on the single-node tree.
  Broader API parity is in the existing tree and forest tests.
- **Docs:** `docs/development/architecture.md`.
- **Gap:** `describe()` still imports `ops` from `utils`. Listed in the
  layer-import allowlist. Not a numerical error.

## 2. Metrics with explicit invariance metadata

- **Code:** `utils/metrics/registry.py`. Aliases in `api_names.py`.
- **Tests:** `tests/test_utils/test_metrics/test_invariance.py`,
  `test_registry.py`.
- **Docs:** `docs/reference/metrics.md`.
- **Gap:** Not every metric has an analytical fixture. PCA and other
  structured returns do not have a single rotation-invariance bit; the
  registry notes say so. Do not write "all metrics are validated" .

## 3. Reduction with a stated preservation list

- **Code:** `utils/graph_utils/graph_editing.py`, `reduction_map.py`,
  synapse transfer in `core/synapses.py`.
- **Tests:** `tests/test_synapses/test_reduction_transfer.py`,
  `test_irregular_curve_reduction_keeps_cable_not_the_chord`.
- **Docs:** `publication/validation_summary.md`, synapses tutorial.
- **Gap:** Radial Sholl on a reduced tree is the chord geometry. That is
  documented in `ops/tree_graphs/tree_sholl.py` and should be a limitation
  paragraph, not a bug.

## 4. Synapses and forest connectivity on the same objects

- **Code:** `core/synapses.py`, `ops/tree_graphs/tree_synapses.py`,
  `ops/forest_ops/forest_connectivity.py`.
- **Tests:** `tests/test_synapses/`. The edge-index regression is the one
  that matches the old tutorial failure.
- **Docs:** `docs/tutorials/synapses.ipynb`.
- **Gap:** No paired comparison with NAVis connectivity tables. `cleft_id`
  uniqueness is a data contract the caller has to meet.

## Do not claim

- That NeuRosetta is faster than NAVis or NeuroM.
- That every registered metric has an analytical proof.
- A DOI or a PyPI release. Neither exists.

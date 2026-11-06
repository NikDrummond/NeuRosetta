# Reproducing the validation checks

From the repository root, with the `nr` environment active
(`mamba activate nr`). graph-tool has to come from that environment.
`pip install neurosetta` does not provide it.

```bash
# Scientific checks added for this pass, plus the synapse and metric suites
# they sit next to. --no-cov skips the coverage run.
python -m pytest \
  tests/test_validation \
  tests/test_architecture \
  tests/test_synapses/test_reduction_transfer.py \
  tests/test_synapses/test_connectivity.py \
  tests/test_tree_graphs/test_path_lengths.py \
  tests/test_utils/test_metrics/test_invariance.py \
  tests/test_utils/test_graph_utils/test_sholl.py \
  tests/test_utils/test_graph_utils/test_subtree_metrics.py \
  -q --no-cov

# Full suite used in CI (slower; coverage on). Needs a display or xvfb
# because a few tests construct a vedo plotter.
pytest -q
```

Optional, and not part of CI:

```bash
# Prints not_run unless navis or neurom imports.
python benchmarks/comparisons/compare_metrics.py

python benchmarks/bench_subtree_metrics.py
python benchmarks/bench_synapse_mapping.py
```

Headless examples:

```bash
python examples/quantitative_morphology.py
python examples/morphology_connectivity.py
```

Those write under `examples/output/`, which is gitignored.

The audit of what those commands do and do not show is
`docs/development/publication_readiness.md`.

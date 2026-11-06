# Cross-tool comparisons

`compare_metrics.py` builds the straight reference chain (cable 3, 4 nodes)
and records whether NAVis and NeuroM import.

Definitions that have to match before a numeric diff is meaningful are in
`publication/software_comparison.md`. The script does not convert a NAVis
neuron or a NeuroM morphology yet. An import check is not an agreement.

Install neither package into the NeuRosetta runtime to make this script
pass. Run it in an environment that already has graph-tool plus the
optional tool:

```bash
python benchmarks/comparisons/compare_metrics.py
```

Output: `benchmarks/results/comparison.json` (gitignored).

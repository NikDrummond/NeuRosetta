# Software comparison

Checked against the NeuRosetta source and the public documentation of NAVis
and NeuroM. This is not a feature-count contest, and it is not a benchmark.

NAVis and NeuroM were not imported in the environment used for the
6 Nov 2026 validation pass. Where the table says "not run", nobody has
produced a paired number.

| Topic | NeuRosetta | NAVis | NeuroM | Comparison status |
| --- | --- | --- | --- | --- |
| Morphology object | graph-tool directed tree (`Tree`) | `TreeNeuron` / mesh neurons | Morphology loader, sections | Different data models. Compare measurements, not objects. |
| Population | `Forest` | `NeuronList` | Populations in the analysis API | Same idea, different types. |
| Cable length | Sum of `Path_length`. On a reduced tree that is accumulated section cable, not the chord. | Usually the sum of segment lengths of the loaded neuron. | Section lengths. | Comparable on an unreduced SWC if both sum Euclidean segments. Not comparable if one side has collapsed edges and still reports cable. Not run. |
| Node / branch counts | Nodes are graph vertices. A branch has out-degree > 1. A bifurcation has out-degree == 2. | NAVis branch counts follow its section model. | NeuroM has its own section and bifurcation definitions. | Do not treat "branch" as the same word. NeuRosetta's split is tested on a trifurcation. |
| Strahler | Leaves are 1. One child inherits. Two or more children at the max order increment. | NAVis exposes Strahler. | NeuroM exposes Strahler. | The leaf-order convention (1 vs 0) has to be checked before a numeric diff means anything. Not run. |
| Sholl | Shell crossings. Radial uses segment–sphere intersections and ignores tangencies. Path mode uses cable. Radius 0 is 0. | NAVis Sholl is widely used and has its own sampling defaults. | NeuroM Sholl is section-based. | Same name, easy to mis-compare. NeuRosetta's path-mode crossings are tested on a unit chain. |
| Synapses and connectivity | Synapses are a table on the tree. Forest connectivity is a directed graph of counts. `cleft_id` dedup is global. | NAVis has connectors and connectivity helpers aimed at CATMAID / FlyWire tables. | Not the main object of NeuroM. | NeuRosetta's direction rules are tested. No paired run against NAVis. |
| Reduction | Deletes degree-2 vertices, keeps cable on `Path_length`, can move synapse edge indices with the cable. | NAVis can downsample; the length contract is not the same object. | NeuroM resamples sections. | Do not claim another package preserves the same quantities. |
| Dependency | graph-tool via conda-forge. Not on PyPI. | pip-installable. | pip-installable. | This is a real installation difference, not a quality score. |

## What would make a numeric comparison valid

1. One SWC, unreduced, units written down.
2. The formula on both sides, including whether multifurcations count as
   bifurcations and whether leaves have Strahler 1.
3. An analytical value from `publication/validation_summary.md` for that
   formula. If both packages match each other and miss the analytical
   value, both are wrong.
4. The script `benchmarks/comparisons/compare_metrics.py` prints
   `not_run` until NAVis or NeuroM imports.

## NeuRosetta limitations relative to those packages

- No claim of broader file-format coverage.
- GUI and graph-tool make a minimal install heavier.
- Cross-tool numbers in this repository have not been produced yet.

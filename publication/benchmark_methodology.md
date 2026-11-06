# Benchmark methodology

No speed claim in the manuscript should be copied from this file. It says
how to measure. Numbers belong in a results file produced by a run, with
the environment block that the script writes.

## What to time

Separate loading from the algorithm. Build or load the graph first, then
time the call. For Numba, run the function once before the timed loop and
report that warm-up separately from the steady times.

Operations worth timing, because they are the ones a methods section would
mention:

- SWC or `.nr` load
- `get_total_cable_length`, `get_strahler_order`, `get_sholl_profile`
- `get_root_distance`
- `get_reduced_tree`
- `map_synapses`
- `get_connectivity_graph`

Existing scripts:

- `benchmarks/bench_subtree_metrics.py`
- `benchmarks/bench_synapse_mapping.py`

They print a minimum of three repeats. They do not record the machine.
Treat them as developer checks.

## Cross-tool timing

Only compare an operation after `publication/software_comparison.md` says
the definitions match. Same trees, same units, warm-up called out, same
thread count. graph-tool uses OpenMP; pin it or record `OMP_NUM_THREADS`.

NAVis and NeuroM are not dependencies of NeuRosetta. Install them in a
separate environment if you run `benchmarks/comparisons/compare_metrics.py`.

## What a results file must contain

- Python version, NeuRosetta version, git commit
- NumPy, graph-tool, Numba versions
- OS, CPU, RAM
- `OMP_NUM_THREADS`
- dataset size (nodes, edges, neurons, synapses)
- repetitions, warm-up excluded or not
- wall time per repeat, not a single number with the setup included

`benchmarks/results/` is gitignored. Commit a results file only when the
machine and the command are written next to it.

## CI

Default CI runs the test suite, not these scripts. A timing job on a shared
runner is not a performance result.

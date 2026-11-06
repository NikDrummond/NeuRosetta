# Changelog

## Unreleased

- Synapse root cable distance is unchanged by reduction on hashed graphs.
  `edge_index` is the graph-tool edge index, not the row order of
  `Graph.get_edges()`.
- `get_root_distance(bind=False)` and `get_vertex_degrees` return an owned
  NumPy array. They no longer return a view of a freed graph-tool map.
- Reduced section lengths are stored as float64.
- Reroot copies each segment's cable onto that same undirected segment.
- Metric method aliases live in `neurosetta.api_names` so the metric
  registry does not import `ops`.
- Reference morphologies and tests under `tests/test_validation/`.
- Synapse tutorial: cable check matches the implementation, and reciprocal
  `cleft_id`s are unique across the two neurons.

## 0.1.0

- Current development line. Not published on PyPI or conda-forge.

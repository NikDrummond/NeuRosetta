# Configuration

NeuRosetta keeps library-wide defaults in a **mutable singleton** accessed via
{func}`~neurosetta.get_settings`, updated with {func}`~neurosetta.configure`, and
temporarily overridden with {func}`~neurosetta.settings`.

These settings control parallel I/O, forest batch operations, graph-tool OpenMP,
and vedo plotting defaults. They do **not** replace per-object metadata such as
tree spatial units — see {doc}`units`.

## Entry points

```python
import neurosetta as nr

nr.configure(parallel_io=True, show_progress=True, openmp_num_threads=4)
s = nr.get_settings()
s.parallel.forest          # scoped parallel default for Forest batch ops
s.openmp.num_threads
s.vedo.backend

with nr.settings(parallel_forest=False):
    forest.count_nodes()   # sequential for this block only

with nr.openmp_context(num_threads=1):
    ...                    # temporary graph-tool thread cap
```

| Function                              | Purpose                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| {func}`~neurosetta.configure`         | Update global settings in place                                                                                   |
| {func}`~neurosetta.get_settings`      | Read the singleton                                                                                                |
| {func}`~neurosetta.settings`          | Context manager for temporary parallel/progress overrides                                                         |
| {func}`~neurosetta.sync_vedo_runtime` | Push vedo settings to ``vedo.settings`` (called automatically by {class}`~neurosetta.ops.plotting.viewer.Viewer`) |
| {func}`~neurosetta.openmp_context`    | Temporary graph-tool OpenMP overrides                                                                             |
| {func}`~neurosetta.openmp_enabled`    | Whether graph-tool was built with OpenMP                                                                          |

## Resolution order

When a function accepts ``parallel``, ``max_workers``, or ``show_progress``:

1. **Explicit keyword** passed to the call
2. **Context override** from ``with nr.settings(...)``
3. **Global singleton** from ``nr.configure(...)`` or environment variables
4. **Scoped default** (parallel only — see table below)

Pass ``None`` (or omit the argument where supported) to defer to this chain.

NeuRosetta uses **thread pools only** — there is no process-pool backend.

## Parallel execution

Parallel settings live on ``get_settings().parallel``.

| Setting            | ``configure`` keyword | Default   | Applies to                                                                               |
| ------------------ | --------------------- | --------- | ---------------------------------------------------------------------------------------- |
| I/O parallel       | ``parallel_io``       | ``False`` | {func}`~neurosetta.load`, {func}`~neurosetta.save`, SWC/mesh import/export               |
| Forest parallel    | ``parallel_forest``   | ``True``  | {class}`~neurosetta.api.Forest` batch methods and ``Forest.apply`` / ``Forest.build_3d`` |
| Fallback scope     | ``parallel_default``  | ``False`` | Any caller using scope ``"default"``                                                     |
| All scopes at once | ``parallel``          | —         | Sets io, forest, and default together                                                    |
| Worker limit       | ``max_workers``       | ``None``  | Thread-pool size; ``None`` uses the executor default                                     |
| Progress bars      | ``show_progress``     | ``False`` | tqdm bars in parallel/sequential loops                                                   |

``Forest.map()`` always runs sequentially (``parallel=False``) by design.

Forest batch methods (for example ``forest.count_nodes()``) pass ``parallel=None`` by
default so they follow the forest-scoped setting above.

## graph-tool OpenMP

OpenMP settings live on ``get_settings().openmp``. Non-``None`` fields are pushed to
graph-tool when settings are first loaded and whenever ``configure(..., apply_openmp_now=True)``
is called (the default).

| Setting            | ``configure`` keyword     | Default  | Notes                                                    |
| ------------------ | ------------------------- | -------- | -------------------------------------------------------- |
| Thread count       | ``openmp_num_threads``    | ``None`` | Maps to ``graph_tool.openmp.openmp_set_num_threads``     |
| Schedule           | ``openmp_schedule``       | ``None`` | ``"static"``, ``"dynamic"``, ``"guided"``, or ``"auto"`` |
| Schedule chunk     | ``openmp_schedule_chunk`` | ``0``    | Chunk size paired with schedule                          |
| Parallel threshold | ``openmp_thresh``         | ``None`` | Minimum problem size for OpenMP parallelisation          |

Use {func}`~neurosetta.openmp_context` for short-lived overrides inside hot loops
without changing the global singleton.

On import, NeuRosetta sets ``OMP_WAIT_POLICY=passive`` if unset (reduces idle CPU
spin-wait in threaded workloads).

## vedo plotting

vedo settings live on ``get_settings().vedo``. They are **stored** by
{func}`~neurosetta.configure` but **not** applied to vedo automatically — vedo is
updated when a {class}`~neurosetta.ops.plotting.viewer.Viewer` is created (via
{func}`~neurosetta.sync_vedo_runtime`), or when you call that function yourself.

| Setting             | ``configure`` keyword        | Default         | Notes                                      |
| ------------------- | ---------------------------- | --------------- | ------------------------------------------ |
| Backend             | ``vedo_backend``             | ``"vtk"``       | e.g. ``"vtk"``, ``"k3d"``                  |
| Parallel projection | ``vedo_parallel_projection`` | ``True``        | Orthographic-style camera                  |
| Offscreen           | ``vedo_offscreen``           | ``False``       | Headless rendering default for new viewers |
| Window size         | ``vedo_window_size``         | ``(1200, 800)`` | Used when ``Viewer(size=None)``            |
| Background          | ``vedo_bg``                  | ``"white"``     | Used when ``Viewer(bg=None)``              |
| Multi-samples       | ``vedo_multi_samples``       | ``None``        | Anti-aliasing; only applied when set       |

`vedo` backends can be found [here](https://vedo.embl.es/site/index.html#running-in-a-jupyter-notebook)

## Default units

| Setting               | ``configure`` keyword | Default             | Notes                                                                       |
| --------------------- | --------------------- | ------------------- | --------------------------------------------------------------------------- |
| Library default units | ``default_units``     | ``"dimensionless"`` | Reserved for future I/O defaults; does **not** assign units to loaded trees |

Tree and forest units remain explicit — use ``tree.set_units(...)`` or
``import_swc(..., set_units=...)``. See {doc}`units`.

## Environment variables

Read once on the first call to {func}`~neurosetta.get_settings`. Boolean values accept
``1``/``0``, ``true``/``false``, ``yes``/``no``, ``on``/``off`` (case-insensitive).

| Variable                                | Maps to                          |
| --------------------------------------- | -------------------------------- |
| ``NEUROSETTA_PARALLEL``                 | All parallel scopes              |
| ``NEUROSETTA_PARALLEL_IO``              | ``parallel.io``                  |
| ``NEUROSETTA_PARALLEL_FOREST``          | ``parallel.forest``              |
| ``NEUROSETTA_PARALLEL_DEFAULT``         | ``parallel.default``             |
| ``NEUROSETTA_MAX_WORKERS``              | ``parallel.max_workers``         |
| ``NEUROSETTA_SHOW_PROGRESS``            | ``parallel.show_progress``       |
| ``NEUROSETTA_DEFAULT_UNITS``            | ``default_units``                |
| ``NEUROSETTA_GT_OPENMP_THREADS``        | ``openmp.num_threads``           |
| ``NEUROSETTA_GT_OPENMP_SCHEDULE``       | ``openmp.schedule``              |
| ``NEUROSETTA_GT_OPENMP_SCHEDULE_CHUNK`` | ``openmp.schedule_chunk``        |
| ``NEUROSETTA_GT_OPENMP_THRESH``         | ``openmp.thresh``                |
| ``NEUROSETTA_VEDO_BACKEND``             | ``vedo.backend``                 |
| ``NEUROSETTA_VEDO_PARALLEL_PROJECTION`` | ``vedo.use_parallel_projection`` |
| ``NEUROSETTA_VEDO_OFFSCREEN``           | ``vedo.offscreen``               |

``configure()`` after startup overrides environment values for the current process.

## Examples

**Notebook session — parallel forest metrics with progress:**

```python
import neurosetta as nr

nr.configure(show_progress=True, max_workers=8)
forest = nr.load_example_data()
forest.count_nodes()  # parallel + tqdm (forest scope default)
```

**HPC / CI — cap graph-tool threads via environment:**

```bash
export NEUROSETTA_GT_OPENMP_THREADS=4
export NEUROSETTA_PARALLEL_IO=1
python analysis.py
```

**Temporary sequential I/O inside a parallel pipeline:**

```python
with nr.settings(parallel_io=False):
    tree = nr.load("single_tree.nr")
```

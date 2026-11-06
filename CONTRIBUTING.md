# Contributing

Bug reports, tests, and docs fixes are welcome. Keep the layer rules in
`docs/development/architecture.md`: `utils` does not import `ops` or `api`.
`tests/test_architecture/test_layer_imports.py` fails if a new import goes
the wrong way. The allowlist is deliberate; do not add to it to silence the
test.

```bash
mamba env create -f environment.yml   # once
mamba activate nr
python -m pip install -e ".[dev]"
python -m pytest tests/test_validation tests/test_architecture -q --no-cov
ruff check src/neurosetta tests
```

graph-tool is installed by the conda environment, not by pip.

Do not commit `.history/`, `benchmarks/results/`, or `examples/output/`.

A numerical change needs a test whose expected value was not computed by
the function being changed. Say so in the test docstring if the tolerance
is looser than `1e-12`, and why.

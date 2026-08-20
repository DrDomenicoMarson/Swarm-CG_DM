# Development notes

Install the package and focused development tools with the repository's
supported Python environment:

```bash
/Users/dm/miniforge3/envs/md/bin/python3 -m pip install -e '.[dev]'
```

Run the non-integration suite and lint checks after each focused change:

```bash
/Users/dm/miniforge3/envs/md/bin/python3 -m pytest tests --ignore=tests/integration
/Users/dm/miniforge3/envs/md/bin/python3 -m ruff check swarmcg tests
```

Run `tests/integration/test_gromacs_integration.py` with `gmx` on `PATH` after
changes to topology parsing/writing, simulation execution, scoring, or history
serialization. CI validates both GROMACS 2025.4 and the current packaged
release.

## Internal architecture

- `swarmcg.topology` owns the mutable typed topology model and semantic ITP
  parser/writer.
- `ParameterVectorLayout` is the sole definition of PSO vector ordering,
  dimensions, bounds, encoding, and topology application.
- Model comparison is separated into trajectory preparation, typed per-group
  distribution comparisons, class-wise score aggregation, and rendering.
- `eval_function(parameters, context) -> float` remains the FST-PSO seam and
  delegates workspace staging, simulation, scoring, artifacts, and history.

These are internal interfaces and are not re-exported from `swarmcg`.

## Optimization history

New runs write only `.internal/optimization_history.jsonl`. Every evaluation is
one flushed strict-JSON object with `schema_version: 1`, identifiers, status,
score breakdown, observables, pairwise geometry scores, canonical parameters,
timings, and optional failure details. Unavailable and non-finite values are
written as JSON `null`; non-standard `NaN` tokens are never emitted.

The monitor supports only this JSONL schema. It may ignore a syntactically
truncated final line from an active writer, but any malformed complete record
is an error. No compatibility reader exists for legacy whitespace recap files.

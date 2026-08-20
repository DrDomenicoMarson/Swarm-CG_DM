# AGENTS.md

## Project Purpose
This is a fork of a Git repo which implemented what you can find in the (reference_papers/Swarm-CG Automatic Parametrization of Bonded Terms in MARTINI-Based Coarse-Grained Models of Simple to Complex Molecules via Fuzzy Self-Tuning Particle Swarm Optimization.pdf) paper. I updated some code to work with recent python/pandas/numpy, and I also added the capability to deal with more gromacs functionals.

## Environment
- Do not use conda run or mamba run.
    - Conda is not installed, we only use mamba.
- for testing, prefere a 'pytest' approach
- If a dependency is missing, ask for it to be installed.
- Use the mamba "md" environment python interpreter for Python commands in this repo. As an example, if you are in MacOs the interpreter should be:
```bash
/Users/dm/miniforge3/envs/md/bin/python3
```

## Code Style
- Prefer dataclasses over dictionaries when they make the data model clearer.
- Do not introduce config dataclasses when a normal function signature is simpler.
- Public functions and classes should have docstrings.
- Docstrings should document arguments and return values where applicable.
- Update docstrings whenever changing a function or class behavior.
- Keep refactors focused, but do not preserve legacy APIs or CLI behavior just for compatibility.


## Project Rules
- This is a living personal project.
- Prefer clear current behavior over backward compatibility.
- It is acceptable to change legacy code, APIs, tests, examples, and CLI behavior when doing so improves correctness or usability.
- Tests, when present, should validate the current intended behavior after the change.
- If a test only protects old behavior, update or remove it.
- When changing CLI/API behavior, update relevant docs, README examples, or tests.


## Repository Layout
- Package code lives in swarmcg.
- Tests live in tests.
- Examples live in examples.
- User-facing documentation lives in README.md and docs.

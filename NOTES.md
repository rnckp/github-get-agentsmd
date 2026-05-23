## Test Import Path

The project keeps executable scripts and shared helpers at the repository root.
Pytest is configured with `pythonpath = ["."]` in `pyproject.toml` so tests can
import those top-level modules the same way `uv run python ...` does.

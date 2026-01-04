# AGENTS Guidelines for This Repository (Python)

Guidelines for agent-assisted development. Python 3.12-3.13, managed with `uv`.

## Environment Management with `uv`

```bash
uv sync                 # Create/sync environment to lockfile
uv add <pkg>            # Add runtime dependency
uv add --dev <pkg>      # Add dev dependency
uv run <command>        # Run in managed environment
```

Re-run `uv sync` after any dependency changes.

## Modern Python Conventions

**Type hints** - Required for all functions. Use modern syntax:

```python
def process(items: list[str], config: dict[str, Any] | None = None) -> bool: ...
```

- Use `X | None` instead of `Optional[X]`
- Use `list`, `dict`, `tuple` directly (not `typing.List`, etc.)
- Use `Self` for methods returning their own class type

**Data structures** - Prefer `dataclasses` or `TypedDict` for structured data.

**Paths** - Use `pathlib.Path`, not string paths.

**Resources** - Use context managers (`with` statements) for files, connections, locks.

**Error handling** - Raise specific exceptions with actionable messages. Never use bare `except:`.

**Pattern matching** - Use `match`/`case` for complex conditional logic with structured data.

**String formatting** - Use f-strings. Use `f"{var=}"` for debug output.

## Documentation

- **Comments**: Explain _why_, not _what_. Code should be self-explanatory for the _what_.
- **Docstrings**: Required for public functions/classes. Follow [PEP 257](https://peps.python.org/pep-0257/).
- **README**: Keep this file updated. Keep it concise and clear. No fluff. Focus on essential info, usage, and examples.
- **Additional docs**: Avoid excessive documentation and additional files unless absolutely necessary. Do not create documents where you just summarize your changes. Update the README instead with the essential information.

## Code Quality

```bash
uv run ruff format .          # Format code
uv run ruff check .           # Lint
uv run ruff check . --fix     # Lint with auto-fix
uv run pytest                 # Run tests
```

## Quick Reference

| Task               | Command                     |
| ------------------ | --------------------------- |
| Sync environment   | `uv sync`                   |
| Add dependency     | `uv add <pkg>`              |
| Add dev dependency | `uv add --dev <pkg>`        |
| Format             | `uv run ruff format .`      |
| Lint               | `uv run ruff check .`       |
| Run tests          | `uv run pytest`             |
| Run module         | `uv run python -m <module>` |

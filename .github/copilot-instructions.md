# Copilot Instructions for github-get-agentsmd

## Project Overview

This tool discovers Python repositories via GitHub Search API and downloads their `AGENTS.md` files. Two-stage workflow:

1. **[get_repos.py](../get_repos.py)** — Query GitHub Search API, partition by date/stars to exceed 1,000-result limit
2. **[get_agentsmd.py](../get_agentsmd.py)** — Download `AGENTS.md` from discovered repos

All configuration centralized in [config.yaml](../config.yaml). Outputs timestamped JSONL and directory structures.

## Architecture Patterns

### Configuration-Driven Design

All behavior controlled via `config.yaml` — no magic constants in code. Access via module-level `CONFIG` dict:

```python
CONFIG = load_config()
REPOS_CONFIG = CONFIG["repos"]
API_CONFIG = REPOS_CONFIG["api"]
```

CLI arguments override config values (see `main()` functions). Always provide defaults from config.

### GitHub API Rate Limit Strategy

`gh_get()` in [get_repos.py](../get_repos.py#L79) implements sophisticated retry logic:

- **Primary rate limit** (remaining=0): Sleep until reset timestamp
- **Secondary rate limit** (Retry-After): Honor header value
- **Abuse detection** (403/429/503): Exponential backoff with configurable ceiling

Never call GitHub APIs directly—always use `gh_get()` with session headers.

### Query Partitioning System

GitHub Search API caps results at 1,000 per query. `build_partitions()` recursively splits date ranges:

1. Count results for star bin + date range
2. If >1,000: binary split date range
3. Continue until all partitions ≤1,000

See [get_repos.py](../get_repos.py#L201) `_split_date()` and `build_partitions()`. Maintains stack of `RepoQuery` dataclasses.

### Timestamped Output Convention

All output files/dirs include `YYYY-MM-DD_HHMMSS` suffix:

- `repos_2026-01-04_122007.jsonl`
- `agents_md_2026-01-04_122036/`

Enables multiple runs without conflicts. `find_latest_repos_file()` auto-detects newest file.

## Code Conventions

Follow [AGENTS.md](../AGENTS.md) (already referenced in workspace). Key points:

- **Type hints**: Modern syntax (`list[str]`, `X | None`, `Self`)
- **Dataclasses**: Structured data (see `Repo`, `RepoQuery`, `DownloadResult`)
- **pathlib.Path**: Never string paths
- **Rich library**: All console output via `console.print()` with markup

### Error Handling Pattern

Distinguish transient (retry) vs permanent (abort) errors:

```python
if response.status_code == 404:
    return DownloadResult(success=False, error="File not found")
if response.status_code == 403:
    # Retry with backoff
    time.sleep(backoff)
    continue
```

Return structured results (`DownloadResult`) instead of raising exceptions for expected failures.

## Development Workflows

```bash
uv sync                          # Initial setup or after dependency changes
uv run python get_repos.py       # Stage 1: Discover repos
uv run python get_agentsmd.py    # Stage 2: Download AGENTS.md (auto-detects latest)
uv run ruff format . && uv run ruff check .  # Format and lint
```

### Testing Rate Limit Handling

Edit `config.yaml` to reduce delays for faster iteration:

```yaml
partition_sleep: 0.05  # Instead of 0.2
page_sleep: 0.1        # Instead of 0.25
```

Monitor with GitHub API: `gh_get(f"{API}/rate_limit")` (see `print_rate_limit()`).

## Critical Files

- **[config.yaml](../config.yaml)**: All runtime behavior (date ranges, star bins, API settings, retry logic)
- **[get_repos.py](../get_repos.py)**: Search API integration, partitioning algorithm, rate limit handling
- **[get_agentsmd.py](../get_agentsmd.py)**: Raw content download, auto-detection of input files
- **[pyproject.toml](../pyproject.toml)**: Dependencies managed with `uv` (requests, pyyaml, rich)

## Common Tasks

### Add new star bins

Edit `config.yaml` → `repos.star_bins` array. Each query partitioned independently:

```yaml
star_bins:
  - [10000, null]
  - [5000, 9999]  # Uncomment for 5k-10k
```

### Change search language

Edit `config.yaml` → `repos.language`. Affects `BASE_QUALIFIERS` in [get_repos.py](../get_repos.py#L60).

### Adjust retry behavior

Edit `config.yaml` → `repos.api` or `agents_md.download`:

- `max_retries`: Attempt ceiling
- `max_backoff`: Upper bound for exponential backoff
- `backoff_exponent`: Growth rate (default: `2^min(attempt, 6)`)

## Integration Points

- **GitHub Search API**: Rate-limited to 30 req/min, paginated to 100 items/page
- **GitHub Raw Content**: `https://raw.githubusercontent.com/{repo}/{branch}/AGENTS.md`
- **Session Management**: Shared `requests.Session()` with auth headers in [get_repos.py](../get_repos.py#L31)

Environment variable required: `export GITHUB_TOKEN="ghp_..."` with `repo` and `user:read:user` scopes.

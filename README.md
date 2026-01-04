# GitHub AGENTS.md Scraper

**Discover and download AGENTS.md files from GitHub repositories.**

[![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://github.com/rnckp/github-get-agentsmd)
![GitHub License](https://img.shields.io/github/license/rnckp/github-get-agentsmd)
[![GitHub Stars](https://img.shields.io/github/stars/rnckp/github-get-agentsmd.svg)](https://github.com/rnckp/github-get-agentsmd/stargazers)
<a href="https://github.com/astral-sh/ruff"><img alt="linting - Ruff" class="off-glb" loading="lazy" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

> [!NOTE]
> For learning and inspiration. Downloaded files retain their original licenses—respect those terms.

## What It Does

1. **`get_repos.py`** — Find repos via GitHub Search API
2. **`get_agentsmd.py`** — Download their AGENTS.md files

Searches recent, non-archived GitHub repos sorted by stars (default: 50,000 repos max). Default language: Python. Configurable via [config.yaml](config.yaml).

## Installation

```bash
git clone https://github.com/yourusername/github-get-agents.git
cd github-get-agents
pip3 install uv && uv sync
```

## Configuration

All settings are centralized in [config.yaml](config.yaml). Edit this file to customize:

- **Repository search:** Language, date ranges, star bins, max repos
- **API settings:** Timeouts, retries, backoff strategies
- **Download settings:** Delays, output directories

Default values work well for most use cases. CLI arguments override config values when specified.

## GitHub Token

Create a [Personal Access Token](https://github.com/settings/tokens) with `repo` and `user:read:user` permissions:

```bash
export GITHUB_TOKEN="ghp_..."
```

## Usage

### 1. Discover Repositories

```bash
uv run python get_repos.py
```

Output: `repos_YYYY-MM-DD_HHMMSS.jsonl`

### 2. Download AGENTS.md Files

```bash
# Use newest repos file
uv run python get_agentsmd.py

# Specify file and options
uv run python get_agentsmd.py -f repos_2026-01-04_143022.jsonl -o my_agents -d 0.2
```

Output: `agents_md_YYYY-MM-DD_HHMMSS/org/repo/AGENTS.md` + `download_results.jsonl`

## Troubleshooting

| Issue                     | Solution                                                    |
| ------------------------- | ----------------------------------------------------------- |
| `ERROR: set GITHUB_TOKEN` | `export GITHUB_TOKEN="..."`                                 |
| `403 Forbidden`           | Regenerate token with `repo` and `user:read:user` scopes    |
| Rate limit                | Scripts auto-wait; run during off-peak hours for large jobs |
| Empty repos.jsonl         | Adjust filters in `get_repos.py` or verify token works      |

**Verify token:**

```bash
curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user | jq -r .login
```

## Scaling to more Repos

GitHub Search API returns max 1,000 results per query. To get more:

**Method 1:** Edit star bins in [config.yaml](config.yaml) to partition queries:

```yaml
star_bins:
  - [10000, null]
  - [5000, 9999] # Uncomment for 5k-10k stars
  - [2000, 4999] # Uncomment for 2k-5k stars
  # ... more bins available in config
```

**Method 2:** Edit date ranges or other filters in [config.yaml](config.yaml)

**Method 3:** Use [GitHub on BigQuery](https://cloud.google.com/bigquery/public-data/github) for exhaustive queries

## API Limits

| Resource       | Limit      | Notes                           |
| -------------- | ---------- | ------------------------------- |
| Search API     | 30 req/min | Used by `get_repos.py`          |
| File downloads | N/A        | 0.1s delay in `get_agentsmd.py` |

Both scripts handle rate limits with automatic retry and backoff.

## License

MIT License

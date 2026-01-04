import os
import json
import time
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import requests
import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

API = "https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "agent-md-repo-discovery/1.0",
}

session = requests.Session()
session.headers.update(HEADERS)

console = Console()


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load configuration
CONFIG = load_config()
REPOS_CONFIG = CONFIG["repos"]
API_CONFIG = REPOS_CONFIG["api"]
PARTITION_CONFIG = REPOS_CONFIG["partition"]

# Parse dates from config
LANG = REPOS_CONFIG["language"]
CUTOFF = dt.date.fromisoformat(REPOS_CONFIG["cutoff"])
TODAY = dt.date.fromisoformat(REPOS_CONFIG["today"])

BASE_QUALIFIERS = [
    f"language:{LANG}",
    "archived:false",
    # forks are excluded by default unless fork:true/fork:only is present.
    # If you want to be explicit, add "NOT is:fork" to the config
]

# Star bins from config: convert null to None
STAR_BINS: list[tuple[int | None, int | None]] = [
    (lo, hi) for lo, hi in REPOS_CONFIG["star_bins"]
]


@dataclass(frozen=True)
class RepoQuery:
    stars_lo: int | None
    stars_hi: int | None
    pushed_lo: dt.date
    pushed_hi: dt.date  # inclusive


def gh_get(
    url: str,
    *,
    params=None,
    headers=None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> requests.Response:
    """
    Robust GET for GitHub REST API:
      - Handles primary rate limit (x-ratelimit-remaining == 0) by sleeping until reset.
      - Handles secondary rate limit via Retry-After or conservative backoff.
    """
    if timeout is None:
        timeout = API_CONFIG["timeout"]
    if max_retries is None:
        max_retries = API_CONFIG["max_retries"]

    attempt = 0
    while True:
        attempt += 1
        r = session.get(url, params=params, headers=headers, timeout=timeout)

        # Success
        if r.status_code < 400:
            return r

        # Helpful debug text (do not crash without context)
        msg = ""
        try:
            msg = r.json().get("message", "")
        except Exception:
            msg = r.text[:200]

        # Rate limiting / throttling
        if r.status_code in (403, 429, 503):
            retry_after = r.headers.get("Retry-After")
            remaining = r.headers.get("X-RateLimit-Remaining")
            reset = r.headers.get("X-RateLimit-Reset")  # epoch seconds UTC

            # Secondary rate limit: prefer Retry-After if present.  [oai_citation:2‡GitHub Docs](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?utm_source=chatgpt.com)
            if retry_after:
                time.sleep(int(retry_after))
                if attempt < max_retries:
                    continue
                raise RuntimeError(
                    f"Giving up after Retry-After retries: {r.status_code} {msg}"
                )

            # Primary rate limit: remaining == 0 → sleep until reset.  [oai_citation:3‡GitHub Docs](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?utm_source=chatgpt.com)
            if remaining == "0" and reset:
                wait_s = max(0, int(reset) - int(time.time()) + 2)
                time.sleep(wait_s)
                if attempt < max_retries:
                    continue
                raise RuntimeError(
                    f"Giving up after primary rate-limit resets: {r.status_code} {msg}"
                )

            # Other throttles / abuse detection: back off conservatively.  [oai_citation:4‡GitHub Docs](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?utm_source=chatgpt.com)
            backoff = min(
                API_CONFIG["max_backoff"],
                2 ** min(attempt, API_CONFIG["backoff_exponent"]),
            )
            time.sleep(backoff)
            if attempt < max_retries:
                continue

            raise RuntimeError(
                f"Giving up after backoff retries: {r.status_code} {msg}"
            )

        # Not a throttle: raise with message for visibility
        raise requests.HTTPError(f"{r.status_code} {r.reason}: {msg}", response=r)


def _stars_qual(lo: int | None, hi: int | None) -> str:
    if lo is None and hi is None:
        return ""
    if hi is None:
        return f"stars:>={lo}"
    if lo == hi:
        return f"stars:{lo}"
    return f"stars:{lo}..{hi}"


def _pushed_range(lo: dt.date, hi: dt.date) -> str:
    """Build GitHub search qualifier for pushed date range."""
    return f"pushed:{lo.isoformat()}..{hi.isoformat()}"


def _build_q(q: RepoQuery) -> str:
    """Build complete GitHub search query string from RepoQuery."""
    parts = list(BASE_QUALIFIERS)
    parts.append(_pushed_range(q.pushed_lo, q.pushed_hi))
    stars = _stars_qual(q.stars_lo, q.stars_hi)
    if stars:
        parts.append(stars)
    return " ".join(parts)


def _search_repos_count(qs: str) -> int:
    """Get total count of repositories matching the search query."""
    r = gh_get(
        f"{API}/search/repositories",
        params={"q": qs, "per_page": 1, "page": 1, "sort": "stars", "order": "desc"},
    )
    return int(r.json().get("total_count", 0))


def _split_date(
    lo: dt.date, hi: dt.date
) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """Split date range into two halves for binary partitioning."""
    mid = lo + dt.timedelta(days=(hi - lo).days // 2)
    left = (lo, mid)
    right = (mid + dt.timedelta(days=1), hi)
    return left, right


def build_partitions(target_max_per_query: int | None = None) -> list[RepoQuery]:
    """
    Build query partitions such that each yields <= 1000 repos.

    Search API provides up to 1,000 results per search.
    """
    if target_max_per_query is None:
        target_max_per_query = PARTITION_CONFIG["max_per_query"]

    console.print("\n[bold cyan]Building query partitions...[/bold cyan]")
    out: list[RepoQuery] = []
    api_calls = 0

    with Status("[cyan]Analyzing star bins...", console=console) as status:
        for idx, (lo, hi) in enumerate(STAR_BINS, 1):
            stars_label = _stars_qual(lo, hi) or "all stars"
            status.update(
                f"[cyan]Processing star bin {idx}/{len(STAR_BINS)}: {stars_label} ({len(out)} partitions so far, {api_calls} API calls)"
            )

            stack = [RepoQuery(lo, hi, CUTOFF, TODAY)]
            while stack:
                q = stack.pop()
                qs = _build_q(q)

                try:
                    total = _search_repos_count(qs)
                    api_calls += 1
                    status.update(
                        f"[cyan]Star bin {idx}/{len(STAR_BINS)}: {stars_label} - found {total} repos ({len(out)} partitions, {api_calls} API calls)"
                    )
                except Exception as e:
                    console.print(f"[red]Error during partition building: {e}[/red]")
                    console.print(f"[yellow]Query was: {qs}[/yellow]")
                    raise

                if total == 0:
                    continue

                if total <= target_max_per_query:
                    out.append(q)
                    continue

                # Too many: split by pushed range (binary split).
                if q.pushed_lo >= q.pushed_hi:
                    # Can't split further; keep it (you'll only get top 1000).
                    console.print(
                        f"[yellow]⚠ Can't split further: {qs} (has {total} results, keeping top 1000)[/yellow]"
                    )
                    out.append(q)
                    continue

                (l_lo, l_hi), (r_lo, r_hi) = _split_date(q.pushed_lo, q.pushed_hi)
                if l_lo <= l_hi:
                    stack.append(RepoQuery(q.stars_lo, q.stars_hi, l_lo, l_hi))
                if r_lo <= r_hi:
                    stack.append(RepoQuery(q.stars_lo, q.stars_hi, r_lo, r_hi))

                # be polite to search API custom limits
                time.sleep(PARTITION_CONFIG["partition_sleep"])

    console.print(
        f"[green]✓ Created {len(out)} query partitions using {api_calls} API calls[/green]\n"
    )
    return out


def fetch_repos_for_partition(q: RepoQuery):
    """Fetch all repositories for a given query partition."""
    qs = _build_q(q)
    for page in range(1, 11):
        r = gh_get(
            f"{API}/search/repositories",
            params={
                "q": qs,
                "per_page": 100,
                "page": page,
                "sort": "stars",
                "order": "desc",
            },
        )
        items = r.json().get("items", [])
        if not items:
            break
        yield from items
        time.sleep(PARTITION_CONFIG["page_sleep"])


def print_rate_limit():
    """Display current GitHub Search API rate limit status."""
    r = gh_get(f"{API}/rate_limit")
    data = r.json()["resources"]["search"]

    table = Table(title="GitHub Search API Rate Limit")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Limit", str(data.get("limit", "N/A")))
    table.add_row("Remaining", str(data.get("remaining", "N/A")))
    table.add_row(
        "Reset",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.get("reset", 0))),
    )

    console.print(table)


def main(max_repos: int | None = None, out_path: str | None = None) -> None:
    """
    Discover Python repositories and save to JSONL file.

    Args:
        max_repos: Maximum number of repositories to fetch (defaults to config)
        out_path: Base path for output file (defaults to config, timestamp will be added)
    """
    if max_repos is None:
        max_repos = REPOS_CONFIG["max_repos"]
    if out_path is None:
        out_path = REPOS_CONFIG["output_file"]
    # Add timestamp to output path
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_name = out_path.rsplit(".", 1)[0]
    ext = out_path.rsplit(".", 1)[1] if "." in out_path else ""
    out_path = f"{base_name}_{timestamp}.{ext}" if ext else f"{base_name}_{timestamp}"

    console.print(
        Panel.fit(
            f"[bold]GitHub Agent.md Scraper[/bold]\n"
            f"Target: [cyan]{max_repos}[/cyan] repos\n"
            f"Output: [cyan]{out_path}[/cyan]",
            border_style="blue",
        )
    )

    print_rate_limit()

    partitions = build_partitions()
    seen_ids = set()
    written = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Fetching repositories...", total=max_repos)

        with open(out_path, "w", encoding="utf-8") as f:
            for part in partitions:
                for repo in fetch_repos_for_partition(part):
                    rid = repo["id"]
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)

                    record = {
                        "id": rid,
                        "full_name": repo["full_name"],
                        "clone_url": repo["clone_url"],
                        "stargazers_count": repo["stargazers_count"],
                        "pushed_at": repo["pushed_at"],
                        "default_branch": repo.get("default_branch"),
                    }
                    f.write(json.dumps(record) + "\n")
                    written += 1
                    progress.update(
                        task,
                        completed=written,
                        description=f"[cyan]Fetching repositories... (latest: {repo['full_name']})",
                    )

                    if written >= max_repos:
                        console.print(
                            f"\n[green]✓ Successfully wrote {written} repositories to {out_path}[/green]"
                        )
                        return

    console.print(
        f"\n[green]✓ Successfully wrote {written} repositories to {out_path}[/green]"
    )


if __name__ == "__main__":
    main()

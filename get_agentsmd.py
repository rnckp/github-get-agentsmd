import argparse
import json
import time
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import requests
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

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
AGENTS_MD_CONFIG = CONFIG["agents_md"]
DOWNLOAD_CONFIG = AGENTS_MD_CONFIG["download"]


@dataclass
class Repo:
    """Repository information from repos.jsonl."""

    id: int
    full_name: str
    clone_url: str
    stargazers_count: int
    pushed_at: str
    default_branch: str

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create Repo from JSON dict."""
        return cls(
            id=data["id"],
            full_name=data["full_name"],
            clone_url=data["clone_url"],
            stargazers_count=data["stargazers_count"],
            pushed_at=data["pushed_at"],
            default_branch=data.get("default_branch", "main"),
        )


@dataclass
class DownloadResult:
    """Result of attempting to download AGENTS.md."""

    repo_full_name: str
    success: bool
    error: str | None = None
    file_size: int | None = None


def load_repos(repos_file: Path) -> list[Repo]:
    """Load repositories from JSONL file."""
    repos = []
    with repos_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repos.append(Repo.from_dict(json.loads(line)))
    return repos


def download_agents_md(
    repo: Repo,
    output_dir: Path,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> DownloadResult:
    """
    Download AGENTS.md from a repository's default branch.

    Tries the raw GitHub content URL first.
    """
    if timeout is None:
        timeout = DOWNLOAD_CONFIG["timeout"]
    if max_retries is None:
        max_retries = DOWNLOAD_CONFIG["max_retries"]

    # GitHub raw content URL pattern from config
    url = AGENTS_MD_CONFIG["raw_url_pattern"].format(
        repo=repo.full_name, branch=repo.default_branch
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                # Create output directory structure (org/repo)
                org, repo_name = repo.full_name.split("/")
                repo_dir = output_dir / org / repo_name
                repo_dir.mkdir(parents=True, exist_ok=True)

                # Save the file
                agents_md_path = repo_dir / "AGENTS.md"
                agents_md_path.write_text(response.text, encoding="utf-8")

                return DownloadResult(
                    repo_full_name=repo.full_name,
                    success=True,
                    file_size=len(response.text),
                )

            if response.status_code == 404:
                return DownloadResult(
                    repo_full_name=repo.full_name,
                    success=False,
                    error="File not found",
                )

            if response.status_code == 403:
                # Rate limited - wait and retry
                if attempt < max_retries:
                    backoff = DOWNLOAD_CONFIG["backoff_base"] ** attempt
                    time.sleep(backoff)
                    continue
                return DownloadResult(
                    repo_full_name=repo.full_name,
                    success=False,
                    error=f"Rate limited (403) after {max_retries} retries",
                )

            return DownloadResult(
                repo_full_name=repo.full_name,
                success=False,
                error=f"HTTP {response.status_code}",
            )

        except requests.Timeout:
            if attempt < max_retries:
                continue
            return DownloadResult(
                repo_full_name=repo.full_name,
                success=False,
                error="Timeout",
            )
        except Exception as e:
            return DownloadResult(
                repo_full_name=repo.full_name,
                success=False,
                error=str(e),
            )

    return DownloadResult(
        repo_full_name=repo.full_name,
        success=False,
        error=f"Failed after {max_retries} retries",
    )


def print_summary(results: list[DownloadResult]) -> None:
    """Print summary statistics of download results."""
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    # Count error types
    error_counts: dict[str, int] = {}
    for r in results:
        if not r.success and r.error:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1

    # Create summary table
    table = Table(title="Download Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total repositories", str(len(results)))
    table.add_row("Successfully downloaded", str(successful))
    table.add_row("Failed", str(failed))

    if error_counts:
        table.add_section()
        for error, count in sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        ):
            table.add_row(f"  {error}", str(count))

    console.print("\n")
    console.print(table)


def find_latest_repos_file() -> Path | None:
    """Find the most recent repos_*.jsonl file in the current directory."""
    repos_files = list(Path.cwd().glob("repos*.jsonl"))
    if not repos_files:
        return None
    # Sort by modification time, newest first
    return max(repos_files, key=lambda p: p.stat().st_mtime)


def main(
    repos_file: str | None = None,
    output_dir: str | None = None,
    delay: float | None = None,
) -> None:
    """
    Download AGENTS.md files from all repositories in repos.jsonl.

    Args:
        repos_file: Path to the JSONL file containing repository data.
                   If None, uses the most recent repos_*.jsonl file.
        output_dir: Base directory name to save downloaded AGENTS.md files (defaults to config)
        delay: Delay in seconds between downloads to avoid rate limiting (defaults to config)
    """
    if output_dir is None:
        output_dir = AGENTS_MD_CONFIG["output_dir"]
    if delay is None:
        delay = AGENTS_MD_CONFIG["delay"]
    # Find repos file if not specified
    if repos_file is None:
        latest = find_latest_repos_file()
        if latest is None:
            console.print("[red]Error: No repos*.jsonl files found[/red]")
            console.print(
                "[yellow]Run get_repos.py first to generate repository data[/yellow]"
            )
            return
        repos_path = latest
        console.print(f"[cyan]Auto-detected: {repos_path.name}[/cyan]")
    else:
        repos_path = Path(repos_file)

    # Add timestamp to output directory
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = Path(f"{output_dir}_{timestamp}")

    if not repos_path.exists():
        console.print(f"[red]Error: {repos_path} not found[/red]")
        return

    console.print(
        Panel.fit(
            f"[bold]AGENTS.md Downloader[/bold]\n"
            f"Source: [cyan]{repos_path}[/cyan]\n"
            f"Output: [cyan]{output_path}/[/cyan]",
            border_style="blue",
        )
    )

    # Load repositories
    console.print("\n[cyan]Loading repositories...[/cyan]")
    repos = load_repos(repos_path)
    console.print(f"[green]✓ Loaded {len(repos)} repositories[/green]")

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Download AGENTS.md files
    results: list[DownloadResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Downloading AGENTS.md files...", total=len(repos)
        )

        for repo in repos:
            result = download_agents_md(repo, output_path)
            results.append(result)

            status = "✓" if result.success else "✗"
            progress.update(
                task,
                advance=1,
                description=f"[cyan]Downloading... {status} {repo.full_name}",
            )

            # Polite delay to avoid rate limiting
            if delay > 0:
                time.sleep(delay)

    # Print summary
    print_summary(results)

    # Save results log
    results_file = output_path / "download_results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    {
                        "repo": result.repo_full_name,
                        "success": result.success,
                        "error": result.error,
                        "file_size": result.file_size,
                    }
                )
                + "\n"
            )
    console.print(f"\n[green]✓ Results saved to {results_file}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AGENTS.md files from GitHub repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use newest repos_*.jsonl file
  %(prog)s -f repos.jsonl               # Use specific file
  %(prog)s -f repos.jsonl -d 0.2        # Custom delay between downloads
  %(prog)s -o my_agents -d 0.05         # Custom output dir and delay
        """,
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="repos_file",
        help="Path to repos JSONL file (default: auto-detect newest repos_*.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="Base output directory name (default: from config.yaml)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        help="Delay in seconds between downloads (default: from config.yaml)",
    )

    args = parser.parse_args()
    main(repos_file=args.repos_file, output_dir=args.output_dir, delay=args.delay)

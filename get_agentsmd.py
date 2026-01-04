import argparse
import json
import os
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# GitHub API authentication (optional but recommended for higher rate limits)
TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {
    "User-Agent": "agent-md-downloader/1.0",
}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

session = requests.Session()
session.headers.update(HEADERS)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_agents_config(config: dict[str, Any]) -> None:
    """Validate agents_md configuration has required keys."""
    required_keys = ["output_dir", "delay", "download", "raw_url_pattern"]
    agents_config = config.get("agents_md")
    if not agents_config:
        raise ValueError("Config missing 'agents_md' section")
    for key in required_keys:
        if key not in agents_config:
            raise ValueError(f"Config agents_md missing required key: {key}")
    download_config = agents_config.get("download", {})
    for key in ["timeout", "max_retries", "backoff_base"]:
        if key not in download_config:
            raise ValueError(f"Config agents_md.download missing required key: {key}")


# Load configuration
CONFIG = load_config()
validate_agents_config(CONFIG)
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
    """Result of attempting to download AGENTS.md and CLAUDE.md."""

    repo_full_name: str
    agents_md_found: bool = False
    agents_md_size: int | None = None
    agents_md_variant: str | None = None
    claude_md_found: bool = False
    claude_md_size: int | None = None
    claude_md_variant: str | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """At least one file was found."""
        return self.agents_md_found or self.claude_md_found


def load_repos(repos_file: Path) -> list[Repo]:
    """Load repositories from JSONL file."""
    repos = []
    with repos_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                repos.append(Repo.from_dict(json.loads(line)))
    return repos


@dataclass
class FileDownloadResult:
    """Result of attempting to download a single file."""

    found: bool
    size: int | None = None
    variant: str | None = None
    error: str | None = None


def try_download_file(
    repo: Repo,
    repo_dir: Path,
    filename_variants: list[str],
    timeout: int,
    max_retries: int,
) -> FileDownloadResult:
    """
    Try to download a file using case-insensitive filename variants.

    Returns:
        FileDownloadResult with found status, size, variant used, and any error.
    """
    for variant in filename_variants:
        url = AGENTS_MD_CONFIG["raw_url_pattern"].format(
            repo=repo.full_name, branch=repo.default_branch, filename=variant
        )

        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, timeout=timeout)

                if response.status_code == 200:
                    # Create directory only when we have a file to save
                    repo_dir.mkdir(parents=True, exist_ok=True)
                    # Save the file with the variant name that worked
                    file_path = repo_dir / variant
                    file_path.write_text(response.text, encoding="utf-8")
                    return FileDownloadResult(True, len(response.text), variant, None)

                if response.status_code == 404:
                    # Try next variant
                    break

                if response.status_code == 403:
                    # Rate limited - wait and retry
                    if attempt < max_retries:
                        backoff = DOWNLOAD_CONFIG["backoff_base"] ** attempt
                        time.sleep(backoff)
                        continue
                    return FileDownloadResult(
                        False,
                        None,
                        None,
                        f"Rate limited (403) after {max_retries} retries",
                    )

                # Other HTTP errors - try next variant
                break

            except requests.Timeout:
                if attempt < max_retries:
                    continue
                return FileDownloadResult(False, None, None, "Timeout")
            except Exception as e:
                return FileDownloadResult(False, None, None, str(e))

    return FileDownloadResult(False, None, None, None)


def download_agents_md(
    repo: Repo,
    output_dir: Path,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> DownloadResult:
    """
    Download AGENTS.md and CLAUDE.md from a repository's default branch.

    Tries multiple case variants for each file.
    """
    if timeout is None:
        timeout = DOWNLOAD_CONFIG["timeout"]
    if max_retries is None:
        max_retries = DOWNLOAD_CONFIG["max_retries"]

    # Prepare output directory path (org/repo) - will be created only if files are found
    org, repo_name = repo.full_name.split("/")
    repo_dir = output_dir / org / repo_name

    # Get filename variants from config
    filename_variants = AGENTS_MD_CONFIG.get("filename_variants", {})
    agents_variants = filename_variants.get("agents_md", ["AGENTS.md"])
    claude_variants = filename_variants.get("claude_md", ["CLAUDE.md"])

    # Try to download AGENTS.md
    agents_result = try_download_file(
        repo, repo_dir, agents_variants, timeout, max_retries
    )

    # Try to download CLAUDE.md
    claude_result = try_download_file(
        repo, repo_dir, claude_variants, timeout, max_retries
    )

    # Combine errors if both failed with errors
    error = None
    if not agents_result.found and not claude_result.found:
        if agents_result.error or claude_result.error:
            error = agents_result.error or claude_result.error
        else:
            error = "No AGENTS.md or CLAUDE.md found"

    return DownloadResult(
        repo_full_name=repo.full_name,
        agents_md_found=agents_result.found,
        agents_md_size=agents_result.size,
        agents_md_variant=agents_result.variant,
        claude_md_found=claude_result.found,
        claude_md_size=claude_result.size,
        claude_md_variant=claude_result.variant,
        error=error,
    )


def print_summary(results: list[DownloadResult], skipped: int = 0) -> None:
    """Print summary statistics of download results."""
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    agents_count = sum(1 for r in results if r.agents_md_found)
    claude_count = sum(1 for r in results if r.claude_md_found)
    both_count = sum(1 for r in results if r.agents_md_found and r.claude_md_found)

    # Count error types
    error_counts: dict[str, int] = {}
    for r in results:
        if not r.success and r.error:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1

    # Create summary table
    table = Table(title="Download Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total repositories", str(len(results) + skipped))
    if skipped > 0:
        table.add_row("Skipped (already processed)", str(skipped))
    table.add_row("Repos with files found", str(successful))
    table.add_row("Repos without files", str(failed))
    table.add_section()
    table.add_row("AGENTS.md found", str(agents_count))
    table.add_row("CLAUDE.md found", str(claude_count))
    table.add_row("Both files found", str(both_count))

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


def load_previous_results(output_path: Path) -> set[str]:
    """Load previously processed repo names from download_results.jsonl."""
    results_file = output_path / "download_results.jsonl"
    processed = set()
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed.add(data["repo"])
    return processed


def download_worker(
    repo: Repo, output_path: Path, timeout: int, max_retries: int
) -> DownloadResult:
    """Worker function for parallel downloads."""
    return download_agents_md(repo, output_path, timeout, max_retries)


def main(
    repos_file: str | None = None,
    output_dir: str | None = None,
    delay: float | None = None,
    workers: int | None = None,
    resume: bool = False,
) -> None:
    """
    Download AGENTS.md and CLAUDE.md files from all repositories in repos.jsonl.

    Args:
        repos_file: Path to the JSONL file containing repository data.
                   If None, uses the most recent repos_*.jsonl file.
        output_dir: Base directory name to save downloaded files (defaults to config)
        delay: Delay in seconds between downloads to avoid rate limiting (defaults to config)
        workers: Number of parallel download workers (1 for sequential)
        resume: Resume from existing output directory, skipping already processed repos
    """
    if output_dir is None:
        output_dir = AGENTS_MD_CONFIG["output_dir"]
    if delay is None:
        delay = AGENTS_MD_CONFIG["delay"]
    if workers is None:
        workers = AGENTS_MD_CONFIG.get("workers", 1)

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

    # Handle resume mode or create new timestamped directory
    if resume:
        # Find latest existing output directory
        existing_dirs = sorted(Path.cwd().glob(f"{output_dir}_*"), reverse=True)
        if existing_dirs:
            output_path = existing_dirs[0]
            console.print(f"[cyan]Resuming in: {output_path.name}[/cyan]")
        else:
            console.print(
                "[yellow]No existing output directory found, creating new one[/yellow]"
            )
            timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_path = Path(f"{output_dir}_{timestamp}")
    else:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = Path(f"{output_dir}_{timestamp}")

    if not repos_path.exists():
        console.print(f"[red]Error: {repos_path} not found[/red]")
        return

    console.print(
        Panel.fit(
            f"[bold]AGENTS.md & CLAUDE.md Downloader[/bold]\n"
            f"Source: [cyan]{repos_path}[/cyan]\n"
            f"Output: [cyan]{output_path}/[/cyan]\n"
            f"Workers: [cyan]{workers}[/cyan]",
            border_style="blue",
        )
    )

    # Load repositories
    console.print("\n[cyan]Loading repositories...[/cyan]")
    repos = load_repos(repos_path)
    console.print(f"[green]✓ Loaded {len(repos)} repositories[/green]")

    # Check for already processed repos if resuming
    skipped_count = 0
    if resume:
        processed = load_previous_results(output_path)
        if processed:
            original_count = len(repos)
            repos = [r for r in repos if r.full_name not in processed]
            skipped_count = original_count - len(repos)
            console.print(
                f"[yellow]Skipping {skipped_count} already processed repos[/yellow]"
            )

    if not repos:
        console.print("[green]✓ All repositories already processed[/green]")
        return

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Download files
    results: list[DownloadResult] = []
    timeout = DOWNLOAD_CONFIG["timeout"]
    max_retries = DOWNLOAD_CONFIG["max_retries"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Downloading files...", total=len(repos))

        if workers > 1:
            # Parallel downloads
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        download_worker, repo, output_path, timeout, max_retries
                    ): repo
                    for repo in repos
                }
                for future in as_completed(futures):
                    repo = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = DownloadResult(
                            repo_full_name=repo.full_name,
                            error=str(e),
                        )
                    results.append(result)

                    status_parts = []
                    if result.agents_md_found:
                        status_parts.append("A")
                    if result.claude_md_found:
                        status_parts.append("C")
                    status = "✓" + ",".join(status_parts) if result.success else "✗"
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Downloading... {status} {repo.full_name}",
                    )
        else:
            # Sequential downloads with delay
            for repo in repos:
                result = download_agents_md(repo, output_path)
                results.append(result)

                status_parts = []
                if result.agents_md_found:
                    status_parts.append("A")
                if result.claude_md_found:
                    status_parts.append("C")
                status = "✓" + ",".join(status_parts) if result.success else "✗"
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Downloading... {status} {repo.full_name}",
                )

                # Polite delay to avoid rate limiting
                if delay > 0:
                    time.sleep(delay)

    # Print summary
    print_summary(results, skipped_count)

    # Save results log (append if resuming)
    results_file = output_path / "download_results.jsonl"
    mode = "a" if resume else "w"
    with results_file.open(mode, encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    {
                        "repo": result.repo_full_name,
                        "success": result.success,
                        "agents_md_found": result.agents_md_found,
                        "agents_md_size": result.agents_md_size,
                        "agents_md_variant": result.agents_md_variant,
                        "claude_md_found": result.claude_md_found,
                        "claude_md_size": result.claude_md_size,
                        "claude_md_variant": result.claude_md_variant,
                        "error": result.error,
                    }
                )
                + "\n"
            )
    console.print(f"\n[green]✓ Results saved to {results_file}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AGENTS.md and CLAUDE.md files from GitHub repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Use newest repos_*.jsonl file
  %(prog)s -f repos.jsonl               # Use specific file
  %(prog)s -f repos.jsonl -d 0.2        # Custom delay between downloads
  %(prog)s -o my_agents -d 0.05         # Custom output dir and delay
  %(prog)s -w 4                         # Use 4 parallel workers
  %(prog)s -r                           # Resume from latest output directory
  %(prog)s -r -w 8                      # Resume with 8 parallel workers
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
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel download workers (default: 1 for sequential)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume from latest output directory, skipping already processed repos",
    )

    args = parser.parse_args()
    main(
        repos_file=args.repos_file,
        output_dir=args.output_dir,
        delay=args.delay,
        workers=args.workers,
        resume=args.resume,
    )

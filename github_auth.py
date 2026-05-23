import os
from pathlib import Path

from dotenv import load_dotenv


def load_github_token(
    env_path: str | Path = ".env", *, required: bool = False
) -> str | None:
    """Load GITHUB_TOKEN from the environment, including values defined in .env."""
    load_dotenv(env_path, override=False)
    token = os.environ.get("GITHUB_TOKEN")
    if required and not token:
        raise RuntimeError("ERROR: set GITHUB_TOKEN in .env or your shell environment")
    return token

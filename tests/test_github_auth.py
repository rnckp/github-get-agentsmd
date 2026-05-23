from github_auth import load_github_token


def test_load_github_token_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("GITHUB_TOKEN=ghp_from_dotenv\n", encoding="utf-8")

    assert load_github_token(env_path) == "ghp_from_dotenv"


def test_load_github_token_keeps_existing_environment_value(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_environment")
    env_path = tmp_path / ".env"
    env_path.write_text("GITHUB_TOKEN=ghp_from_dotenv\n", encoding="utf-8")

    assert load_github_token(env_path) == "ghp_from_environment"

# Repository Guidelines

## Project Structure & Module Organization
The `crawl4ai/` package hosts runtime code: `components/` for adapters, `crawlers/` plus `deep_crawling/` for orchestration, `processors/` and `html2text/` for parsing, and `js_snippet/` for Playwright helpers. CLI entrypoints (`crwl`, installers, migrations) live in `crawl4ai/cli.py`, `install.py`, and `migrations.py`. Tests mirror these areas inside `tests/`, MkDocs sources stay in `docs/`, Docker recipes sit in `deploy/`, and prompt or memory assets live in `prompts/`.

## Build, Test, and Development Commands
Create a venv with `python -m venv .venv && source .venv/bin/activate` (PowerShell: `..\.venv\Scripts\Activate.ps1`). Install dependencies via `pip install -e ".[all]"`; switch to `".[sync]"` if you only need Selenium extras. Refresh browsers and snippets with `crawl4ai-setup`, then align the async cache using `crawl4ai-migrate --database data/crawl4ai.db`. Execute crawls through `crwl crawl <url> --crawler-config configs/default.yaml`, and run `crawl4ai-download-models --all` before offline work.

## Coding Style & Naming Conventions
Follow PEP 8, four-space indents, and <=120-character lines. Modules and functions stay `snake_case`, classes or data models (`crawl4ai/models.py`, `types.py`) use `PascalCase`, and CLI command names remain hyphenated. Annotate new APIs, reuse shared enums or dataclasses before adding ones, and route console output through `crawl4ai/async_logger.py` so structured logging and colorized levels stay intact.

## Testing Guidelines
Pytest drives verification. Typical calls: `pytest tests/general -m asyncio -vv` for dispatcher logic, `pytest tests/browser/test_web_crawler.py` for Playwright adapters, and `pytest tests/proxy/test_proxy_config.py` for network guards. Name tests `test_<module>.py`, mirror package folders, tag coroutine cases with `@pytest.mark.asyncio`, and wrap credentialed scenarios in `pytest.skip` messages that explain the required env vars or Docker services.

## Commit & Pull Request Guidelines
History currently relies on short imperative subjects (`Add files via upload`), so keep commits in that voice: `<Verb> <scope>` under 72 characters, elaborating in the body with bullet points or `Refs #ID`. Pull requests should summarize the behavior change, link the issue, paste the test commands you ran, and attach screenshots or logs for CLI or docs updates plus any migration or config steps reviewers must follow.

## Security & Configuration Tips
Copy secrets from `.env.txt`, keep them local, and never commit API keys embedded in fixtures or prompts. Run `crawl4ai-doctor` before filing bugs; it validates Playwright/OpenSSL installs while redacting credentials. When editing `deploy/` assets, surface configuration through env vars (for example `CRAWL4AI_BROWSER_PORT`), and keep downloaded models or cache files inside ignored directories.

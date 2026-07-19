# Repository Guidelines

## Project Structure & Module Organization

`main.py` bootstraps the asynchronous agent and its optional PyQt dashboard.
Application code lives in `core/`: `kernel/` coordinates System 1 and System 2,
`io/` defines events and adapters, `memory/`, `limbic/`, and `social/` hold
domain state, `tool_manager/` manages tools, and `infrastructure/` provides
configuration, database, logging, and API clients. `tools/System1/` and
`tools/System2/` contain callable tool implementations. Keep prompts and
runtime configuration under `data/`; use `data/default-config.yaml` as the
checked-in template. `db_mg.py` and `db_web.py` are database-management UIs.

## Build, Test, and Development Commands

Use a virtual environment and install the pinned runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --nogui       # run without the PyQt dashboard
python main.py               # run with the dashboard
python test_attention_sandbox.py
```

The sandbox script is an opt-in, long-running OneBot/WebSocket integration
check. It requires a local `data/config-for-testing.yaml`, a reachable OneBot
endpoint, and a deliberately selected `TARGET_GROUP_ID`; stop it with
`Ctrl+C`. There is currently no configured unit-test runner, formatter, linter,
or coverage threshold. Add focused tests as `test_*.py` files and avoid tests
that send messages to real services unless explicitly marked as integration
tests.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for
functions, variables, and modules; `PascalCase` for classes; and `UPPER_CASE`
for constants. Preserve the async design: use `async def` and await I/O rather
than blocking event-loop code. Keep Pydantic schemas in the relevant domain
module and extend `core/io/event_schema.py` when changing cross-module event
contracts. No automatic formatter is configured, so make minimal, readable
edits and keep imports grouped as standard library, third-party, then local.

## Configuration, Commits & Pull Requests

Copy and customize the default configuration locally; never commit live API
keys, OneBot tokens, personal database data, or generated `__pycache__/` files.
Review config diffs carefully because `data/config.yaml` may contain secrets.

Recent history uses short messages (often `update`) and an imperative change
description (for example, `Change license from GPL-3.0 to AGPL-3.0`). Prefer a
specific imperative subject such as `Handle OneBot reconnect failures`. Pull
requests should explain the affected subsystem and runtime behavior, link any
issue, list validation performed, and include dashboard screenshots when UI
behavior changes.

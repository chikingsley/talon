set dotenv-load

default:
    @just --list

# Sync dependencies
sync:
    uv sync

# Run tests with coverage
test:
    uv run pytest

# Lint and format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Type check
check:
    uv run ty check

# Run all checks (lint, format, type check, test)
all: lint check test

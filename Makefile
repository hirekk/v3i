SHELL := /bin/bash

.PHONY: clear-workflow-runs
clear-workflow-runs:
	@gh run list --limit 1000 --json databaseId --jq '.[].databaseId' | xargs -I {} gh run delete {}

.PHONY: clear-actions-cache
clear-actions-cache:
	@gh cache delete --all

.PHONY: full-actions-clear
full-actions-clear: clear_actions_cache clear_workflow_runs

.PHONY: code-check
code-check:
	@echo "Running linter and formatter..."
	@uv run pre-commit run --all-files

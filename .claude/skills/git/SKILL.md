---
name: git
description: Rules for git operations in TVBO. The user manages all version control.
  Use when clarifying what git actions are allowed or when a task involves version
  control.
---

# Git Rules

The user owns all version control. Never perform write git operations on their behalf.

## Prohibited Actions

- **Never** run `git add`, `git commit`, or `git push`.
- **Never** use the GitKraken `git_add_or_commit` or `git_push` tools.
- Do not stage, commit, or push changes on behalf of the user, even if asked to "save" or "finish up" work.

## Allowed Actions

- `git status`, `git diff`, `git log`, `git branch`, `git stash` (read-only / non-destructive)
- `git checkout` or `git switch` to change branches when the user requests it
- `git fetch` and `git pull` when the user requests it
- Any read-only git inspection command

## Workflow

1. Make file edits as requested.
2. When done, summarize what changed so the user can review and commit themselves.

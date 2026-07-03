#!/usr/bin/env bash
# Copyright © Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.
#
# Cut a TVBO release: pick/verify the next version, show what is shipping, and
# (after confirmation) commit, push and create the GitHub release. The GitHub
# Actions PyPI workflow triggers off the created tag.
#
# Driven entirely by environment variables (see `make release`):
#   VERSION=x.y.z   Release exactly this version.
#   BUMP=patch|minor|major
#                   Compute the next version from the latest release/current one.
#   CONFIRM=yes     Skip the interactive confirmation (for automation).
#   DRYRUN=1        Show everything that would happen; change nothing.
#
# Precedence: VERSION wins over BUMP; with neither, the version currently in
# tvbo/__init__.py is released as-is (and still checked against the last release).
set -eu

INIT="tvbo/__init__.py"
VERSION="${VERSION:-}"
BUMP="${BUMP:-}"
CONFIRM="${CONFIRM:-}"
DRYRUN="${DRYRUN:-}"
ALLOW_DIRTY="${ALLOW_DIRTY:-}"

die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*"; }

valid_version() {
	printf '%s' "$1" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([.-]?(a|b|rc|dev|post)[0-9]+)?$'
}

# Highest of two versions per `sort -V` (semantic version sort).
highest() { printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1; }

read_current() { grep '^__version__' "$INIT" | cut -d'"' -f2; }

set_version() {
	# Portable in-place edit (BSD + GNU sed).
	sed -i.bak "s/^__version__ = .*/__version__ = \"$1\"/" "$INIT" && rm -f "$INIT.bak"
	[ "$(read_current)" = "$1" ] || die "failed to write version $1 to $INIT"
}

CITATION="CITATION.cff"

cit_field() { grep -E "^$1:" "$CITATION" | head -1 | sed -E "s/^$1:[[:space:]]*//"; }

set_citation() {
	# $1 = version, $2 = date-released (YYYY-MM-DD). Keeps CITATION.cff's package
	# version in sync with the release so citation metadata does not drift.
	[ -f "$CITATION" ] || return 0
	sed -i.bak -E "s/^version:.*/version: \"$1\"/" "$CITATION" && rm -f "$CITATION.bak"
	[ -n "${2:-}" ] && sed -i.bak -E "s/^date-released:.*/date-released: $2/" "$CITATION" && rm -f "$CITATION.bak"
	return 0
}

# Write the release version to every version-carrying file, and undo it. Both
# reference globals set below (NEW/CURRENT/TODAY/CIT_*), evaluated at call time.
apply_bump() {
	if [ "$NEW" != "$CURRENT" ]; then
		set_version "$NEW"
		info "✓ Set version $CURRENT → $NEW in $INIT"
	fi
	if [ -f "$CITATION" ]; then
		set_citation "$NEW" "$TODAY"
		info "✓ Synced $CITATION → version $NEW, date-released $TODAY"
	fi
}
revert_bump() {
	[ "$NEW" != "$CURRENT" ] && set_version "$CURRENT"
	[ -f "$CITATION" ] && set_citation "$CIT_VER_OLD" "$CIT_DATE_OLD"
	info "Reverted version files."
}

bump_version() {
	base=$1 level=$2
	maj=$(printf '%s' "$base" | cut -d. -f1)
	min=$(printf '%s' "$base" | cut -d. -f2)
	# strip any pre-release suffix (e.g. 1rc1 -> 1) before incrementing
	pat=$(printf '%s' "$base" | cut -d. -f3 | sed 's/[^0-9].*$//')
	: "${maj:=0}" "${min:=0}" "${pat:=0}"
	case "$level" in
		patch) pat=$((pat + 1)) ;;
		minor) min=$((min + 1)) pat=0 ;;
		major) maj=$((maj + 1)) min=0 pat=0 ;;
		*) die "invalid BUMP '$level' (expected patch|minor|major)" ;;
	esac
	printf '%s.%s.%s' "$maj" "$min" "$pat"
}

[ -f "$INIT" ] || die "$INIT not found (run from the repo root)"
command -v gh >/dev/null 2>&1 || die "GitHub CLI 'gh' is required"

CURRENT=$(read_current)
LATEST=$(gh release view --json tagName --jq .tagName 2>/dev/null | sed 's/^v//' || true)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
TODAY=$(date +%F)

# Original CITATION.cff fields, captured so an abort can restore them exactly.
CIT_VER_OLD="" CIT_DATE_OLD=""
if [ -f "$CITATION" ]; then
	CIT_VER_OLD=$(cit_field version | tr -d '"')
	CIT_DATE_OLD=$(cit_field date-released)
fi

# --- Decide the target version -------------------------------------------------
if [ -n "$VERSION" ] && [ -n "$BUMP" ]; then
	die "pass either VERSION= or BUMP=, not both"
elif [ -n "$VERSION" ]; then
	NEW=$VERSION
elif [ -n "$BUMP" ]; then
	base=$(highest "${LATEST:-0.0.0}" "$CURRENT")
	NEW=$(bump_version "$base" "$BUMP")
else
	NEW=$CURRENT
fi
valid_version "$NEW" || die "invalid version '$NEW' (expected e.g. 0.5.1)"

# --- If the current version is already published, pick a bump interactively ----
# Only when the user gave no explicit VERSION/BUMP and the current version would
# not move forward past the latest release. With no terminal, fall back to a hint.
if [ -z "$VERSION" ] && [ -z "$BUMP" ] && [ -n "$LATEST" ] &&
	[ "$(highest "$LATEST" "$NEW")" = "$LATEST" ]; then
	if [ ! -t 0 ]; then
		die "v$NEW is already published — choose a bump: make release BUMP=patch|minor|major"
	fi
	base=$(highest "$LATEST" "$CURRENT")
	p=$(bump_version "$base" patch)
	m=$(bump_version "$base" minor)
	M=$(bump_version "$base" major)
	printf '\nv%s is already published. Which release do you want to cut?\n' "$CURRENT" >&2
	PS3="Select bump [1-4]: "
	select _ in "patch  → $p" "minor  → $m" "major  → $M" "cancel"; do
		case "$REPLY" in
		1) NEW=$p; break ;;
		2) NEW=$m; break ;;
		3) NEW=$M; break ;;
		4 | q | Q) die "aborted — nothing committed or pushed" ;;
		*) printf 'Please choose 1-4.\n' >&2 ;;
		esac
	done
	info "→ selected v$NEW"
fi

# --- Verify it is a forward bump ----------------------------------------------
if [ -n "$LATEST" ]; then
	if [ "$NEW" = "$LATEST" ]; then
		die "v$NEW is already the latest release — bump it (make release BUMP=patch)"
	fi
	if [ "$(highest "$LATEST" "$NEW")" != "$NEW" ]; then
		die "v$NEW is lower than the latest release v$LATEST — refusing to release a non-bump"
	fi
fi

# --- Show what is shipping -----------------------------------------------------
info ""
info "  branch           : $BRANCH"
info "  latest published : ${LATEST:-<none>}"
info "  current (__init__): $CURRENT"
info "  about to release : $NEW"
info ""
info "Recent releases:"
gh release list --limit 5 2>/dev/null | sed 's/^/  /' || info "  (unable to list releases)"
info ""
if [ -n "$LATEST" ]; then
	info "Commits since v$LATEST:"
	git log "v$LATEST..HEAD" --oneline 2>/dev/null | sed 's/^/  /' || info "  (tag v$LATEST not found locally)"
	info ""
fi
# Tracked changes other than the version bump. By default the release commit
# must contain ONLY the version bump, so these block the release (ALLOW_DIRTY=1
# reverts to committing everything via `git add -A`).
DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null || true)
if [ "$ALLOW_DIRTY" = "1" ]; then
	info "Changes that will be committed (ALLOW_DIRTY=1, git add -A):"
	git status --short 2>/dev/null | sed 's/^/  /' || true
else
	if [ -f "$CITATION" ]; then
		info "Release commit will contain only the version bump ($INIT + $CITATION)."
	else
		info "Release commit will contain only the version bump ($INIT)."
	fi
	if [ -n "$DIRTY" ]; then
		info "Uncommitted tracked changes that BLOCK the release (commit or stash first):"
		printf '%s\n' "$DIRTY" | sed 's/^/  /'
	fi
fi
info ""
[ "$BRANCH" = "main" ] || info "⚠ You are on '$BRANCH', not 'main'."

if [ "$DRYRUN" = "1" ]; then
	if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" != "1" ]; then
		info "DRYRUN — blocked: commit or stash the tracked changes above, then release v$NEW."
	else
		info "DRYRUN — would set version to $NEW, commit \"Release v$NEW\", push, and create GitHub release v$NEW."
	fi
	info "Nothing changed."
	exit 0
fi

# --- Enforce a clean tree so the release commit is exactly the version bump ----
if [ -n "$DIRTY" ] && [ "$ALLOW_DIRTY" != "1" ]; then
	die "working tree not clean — commit or stash the tracked changes listed above first (or pass ALLOW_DIRTY=1 to include them in the release commit)"
fi

# --- Apply version bump to all version files (revert cleanly on abort) ---------
apply_bump

if [ "$CONFIRM" != "yes" ]; then
	if [ ! -t 0 ]; then
		revert_bump
		die "no interactive terminal for confirmation — run in a terminal, or pass CONFIRM=yes (e.g. 'make release BUMP=patch CONFIRM=yes')"
	fi
	printf "Release v%s? This will commit, push and tag. [y/N] " "$NEW"
	read -r ans || ans=""
	case "$ans" in
	[Yy] | [Yy][Ee][Ss]) ;;
	*)
		revert_bump
		die "aborted — nothing committed or pushed"
		;;
	esac
fi

# --- Release -------------------------------------------------------------------
info "Creating GitHub release v$NEW …"
if [ "$ALLOW_DIRTY" = "1" ]; then
	git add -A
else
	git add "$INIT"
	[ -f "$CITATION" ] && git add "$CITATION"
fi
git commit -m "Release v$NEW" || true
git push
# Pin the tag to the exact commit we just pushed, so it is correct even when
# releasing from a branch other than the repo default.
gh release create "v$NEW" \
	--target "$(git rev-parse HEAD)" \
	--title "v$NEW" \
	--generate-notes
info "✓ GitHub release v$NEW created"
info "✓ GitHub Actions will publish to PyPI"

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
info "Changes that will be committed (git add -A):"
git status --short 2>/dev/null | sed 's/^/  /' || true
info ""
[ "$BRANCH" = "main" ] || info "⚠ You are on '$BRANCH', not 'main'."

if [ "$DRYRUN" = "1" ]; then
	info "DRYRUN — would set version to $NEW, commit \"Release v$NEW\", push, and create GitHub release v$NEW."
	info "Nothing changed."
	exit 0
fi

# --- Apply version bump (revert cleanly on abort) -----------------------------
if [ "$NEW" != "$CURRENT" ]; then
	set_version "$NEW"
	info "✓ Set version $CURRENT → $NEW in $INIT"
fi

if [ "$CONFIRM" != "yes" ]; then
	printf "Release v%s? This will commit, push and tag. [y/N] " "$NEW"
	read -r ans || ans=""
	case "$ans" in
	[Yy] | [Yy][Ee][Ss]) ;;
	*)
		[ "$NEW" != "$CURRENT" ] && set_version "$CURRENT" && info "Reverted $INIT."
		die "aborted — nothing committed or pushed"
		;;
	esac
fi

# --- Release -------------------------------------------------------------------
info "Creating GitHub release v$NEW …"
git add -A
git commit -m "Release v$NEW" || true
git push
gh release create "v$NEW" \
	--title "v$NEW" \
	--notes "See CHANGELOG.md for details" \
	--generate-notes
info "✓ GitHub release v$NEW created"
info "✓ GitHub Actions will publish to PyPI"

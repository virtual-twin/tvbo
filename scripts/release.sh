#!/usr/bin/env bash
# Copyright © Charité Universitätsmedizin Berlin. This software is licensed under the terms of the European Union Public Licence (EUPL) version 1.2 or later.
# Cut a release from the variables `make release` passes (VERSION or BUMP, CONFIRM, DRYRUN, ALLOW_DIRTY): settle a forward version, show what ships, then commit, push and create the GitHub release whose tag the docker, docs and PyPI workflows publish.
set -eu

INIT="tvbo/__init__.py"
VERSION="${VERSION:-}"
BUMP="${BUMP:-}"
CONFIRM="${CONFIRM:-}"
DRYRUN="${DRYRUN:-}"
ALLOW_DIRTY="${ALLOW_DIRTY:-}"

die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*"; }

# Canonical PEP 440 with an optional a/b/rc pre-release: the forms whose semver tag spelling orders the same way.
valid_version() {
	printf '%s' "$1" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$'
}

# Higher of two versions; `~` sorts a pre-release before its final release under `sort -V`, as PEP 440 orders them.
highest() { printf '%s\n%s\n' "$1" "$2" | sed -E 's/(a|b|rc)([0-9]+)$/~\1\2/' | sort -V | tail -1 | tr -d '~'; }

tag_of() { make -s print-tag PKG_VERSION="$1"; }

version_of_tag() { printf '%s' "$1" | sed -E 's/^v//; s/-(a|b|rc)\.([0-9]+)$/\1\2/'; }

read_current() { make -s print-version; }

set_version() {
	# Portable in-place edit (BSD + GNU sed).
	sed -i.bak "s/^__version__ = .*/__version__ = \"$1\"/" "$INIT" && rm -f "$INIT.bak"
	[ "$(read_current)" = "$1" ] || die "failed to write version $1 to $INIT"
}

CITATION="CITATION.cff"

cit_field() { grep -E "^$1:" "$CITATION" | head -1 | sed -E "s/^$1:[[:space:]]*//"; }

set_citation() {
	# $1 = version, $2 = date-released (YYYY-MM-DD), so the citation names the release it ships with.
	[ -f "$CITATION" ] || return 0
	sed -i.bak -E "s/^version:.*/version: \"$1\"/" "$CITATION" && rm -f "$CITATION.bak"
	[ -n "${2:-}" ] && sed -i.bak -E "s/^date-released:.*/date-released: $2/" "$CITATION" && rm -f "$CITATION.bak"
	return 0
}

# Write the release version to every version-carrying file, and undo it, from the globals NEW/CURRENT/TODAY/CIT_* set below.
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

# The next version at *level*; a candidate already at that level is released instead (1.0.0rc1 → 1.0.0 at any level, 1.2.3rc1 → patch 1.2.3, minor 1.3.0), as npm's `semver.inc` bumps.
bump_version() {
	base=$1 level=$2
	maj=$(printf '%s' "$base" | cut -d. -f1)
	min=$(printf '%s' "$base" | cut -d. -f2)
	pat=$(printf '%s' "$base" | cut -d. -f3 | sed 's/[^0-9].*$//')
	: "${maj:=0}" "${min:=0}" "${pat:=0}"
	pre=""
	printf '%s' "$base" | grep -Eq '(a|b|rc)[0-9]+$' && pre=1
	case "$level" in
		patch) [ -n "$pre" ] || pat=$((pat + 1)) ;;
		minor)
			{ [ -n "$pre" ] && [ "$pat" = 0 ]; } || min=$((min + 1))
			pat=0
			;;
		major)
			{ [ -n "$pre" ] && [ "$min" = 0 ] && [ "$pat" = 0 ]; } || maj=$((maj + 1))
			min=0 pat=0
			;;
		*) die "invalid BUMP '$level' (expected patch|minor|major)" ;;
	esac
	printf '%s.%s.%s' "$maj" "$min" "$pat"
}

[ -f "$INIT" ] || die "$INIT not found (run from the repo root)"
command -v gh >/dev/null 2>&1 || die "GitHub CLI 'gh' is required"

CURRENT=$(read_current)
# The highest version any published release carries, pre-releases included: `gh release view` answers only the latest full release, so a published candidate would pass as unreleased.
LATEST="" LATEST_TAG=""
for tag in $(gh release list --exclude-drafts --limit 1000 --json tagName --jq '.[].tagName' 2>/dev/null || true); do
	version=$(version_of_tag "$tag")
	valid_version "$version" || continue
	if [ -z "$LATEST" ] || [ "$(highest "$LATEST" "$version")" = "$version" ]; then
		LATEST=$version LATEST_TAG=$tag
	fi
done
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
valid_version "$NEW" || die "invalid version '$NEW' (expected x.y.z, or x.y.z followed by aN, bN or rcN)"

# --- With neither VERSION nor BUMP and nothing past the latest release, pick a bump interactively ----
if [ -z "$VERSION" ] && [ -z "$BUMP" ] && [ -n "$LATEST" ] &&
	[ "$(highest "$LATEST" "$NEW")" = "$LATEST" ]; then
	if [ ! -t 0 ]; then
		die "$NEW is already published — choose a bump: make release BUMP=patch|minor|major"
	fi
	base=$(highest "$LATEST" "$CURRENT")
	p=$(bump_version "$base" patch)
	m=$(bump_version "$base" minor)
	M=$(bump_version "$base" major)
	printf '\n%s is already published. Which release do you want to cut?\n' "$CURRENT" >&2
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
	info "→ selected $NEW"
fi

TAG=$(tag_of "$NEW")
PRERELEASE=""
case "$TAG" in *-*) PRERELEASE="--prerelease" ;; esac

# --- Verify it is a forward bump ----------------------------------------------
if [ -n "$LATEST" ]; then
	if [ "$NEW" = "$LATEST" ]; then
		die "$TAG is already the latest release — bump it (make release BUMP=patch)"
	fi
	if [ "$(highest "$LATEST" "$NEW")" != "$NEW" ]; then
		die "$TAG is lower than the latest release $LATEST_TAG — refusing to release a non-bump"
	fi
fi

# --- Show what is shipping -----------------------------------------------------
info ""
info "  branch           : $BRANCH"
info "  latest published : ${LATEST:-<none>}"
info "  current (__init__): $CURRENT"
info "  about to release : $NEW, tagged $TAG${PRERELEASE:+ (pre-release)}"
info ""
info "Recent releases:"
gh release list --limit 5 2>/dev/null | sed 's/^/  /' || info "  (unable to list releases)"
info ""
if [ -n "$LATEST" ]; then
	info "Commits since $LATEST_TAG:"
	git log "$LATEST_TAG..HEAD" --oneline 2>/dev/null | sed 's/^/  /' || info "  (tag $LATEST_TAG not found locally)"
	info ""
fi
# The release commit holds only the version bump, so other tracked changes block it unless ALLOW_DIRTY=1 commits them all.
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
		info "DRYRUN — blocked: commit or stash the tracked changes above, then release $TAG."
	else
		info "DRYRUN — would set version to $NEW, commit \"Release $TAG\", push, and create GitHub ${PRERELEASE:+pre-}release $TAG."
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
	printf "Release %s? This will commit, push and tag. [y/N] " "$TAG"
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
info "Creating GitHub ${PRERELEASE:+pre-}release $TAG …"
if [ "$ALLOW_DIRTY" = "1" ]; then
	git add -A
else
	git add "$INIT"
	[ -f "$CITATION" ] && git add "$CITATION"
fi
git commit -m "Release $TAG" || true
git push
# Pinned to the commit just pushed, so the tag is right from any branch.
gh release create "$TAG" \
	--target "$(git rev-parse HEAD)" \
	--title "$TAG" \
	--generate-notes $PRERELEASE
info "✓ GitHub ${PRERELEASE:+pre-}release $TAG created"
info "✓ GitHub Actions will publish to PyPI"

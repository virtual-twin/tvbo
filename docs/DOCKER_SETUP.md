# Docker Publishing Setup Guide

## Overview

The Docker publishing workflow is **configured and active**. It automatically builds and publishes Docker images when you create a release, synchronized with the PyPI package publishing workflow.

## Current Status

✅ **Active**: Workflow triggers automatically on releases
✅ **Synchronized**: Runs together with PyPI publishing
✅ **Safe**: Gracefully handles missing Docker Hub credentials
🎯 **Manual**: Also available for manual testing via GitHub Actions tab

## What's Been Set Up

### 1. GitHub Workflow: `.github/workflows/publish-docker.yml`

This workflow:
- ✅ Builds Docker images with proper versioning
- ✅ Tests images before pushing
- ✅ Supports both Docker Hub and GitHub Container Registry (ghcr.io)
- ✅ Uses Docker BuildKit for efficient builds
- ✅ Implements layer caching for faster builds
- ✅ **Triggers automatically on releases** (same as PyPI publishing)
- ✅ **Always pushes to GitHub Container Registry** (free, no setup needed)
- ⚙️ **Conditionally pushes to Docker Hub** (if credentials configured)

### 2. Supported Registries

| Registry | Status | Public? | Authentication |
|----------|--------|---------|----------------|
| **GitHub Container Registry** (ghcr.io) | ✅ Ready | Yes (when repo is public) | Automatic via GITHUB_TOKEN |
| **Docker Hub** | 🔧 Needs secrets | Optional | Requires DOCKERHUB_USERNAME & DOCKERHUB_TOKEN |

## Setup Instructions

### Step 1: Configure Docker Hub (Optional)

If you want to publish to Docker Hub:

1. **Create Docker Hub account** (if you don't have one):
   - Go to https://hub.docker.com/signup

2. **Create access token**:
   - Go to https://hub.docker.com/settings/security
   - Click "New Access Token"
   - Name: `github-actions-tvbo`
   - Permissions: Read & Write
   - Copy the token (you won't see it again!)

3. **Add GitHub Secrets**:
   - Go to https://github.com/virtual-twin/tvbo-python/settings/secrets/actions
   - Click "New repository secret"

   Add these two secrets:
   ```
   Name: DOCKERHUB_USERNAME
   Value: your-dockerhub-username

   Name: DOCKERHUB_TOKEN
   Value: your-access-token-from-step-2
   ```

4. **Update workflow file**:
   Edit `.github/workflows/publish-docker.yml`:
   ```yaml
   env:
     DOCKERHUB_USERNAME: ${{ secrets.DOCKERHUB_USERNAME }}
     DOCKERHUB_REPOSITORY: tvbo  # Change this if needed
   ```

### Step 2: Test the Workflow (Manual Run)

Before your first release, test the workflow manually:

1. Go to: https://github.com/virtual-twin/tvbo-python/actions/workflows/publish-docker.yml
2. Click "Run workflow"
3. Choose options:
   - **Push to Docker Hub?**: `false` (for testing)
   - **Version tag**: Leave empty or specify a test version (e.g., `0.0.1-test`)
4. Click "Run workflow"

This will:
- ✅ Build the Docker image
- ✅ Run smoke tests
- ✅ Push to GitHub Container Registry only (ghcr.io)
- ❌ NOT push to Docker Hub

### Step 3: Create a Release

When you're ready to publish (after testing):

1. **Update version in `tvbo/__init__.py`**:
   ```python
   __version__ = "1.0.0"
   ```

2. **Update CHANGELOG.md** with the new version section

3. **Create a GitHub release**:
   - Go to https://github.com/virtual-twin/tvbo-python/releases/new
   - Tag: `v1.0.0`
   - Title: `Release v1.0.0`
   - Description: Release notes from CHANGELOG
   - Click "Publish release"

4. **Automatic workflows trigger**:
   - ✅ `publish-pypi.yml` → Publishes to PyPI
   - ✅ `publish-docker.yml` → Builds and publishes Docker images
   - ✅ `release-ontology.yml` → Creates ontology release (if ontology changed)

## Using Published Images

### From GitHub Container Registry (Always Available)

```bash
# Pull latest
docker pull ghcr.io/virtual-twin/tvbo-python:latest

# Pull specific version
docker pull ghcr.io/virtual-twin/tvbo-python:1.0.0

# Run
docker run -it ghcr.io/virtual-twin/tvbo-python:latest
```

### From Docker Hub (If Configured)

```bash
# Pull latest
docker pull yourusername/tvbo:latest

# Pull specific version
docker pull yourusername/tvbo:1.0.0

# Run
docker run -it yourusername/tvbo:latest
```

## Image Tagging Strategy

Images are automatically tagged with:

| Tag | Example | When |
|-----|---------|------|
| Version | `1.0.0` | Every build |
| Major.Minor | `1.0` | Every build |
| Major | `1` | Every build |
| Latest | `latest` | Only on main branch |
| SHA | `main-abc1234` | Every build |

## Versioning

The workflow automatically detects versions from:
1. Git release tag (when triggered by release) - **Primary method**
2. Manual input (when triggered manually)
3. `tvbo/__init__.py` `__version__` (fallback)

**Example**: Release tag `v1.0.0` → Docker images tagged as `1.0.0`, `1.0`, `1`, and `latest`

## GitHub Container Registry Permissions

If images don't appear public after the repo is public:

1. Go to https://github.com/orgs/virtual-twin/packages
2. Find the `tvbo-python` package
3. Click "Package settings"
4. Under "Danger Zone" → "Change visibility" → Make public
5. Link to repository if needed

## Dockerfile Requirements

Your current `Dockerfile` needs to work with the workflow. If it requires secrets (like `GITLAB_TOKEN`), uncomment these lines in the workflow:

```yaml
secrets: |
  gitlab_token=${{ secrets.GITLAB_TOKEN }}
```

And add the secret to GitHub: https://github.com/virtual-twin/tvbo-python/settings/secrets/actions

## Testing Locally

Test the Docker build locally before pushing:

```bash
# Build
docker build -t tvbo:test .

# Test
docker run --rm tvbo:test python -c "import tvbo; print(tvbo.__version__)"

# Run interactively
docker run -it tvbo:test bash
```

## Troubleshooting

### Build fails with "secret not found"

If your Dockerfile uses `--secret id=gitlab_token`:
1. Add `GITLAB_TOKEN` to GitHub secrets
2. Uncomment the secrets section in the workflow

### Images not appearing in registry

- **GitHub Container Registry**: Check package visibility settings
- **Docker Hub**: Verify secrets are set correctly and push is enabled

### Version detection fails

Ensure `tvbo/__init__.py` has a valid `__version__`:
```python
__version__ = "1.0.0"
```

## When Docker Images Are Published

Docker images are automatically built and published in these scenarios:

| Trigger | Build | Push to ghcr.io | Push to Docker Hub | When |
|---------|-------|-----------------|-------------------|------|
| **Release created** | ✅ Yes | ✅ Yes | ✅ Yes (if configured) | Automatic on GitHub release |
| **Manual workflow** | ✅ Yes | ✅ Yes | ⚙️ Optional (checkbox) | Manual from Actions tab |

### Release Flow

```
Create Release (v1.0.0)
    ↓
├─→ publish-pypi.yml runs    → Package on PyPI
├─→ publish-docker.yml runs  → Images on ghcr.io (+ Docker Hub if configured)
└─→ release-ontology.yml     → Ontology release (if changed)
```

## Pre-Release Checklist

Before creating your first release that publishes Docker images:

- [ ] Test build locally with `docker build -t tvbo:test .`
- [ ] Test manual workflow run (without Docker Hub push)
- [ ] Verify images work: `docker run --rm tvbo:test python -c "import tvbo; print(tvbo.__version__)"`
- [ ] **(Optional)** Add Docker Hub secrets to GitHub if you want to publish there
- [ ] Update version in `tvbo/__init__.py`
- [ ] Update CHANGELOG.md
- [ ] Update README with Docker installation instructions
- [ ] Create release on GitHub
- [ ] Verify both workflows complete successfully
- [ ] Test pulling and running published images from ghcr.io
- [ ] **(If Docker Hub configured)** Test pulling from Docker Hub

## Maintenance

### Updating Images

Images automatically rebuild when:
- ✅ **New release is created** (automatic, synchronized with PyPI)
- 🎯 **Manual workflow trigger** (for testing or ad-hoc builds)

### Publishing Flow

1. **Development**: Make changes to code, Dockerfile, or dependencies
2. **Version bump**: Update `tvbo/__init__.py` and CHANGELOG.md
3. **Create release**: Tag and publish release on GitHub
4. **Automatic build**: Workflow triggers and publishes images
5. **Verification**: Check Actions tab for build status

### Cleaning Old Images

**GitHub Container Registry:**
- Go to package settings
- Delete old versions manually or set retention policy

**Docker Hub:**
- Go to repository settings
- Set up automated cleanup rules

## Documentation Updates

After your first successful release, update your README:

```markdown
## Installation

### Using pip (PyPI)
\`\`\`bash
pip install tvbo
\`\`\`

### Using Docker

Pull the latest image from GitHub Container Registry:
\`\`\`bash
docker pull ghcr.io/virtual-twin/tvbo-python:latest
docker run -it ghcr.io/virtual-twin/tvbo-python:latest
\`\`\`

Or pull a specific version:
\`\`\`bash
docker pull ghcr.io/virtual-twin/tvbo-python:1.0.0
\`\`\`

**(If Docker Hub is configured)** Alternatively, from Docker Hub:
\`\`\`bash
docker pull yourusername/tvbo:latest
\`\`\`
```

## Release Workflow Summary

When you create a release, three automated workflows run in parallel:

1. **`publish-pypi.yml`**
   - Validates version matches release tag
   - Checks CHANGELOG.md contains version section
   - Builds Python package (sdist + wheel)
   - Publishes to PyPI
   - Updates release notes from CHANGELOG

2. **`publish-docker.yml`**
   - Extracts version from release tag
   - Builds Docker image with BuildKit
   - Runs smoke tests
   - Pushes to GitHub Container Registry (always)
   - Pushes to Docker Hub (if configured)
   - Tags with version, major.minor, major, latest, and SHA

3. **`release-ontology.yml`**
   - Checks if ontology file changed
   - Creates separate ontology release (`ontology-v{version}`)
   - Attaches OWL file and SHA256 checksum
   - Generates release notes with usage instructions

All workflows use the same version number, ensuring synchronized releases across all artifacts.

## Support

For issues with the Docker workflow:
- Check the Actions log: https://github.com/virtual-twin/tvbo-python/actions/workflows/publish-docker.yml
- Open an issue: https://github.com/virtual-twin/tvbo-python/issues
- Contact: leon.martin@bih-charite.de

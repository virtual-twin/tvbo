# Ontology Release Automation - Setup Summary

## What Was Created

### 1. GitHub Workflow: `.github/workflows/release-ontology.yml`

This workflow automatically creates versioned releases of the TVB-O ontology whenever it changes in the main branch.

**Key Features:**
- ✅ Triggered only when `tvbo/data/ontology/tvb-o.owl` changes
- ✅ Extracts version from `tvbo/__init__.py`
- ✅ Creates releases tagged as `ontology-v{version}` (e.g., `ontology-v1.0.0`)
- ✅ Prevents duplicate releases for the same version
- ✅ Generates SHA256 checksums for verification
- ✅ Includes comprehensive release notes with usage instructions
- ✅ Attaches both the OWL file and checksum to the release

**Workflow Steps:**
1. Detects if ontology file changed in the push
2. Extracts current version from Python package
3. Checks if release already exists
4. Generates release notes and checksums
5. Creates GitHub release with tagged version
6. Attaches ontology file and checksum

### 2. Documentation: `docs/ONTOLOGY_RELEASES.md`

Comprehensive guide covering:
- How the automated release process works
- How to download and use released ontologies
- Persistent URL formats for citations
- Version synchronization between package and ontology
- Developer guidelines for triggering new releases
- Troubleshooting common issues
- Citation format recommendations

### 3. Updated Manuscript: `manuscript/manuscript.qmd`

The manuscript now references:
- The automated ontology releases
- The releases page URL
- The versioning scheme (`ontology-v{version}`)
- Links to both current ontology and versioned releases

## How It Works

### Version Synchronization

```
tvbo Package Version: 1.0.0 (in tvbo/__init__.py)
         ↓
Ontology Release Tag: ontology-v1.0.0
         ↓
Release URL: https://github.com/virtual-twin/tvbo-python/releases/tag/ontology-v1.0.0
```

### Triggering a Release

**Scenario 1: Ontology Update**
```bash
# Edit the ontology file
vim tvbo/data/ontology/tvb-o.owl

# Commit and push to main (or via PR)
git add tvbo/data/ontology/tvb-o.owl
git commit -m "Update ontology: add new model classes"
git push origin main

# → Automatic release created with current version from __init__.py
```

**Scenario 2: Version Bump + Ontology Update**
```bash
# Update version
vim tvbo/__init__.py  # Change __version__ = "1.1.0"

# Update changelog
vim CHANGELOG.md  # Add ## Version 1.1.0 section

# Update ontology
vim tvbo/data/ontology/tvb-o.owl

# Commit all changes
git add tvbo/__init__.py CHANGELOG.md tvbo/data/ontology/tvb-o.owl
git commit -m "Release v1.1.0 with updated ontology"
git push origin main

# → Automatic release created: ontology-v1.1.0
```

## Benefits

### For Researchers
- 📚 **Citable Versions**: Each ontology version has a permanent URL and DOI-ready reference
- 🔒 **Immutable**: Released versions never change, ensuring reproducibility
- ✅ **Verified**: SHA256 checksums ensure file integrity
- 📖 **Documented**: Each release includes usage instructions

### For Developers
- 🤖 **Automated**: No manual release process needed
- 🔄 **Synchronized**: Ontology version always matches package version
- 🚫 **Safe**: Duplicate prevention avoids confusion
- 📊 **Traceable**: Clear connection between commits and releases

### For the Community
- 🌐 **Accessible**: Easy download from GitHub releases
- 🔗 **Persistent**: Stable URLs for citations and integrations
- 📦 **Standard**: Follows semantic versioning conventions
- 🔍 **Discoverable**: Clear tagging scheme (`ontology-v*`)

## Usage Examples

### Download Specific Version
```bash
# Download ontology v1.0.0
wget https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v1.0.0/tvb-o.owl

# Verify integrity
wget https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v1.0.0/tvb-o.owl.sha256
sha256sum -c tvb-o.owl.sha256
```

### Reference in Paper
```markdown
The Virtual Brain Ontology (version 1.0.0) is available at:
https://github.com/virtual-twin/tvbo-python/releases/tag/ontology-v1.0.0
```

### Load in Python
```python
import owlready2
import urllib.request

# Download specific version
version = "1.0.0"
url = f"https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v{version}/tvb-o.owl"
urllib.request.urlretrieve(url, "tvb-o.owl")

# Load ontology
onto = owlready2.get_ontology("tvb-o.owl").load()
```

### Use with RDFlib
```python
from rdflib import Graph

g = Graph()
g.parse("https://github.com/virtual-twin/tvbo-python/raw/ontology-v1.0.0/tvbo/data/ontology/tvb-o.owl")
```

## Checking Releases

### List All Ontology Releases
```bash
gh release list | grep "ontology-v"
```

### View Specific Release
```bash
gh release view ontology-v1.0.0
```

### Check Latest Ontology Version
```bash
gh release list | grep "ontology-v" | head -1
```

## Integration with Existing Workflows

The ontology release workflow is **independent** from the PyPI package release workflow:

| Workflow | Trigger | Tag Format | Purpose |
|----------|---------|------------|---------|
| `release-ontology.yml` | Ontology file changes in main | `ontology-v1.0.0` | Release OWL file |
| `publish-pypi.yml` | Manual release creation | `v1.0.0` | Release Python package |

This allows:
- Ontology updates without full package releases
- Package releases without ontology changes
- Both synchronized to the same version number

## Future Enhancements

Possible additions:
- 🤖 Automatic DOI minting via Zenodo
- 📊 Ontology metrics in release notes (classes, properties, axioms)
- 🔄 Validation checks before release
- 📧 Notifications to mailing list
- 🏷️ PURL (Persistent URL) registration
- 📚 Automatic documentation generation

## Questions?

See `docs/ONTOLOGY_RELEASES.md` for detailed documentation, or contact:
- Leon K. Martin: leon.martin@bih-charite.de
- Open an issue: https://github.com/virtual-twin/tvbo-python/issues

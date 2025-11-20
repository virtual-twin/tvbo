# TVB-O Ontology Releases

## Overview

The TVB-O ontology (OWL file) is automatically released whenever it is updated in the main branch. Each ontology release is versioned to match the tvbo Python package version, ensuring synchronization between the software and its semantic definitions.

## Automated Release Process

### How It Works

1. **Trigger**: When changes to `tvbo/data/ontology/tvb-o.owl` are pushed to the `main` branch
2. **Version Detection**: The workflow extracts the current version from `tvbo/__init__.py`
3. **Release Creation**: A new GitHub release is created with:
   - Tag format: `ontology-v{version}` (e.g., `ontology-v1.0.0`)
   - Title: `TVB-O Ontology v{version}`
   - Attached files:
     - `tvb-o.owl` - The ontology file
     - `tvb-o.owl.sha256` - SHA256 checksum for verification

### Workflow Details

The automated release is handled by `.github/workflows/release-ontology.yml`, which:
- Only runs when the ontology file changes
- Checks if a release for the current version already exists (prevents duplicates)
- Generates checksums for file integrity verification
- Creates comprehensive release notes

## Using Released Ontologies

### Direct Download

Download the ontology from any release:
```bash
# Download specific version
wget https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v1.0.0/tvb-o.owl

# Verify checksum
wget https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v1.0.0/tvb-o.owl.sha256
sha256sum -c tvb-o.owl.sha256
```

### Persistent URLs

For stable references in papers or documentation, use the raw GitHub URL with the tag:
```
https://github.com/virtual-twin/tvbo-python/raw/ontology-v1.0.0/tvbo/data/ontology/tvb-o.owl
```

This URL will always point to the specific version, even if the main branch is updated.

### Programmatic Access

```python
import urllib.request

version = "1.0.0"
url = f"https://github.com/virtual-twin/tvbo-python/releases/download/ontology-v{version}/tvb-o.owl"
urllib.request.urlretrieve(url, "tvb-o.owl")
```

## Version Synchronization

The ontology version always matches the tvbo package version:

| Package Version | Ontology Tag | Release Type |
|----------------|--------------|--------------|
| `1.0.0` | `ontology-v1.0.0` | Ontology Release |
| `1.0.0` | `v1.0.0` | Package Release (PyPI) |

This ensures that:
- Users can easily identify which ontology corresponds to their installed package
- Citations can reference both the software and ontology versions consistently
- Semantic compatibility is maintained across versions

## For Developers

### Triggering a New Ontology Release

1. Make changes to `tvbo/data/ontology/tvb-o.owl`
2. Commit and push to a feature branch
3. Create a pull request to `main`
4. Once merged, the workflow automatically:
   - Detects the ontology change
   - Reads the current version from `tvbo/__init__.py`
   - Creates a new ontology release

### Manual Version Update

If you need to create a new ontology release with a new version:

1. Update the version in `tvbo/__init__.py`:
   ```python
   __version__ = "1.1.0"
   ```

2. Update `CHANGELOG.md` with the new version section

3. Commit both changes:
   ```bash
   git add tvbo/__init__.py CHANGELOG.md tvbo/data/ontology/tvb-o.owl
   git commit -m "Bump version to 1.1.0 with ontology updates"
   ```

4. Push to main (or merge PR)

### Checking Existing Releases

View all ontology releases:
```bash
gh release list --limit 100 | grep "ontology-v"
```

Or visit: https://github.com/virtual-twin/tvbo-python/releases?q=ontology-v&expanded=true

## Citation

When citing the TVB-O ontology in publications, please reference both the version and the permanent URL:

```bibtex
@misc{tvbo_ontology,
  title = {The Virtual Brain Ontology (TVB-O)},
  author = {Martin, Leon K. and others},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/virtual-twin/tvbo-python/releases/tag/ontology-v1.0.0},
  note = {OWL ontology}
}
```

## Troubleshooting

### Release Not Created

If the ontology file changed but no release was created:

1. Check the GitHub Actions log: https://github.com/virtual-twin/tvbo-python/actions/workflows/release-ontology.yml
2. Verify that `tvbo/__init__.py` contains a valid `__version__` string
3. Ensure the ontology file path is correct: `tvbo/data/ontology/tvb-o.owl`

### Duplicate Release Prevention

The workflow automatically checks if a release with the same tag already exists. If you need to update an existing release:

1. Delete the existing release (if needed)
2. Push a new commit that modifies the ontology file

### Version Mismatch

If you see a version mismatch error, ensure that:
- The version in `tvbo/__init__.py` follows semantic versioning (e.g., `1.0.0`)
- The version has been updated if you're making a new release

## Contact

For questions or issues with ontology releases, please:
- Open an issue: https://github.com/virtual-twin/tvbo-python/issues
- Contact: leon.martin@bih-charite.de

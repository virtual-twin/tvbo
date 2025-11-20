"""
Test that all models in the database can be loaded without errors.
"""

import os
import glob
import pytest
import yaml
from pathlib import Path

from tvbo import Dynamics

# Get the database directory
REPO_ROOT = Path(__file__).parent.parent
DB_ROOT = REPO_ROOT / "database"
MODEL_DIR = DB_ROOT / "models"


def get_all_model_files():
    """Get all YAML model files from the database."""
    model_files = []
    for ext in ("*.yaml", "*.yml"):
        pattern = str(MODEL_DIR / "**" / ext)
        model_files.extend(glob.glob(pattern, recursive=True))
    return model_files


def get_model_name(path):
    """Extract model name from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict):
            return raw.get("name", Path(path).stem)
    except Exception:
        pass
    return Path(path).stem


# Generate test parameters: (path, model_name) tuples
model_files = get_all_model_files()
test_params = [(path, get_model_name(path)) for path in model_files]


@pytest.mark.parametrize("model_path,model_name", test_params, ids=lambda x: x[1] if isinstance(x, str) else str(x))
def test_model_loads(model_path, model_name):
    """Test that a model can be loaded from its YAML file."""
    # Read the YAML to ensure it's valid
    with open(model_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    assert isinstance(raw, dict), f"Model YAML must be a dictionary, got {type(raw)}"

    # Load the model using tvbo
    model = Dynamics.from_file(model_path)

    # Basic validation
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'name'), "Model should have a name attribute"

    # Verify the model name matches
    if raw.get("name"):
        assert model.name == raw.get("name"), f"Model name mismatch: {model.name} != {raw.get('name')}"


def test_all_models_found():
    """Ensure we found at least some models to test."""
    assert len(model_files) > 0, f"No model files found in {MODEL_DIR}"
    print(f"\nFound {len(model_files)} model files to test")


if __name__ == "__main__":
    # Allow running as a script for quick testing
    import sys

    print(f"Testing models in: {MODEL_DIR}")
    print("=" * 80)

    failed_models = []
    success_count = 0

    for path, name in test_params:
        print(f"\n📝 Testing: {name}")
        print(f"   File: {Path(path).relative_to(REPO_ROOT)}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            if not isinstance(raw, dict):
                print(f"   ⚠️  SKIP: Not a dict")
                continue

            model = Dynamics.from_file(path)
            print(f"   ✅ Loaded successfully")
            success_count += 1

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ FAILED: {type(e).__name__}: {error_msg[:100]}")
            failed_models.append({
                "path": str(Path(path).relative_to(REPO_ROOT)),
                "name": name,
                "error": f"{type(e).__name__}: {error_msg}"
            })

    print("\n" + "=" * 80)
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {len(failed_models)}")

    if failed_models:
        print(f"\n❌ Failed Models:")
        for fm in failed_models:
            print(f"\n   Model: {fm['name']}")
            print(f"   File: {fm['path']}")
            print(f"   Error: {fm['error'][:200]}")

    sys.exit(0 if not failed_models else 1)

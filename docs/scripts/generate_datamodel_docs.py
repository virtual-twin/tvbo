#!/usr/bin/env python
"""Generate LinkML datamodel documentation for Quarto integration.

This script generates markdown documentation from the tvbo_datamodel.yaml schema and places it in the docs/datamodel directory for Quarto rendering.
"""

import re
import subprocess
import sys
from pathlib import Path

_MKDOCS_SEARCH_FRONTMATTER = re.compile(
    r"\A---\s*\n"  # opening fence
    r"search:\s*\n"  # `search:` key (MkDocs Material search-boost block)
    r"(?:\s+\S.*\n)+"  # its indented children
    r"---\s*\n"  # closing fence
    r"\n?",  # optional trailing blank line
)


def convert_md_to_qmd(output_dir: Path):
    """Convert all .md files to .qmd files for Quarto and fix internal links."""
    print("\nConverting .md files to .qmd files...")

    # First pass: rename all .md files to .qmd
    md_files = list(output_dir.rglob("*.md"))
    for md_file in md_files:
        qmd_file = md_file.with_suffix(".qmd")
        md_file.rename(qmd_file)

    # Second pass: fix all internal .md links to .qmd and mermaid blocks
    for qmd_file in output_dir.rglob("*.qmd"):
        original = qmd_file.read_text()
        # LinkML docgen emits MkDocs-style `search: {boost: N}`; Quarto's `search:` must be a boolean, so drop it.
        content = _MKDOCS_SEARCH_FRONTMATTER.sub("", original)
        # Replace markdown links: [text](path.md) -> [text](path.qmd)
        updated_content = content.replace(".md)", ".qmd)")
        # Replace mermaid code blocks: ```mermaid -> ```{mermaid}
        updated_content = updated_content.replace("```mermaid", "```{mermaid}")

        # Fix mermaid click links based on the subdirectory
        if qmd_file.parent.name == "classes":
            # In classes folder: ../../classes/ClassName/ -> ./ClassName.html (same directory)
            updated_content = updated_content.replace('href "../../classes/', 'href "./')
            updated_content = updated_content.replace('/"', '.html"')  # Replace trailing / with .html
            # Also fix links to slots: ../../slots/slotname/ -> ../slots/slotname.html
            updated_content = updated_content.replace('href "../../slots/', 'href "../slots/')
        elif qmd_file.parent.name == "slots":
            # In slots folder: ../../classes/ClassName/ -> ../classes/ClassName.html
            updated_content = updated_content.replace('href "../../classes/', 'href "../classes/')
            # In slots folder: ../../slots/slotname/ -> ./slotname.html (same directory)
            updated_content = updated_content.replace('href "../../slots/', 'href "./')
            updated_content = updated_content.replace('/"', '.html"')  # Replace trailing / with .html
        elif qmd_file.parent.name == "enums":
            # In enums folder: ../../classes/ClassName/ -> ../classes/ClassName.html
            updated_content = updated_content.replace('href "../../classes/', 'href "../classes/')
            # In enums folder: ../../enums/enumname/ -> ./enumname.html (same directory)
            updated_content = updated_content.replace('href "../../enums/', 'href "./')
            updated_content = updated_content.replace('/"', '.html"')  # Replace trailing / with .html

        if original != updated_content:
            qmd_file.write_text(updated_content)

    print(f"✓ Converted {len(md_files)} files and fixed internal links")


def main():
    """Generate LinkML documentation."""
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    schema_path = repo_root / "schema" / "tvbo_datamodel.yaml"
    output_dir = repo_root / "docs" / "datamodel"
    stamp_file = output_dir / ".gen_stamp"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Freshness check: skip only if the schema and *this generator script* are both unchanged since the last successful run. Every file under schema/ counts, because tvbo_datamodel.yaml imports the rest and a class added to one of those would otherwise never reach these pages; the script's own mtime counts so edits to ``convert_md_to_qmd`` re-process existing pages.
    script_path = Path(__file__).resolve()
    input_mtime = max(p.stat().st_mtime for p in [script_path, *sorted(schema_path.parent.rglob("*.yaml"))])
    if stamp_file.exists() and any(output_dir.rglob("*.qmd")):
        stamp_mtime = stamp_file.stat().st_mtime
        if input_mtime <= stamp_mtime:
            print("Datamodel docs up-to-date (schema unchanged). Skipping.")
            return 0

    print("Generating LinkML documentation...")
    print(f"  Schema: {schema_path}")
    print(f"  Output: {output_dir}")

    # Generate documentation using LinkML gen-doc Resolve gen-doc from the same Python interpreter's bin directory to ensure it works even when Quarto doesn't inherit shell PATH.
    import shutil

    gen_doc = shutil.which("gen-doc")
    if gen_doc is None:
        gen_doc = str(Path(sys.executable).parent / "gen-doc")
        if not Path(gen_doc).is_file():
            gen_doc = "gen-doc"  # fallback to PATH lookup

    try:
        subprocess.run(
            [
                gen_doc,
                "--directory",
                str(output_dir),
                "--format",
                "markdown",
                "--no-mergeimports",  # Keep imports separate
                "--subfolder-type-separation",  # Separate classes, slots, enums into subdirectories
                str(schema_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ LinkML documentation generated successfully")

        # Convert .md files to .qmd files for Quarto
        convert_md_to_qmd(output_dir)
        print("✓ Conversion complete")

        # Write stamp file so we can skip next time if schema unchanged
        stamp_file.write_text(str(input_mtime))

        return 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating documentation: {e.stderr}", file=sys.stderr)
        # Continue with existing .qmd files if available
        if any(output_dir.rglob("*.qmd")):
            print("  Using existing datamodel docs from previous generation.", file=sys.stderr)
            return 0
        return 1
    except FileNotFoundError:
        print("✗ Error: 'gen-doc' command not found. Install linkml: pip install linkml", file=sys.stderr)
        # Continue with existing .qmd files if available
        if any(output_dir.rglob("*.qmd")):
            print("  Using existing datamodel docs from previous generation.", file=sys.stderr)
            return 0
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract base64-encoded images from a markdown file into separate files.

Replaces inline data URIs with relative file references so images
render correctly in standard markdown previewers.

Usage:
    python scripts/extract_images.py path/to/file.md
"""
import argparse
import base64
import os
import re
import sys


def extract_images(md_path):
    with open(md_path, "r") as f:
        content = f.read()

    pattern = r"!\[\]\(data:image;base64,([A-Za-z0-9+/=\s]+)\)"
    matches = list(re.finditer(pattern, content))

    if not matches:
        print("No embedded images found.")
        return

    print(f"Found {len(matches)} embedded images")

    img_dir = os.path.join(os.path.dirname(md_path), "img")
    os.makedirs(img_dir, exist_ok=True)

    for i, match in enumerate(matches, 1):
        b64_data = match.group(1).strip()
        img_data = base64.b64decode(b64_data)
        filename = f"img/fig_{i:02d}.png"
        filepath = os.path.join(os.path.dirname(md_path), filename)
        with open(filepath, "wb") as f:
            f.write(img_data)
        print(f"  Extracted {filename} ({len(img_data)} bytes)")

    new_content = content
    for i, match in enumerate(matches, 1):
        new_content = new_content.replace(match.group(0), f"![](img/fig_{i:02d}.png)", 1)

    with open(md_path, "w") as f:
        f.write(new_content)

    print("Done! Updated markdown with file references.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_file", help="Path to the markdown file")
    args = parser.parse_args()

    if not os.path.isfile(args.markdown_file):
        print(f"Error: {args.markdown_file} not found", file=sys.stderr)
        sys.exit(1)

    extract_images(args.markdown_file)

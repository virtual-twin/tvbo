#!/usr/bin/env python3
"""Extract and download images from a markdown file into an img/ folder.

Handles two cases:
  1. Inline base64 data URIs: ![](data:image;base64,...)
  2. Remote URL img tags:      <img src="https://..." ... />

Replaces both with local relative references so images render correctly
in standard markdown previewers.

Usage:
    python scripts/extract_images.py path/to/file.md
"""

import argparse
import base64
import os
import re
import sys
import urllib.parse
import urllib.request


def _ext_from_url(url: str, fallback: str = "jpg") -> str:
    path = urllib.parse.urlparse(url).path
    ext = os.path.splitext(path)[1].lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png", "gif", "webp", "svg"} else fallback


def extract_images(md_path: str) -> None:
    md_dir = os.path.dirname(os.path.abspath(md_path))
    img_dir = os.path.join(md_dir, "img")

    with open(md_path, "r") as f:
        content = f.read()

    new_content = content
    counter = [0]  # mutable so nested helper can increment

    # ── 1. Base64 data URIs ──────────────────────────────────────────────────
    b64_pattern = re.compile(r"!\[\]\(data:image;base64,([A-Za-z0-9+/=\s]+)\)")
    b64_matches = list(b64_pattern.finditer(content))

    if b64_matches:
        print(f"Found {len(b64_matches)} base64-encoded image(s)")
        os.makedirs(img_dir, exist_ok=True)
        for match in b64_matches:
            counter[0] += 1
            b64_data = match.group(1).strip()
            img_data = base64.b64decode(b64_data)
            filename = f"fig_{counter[0]:02d}.png"
            filepath = os.path.join(img_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            print(f"  Extracted img/{filename} ({len(img_data)} bytes)")
            new_content = new_content.replace(
                match.group(0), f"![](img/{filename})", 1
            )

    # ── 2. Remote URL <img> tags ─────────────────────────────────────────────
    url_pattern = re.compile(
        r'<img\s[^>]*src="(https?://[^"]+)"[^>]*/?>',
        re.IGNORECASE,
    )
    url_matches = list(url_pattern.finditer(content))

    if url_matches:
        print(f"Found {len(url_matches)} remote image(s)")
        os.makedirs(img_dir, exist_ok=True)
        for match in url_matches:
            counter[0] += 1
            url = match.group(1)
            ext = _ext_from_url(url)
            filename = f"fig_{counter[0]:02d}.{ext}"
            filepath = os.path.join(img_dir, filename)
            try:
                urllib.request.urlretrieve(url, filepath)
                size = os.path.getsize(filepath)
                print(f"  Downloaded img/{filename} ({size} bytes)")
            except Exception as e:
                print(f"  WARNING: could not download {url}: {e}", file=sys.stderr)
                continue
            new_content = new_content.replace(
                match.group(0), f"![](img/{filename})", 1
            )

    if not b64_matches and not url_matches:
        print("No embedded or remote images found.")
        return

    with open(md_path, "w") as f:
        f.write(new_content)

    print("Done! Updated markdown with local file references.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_file", help="Path to the markdown file")
    args = parser.parse_args()

    if not os.path.isfile(args.markdown_file):
        print(f"Error: {args.markdown_file} not found", file=sys.stderr)
        sys.exit(1)

    extract_images(args.markdown_file)

#!/usr/bin/env python3
"""Extract and download images from a markdown file into an img/ folder.

Handles two cases:
  1. Inline base64 data URIs: ![](data:image/<sub>;base64,...)
                              ![](data:image;base64,...)    (legacy form)
  2. Remote URL img tags:      <img src="https://..." ... />

Replaces each with a local relative reference. When the matched markdown image sits on the same line between an opening and closing HTML tag
(e.g. `<div ...>![](data:...)</div>`), the replacement is emitted as an
HTML `<img>` tag instead — markdown syntax inside a block HTML element is not parsed by CommonMark/GFM and would render as literal text.

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


_VALID_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "svg"}


def _normalize_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return "jpg" if ext == "jpeg" else ext


def _ext_from_url(url: str, fallback: str = "jpg") -> str:
    path = urllib.parse.urlparse(url).path
    ext = _normalize_ext(os.path.splitext(path)[1])
    return ext if ext in _VALID_EXTS else fallback


def _ext_from_bytes(data: bytes, fallback: str = "png") -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    stripped = data.lstrip()
    if stripped.startswith(b"<svg") or stripped.startswith(b"<?xml"):
        return "svg"
    return fallback


def _wrapped_by_html(text: str, start: int, end: int) -> bool:
    """True if the substring [start:end) is sandwiched between an opening
    HTML tag and a closing tag on the same line — the case where markdown-in-HTML rendering fails and we must emit an <img> tag."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    before = text[line_start:start].rstrip()
    after = text[end:line_end].lstrip()
    return before.endswith(">") and after.startswith("</")


def _save(img_data: bytes, img_dir: str, idx: int, ext: str) -> str:
    """Write img_data to img_dir/fig_<idx>.<ext>; skip rewrite if identical bytes already on disk (idempotent reruns)."""
    os.makedirs(img_dir, exist_ok=True)
    filename = f"fig_{idx:02d}.{ext}"
    filepath = os.path.join(img_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            if f.read() == img_data:
                return filename
    with open(filepath, "wb") as f:
        f.write(img_data)
    return filename


def _local_ref(filename: str, wrapped: bool) -> str:
    if wrapped:
        return f'<img src="img/{filename}" />'
    return f"![](img/{filename})"


def extract_images(md_path: str) -> None:
    md_dir = os.path.dirname(os.path.abspath(md_path))
    img_dir = os.path.join(md_dir, "img")

    with open(md_path, "r") as f:
        content = f.read()

    # Single combined sweep so figures are numbered in document order regardless of source type (base64 markdown vs. remote <img>).
    pattern = re.compile(
        r"!\[(?P<alt>[^\]]*)\]"
        r"\(data:image(?:/(?P<sub>\w+))?;base64,(?P<b64>[A-Za-z0-9+/=\s]+)\)"
        r"|"
        r'<img\s[^>]*?src="(?P<url>https?://[^"]+)"[^>]*?/?>',
        re.IGNORECASE,
    )

    out_parts: list[str] = []
    pos = 0
    counter = 0
    n_b64 = 0
    n_url = 0

    for match in pattern.finditer(content):
        out_parts.append(content[pos : match.start()])
        pos = match.end()

        if match.group("b64") is not None:
            raw = base64.b64decode(match.group("b64").strip())
            ext = _normalize_ext(match.group("sub") or "")
            if ext not in _VALID_EXTS:
                ext = _ext_from_bytes(raw)
            counter += 1
            filename = _save(raw, img_dir, counter, ext)
            n_b64 += 1
            print(f"  Extracted img/{filename} ({len(raw)} bytes)")
        else:
            url = match.group("url")
            ext = _ext_from_url(url)
            # Tentative filename; only commit counter on success.
            tentative = counter + 1
            tentative_path = os.path.join(img_dir, f"fig_{tentative:02d}.{ext}")
            os.makedirs(img_dir, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, tentative_path)
            except Exception as e:
                print(f"  WARNING: could not download {url}: {e}", file=sys.stderr)
                out_parts.append(match.group(0))
                continue
            counter = tentative
            filename = os.path.basename(tentative_path)
            n_url += 1
            print(f"  Downloaded img/{filename} ({os.path.getsize(tentative_path)} bytes)")

        wrapped = _wrapped_by_html(content, match.start(), match.end())
        out_parts.append(_local_ref(filename, wrapped))

    out_parts.append(content[pos:])

    if counter == 0:
        print("No embedded or remote images found.")
        return

    print(f"Extracted {counter} image(s): {n_b64} base64, {n_url} remote")

    with open(md_path, "w") as f:
        f.write("".join(out_parts))

    print("Done! Updated markdown with local file references.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_file", help="Path to the markdown file")
    args = parser.parse_args()

    if not os.path.isfile(args.markdown_file):
        print(f"Error: {args.markdown_file} not found", file=sys.stderr)
        sys.exit(1)

    extract_images(args.markdown_file)

#!/usr/bin/env python3
"""Fail if any tracked text file has broken text encoding.

Guards against a specific, easy-to-miss corruption: reading a UTF-8 file with
a tool that assumes the legacy Windows ANSI codepage, then writing it back as
UTF-8. That double-encodes every non-ASCII character -- "…" becomes "â€¦" and
"→" becomes "â†’" -- and because the result is still *valid* UTF-8, nothing
downstream complains. It compiles, it passes tests, and it only shows up as
garbage in the rendered UI.

It also rejects UTF-8 BOMs, which some Windows editors and PowerShell's
Set-Content add silently.

Run from the repository root:  python scripts/check_encoding.py
"""

from __future__ import annotations

import subprocess
import sys
import unicodedata

TEXT_SUFFIXES = (
    ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json", ".yml", ".yaml",
    ".css", ".html", ".txt", ".example", ".toml", ".cfg", ".sh",
)

# Each of these is what a common UTF-8 character looks like after being
# misread as cp1252. None of them is plausible in hand-written source.
MOJIBAKE = ("â€", "â†", "â€™", "â€œ", "Ã¢", "Ã©", "Ã¨", "Ã¯", "Â·", "Â¶", "Â°")

REPLACEMENT = "�"  # U+FFFD, left behind when a decode gives up
BOM = b"\xef\xbb\xbf"


def tracked_text_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.endswith(TEXT_SUFFIXES)]


def main() -> int:
    problems: list[str] = []
    scanned = 0

    for rel in tracked_text_files():
        try:
            raw = open(rel, "rb").read()
        except FileNotFoundError:
            continue  # tracked but deleted in the working tree
        scanned += 1

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{rel}: not valid UTF-8 ({exc})")
            continue

        if raw.startswith(BOM):
            problems.append(f"{rel}: starts with a UTF-8 BOM")

        for marker in MOJIBAKE:
            if marker in text:
                line = next(
                    (i for i, l in enumerate(text.splitlines(), 1) if marker in l),
                    0,
                )
                problems.append(
                    f"{rel}:{line}: mojibake {marker!r} "
                    f"(UTF-8 read as cp1252, then re-encoded)"
                )
                break

        if REPLACEMENT in text:
            problems.append(f"{rel}: contains U+FFFD, so characters were lost")

    print(f"checked {scanned} tracked text files")

    if problems:
        print("\nEncoding problems found:\n")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nRe-read the affected file as UTF-8 and rewrite it as UTF-8. On "
            "Windows PowerShell 5.1 that means passing -Encoding utf8 to "
            "Get-Content as well as Set-Content -- the default for reading is "
            "the ANSI codepage, which is what causes this."
        )
        return 1

    print("no encoding problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())

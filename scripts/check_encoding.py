#!/usr/bin/env python3
"""Fail if any tracked text file has broken text encoding.

Guards against a specific, easy-to-miss corruption: reading a UTF-8 file with
a tool that assumes the legacy Windows ANSI codepage, then writing it back as
UTF-8. That double-encodes every non-ASCII character -- an ellipsis becomes a
three-character sequence starting with a-circumflex -- and because the result
is still *valid* UTF-8, nothing downstream complains. It compiles, it passes
tests, and it only shows up as garbage in the rendered UI.

It also rejects UTF-8 BOMs, which some Windows editors and PowerShell's
Set-Content add silently.

This file is deliberately pure ASCII: the characters it searches for are built
from numeric code points rather than typed out. Spelled literally they would
appear in this file's own source and the checker would flag itself -- which is
exactly what happened the first time it ran in CI. Staying ASCII means it can
scan itself like any other file instead of needing an exemption, so real
corruption here is still caught. Keep it that way.

Run from the repository root:  python scripts/check_encoding.py
"""

from __future__ import annotations

import subprocess
import sys

TEXT_SUFFIXES = (
    ".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json", ".yml", ".yaml",
    ".css", ".html", ".txt", ".example", ".toml", ".cfg", ".sh",
)

# What common UTF-8 lead bytes look like once misread as cp1252. A UTF-8
# sequence beginning C2/C3 (Latin-1 supplement) or E2 (punctuation, arrows)
# decodes under cp1252 to one of these pairs. None is plausible in
# hand-written source text.
_MOJIBAKE_CODEPOINTS = (
    (0x00E2, 0x20AC),  # E2 80 xx: ellipsis, em dash, curly quotes
    (0x00E2, 0x2020),  # E2 86 xx: arrows
    (0x00C3, 0x00A2),  # C3 A2: a-circumflex
    (0x00C3, 0x00A9),  # C3 A9: e-acute
    (0x00C3, 0x00A8),  # C3 A8: e-grave
    (0x00C3, 0x00AF),  # C3 AF: i-diaeresis
    (0x00C2, 0x00B7),  # C2 B7: middle dot
    (0x00C2, 0x00B6),  # C2 B6: pilcrow
    (0x00C2, 0x00B0),  # C2 B0: degree sign
    (0x00C2, 0x00A0),  # C2 A0: non-breaking space
)

MOJIBAKE = tuple("".join(chr(point) for point in pair)
                 for pair in _MOJIBAKE_CODEPOINTS)

REPLACEMENT = chr(0xFFFD)  # left behind when a decode gives up
BOM = b"\xef\xbb\xbf"


def tracked_text_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.endswith(TEXT_SUFFIXES)]


def first_line_containing(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    return 0


def describe(text: str) -> str:
    """Render a needle as escapes, so the report stays ASCII and unambiguous."""
    return "".join(f"\\u{ord(char):04x}" for char in text)


def main() -> int:
    problems: list[str] = []
    scanned = 0

    for rel in tracked_text_files():
        try:
            with open(rel, "rb") as handle:
                raw = handle.read()
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
                problems.append(
                    f"{rel}:{first_line_containing(text, marker)}: "
                    f"mojibake {describe(marker)} "
                    f"(UTF-8 read as cp1252, then re-encoded)"
                )
                break

        if REPLACEMENT in text:
            problems.append(
                f"{rel}:{first_line_containing(text, REPLACEMENT)}: "
                f"contains U+FFFD, so characters were lost"
            )

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

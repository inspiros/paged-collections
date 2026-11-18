"""Run documentation examples and verify the output captured for mdBook."""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs/site"
EXAMPLES = SITE / "examples"
RUST_EXAMPLES = ROOT / "crates/paged-collections/examples"


def warn_missing_includes() -> None:
    for page in sorted((SITE / "src").rglob("*.md")):
        for target in re.findall(r"\{\{#include\s+([^\s}]+)\s*\}\}", page.read_text(encoding="utf-8")):
            # mdBook suffixes select line numbers or named anchors.
            path = page.parent / target.split(":", 1)[0]
            if not path.is_file():
                print(f"Warning: {page.relative_to(ROOT)} includes missing {target}", file=sys.stderr)


def check(command: list[str], capture: Path, update: bool) -> bool:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    except OSError as error:
        print(f"Could not run {' '.join(command)}: {error}", file=sys.stderr)
        return False
    if result.returncode:
        print(f"Failed ({result.returncode}): {' '.join(command)}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return False
    if update:
        capture.parent.mkdir(parents=True, exist_ok=True)
        capture.write_text(result.stdout, encoding="utf-8")
        print(f"Updated {capture.relative_to(ROOT)}")
        return True
    if not capture.is_file():
        print(f"Missing capture: {capture.relative_to(ROOT)}", file=sys.stderr)
        expected = ""
    else:
        expected = capture.read_text(encoding="utf-8")
    if expected != result.stdout or not capture.is_file():
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True), result.stdout.splitlines(keepends=True),
            fromfile=str(capture.relative_to(ROOT)), tofile="actual stdout",
        )
        sys.stderr.writelines(diff)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite captured output")
    parser.add_argument("--only", choices=("python", "rust"), help="verify one language")
    args = parser.parse_args()
    warn_missing_includes()
    ok = True
    if args.only in (None, "python"):
        sources = sorted(EXAMPLES.glob("*.py"))
        for source in sources:
            # Isolate imports so examples/numpy.py cannot shadow installed NumPy.
            ok = check([sys.executable, "-I", str(source)], source.with_suffix(".python.txt"), args.update) and ok
        print(f"Python examples: {len(sources)} checked")
    if args.only in (None, "rust"):
        sources = sorted(RUST_EXAMPLES.glob("doc_*.rs"))
        for source in sources:
            name = source.stem.removeprefix("doc_")
            command = ["cargo", "run", "-q", "-p", "paged-collections", "--example", source.stem]
            if name == "concurrent":
                command.extend(["--features", "concurrent"])
            ok = check(command, EXAMPLES / f"{name}.rust.txt", args.update) and ok
        print(f"Rust examples: {len(sources)} checked")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Copy a pinned, clean PlaCo source checkout without bulky upstream test meshes.

Usage: python tools/vendor_placo.py /path/to/placo --revision <40-char-commit>
The destination must not already exist. No downloads or builds are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ("python/tests/", ".github/")
EXCLUDED_FILES = {"python/run_tests.sh"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = ROOT / "third_party/placo"

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments])

    revision = git("rev-parse", "HEAD").decode().strip()
    if revision != args.revision or git("status", "--porcelain"):
        raise ValueError("Source must be clean and checked out at the explicitly requested revision")
    remote = git("config", "--get", "remote.origin.url").decode().strip()
    if remote.removesuffix(".git") != "https://github.com/Rhoban/placo":
        raise ValueError("Expected the official Rhoban/placo checkout")
    if destination.exists():
        raise FileExistsError("Vendor destination exists; review and move the old snapshot before updating")
    entries = git("ls-files", "-z").decode().split("\0")
    selected = [relative for relative in sorted(entries) if relative
                and not relative.startswith(EXCLUDED_PREFIXES) and relative not in EXCLUDED_FILES]
    for relative in selected:
        original = source / relative
        if original.is_symlink() and not original.resolve().is_relative_to(source):
            raise ValueError(f"Upstream symlink leaves the source tree: {relative}")
    manifest, symlinks = {}, {}
    for relative in selected:
        original, copy = source / relative, destination / relative
        if original.is_symlink():
            copy.parent.mkdir(parents=True, exist_ok=True)
            link_target = os.readlink(original)
            copy.symlink_to(link_target)
            symlinks[relative] = link_target
            continue
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, copy)
        manifest[relative] = hashlib.sha256(copy.read_bytes()).hexdigest()
    version = re.search(r'^version\s*=\s*"([^"]+)"', (source / "pyproject.toml").read_text(), re.MULTILINE)
    if version is None:
        raise ValueError("Upstream project version was not found")
    metadata = {
        "repository": "https://github.com/Rhoban/placo", "revision": revision,
        "version": version.group(1), "license": "MIT", "bundled_qpmad_license": "Apache-2.0",
        "excluded_prefixes": list(EXCLUDED_PREFIXES), "excluded_files": sorted(EXCLUDED_FILES),
        "files_sha256": manifest, "symlinks": symlinks,
    }
    (destination / "UPSTREAM.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(manifest)} unchanged upstream files at {revision}")


if __name__ == "__main__":
    main()

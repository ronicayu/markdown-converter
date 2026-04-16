#!/usr/bin/env python3
"""Decrypt password-protected Office files (.doc/.docx) in-place.

Uses msoffcrypto-tool. Tries passwords in order: empty string, then any
supplied via --password. Decrypted files replace the originals.

Usage:
    # Decrypt all encrypted files under DDS/ and DRS/
    .venv/bin/python3 convert_scripts/decrypt_office.py

    # Decrypt a specific directory
    .venv/bin/python3 convert_scripts/decrypt_office.py DRS/

    # Decrypt with a known password
    .venv/bin/python3 convert_scripts/decrypt_office.py --password secret123

    # Dry run — show which files are encrypted
    .venv/bin/python3 convert_scripts/decrypt_office.py --dry-run
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import msoffcrypto


def find_office_files(root: Path) -> list[Path]:
    """Find all .doc and .docx files under root."""
    files = []
    for ext in ("*.doc", "*.docx"):
        files.extend(root.rglob(ext))
    return sorted(files)


def is_encrypted(path: Path) -> bool:
    """Check if an Office file is encrypted."""
    try:
        with open(path, "rb") as f:
            return msoffcrypto.OfficeFile(f).is_encrypted()
    except Exception:
        return False


def decrypt_file(path: Path, passwords: list[str]) -> str | None:
    """Try to decrypt a file in-place. Returns None on success, error string on failure."""
    for pw in passwords:
        try:
            with open(path, "rb") as f:
                office_file = msoffcrypto.OfficeFile(f)
                office_file.load_key(password=pw)
                with tempfile.NamedTemporaryFile(delete=False, suffix=path.suffix) as tmp:
                    office_file.decrypt(tmp)
                    tmp_path = Path(tmp.name)
            shutil.move(tmp_path, path)
            return None
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

    tried = ", ".join(repr(p) for p in passwords)
    return f"decryption failed with passwords: {tried}"


def main():
    parser = argparse.ArgumentParser(description="Decrypt password-protected Office files in-place")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["DRS", "Error Codes", "Message Specs V2.1", "MHX-MHub", "DDS", "UserGuide"],
        help="Files or directories to scan (default: DDS/ DRS/ Error Codes/ Message Specs V2.1/ MHX-MHub/ UserGuide/)",
    )
    parser.add_argument("--password", action="append", metavar="PW", help="Password(s) to try (empty string always tried first)")
    parser.add_argument("--dry-run", action="store_true", help="Only list encrypted files, don't decrypt")
    args = parser.parse_args()

    passwords = [""]
    if args.password:
        passwords.extend(p for p in args.password if p != "")

    # Collect files
    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(find_office_files(path))
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)

    if not files:
        print("No Office files found.")
        return

    # Find encrypted files
    encrypted = [f for f in files if is_encrypted(f)]
    print(f"Found {len(files)} Office files, {len(encrypted)} encrypted")

    if not encrypted:
        return

    if args.dry_run:
        for f in encrypted:
            print(f"  {f}")
        return

    ok = 0
    failed = 0
    errors = []
    for i, f in enumerate(encrypted, 1):
        print(f"[{i}/{len(encrypted)}] {f.name} ... ", end="", flush=True)
        err = decrypt_file(f, passwords)
        if err is None:
            ok += 1
            print("OK")
        else:
            failed += 1
            errors.append((f, err))
            print(f"FAILED: {err}")

    print(f"\nDone: {ok} decrypted, {failed} failed")
    if errors:
        print("\nFailed files:")
        for f, err in errors:
            print(f"  {f}: {err}")


if __name__ == "__main__":
    main()

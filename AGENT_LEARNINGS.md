# Agent learnings

## 2026-10-02: Windows test temporary-directory permissions

- Failure: pytest could not access the existing system temporary test directory or its cache; the first run reported setup errors instead of checking those tests.
- Prevention: run with a fresh, uniquely named `--basetemp` inside the repository and `-p no:cacheprovider` when those paths are inaccessible. Do not reuse an existing directory because pytest clears its base temporary directory.

## 2026-10-02: Empty signatures in scripted Python edits

- Failure: adding a parameter with an unconditional comma generated an invalid empty-argument signature. Parsing caught it before any source file was written.
- Prevention: handle empty argument lists separately and parse the complete rewritten Python source before writing it.

## 2026-10-02: Graft CLI wrapper on PowerShell

- Failure: `graft build` selected `graft.ps1`, which the Windows execution policy blocked.
- Prevention: use the installed `graft.cmd` wrapper for CLI builds; keep PowerShell execution policy unchanged.

## 2026-10-02: Test paths and Windows source encoding

- Failure: a guessed test filename stopped a focused pytest run before any tests ran; printing a source file's UTF-8 BOM also failed under the shell's default encoding.
- Prevention: choose test paths from the repository inventory. For Python source inspection, read with `utf-8-sig` and set stdout to UTF-8; preserve the file's original BOM and line endings when editing.

## 2026-10-20: `sed -i` on CRLF files in Git Bash

- Failure: a `sed -i` substitution on `scripts/generate_manuscript_figures.py` rewrote the whole file with LF endings; a pattern anchored on `\r$` also failed to match.
- Prevention: edit CRLF files with the Edit tool or a Python byte-level rewrite that restores `\r\n`; check `file <path>` after any shell edit.

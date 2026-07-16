#!/usr/bin/env python3
"""Export an allowlisted aggregate-only public bundle from a private run."""

from __future__ import annotations

import argparse

from src.reporting.publication_bundle import export_publication_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an aggregate-only publication bundle")
    parser.add_argument("--source", required=True, help="Private Results directory")
    parser.add_argument("--output", required=True, help="New destination directory for the public bundle")
    args = parser.parse_args()
    exported = export_publication_bundle(args.source, args.output)
    print(f"Exported {len(exported)} aggregate publication artifacts to {args.output}")


if __name__ == "__main__":
    main()

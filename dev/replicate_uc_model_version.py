"""Replicate a single Unity Catalog model version across metastores/regions.

The destination version number is reassigned by the destination metastore; aliases are
remapped to it. run_id, lineage, timestamps, user_id, and audit history are metastore-local
and are intentionally not preserved.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate one UC model version to a registered model in another metastore."
    )
    parser.add_argument("--src-registry-uri", required=True, help="Source registry URI.")
    parser.add_argument("--dst-registry-uri", required=True, help="Destination registry URI.")
    parser.add_argument("--src-model", required=True, help="Source registered model name.")
    parser.add_argument("--src-version", required=True, help="Source model version number.")
    parser.add_argument("--dst-model", required=True, help="Destination registered model name.")
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Alias to remap onto the new version. Repeatable. "
        "Defaults to all aliases on the source version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned destination API sequence without making any writes.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)

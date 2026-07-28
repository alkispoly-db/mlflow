"""Replicate a single Unity Catalog model version across metastores/regions.

The destination version number is reassigned by the destination metastore; aliases are
remapped to it. run_id, lineage, timestamps, user_id, and audit history are metastore-local
and are intentionally not preserved.
"""

import argparse
import logging

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

_logger = logging.getLogger(__name__)


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


def ensure_registered_model(client: MlflowClient, name: str, dry_run: bool) -> bool:
    try:
        client.get_registered_model(name)
        return True
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
    if dry_run:
        _logger.info("[dry-run] would create registered model %s", name)
    else:
        client.create_registered_model(name)
    return False


def resolve_aliases(src_mv: ModelVersion, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return list(src_mv.aliases)

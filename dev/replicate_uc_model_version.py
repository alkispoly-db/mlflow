"""Replicate a single Unity Catalog model version across metastores/regions.

The destination version number is reassigned by the destination metastore; aliases are
remapped to it. run_id, lineage, timestamps, user_id, and audit history are metastore-local
and are intentionally not preserved.
"""

import argparse
import logging
import tempfile

import mlflow.artifacts
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

_logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate one UC model version to a registered model in another metastore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  python dev/replicate_uc_model_version.py \\\n"
            "    --src-registry-uri databricks-uc://src-profile \\\n"
            "    --dst-registry-uri databricks-uc://dst-profile \\\n"
            "    --src-model main.my_schema.my_model --src-version 3 \\\n"
            "    --dst-model main.other_schema.my_model \\\n"
            "    --alias champion --dry-run"
        ),
    )
    parser.add_argument(
        "--src-registry-uri",
        required=True,
        help="Source registry URI, e.g. 'databricks-uc://src-profile' (a profile in "
        "~/.databrickscfg) or 'databricks-uc'.",
    )
    parser.add_argument(
        "--dst-registry-uri",
        required=True,
        help="Destination registry URI, e.g. 'databricks-uc://dst-profile' (a profile in "
        "~/.databrickscfg) or 'databricks-uc'.",
    )
    parser.add_argument(
        "--src-model",
        required=True,
        help="Source registered model name, e.g. 'main.my_schema.my_model'.",
    )
    parser.add_argument("--src-version", required=True, help="Source model version number, e.g. 3.")
    parser.add_argument(
        "--dst-model",
        required=True,
        help="Destination registered model name, e.g. 'main.other_schema.my_model'.",
    )
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


def replicate(args: argparse.Namespace) -> str | None:
    src_client = MlflowClient(registry_uri=args.src_registry_uri)
    dst_client = MlflowClient(registry_uri=args.dst_registry_uri)

    src_mv = src_client.get_model_version(args.src_model, args.src_version)
    aliases = resolve_aliases(src_mv, args.alias)

    if args.dry_run:
        _logger.info(
            "[dry-run] would replicate %s/%s -> %s (tags=%s, description=%r, aliases=%s)",
            args.src_model,
            args.src_version,
            args.dst_model,
            dict(src_mv.tags),
            src_mv.description,
            aliases,
        )
        ensure_registered_model(dst_client, args.dst_model, dry_run=True)
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{args.src_model}/{args.src_version}",
            dst_path=tmp_dir,
            registry_uri=args.src_registry_uri,
        )
        ensure_registered_model(dst_client, args.dst_model, dry_run=False)
        new_mv = dst_client.create_model_version(
            name=args.dst_model,
            source=local_path,
            tags=dict(src_mv.tags),
            description=src_mv.description,
            run_id=None,
        )

    for alias in aliases:
        dst_client.set_registered_model_alias(args.dst_model, alias, new_mv.version)

    _logger.info(
        "Replicated %s/%s -> %s/%s",
        args.src_model,
        args.src_version,
        args.dst_model,
        new_mv.version,
    )
    return new_mv.version


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    replicate(args)


if __name__ == "__main__":
    main()

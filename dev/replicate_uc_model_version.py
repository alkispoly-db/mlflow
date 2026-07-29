"""Replicate a single Unity Catalog model version across metastores/regions.

The destination version number is reassigned by the destination metastore; aliases are
remapped to it. run_id, lineage, timestamps, user_id, and audit history are metastore-local
and are intentionally not preserved.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import mlflow.artifacts
import yaml
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


def decouple_logged_model(local_path: str) -> None:
    """Strip source-metastore logged-model identity from MLmodel files.

    Newer MLflow create_model_version resolves the embedded logged model in the
    TARGET metastore. The source logged model does not exist there, causing
    NOT_FOUND errors. This decouples the artifacts into a standalone model by
    removing model_id, model_uuid, run_id, and artifact_path from all MLmodel
    files in the downloaded directory (recursively). These keys embed source
    lineage and are meaningless in the destination metastore.

    This is version-dependent (MLflow 3.13+) so must be preserved across
    future refactors.
    """
    # Find all MLmodel files (may be multiple for nested models).
    local_path_obj = Path(local_path)
    mlmodel_files = list(local_path_obj.glob("**/MLmodel"))

    if not mlmodel_files:
        _logger.warning("No MLmodel files found in %s", local_path)
        return

    for mlmodel_file in mlmodel_files:
        _logger.debug("Processing MLmodel at %s", mlmodel_file)

        with open(mlmodel_file, "r") as f:
            model_config = yaml.safe_load(f)

        if not model_config:
            _logger.warning("MLmodel at %s is empty or invalid YAML", mlmodel_file)
            continue

        # Strip identity/lineage keys that reference the source metastore.
        keys_to_remove = ["model_id", "model_uuid", "run_id", "artifact_path"]
        removed_keys = [k for k in keys_to_remove if k in model_config]

        for key in removed_keys:
            del model_config[key]

        if removed_keys:
            _logger.info(
                "Decoupled MLmodel at %s: removed keys %s",
                mlmodel_file,
                removed_keys,
            )

        # Write back preserving key order and YAML structure.
        with open(mlmodel_file, "w") as f:
            yaml.safe_dump(model_config, f, sort_keys=False, allow_unicode=True)


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
        _logger.info("[dry-run] would decouple logged-model identity from MLmodel files")
        ensure_registered_model(dst_client, args.dst_model, dry_run=True)
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{args.src_model}/{args.src_version}",
            dst_path=tmp_dir,
            registry_uri=args.src_registry_uri,
        )

        # Decouple the source logged-model identity before registering in destination.
        decouple_logged_model(local_path)

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

from unittest import mock

from dev.replicate_uc_model_version import (
    ensure_registered_model,
    parse_args,
    resolve_aliases,
)
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


def test_parse_args_minimal():
    args = parse_args([
        "--src-registry-uri",
        "databricks-uc://regionA",
        "--dst-registry-uri",
        "databricks-uc://regionB",
        "--src-model",
        "cat_a.sch.model",
        "--src-version",
        "3",
        "--dst-model",
        "cat_b.sch.model",
    ])
    assert args.src_registry_uri == "databricks-uc://regionA"
    assert args.dst_registry_uri == "databricks-uc://regionB"
    assert args.src_model == "cat_a.sch.model"
    assert args.src_version == "3"
    assert args.dst_model == "cat_b.sch.model"
    assert args.alias == []
    assert args.dry_run is False


def test_parse_args_with_aliases_and_dry_run():
    args = parse_args([
        "--src-registry-uri",
        "databricks-uc://regionA",
        "--dst-registry-uri",
        "databricks-uc://regionB",
        "--src-model",
        "cat_a.sch.model",
        "--src-version",
        "3",
        "--dst-model",
        "cat_b.sch.model",
        "--alias",
        "prod",
        "--alias",
        "champion",
        "--dry-run",
    ])
    assert args.alias == ["prod", "champion"]
    assert args.dry_run is True


def test_ensure_registered_model_already_exists():
    client = mock.Mock()
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=False)
    assert existed is True
    client.get_registered_model.assert_called_once_with("cat.sch.model")
    client.create_registered_model.assert_not_called()


def test_ensure_registered_model_creates_when_missing():
    client = mock.Mock()
    client.get_registered_model.side_effect = MlflowException(
        "nope", error_code=RESOURCE_DOES_NOT_EXIST
    )
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=False)
    assert existed is False
    client.create_registered_model.assert_called_once_with("cat.sch.model")


def test_ensure_registered_model_dry_run_skips_create():
    client = mock.Mock()
    client.get_registered_model.side_effect = MlflowException(
        "nope", error_code=RESOURCE_DOES_NOT_EXIST
    )
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=True)
    assert existed is False
    client.create_registered_model.assert_not_called()


def _mv_with_aliases(aliases):
    return ModelVersion(name="cat.sch.model", version="3", creation_timestamp=0, aliases=aliases)


def test_resolve_aliases_uses_requested_when_given():
    src_mv = _mv_with_aliases(["prod", "old"])
    assert resolve_aliases(src_mv, ["champion"]) == ["champion"]


def test_resolve_aliases_defaults_to_source_aliases():
    src_mv = _mv_with_aliases(["prod", "champion"])
    assert resolve_aliases(src_mv, []) == ["prod", "champion"]


def test_resolve_aliases_empty_when_none():
    src_mv = _mv_with_aliases([])
    assert resolve_aliases(src_mv, []) == []

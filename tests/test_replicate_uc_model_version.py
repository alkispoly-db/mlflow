import argparse
from unittest import mock

from dev import replicate_uc_model_version as R
from dev.replicate_uc_model_version import (
    ensure_registered_model,
    parse_args,
    resolve_aliases,
)
from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
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


def _replicate_args(dry_run=False, alias=None):
    return argparse.Namespace(
        src_registry_uri="databricks-uc://a",
        dst_registry_uri="databricks-uc://b",
        src_model="cat_a.sch.model",
        src_version="3",
        dst_model="cat_b.sch.model",
        alias=alias or [],
        dry_run=dry_run,
    )


def test_replicate_full_flow():
    src_mv = ModelVersion(
        name="cat_a.sch.model",
        version="3",
        creation_timestamp=0,
        description="my model",
        tags=[ModelVersionTag("stage", "gold")],
        aliases=["prod"],
    )
    new_mv = ModelVersion(name="cat_b.sch.model", version="7", creation_timestamp=0)

    src_client = mock.Mock()
    src_client.get_model_version.return_value = src_mv
    dst_client = mock.Mock()
    dst_client.get_registered_model.return_value = mock.Mock()
    dst_client.create_model_version.return_value = new_mv

    with (
        mock.patch.object(R, "MlflowClient", side_effect=[src_client, dst_client]),
        mock.patch.object(R.mlflow.artifacts, "download_artifacts", return_value="/tmp/model"),
    ):
        version = R.replicate(_replicate_args())

    assert version == "7"
    src_client.get_model_version.assert_called_once_with("cat_a.sch.model", "3")
    dst_client.create_model_version.assert_called_once()
    _, kwargs = dst_client.create_model_version.call_args
    assert kwargs["name"] == "cat_b.sch.model"
    assert kwargs["source"] == "/tmp/model"
    assert kwargs["tags"] == {"stage": "gold"}
    assert kwargs["description"] == "my model"
    assert kwargs["run_id"] is None
    dst_client.set_registered_model_alias.assert_called_once_with("cat_b.sch.model", "prod", "7")


def test_replicate_dry_run_makes_no_writes():
    src_mv = ModelVersion(
        name="cat_a.sch.model", version="3", creation_timestamp=0, aliases=["prod"]
    )
    src_client = mock.Mock()
    src_client.get_model_version.return_value = src_mv
    dst_client = mock.Mock()

    with mock.patch.object(R, "MlflowClient", side_effect=[src_client, dst_client]):
        version = R.replicate(_replicate_args(dry_run=True))

    assert version is None
    dst_client.create_model_version.assert_not_called()
    dst_client.set_registered_model_alias.assert_not_called()

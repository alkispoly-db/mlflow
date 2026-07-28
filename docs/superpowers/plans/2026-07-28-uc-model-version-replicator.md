# Cross-Region UC Model Version Replicator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script that replicates one UC model version from a source metastore to a destination registered model in a different region/metastore.

**Architecture:** Two `MlflowClient` instances (one per registry URI) drive the replication over MLflow's public API. Read side: `get_model_version` + `mlflow.artifacts.download_artifacts` to a local temp dir. Write side: auto-create the destination registered model if absent, then `create_model_version(source=<local dir>)` (which internally runs the CreateModelVersion → temp-creds(READ_WRITE) → upload → FinalizeModelVersion handshake), then remap aliases to the destination's reassigned version number. A `--dry-run` prints the planned write sequence without mutating.

**Tech Stack:** Python 3.10+, `mlflow.tracking.MlflowClient`, `mlflow.artifacts.download_artifacts`, `argparse`, `tempfile`, `pytest`.

## Global Constraints

- Python 3.10+ compatible; use `Optional[T]` / `X | None` consistent with the file's style.
- Top-level imports only (no lazy imports unless necessary) — per repo CLAUDE.md.
- Location: `dev/replicate_uc_model_version.py` (standalone tool, not shipped library code).
- Test file: `tests/test_replicate_uc_model_version.py`.
- One model version per invocation; no bulk/multi-version support.
- Do NOT preserve `run_id`, lineage, timestamps, `user_id`, or audit history — pass `run_id=None`.
- Commit each task with DCO sign-off (`git commit -s`) and `Co-Authored-By: Claude <noreply@anthropic.com>`.
- This workspace's pre-commit hook runs `uv lock`, which rewrites `uv.lock` against the internal proxy. Prefix commits with `UV_DEFAULT_INDEX=https://pypi.org/simple` to keep the lock a no-op.

**Reference signatures (verified in tree, do not re-guess):**
- `MlflowClient(tracking_uri=None, registry_uri=None, workspace_store_uri=None)`
- `MlflowClient.get_model_version(name: str, version: str) -> ModelVersion`
- `MlflowClient.get_registered_model(name: str) -> RegisteredModel`
- `MlflowClient.create_registered_model(name, tags=None, description=None, deployment_job_id=None) -> RegisteredModel`
- `MlflowClient.create_model_version(name, source, run_id=None, tags=None, run_link=None, description=None, await_creation_for=..., model_id=None) -> ModelVersion`
- `MlflowClient.set_registered_model_alias(name: str, alias: str, version: str) -> None`
- `mlflow.artifacts.download_artifacts(artifact_uri=None, run_id=None, artifact_path=None, dst_path=None, tracking_uri=None, registry_uri=None) -> str`
- `ModelVersion` properties: `.name`, `.version`, `.description`, `.source`, `.tags` (`dict[str, str]`), `.aliases` (`list[str]`)
- Not-found detection: `except MlflowException as e: if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST): ...`

---

### Task 1: Argument parsing

**Files:**
- Create: `dev/replicate_uc_model_version.py`
- Test: `tests/test_replicate_uc_model_version.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser` and `parse_args(argv: list[str] | None = None) -> argparse.Namespace`. Namespace attributes: `src_registry_uri`, `dst_registry_uri`, `src_model`, `src_version`, `dst_model`, `alias` (`list[str]`), `dry_run` (`bool`).

- [ ] **Step 1: Write the failing test**

```python
from dev.replicate_uc_model_version import parse_args


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module or `parse_args` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
UV_DEFAULT_INDEX=https://pypi.org/simple git commit -s -m "Add arg parsing for UC model version replicator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Ensure destination registered model exists

**Files:**
- Modify: `dev/replicate_uc_model_version.py`
- Test: `tests/test_replicate_uc_model_version.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `ensure_registered_model(client, name: str, dry_run: bool) -> bool`. Returns `True` if the model already existed, `False` if it was created (or would be created in dry-run). Uses `client.get_registered_model`, catches `MlflowException` with `error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)`, then `client.create_registered_model(name)` unless `dry_run`.

- [ ] **Step 1: Write the failing test**

```python
from unittest import mock

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from dev.replicate_uc_model_version import ensure_registered_model


def test_ensure_registered_model_already_exists():
    client = mock.Mock()
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=False)
    assert existed is True
    client.get_registered_model.assert_called_once_with("cat.sch.model")
    client.create_registered_model.assert_not_called()


def test_ensure_registered_model_creates_when_missing():
    client = mock.Mock()
    client.get_registered_model.side_effect = MlflowException(
        "nope", error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    )
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=False)
    assert existed is False
    client.create_registered_model.assert_called_once_with("cat.sch.model")


def test_ensure_registered_model_dry_run_skips_create():
    client = mock.Mock()
    client.get_registered_model.side_effect = MlflowException(
        "nope", error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    )
    existed = ensure_registered_model(client, "cat.sch.model", dry_run=True)
    assert existed is False
    client.create_registered_model.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k ensure_registered_model -v`
Expected: FAIL with `ImportError` (`ensure_registered_model` not defined).

- [ ] **Step 3: Write minimal implementation**

Add imports at top of `dev/replicate_uc_model_version.py`:

```python
import logging

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

_logger = logging.getLogger(__name__)
```

Add the function:

```python
def ensure_registered_model(client, name: str, dry_run: bool) -> bool:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k ensure_registered_model -v`
Expected: PASS (three tests).

- [ ] **Step 5: Commit**

```bash
git add dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
UV_DEFAULT_INDEX=https://pypi.org/simple git commit -s -m "Add destination registered model auto-create

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Resolve aliases to remap

**Files:**
- Modify: `dev/replicate_uc_model_version.py`
- Test: `tests/test_replicate_uc_model_version.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `resolve_aliases(src_mv, requested: list[str]) -> list[str]`. If `requested` is non-empty, returns it unchanged; otherwise returns `list(src_mv.aliases)`.

- [ ] **Step 1: Write the failing test**

```python
from mlflow.entities.model_registry import ModelVersion

from dev.replicate_uc_model_version import resolve_aliases


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k resolve_aliases -v`
Expected: FAIL with `ImportError` (`resolve_aliases` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_aliases(src_mv, requested: list[str]) -> list[str]:
    if requested:
        return requested
    return list(src_mv.aliases)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k resolve_aliases -v`
Expected: PASS (three tests).

- [ ] **Step 5: Commit**

```bash
git add dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
UV_DEFAULT_INDEX=https://pypi.org/simple git commit -s -m "Add alias resolution for UC replicator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Replication orchestration

**Files:**
- Modify: `dev/replicate_uc_model_version.py`
- Test: `tests/test_replicate_uc_model_version.py`

**Interfaces:**
- Consumes: `ensure_registered_model`, `resolve_aliases` from Tasks 2-3.
- Produces: `replicate(args: argparse.Namespace) -> str | None`. Returns the new destination version number, or `None` in dry-run. Builds two `MlflowClient`s (source with `registry_uri=args.src_registry_uri`, destination with `registry_uri=args.dst_registry_uri`), reads the source version, downloads artifacts to a temp dir via `mlflow.artifacts.download_artifacts(f"models:/{args.src_model}/{args.src_version}", dst_path=<tmp>, registry_uri=args.src_registry_uri)`, ensures the destination model, then `create_model_version(name=args.dst_model, source=<tmp dir>, tags=<src tags>, description=<src desc>, run_id=None)`, then remaps aliases.

- [ ] **Step 1: Write the failing test**

```python
import argparse
from unittest import mock

from mlflow.entities.model_registry import ModelVersion

from dev import replicate_uc_model_version as R


def _args(dry_run=False, alias=None):
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
        tags={"stage": "gold"},
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
        version = R.replicate(_args())

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
        version = R.replicate(_args(dry_run=True))

    assert version is None
    dst_client.create_model_version.assert_not_called()
    dst_client.set_registered_model_alias.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k replicate -v`
Expected: FAIL with `ImportError` / `AttributeError` (`replicate` not defined).

- [ ] **Step 3: Write minimal implementation**

Add imports at top:

```python
import tempfile

import mlflow.artifacts
from mlflow.tracking import MlflowClient
```

Add the function:

```python
def replicate(args) -> str | None:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k replicate -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
UV_DEFAULT_INDEX=https://pypi.org/simple git commit -s -m "Add replication orchestration for UC replicator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: CLI entry point

**Files:**
- Modify: `dev/replicate_uc_model_version.py`
- Test: `tests/test_replicate_uc_model_version.py`

**Interfaces:**
- Consumes: `parse_args`, `replicate`.
- Produces: `main(argv: list[str] | None = None) -> None` and a `if __name__ == "__main__": main()` guard. `main` configures logging, parses args, and calls `replicate`.

- [ ] **Step 1: Write the failing test**

```python
from unittest import mock

from dev import replicate_uc_model_version as R


def test_main_invokes_replicate():
    argv = [
        "--src-registry-uri",
        "databricks-uc://a",
        "--dst-registry-uri",
        "databricks-uc://b",
        "--src-model",
        "cat_a.sch.model",
        "--src-version",
        "3",
        "--dst-model",
        "cat_b.sch.model",
    ]
    with mock.patch.object(R, "replicate", return_value="7") as mock_replicate:
        R.main(argv)
    mock_replicate.assert_called_once()
    called_args = mock_replicate.call_args[0][0]
    assert called_args.src_model == "cat_a.sch.model"
    assert called_args.dst_model == "cat_b.sch.model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k main -v`
Expected: FAIL with `AttributeError` (`main` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    replicate(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_replicate_uc_model_version.py -k main -v`
Expected: PASS.

- [ ] **Step 5: Run the full test file and lint**

Run:
```bash
uv run --frozen pytest tests/test_replicate_uc_model_version.py -v
uv run ruff check dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py --fix
uv run ruff format dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
```
Expected: all tests PASS; lint clean.

- [ ] **Step 6: Commit**

```bash
git add dev/replicate_uc_model_version.py tests/test_replicate_uc_model_version.py
UV_DEFAULT_INDEX=https://pypi.org/simple git commit -s -m "Add CLI entry point for UC replicator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Two `MlflowClient`s, one per registry URI → Task 4 ✓
- Read: `get_model_version` + `download_artifacts` to temp → Task 4 ✓
- Auto-create destination registered model → Task 2 ✓
- `create_model_version(source=<local dir>)` handshake with copied tags/description, `run_id=None` → Task 4 ✓
- Alias remap to reassigned version number → Tasks 3 + 4 ✓
- `--dry-run` prints planned sequence, zero writes → Tasks 1, 2, 4 ✓
- Location `dev/replicate_uc_model_version.py`; test file present → all tasks ✓
- Single version per invocation → CLI takes one `--src-version` ✓
- No run_id/lineage preservation → `run_id=None` explicit ✓

**Placeholder scan:** none — all steps contain concrete code.

**Type consistency:** `ensure_registered_model(client, name, dry_run) -> bool`, `resolve_aliases(src_mv, requested) -> list[str]`, `replicate(args) -> str | None`, `main(argv) -> None` — consistent across definitions and call sites. `src_mv.tags` (`dict[str, str]`) passed to `create_model_version(tags=...)` (`dict[str, Any]`) — compatible.

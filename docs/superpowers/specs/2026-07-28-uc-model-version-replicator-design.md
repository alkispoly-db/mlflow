# Cross-Region UC Model Version Replicator — Design

**Date:** 2026-07-28
**Status:** Approved (pending spec review)

## Problem

MLflow ships `MlflowClient.copy_model_version(src_model_uri, dst_name)`
(`mlflow/tracking/client.py:4697`), but it runs as an async job **within a single
Unity Catalog metastore** and takes no destination region/metastore/host. There is no
first-class tool to replicate a UC model version across regions or metastores.

This tool provides a standalone script that replicates **one** UC model version from a
source metastore to a destination registered model in a *different* region/metastore,
following the create → upload → finalize lifecycle described in "Option A" (the direct
UC public-REST replication path).

## Goals

- Replicate a single UC model version cross-region/cross-metastore.
- Carry over: model artifacts, description, model-version tags, and aliases (remapped
  to the destination's reassigned version number).
- Auto-create the destination registered model if it does not exist.
- Provide a `--dry-run` that prints the planned destination API sequence with zero writes.

## Non-Goals

- Preserving `run_id` / source-run linkage, lineage, timestamps, `user_id`, or audit
  history — these are metastore-local and do not survive the cross-region hop.
- Streaming directly between vended cloud locations (no full local copy). We stage
  through a local temp dir instead; simpler and region/cloud-agnostic.
- Bulk / multi-version replication. One version per invocation; callers loop externally.
- Idempotency / dedupe. UC versions are write-once and always append, so re-running
  creates a new destination version.

## Approach

Use **two `MlflowClient` instances**, one per registry URI, over MLflow's public API —
no raw REST. This rides on the UC store's own `create_model_version` handshake
(`CreateModelVersion` → `GenerateTemporaryModelVersionCredentials(READ_WRITE)` → upload
→ `FinalizeModelVersion`) rather than reimplementing those RPCs. Artifacts move via
local temp staging: download from source, then re-upload during destination
`create_model_version`.

Deliberately *not* using `copy_model_version`: single-metastore/async-job only, no
destination host — exactly the gap this tool fills.

## Location

`dev/replicate_uc_model_version.py` — a standalone tool, not shipped library code.

## CLI

```
python dev/replicate_uc_model_version.py \
  --src-registry-uri databricks-uc://regionA \
  --dst-registry-uri databricks-uc://regionB \
  --src-model  catalog_a.schema.model  --src-version 3 \
  --dst-model  catalog_b.schema.model \
  [--alias prod --alias champion]   # aliases to remap; default: all aliases on src version \
  [--dry-run]
```

- Each registry URI resolves its own Databricks credentials via MLflow's normal
  `databricks-uc://profile` mechanism. No credentials in the script.
- The tracking URI is set per-client to match its registry so metadata reads and
  `get_full_name_from_sc` name resolution work correctly.

## Flow (mapped to underlying UC API calls)

```
READ (source client)
  1. src.get_model_version(src_model, src_version)
        → GetModelVersion  (description, tags, aliases, source, status)
  2. mlflow.artifacts.download_artifacts(
        f"models:/{src_model}/{src_version}", dst_path=<tmp>)
        → GenerateTemporaryModelVersionCredentials(READ) / GetModelVersionDownloadUri
          + direct cloud download

WRITE (destination client)
  3. ensure destination registered model exists:
        try:    dst.get_registered_model(dst_model)
        except RESOURCE_DOES_NOT_EXIST:
                dst.create_registered_model(dst_model)      → CreateRegisteredModel
  4. new_mv = dst.create_model_version(
        name=dst_model, source=<tmp local dir>,
        tags=<copied from src>, description=<copied from src>,
        run_id=None)                                        → CreateModelVersion (PENDING_REGISTRATION)
                                                            → GenerateTemporaryModelVersionCredentials(READ_WRITE) + upload
                                                            → FinalizeModelVersion (READY)
  5. for alias in aliases_to_remap:
        dst.set_registered_model_alias(dst_model, alias, new_mv.version)
                                                            → SetRegisteredModelAlias
```

Step 4 is a single `create_model_version` call; the store internally performs the
create → temp-creds(READ_WRITE) → upload → finalize handshake. Tags and description are
passed inline, so no separate `SetModelVersionTag` calls are needed. `run_id` is
intentionally `None`.

## Error handling

- **`--dry-run`:** performs only step 1 (and optionally 2) and prints the exact planned
  destination call sequence with resolved names/versions. Zero writes.
- **Version remap:** the destination assigns its own version number. Capture
  `new_mv.version` and use it for every alias — never assume source and destination
  numbers match.
- **Partial failure:** a failure between `CreateModelVersion` and `FinalizeModelVersion`
  leaves a `PENDING_REGISTRATION` / `FAILED_REGISTRATION` version on the destination. The
  script surfaces the destination version number for manual inspection/deletion; it does
  not auto-rollback (matching MLflow's own behavior).
- **Aliases:** if `--alias` is omitted, read `aliases` from the source model version and
  remap all of them; explicit `--alias` flags override.

## Testing

- Unit test with two fake/mocked stores verifying the **call sequence and argument
  mapping**: destination receives the copied description and tags, alias remap uses
  `new_mv.version`, `run_id` is `None`, and `--dry-run` performs no writes.
- Prefer real entities (per repo conventions); use in-memory / file-based backends where
  a real UC backend is unavailable, mocking only the network boundary.
- No live cross-region integration test (requires two real metastores) — documented as a
  manual verification step.

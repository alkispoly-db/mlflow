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

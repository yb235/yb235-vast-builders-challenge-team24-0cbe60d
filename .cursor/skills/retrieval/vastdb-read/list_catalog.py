#!/usr/bin/env python3
"""List VastDB databases, schemas, and tables via the vastdb Python SDK only."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def _team_config() -> Path:
    configs = sorted(Path("/config").glob("*.config"))
    if len(configs) != 1:
        print(
            f"Expected exactly one /config/*.config team file; found {len(configs)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return configs[0]


ENV_PATH = Path(os.environ.get("VAST_ENV_FILE", _team_config()))

INTERNAL_TABLES = frozenset({"tabular_schema_table"})


def load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        print(f"Missing team config: {path}", file=sys.stderr)
        sys.exit(1)

    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def normalize_endpoint(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


def resolve_endpoint(env: dict[str, str]) -> str:
    endpoint = (
        os.environ.get("VDB_ENDPOINT")
        or env.get("VDB_ENDPOINT")
        or env.get("S3_ENDPOINT", "")
    )
    if not endpoint:
        print("Set VDB_ENDPOINT or S3_ENDPOINT in /config/<team>.config", file=sys.stderr)
        sys.exit(1)
    return normalize_endpoint(endpoint)


def probe_endpoint(endpoint: str) -> None:
    """Warn if localhost tunnel port is not reachable."""
    if "127.0.0.1" not in endpoint and "localhost" not in endpoint:
        return
    try:
        urllib.request.urlopen(endpoint, timeout=5)
    except urllib.error.HTTPError:
        return  # HTTP 404/200 both mean tunnel is up
    except OSError as exc:
        print(
            f"Cannot reach {endpoint} — start SSH tunnel first:\n"
            f"  ssh -N -f -L 18080:172.27.121.1:80 vastdata@v151lg1",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_path_parts(parent_path: str) -> list[str]:
    return [p for p in parent_path.strip("/").split("/") if p]


def catalog_tree(endpoint: str, access: str, secret: str) -> dict[str, dict[str, list[str]]]:
    """Build database → schema → [tables] from tx.catalog() SCHEMA/TABLE rows."""
    import vastdb

    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    session = vastdb.connect(endpoint=endpoint, access=access, secret=secret, ssl_verify=False)
    with session.transaction() as tx:
        catalog = tx.catalog(fail_if_missing=False)
        if catalog is None:
            return {}

        df = catalog.select(columns=["element_type", "name", "parent_path"]).read_all().to_pandas()

    for _, row in df[df["element_type"] == "SCHEMA"].iterrows():
        parts = parse_path_parts(row["parent_path"])
        if len(parts) != 1:
            continue
        tree[parts[0]][row["name"]]  # ensure schema key exists

    for _, row in df[df["element_type"] == "TABLE"].iterrows():
        if row["name"] in INTERNAL_TABLES:
            continue
        parts = parse_path_parts(row["parent_path"])
        if len(parts) != 2:
            continue
        bucket, schema = parts
        tables = tree[bucket][schema]
        if row["name"] not in tables:
            tables.append(row["name"])

    return {db: {sch: sorted(tables) for sch, tables in sorted(schemas.items())} for db, schemas in sorted(tree.items())}


def live_bucket_tree(
    endpoint: str, access: str, secret: str, bucket: str
) -> dict[str, dict[str, list[str]]]:
    """Optional live drill-down via bucket.schemas() / schema.tables() (SDK only)."""
    import vastdb
    from vastdb.errors import Conflict

    tree: dict[str, dict[str, list[str]]] = {bucket: {}}
    session = vastdb.connect(endpoint=endpoint, access=access, secret=secret, ssl_verify=False)
    with session.transaction() as tx:
        try:
            schemas = list(tx.bucket(bucket).schemas())
        except Conflict:
            return {}

        for schema in schemas:
            tables = []
            for table in schema.tables():
                if table.name not in INTERNAL_TABLES:
                    tables.append(table.name)
            tree[bucket][schema.name] = sorted(tables)
    return tree


def print_tree(tree: dict[str, dict[str, list[str]]]) -> None:
    if not tree:
        print("No VastDB databases found in catalog.")
        return

    print(f"{'DATABASE (bucket)':<40} {'SCHEMA':<30} TABLES")
    print("-" * 100)
    for bucket, schemas in tree.items():
        if not schemas:
            print(f"{bucket:<40} (no schemas)")
            continue
        for i, (schema, tables) in enumerate(sorted(schemas.items())):
            table_str = ", ".join(tables) if tables else "(no tables)"
            if i == 0:
                print(f"{bucket:<40} {schema:<30} {table_str}")
            else:
                print(f"{'':<40} {schema:<30} {table_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List VastDB catalog via vastdb SDK")
    parser.add_argument("--bucket", help="Also verify one bucket via bucket.schemas()")
    parser.add_argument("--live-only", action="store_true", help="Use bucket.schemas() only (requires --bucket)")
    args = parser.parse_args()

    env = load_env(ENV_PATH)
    access = env.get("VAST_ACCESS_KEY") or env.get("ACCESS_KEY", "")
    secret = env.get("VAST_SECRET_KEY") or env.get("SECRET_KEY", "")
    if not access or not secret:
        print(
            "Set ACCESS_KEY/SECRET_KEY in /config/<team>.config",
            file=sys.stderr,
        )
        sys.exit(1)

    endpoint = resolve_endpoint(env)
    probe_endpoint(endpoint)
    print(f"VastDB endpoint: {endpoint}\n")

    if args.live_only:
        if not args.bucket:
            print("--live-only requires --bucket", file=sys.stderr)
            sys.exit(1)
        print_tree(live_bucket_tree(endpoint, access, secret, args.bucket))
        return

    tree = catalog_tree(endpoint, access, secret)
    print_tree(tree)

    if args.bucket:
        print(f"\n--- live bucket.schemas() for {args.bucket} ---")
        print_tree(live_bucket_tree(endpoint, access, secret, args.bucket))


if __name__ == "__main__":
    main()

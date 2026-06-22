"""Run the SQL churn-analysis layer against the raw dataset using DuckDB.

This executes the queries in ``sql/churn_analysis.sql`` over an in-memory
DuckDB view built directly from ``data/raw/telco_customer_churn.csv`` — no
database server required. Each query result is printed and, optionally, written
to ``results/sql/<name>.csv``.

Usage:
    python3 scripts/sql_analysis.py                 # print all query results
    python3 scripts/sql_analysis.py --save          # also export CSVs
    python3 scripts/sql_analysis.py --query churn_by_contract
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = PROJECT_ROOT / "sql" / "churn_analysis.sql"
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sql"

# Matches a "-- name: <identifier>" header that precedes each query.
NAME_HEADER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def build_customers_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create the `customers` view with derived churn_flag / total_charges."""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW customers AS
        SELECT
            *,
            CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END        AS churn_flag,
            TRY_CAST(NULLIF(TRIM(TotalCharges), '') AS DOUBLE) AS total_charges
        FROM read_csv_auto('{DATA_FILE.as_posix()}', header = true);
        """
    )


def parse_named_queries(sql_text: str) -> List[Tuple[str, str]]:
    """Split the SQL file into (name, query) pairs using `-- name:` headers."""
    matches = list(NAME_HEADER.finditer(sql_text))
    queries: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql_text)
        body = sql_text[start:end].strip().rstrip(";").strip()
        if body:
            queries.append((name, body))
    return queries


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", action="store_true", help="export each result to results/sql/<name>.csv"
    )
    parser.add_argument(
        "--query", help="run only the query with this name (default: run all)"
    )
    args = parser.parse_args()

    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    queries = parse_named_queries(SQL_FILE.read_text())
    if args.query:
        queries = [(n, q) for n, q in queries if n == args.query]
        if not queries:
            raise SystemExit(f"No query named '{args.query}' in {SQL_FILE.name}")

    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    build_customers_view(con)

    results: Dict[str, int] = {}
    for name, query in queries:
        print(f"\n{'=' * 70}\n{name}\n{'-' * 70}")
        relation = con.sql(query)
        relation.show()
        if args.save:
            out_path = OUTPUT_DIR / f"{name}.csv"
            con.sql(query).write_csv(str(out_path))
            results[name] = 1
            print(f"saved -> {out_path.relative_to(PROJECT_ROOT)}")

    if args.save:
        print(f"\nExported {len(results)} result set(s) to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

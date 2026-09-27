"""SQLite-backed inverted-index candidate generation for large TSV corpora."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterator

from normalization import blocking_keys, blocking_keys_normalized, normalize

FIELDS = ("entity_id", "business_name", "business_address", "country")


def read_records(path: str | Path) -> Iterator[dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            yield {field: (row.get(field) or "").strip() for field in FIELDS}


def _init_index(db: sqlite3.Connection) -> None:
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-64000")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            entity_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS postings (
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (key, entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS key_counts (
            key TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL
        ) WITHOUT ROWID;
    """)


def build_index(paths: list[str | Path], db_path: str | Path, *, batch_size: int = 25000) -> int:
    """Build a disk-backed candidate inverted index across the S2 and S3 files."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    _init_index(db)
    count = 0
    try:
        for path in paths:
            records_batch = []
            posting_batch = []
            for record in read_records(path):
                entity_id = record["entity_id"]
                name = normalize(record["business_name"])
                address = normalize(record["business_address"])
                country = normalize(record["country"])
                records_batch.append((entity_id, name, address, country))
                for key in blocking_keys_normalized(name, address, country):
                    posting_batch.append((key, entity_id))
                if len(records_batch) >= batch_size:
                    db.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", records_batch)
                    db.executemany("INSERT OR IGNORE INTO postings VALUES (?, ?)", posting_batch)
                    db.commit()
                    count += len(records_batch)
                    records_batch.clear()
                    posting_batch.clear()
                    if count % 250000 < batch_size:
                        print(f"Indexed {count:,} target records...", flush=True)

            if records_batch:
                db.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", records_batch)
                db.executemany("INSERT OR IGNORE INTO postings VALUES (?, ?)", posting_batch)
                db.commit()
                count += len(records_batch)
        print("Computing blocking-key frequencies...", flush=True)
        db.execute("INSERT INTO key_counts SELECT key, COUNT(*) FROM postings GROUP BY key")
        db.commit()
        print("Blocking-key frequencies ready.", flush=True)
        return count
    finally:
        db.close()


def retrieve_candidates(db: sqlite3.Connection, record: dict[str, str], *, max_candidates: int = 30) -> list[dict[str, str]]:
    """Retrieve candidate IDs by rare informative name/address tokens."""
    keys = blocking_keys(record["business_name"], record["business_address"], record["country"])
    if not keys:
        return []
    placeholders = ",".join("?" for _ in keys)
    # Rank by shared blocking-key IDF; frequencies were materialized once at index build.
    rows = db.execute(
        f"""SELECT r.entity_id, r.name, r.address, r.country,
                   SUM(1.0 / (1.0 + k.document_frequency)) AS score
            FROM postings p JOIN key_counts k ON p.key = k.key
            JOIN records r ON r.entity_id = p.entity_id
            WHERE p.key IN ({placeholders})
            GROUP BY r.entity_id ORDER BY score DESC, r.entity_id LIMIT ?""",
        (*keys, max_candidates),
    )
    return [
        {"entity_id": row[0], "business_name": row[1], "business_address": row[2], "country": row[3]}
        for row in rows
    ]


def write_candidate_file(path: str | Path, ids: list[str], candidates: dict[str, list[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        for source1_id in ids:
            writer.writerow((source1_id, ",".join(dict.fromkeys(candidates.get(source1_id, [])))))

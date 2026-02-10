import json
import os
from typing import Optional

import psycopg2
import sqlite3


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS genomic_variants (
    id SERIAL PRIMARY KEY,
    chrom TEXT NOT NULL,
    pos INTEGER NOT NULL,
    gene TEXT,
    ref TEXT NOT NULL,
    alt TEXT NOT NULL,
    gt TEXT,
    depth INTEGER,
    effect TEXT,
    clinsig TEXT
);
"""

SQLITE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS genomic_variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chrom TEXT NOT NULL,
    pos INTEGER NOT NULL,
    gene TEXT,
    ref TEXT NOT NULL,
    alt TEXT NOT NULL,
    gt TEXT,
    depth INTEGER,
    effect TEXT,
    clinsig TEXT
);
"""


def load_config(config_path: str = "config/db_config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def connect_postgres(cfg: dict):
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )


def connect_sqlite(cfg: dict):
    os.makedirs(os.path.dirname(cfg["path"]), exist_ok=True)
    return sqlite3.connect(cfg["path"])


def connect_db(config_path: str = "config/db_config.json"):
    config = load_config(config_path)
    db_type = config.get("db_type", "postgres")
    if db_type == "sqlite":
        return connect_sqlite(config["sqlite"])
    return connect_postgres(config["postgres"])


def init_db(config_path: str = "config/db_config.json") -> None:
    config = load_config(config_path)
    db_type = config.get("db_type", "postgres")

    if db_type == "sqlite":
        conn = connect_sqlite(config["sqlite"])
        try:
            conn.execute(SQLITE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()
        return

    conn = connect_postgres(config["postgres"])
    try:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(TABLE_SQL)
    finally:
        conn.close()


def init_database_if_missing(config_path: str = "config/db_config.json") -> Optional[str]:
    config = load_config(config_path)
    db_type = config.get("db_type", "postgres")

    if db_type == "sqlite":
        init_db(config_path)
        return None

    admin_cfg = dict(config["postgres"])
    target_db = admin_cfg.pop("dbname")
    admin_cfg["dbname"] = "postgres"

    conn = psycopg2.connect(**admin_cfg)
    try:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (target_db,)
            )
            exists = cursor.fetchone() is not None
            if not exists:
                cursor.execute(f"CREATE DATABASE {target_db}")
    finally:
        conn.close()

    init_db(config_path)
    return target_db


if __name__ == "__main__":
    init_database_if_missing()

import json
import os
from typing import Optional

import psycopg2
import sqlite3


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nyu_samples (
    id SERIAL PRIMARY KEY,
    rgb_path TEXT NOT NULL,
    depth_path TEXT NOT NULL,
    label_path TEXT NOT NULL,
    split TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS segmentation_results (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER REFERENCES nyu_samples(id),
    model TEXT NOT NULL,
    miou REAL,
    pixel_acc REAL,
    mean_acc REAL,
    fw_iou REAL,
    dice REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reconstruction_results (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER REFERENCES nyu_samples(id),
    method TEXT NOT NULL,
    rmse REAL,
    absrel REAL,
    delta1 REAL,
    delta2 REAL,
    delta3 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SQLITE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS nyu_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rgb_path TEXT NOT NULL,
    depth_path TEXT NOT NULL,
    label_path TEXT NOT NULL,
    split TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS segmentation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER,
    model TEXT NOT NULL,
    miou REAL,
    pixel_acc REAL,
    mean_acc REAL,
    fw_iou REAL,
    dice REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(sample_id) REFERENCES nyu_samples(id)
);

CREATE TABLE IF NOT EXISTS reconstruction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id INTEGER,
    method TEXT NOT NULL,
    rmse REAL,
    absrel REAL,
    delta1 REAL,
    delta2 REAL,
    delta3 REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(sample_id) REFERENCES nyu_samples(id)
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
            for statement in SQLITE_TABLE_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    conn.execute(stmt)
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

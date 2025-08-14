import os
import sqlite3
from typing import Dict, Any


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def init_db(db_path: str) -> None:
    _ensure_parent_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                initial_ts REAL,
                best_ts REAL,
                delta_t REAL,
                speed_mph REAL,
                vla_deg REAL,
                hla_deg REAL,
                total_spin_rpm REAL,
                side_spin_rpm REAL,
                back_spin_rpm REAL,
                spin_axis_deg REAL,
                carry_yd REAL,
                total_yd REAL,
                apex_ft REAL,
                flight_time_s REAL,
                descending_angle_deg REAL,
                initial_img_path TEXT,
                best_img_path TEXT,
                positions_json TEXT
            )
            """
        )


def insert_shot_record(db_path: str, record: Dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO shots (
                initial_ts, best_ts, delta_t,
                speed_mph, vla_deg, hla_deg,
                total_spin_rpm, side_spin_rpm, back_spin_rpm,
                spin_axis_deg, carry_yd, total_yd,
                apex_ft, flight_time_s, descending_angle_deg,
                initial_img_path, best_img_path, positions_json
            ) VALUES (
                :initial_ts, :best_ts, :delta_t,
                :speed_mph, :vla_deg, :hla_deg,
                :total_spin_rpm, :side_spin_rpm, :back_spin_rpm,
                :spin_axis_deg, :carry_yd, :total_yd,
                :apex_ft, :flight_time_s, :descending_angle_deg,
                :initial_img_path, :best_img_path, :positions_json
            )
            """,
            record,
        )


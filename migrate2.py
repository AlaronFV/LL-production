#!/usr/bin/env python3
"""
migrate.py

Performs a one-time migration to a new, unified SQLite database (i_plus_one.db),
now including a 'chapter' column for a two-level selection UI.
"""

import orjson
import sqlite3
import regex
from pathlib import Path

# --- CONFIGURATION ---
OLD_LOGS_DB = Path("data") / "logs.db"
NEW_DB_PATH = Path("data") / "i_plus_one.db"
INPUT_SRC_DIR = Path("input")

def parse_filename_and_chapter(stem: str):
    """
    Parses a stem like 'My Book 12' into ('My Book', '12').
    Handles cases with no numbers.
    """
    match = regex.match(r'^(.*\D)?(\d+)$', stem)
    if match:
        base_name = (match.group(1) or '').strip()
        chapter = match.group(2)
        return base_name if base_name else f"Chapter {chapter}", chapter
    return stem, "1" # Default chapter if no number is found

def create_new_schema(conn):
    """Creates the new table structures, including the 'chapter' column."""
    cur = conn.cursor()
    print("Creating new database schema with chapter support...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS texts (
        lang TEXT NOT NULL,
        filename TEXT NOT NULL,
        chapter TEXT NOT NULL,
        idx INTEGER NOT NULL,
        source TEXT NOT NULL,
        target TEXT NOT NULL,
        words_json TEXT NOT NULL,
        PRIMARY KEY(lang, filename, chapter, idx)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_progress (
        lang TEXT NOT NULL,
        filename TEXT NOT NULL,
        chapter TEXT NOT NULL,
        text_idx INTEGER NOT NULL,
        is_target INTEGER NOT NULL DEFAULT 0,
        is_revealed INTEGER NOT NULL DEFAULT 0,
        is_reviewed INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(lang, filename, chapter, text_idx),
        FOREIGN KEY(lang, filename, chapter, text_idx) REFERENCES texts(lang, filename, chapter, idx)
    )""")
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_texts_lookup ON texts (lang, filename, chapter)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_progress_lookup ON user_progress (lang, filename, chapter)")
    
    conn.commit()
    print("Schema created successfully.")

def migrate_inputs(conn):
    """Migrates raw text data from JSON files into the 'texts' table."""
    print("\nStarting text migration...")
    all_records = []
    for lang_dir in INPUT_SRC_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        print(f"  Processing language: {lang}")
        for js_file in sorted(lang_dir.glob("*.json")):
            try:
                data = orjson.loads(js_file.read_bytes())
                base_name, chapter = parse_filename_and_chapter(js_file.stem)
                for idx, unit in enumerate(data):
                    if "source" in unit and "target" in unit and "words" in unit:
                        all_records.append((
                            lang, base_name, chapter, idx,
                            unit["source"], unit["target"],
                            orjson.dumps(unit["words"]).decode('utf-8')
                        ))
                print(f"    - Parsed {js_file.name} -> ('{base_name}', '{chapter}')")
            except Exception as e:
                print(f"    - ERROR: Could not process file {js_file.name}: {e}")

    if not all_records:
        print("No text records found to migrate.")
        return

    print(f"\nInserting {len(all_records)} text records into the database...")
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO texts(lang, filename, chapter, idx, source, target, words_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        all_records
    )
    conn.commit()
    print("Text migration complete.")

def migrate_logs(conn):
    """Migrates data from the old logs.db into the new 'user_progress' table."""
    print("\nStarting log migration...")
    if not OLD_LOGS_DB.exists():
        print("Old logs database (logs.db) not found. Skipping.")
        return

    old_conn = sqlite3.connect(str(OLD_LOGS_DB))
    old_cur = old_conn.cursor()
    try:
        old_cur.execute("SELECT lang, filename, payload FROM logs")
    except sqlite3.OperationalError:
        print("Could not find 'logs' table in old database. Skipping.")
        old_conn.close()
        return

    progress_records = []
    rows = old_cur.fetchall()
    print(f"Found {len(rows)} log entries to process.")

    for lang, filename_stem, payload_blob in rows:
        payload = orjson.loads(payload_blob)
        base_name, chapter = parse_filename_and_chapter(filename_stem)
        
        all_indices = set(payload.get("target_indices", [])) | set(payload.get("revealed", [])) | set(payload.get("reviewed", []))

        for idx in all_indices:
            progress_records.append((
                lang, base_name, chapter, idx,
                1 if idx in payload.get("target_indices", []) else 0,
                1 if idx in payload.get("revealed", []) else 0,
                1 if idx in payload.get("reviewed", []) else 0,
            ))

    old_conn.close()

    if not progress_records:
        print("No user progress records found to migrate.")
        return

    print(f"Inserting {len(progress_records)} user progress records...")
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO user_progress(lang, filename, chapter, text_idx, is_target, is_revealed, is_reviewed) VALUES (?, ?, ?, ?, ?, ?, ?)",
        progress_records
    )
    conn.commit()
    print("Log migration complete.")

if __name__ == "__main__":
    # --- Main execution logic is the same as before ---
    print("--- i+1 Data Migration to Unified SQLite Database (with Chapters) ---")
    if NEW_DB_PATH.exists():
        if input(f"DB '{NEW_DB_PATH}' exists. Delete and remigrate? (y/N): ").lower() != 'y':
            print("Migration aborted.")
            exit()
        NEW_DB_PATH.unlink()
    NEW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db_conn = sqlite3.connect(str(NEW_DB_PATH))
    try:
        create_new_schema(db_conn)
        migrate_inputs(db_conn)
        migrate_logs(db_conn)
        print("\n✅ Migration complete!")
    except Exception as e:
        print(f"\n❌ An error occurred during migration: {e}")
    finally:
        db_conn.close()
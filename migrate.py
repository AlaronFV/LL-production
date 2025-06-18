#!/usr/bin/env python3
"""
migrate_data.py

1) inputs: input/<lang>/*.json → input/<lang>/<lang>.ndjson  
2) logs:   logs/<lang>/*.json → data/logs.db (SQLite)
"""

import orjson
import sqlite3
from pathlib import Path

def migrate_inputs():
    for lang_dir in Path("input").iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        ndjson_path = Path("input") / f"{lang}.ndjson"
        print(f"→ Writing {ndjson_path}")
        with ndjson_path.open("wb") as out:
            for js in sorted(lang_dir.glob("*.json")):
                data = orjson.loads(js.read_bytes())
                stem = js.stem
                for idx, unit in enumerate(data):
                    rec = {"filename": stem, "index": idx, "unit": unit}
                    out.write(orjson.dumps(rec) + b"\n")

def migrate_logs():
    db_path = Path("data") / "logs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        lang TEXT NOT NULL,
        filename TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY(lang, filename)
    )""")
    for lang_dir in Path("logs").iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for lf in lang_dir.glob("*.json.json"):
            raw = orjson.loads(lf.read_bytes())
            if isinstance(raw, dict):
                payload = {
                    "revealed": raw.get("revealed", []),
                    "target_indices": raw.get("target_indices", raw.get("revealed", [])),
                    "visible_states": raw.get("visible_states", {}),
                    "reviewed": raw.get("reviewed", []),
                }
            else:
                payload = {
                    "revealed": raw,
                    "target_indices": raw,
                    "visible_states": {str(i): True for i in raw},
                    "reviewed": raw,
                }
            
            blob = orjson.dumps(payload)
            conn.execute(
                "INSERT OR REPLACE INTO logs(lang,filename,payload) VALUES(?,?,?)",
                (lang, lf.name.split(".")[0], blob)
            )
    conn.commit()
    conn.close()
    print(f"→ Logs migrated into {db_path}")

if __name__ == "__main__":
    #migrate_inputs()
    migrate_logs()
    print("Migration complete. You can now remove your old JSON files if you like.")
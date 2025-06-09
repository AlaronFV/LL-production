import spacy
import orjson
import regex
import os
import sqlite3
from pathlib import Path

# --- CONFIGURATION ---
# The unified database for the entire application
DB_PATH = Path("data") / "i_plus_one.db"
# The directory where you place your raw text files to be processed
# NOTE: Please adjust this path to your actual source directory
RAW_TEXT_SRC_DIR = Path("/data/data/com.termux/files/home/storage/shared/movies/Writer/language_learning_app")

# Global nlp model to avoid reloading
nlp = None


def parse_filename_and_chapter(stem: str):
    """
    Parses a stem like 'My Book 12' into ('My Book', '12').
    """
    match = regex.match(r'^(.*\D)?(\d+)$', stem)
    if match:
        base_name = (match.group(1) or '').strip()
        chapter = match.group(2)
        return base_name if base_name else f"Chapter {chapter}", chapter
    return stem, "1"


def strip_nonalpha_ends(s):
    return regex.sub(r'^[^\p{L}]*|[^\p{L}]*$', '', s)

def remove_nonalpha_after_spaces(s):
    return regex.sub(r'(?<=\s+)[^\p{L}0-9]+', '', s)

def get_words(text):
    doc = nlp(remove_nonalpha_after_spaces(strip_nonalpha_ends(text)))
    
    words = []
    for token in doc:
        if token.pos_ == "VERB":
            head_lower = token.lower_
            # Simplified passive check
            if any(child.dep_ == "aux:pass" for child in token.children):
                passive_aux = [child.lower_ for child in token.children if child.dep_ == "aux:pass"]
                words.append(f"{head_lower} {' '.join(passive_aux)}")
            else:
                words.append(token.lower_)
        elif token.is_alpha and not (token.dep_ == "aux:pass" and token.head.pos_ == "VERB"):
            words.append(token.lower_)
    return words

def prepare(language: str):
    """
    Connects to the DB, checks for existing (file, chapter) pairs,
    and inserts new texts from the source directory.
    """
    global nlp
    language = language.lower()

    if not DB_PATH.exists():
        print(f"ERROR: Database '{DB_PATH}' not found. Run migration first.")
        return
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    if nlp is None:
        print("Loading spaCy model 'la_core_web_lg'...")
        try:
            nlp = spacy.load("la_core_web_lg")
            auto_doubler = {"…"}
            print("spaCy model loaded.")
        except OSError:
            print("ERROR: spaCy model 'la_core_web_lg' not found.")
            conn.close()
            return

    # Discover existing (filename, chapter) pairs from the DB
    cur.execute("SELECT DISTINCT filename, chapter FROM texts WHERE lang = ?", (language,))
    existing_items = {tuple(row) for row in cur.fetchall()}
    print(f"Found {len(existing_items)} existing file/chapter pairs for '{language}'.")

    src_dir = RAW_TEXT_SRC_DIR / language
    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}")
        conn.close()
        return

    new_files_processed = 0
    total_units_added = 0
    
    for fname in sorted(os.listdir(src_dir)):
        stem = Path(fname).stem
        base_name, chapter = parse_filename_and_chapter(stem)
        
        if (base_name, chapter) in existing_items:
            continue

        print(f"[prepare] Processing new item: ('{base_name}', '{chapter}')...")
        new_files_processed += 1
        raw_text = (src_dir / fname).read_text(encoding="utf-8")
        paragraphs = raw_text.split("\n\n")

        units_to_insert = []
        for p in paragraphs:
            lines = p.strip().split("\n")
            
            if p in auto_doubler:
                # duplicate the only line
                lines = [lines[0], lines[0]]
            
            if len(lines) < 2:
                continue
            
            # Simple source/target logic
            src, tgt = lines[0], lines[1]
            words = get_words(tgt)
            
            units_to_insert.append((
                language, base_name, chapter, len(units_to_insert),
                src, tgt, orjson.dumps(words).decode('utf-8')
            ))

        if units_to_insert:
            cur.executemany(
                "INSERT INTO texts(lang, filename, chapter, idx, source, target, words_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                units_to_insert
            )
            conn.commit()
            print(f"[prepare] Added {len(units_to_insert)} units from ('{base_name}', '{chapter}').")
            total_units_added += len(units_to_insert)

    if new_files_processed == 0:
        print("No new files to process.")
    else:
        print(f"\n✅ Preparation complete. Added {total_units_added} new units.")
    conn.close()

if __name__ == '__main__':
    # Example of how to run this script
    # You would typically call this from the main app's "Add files" button.
    target_lang = input("Enter the language to prepare (e.g., 'latin'): ").lower()
    if target_lang:
        prepare(target_lang)
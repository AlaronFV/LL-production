import spacy
import orjson
import regex
import os
from pathlib import Path


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
            if any(child.dep_ == "aux:pass" for child in token.children):
                for child in token.children:
                    if child.dep_ == "aux:pass":
                        words.append(f"{head_lower} {child.lower_}")
            else:
                words.append(token.lower_)
        elif token.is_alpha and not (token.dep_ == "aux:pass" and token.head.pos_ == "VERB"):
            words.append(token.lower_)
    return words

def prepare(language):
    """
    1) Reads /data/.../language/*.txt (or whatever)  
    2) Skips any file whose stem already appears in input/<lang>/<lang>.ndjson  
    3) Appends new records of the form 
         { "filename": stem, "index": i, "unit": {...} }
       to input/<lang>/<lang>.ndjson
    """
    # lazy‐load the spaCy model just once
    global nlp
    if "nlp" not in globals():
        nlp = spacy.load("la_core_web_lg")

    auto_doubler = {"…"}

    # source directory with your raw text files
    src_dir = Path(
        "/data/data/com.termux/files/home"
        "/storage/shared/movies/Writer/language_learning_app"
    ) / language

    # ensure our input folder & NDJSON exist
    ndjson_path = Path("input") / f"{language}.ndjson"
    ndjson_path.parent.mkdir(parents=True, exist_ok=True)
    if not ndjson_path.exists():
        ndjson_path.write_bytes(b"")

    # discover which stems we've already imported
    existing_stems = {
        orjson.loads(line)["filename"]
        for line in ndjson_path.read_bytes().splitlines()
    }

    # open once for append
    with ndjson_path.open("ab") as out:
        for fname in sorted(os.listdir(src_dir)):
            stem = Path(fname).stem
            if stem in existing_stems:
                continue

            print(f"[prepare] processing {fname}…")
            raw = (src_dir / fname).read_text(encoding="utf-8")
            paras = raw.split("\n\n")

            # build your units
            units = []
            for p in paras:
                lines = p.split("\n")
                if p in auto_doubler:
                    # duplicate the only line
                    lines = [lines[0], lines[0]]
                if len(lines) < 2:
                    continue
                src, tgt = lines[0], lines[1]
                words = get_words(tgt)
                units.append({"source": src, "target": tgt, "words": words})

            # append each as one NDJSON record
            for idx, unit in enumerate(units):
                rec = {
                    "filename": stem,
                    "index": idx,
                    "unit": unit,
                }
                out.write(orjson.dumps(rec) + b"\n")

            print(f"[prepare] appended {len(units)} units from {fname}")
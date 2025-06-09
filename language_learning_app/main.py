# main.py
# -------------------------------------------------------------------
# i+1 LANGUAGE-LEARNING TOOL
# -------------------------------------------------------------------

import streamlit as st
from pathlib import Path
from collections import defaultdict
from functools import partial, lru_cache

import orjson
import sqlite3

# -------------------------------------------------------------------
# import underlying vocabulary model
# -------------------------------------------------------------------
from i_plus_one import (
    VocabularyModel,
    LearningQueue,
    get_natural_candidates,
    get_vocabulary_statistics
)
from preparer import prepare


# -------------------------------------------------------------------
# folder bootstrap
# -------------------------------------------------------------------
for d in ["input", "data/vocab_models"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# set up SQLite logs DB
LOG_DB = Path("data") / "logs.db"
LOG_DB.parent.mkdir(exist_ok=True, parents=True)
log_conn = sqlite3.connect(str(LOG_DB), check_same_thread=False)
log_conn.execute("""
CREATE TABLE IF NOT EXISTS logs (
    lang TEXT NOT NULL,
    filename TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(lang, filename)
)""")
log_conn.commit()

# -------------------------------------------------------------------
#  MULTI-LANGUAGE WRAPPER AROUND VocabularyModel
# -------------------------------------------------------------------
class LanguageLearningModel:
    """Manages one VocabularyModel per language + convenience helpers."""

    def __init__(self):
        self.vocab_models = {}
        self.load_models()

    # ---------- disk I/O ----------

    def get_model_path(self, language):  # noqa
        return Path("data/vocab_models") / f"{language.lower()}_vocab_model.bin"

    def load_models(self):
        vocab_dir = Path("data/vocab_models")
        if vocab_dir.exists():
            for f in vocab_dir.glob("*_vocab_model.bin"):
                try:
                    lang = f.stem.split("_")[0]
                    model = VocabularyModel.load_fast(str(f))
                    model.save_fast(str(f)) # This sets the internal path
                    self.vocab_models[lang] = model
                    st.success(f"Loaded vocabulary model for {lang}")
                except Exception as e:
                    st.warning(f"Error loading model {f}: {e}")
                    print(f, e)

    # ---------- model retrieval / creation ----------

    def get_or_create_model(self, language):
        key = language.lower()
        if key not in self.vocab_models:
            self.vocab_models[key] = VocabularyModel(
                learning_rate=0.15, base_decay_rate=0.05
            )
            self.save_model(key)
            st.info(f"Created new vocabulary model for {language}")
        return self.vocab_models[key]

    def save_model(self, language):
        key = language.lower()
        if key in self.vocab_models:
            path = self.get_model_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.vocab_models[key].save_fast(str(path))
    # ---------- high-level update / stats ----------

    def update_knowledge(self, words, language, feedback_level):
        """Update vocabulary model AND RETURN detailed result."""
        if not words:
            return None
        mdl = self.get_or_create_model(language)

        mdl.update_from_words(words, feedback_level / 2)
        self.save_model(language)

    def reset_vocabulary(self, language=None):
        """Reset the vocabulary model for a specific language or all languages"""
        if language:
            # Reset just one language
            language_key = language.lower()
            if language_key in self.vocab_models:
                self.vocab_models[language_key] = VocabularyModel(
                    learning_rate=0.15, base_decay_rate=0.05
                )
                self.save_model(language_key)
                return f"Reset vocabulary for {language}"
        else:
            # Reset all languages
            self.vocab_models = {}
            # Delete all model files
            vocab_dir = Path("data/vocab_models")
            if vocab_dir.exists():
                for model_file in vocab_dir.glob("*_vocab_model.bin"):
                    try:
                        model_file.unlink()
                    except Exception:
                        pass
            return "Reset all vocabulary models"


# -------------------------------------------------------------------
# LOG-FILE HELPERS 
# -------------------------------------------------------------------
def load_log_file(lang: str, filename: str):
    """
    Returns (revealed:set, visible:dict, reviewed:set, target:set)
    for this (lang, filename_stem).
    """

    cur = log_conn.execute(
        "SELECT payload FROM logs WHERE lang=? AND filename=?", (lang, filename)
    )
    row = cur.fetchone()
    if not row:
        return set(), {}, set(), set()
    p = orjson.loads(row[0])
    return (
        set(p.get("revealed", [])),
        {int(k): v for k, v in p.get("visible_states", {}).items()},
        set(p.get("reviewed", [])),
        set(p.get("target_indices", [])),
    )

def save_session_state(lang: str, filename: str, revealed, target, visible, reviewed):
    """
    Upsert one record back into logs table.
    """
    payload = {
        "revealed": list(revealed),
        "target_indices": list(target),
        "visible_states": {str(k): v for k, v in visible.items()},
        "reviewed": list(reviewed),
    }
    blob = orjson.dumps(payload)
    log_conn.execute("""
        INSERT INTO logs(lang, filename, payload)
        VALUES(?, ?, ?)
        ON CONFLICT(lang, filename) DO UPDATE SET payload=excluded.payload
    """, (lang, filename, blob))
    log_conn.commit()

# -------------------------------------------------------------------
#  INPUT SCAN (one pass)
# -------------------------------------------------------------------
def scan_input(lang: str):
    """
    Streams input/<lang>/<lang>.ndjson → a list of items
    {filename, index, unit}.
    """
    items = []
    nd = Path("input") / f"{lang}.ndjson"
    if not nd.exists():
        st.error(f"No input NDJSON for '{lang}'. Run migrate_data.py")
        return items

    for line in nd.read_bytes().splitlines():
        rec = orjson.loads(line)
        stem = rec["filename"]
        idx = rec["index"]
        unit = rec["unit"]
        _, _, reviewed, _ = load_log_file(lang, stem)
        if idx in reviewed or not unit.get("words"):
            continue
        items.append({"filename": stem, "index": idx, "unit": unit})
    return items


# -------------------------------------------------------------------
#  SIDEBAR VOCAB STATISTICS 
# -------------------------------------------------------------------
def display_vocabulary_stats(model, target_language):
    stats = get_vocabulary_statistics(model.get_or_create_model(target_language))
    if stats["total_words"] > 0:
        knowledge_stats = f"""Current knowledge: {stats['all_known_knowledge']:.2f}\n
Learning potential: {stats['all_possible_knowledge'] - stats['all_known_knowledge']:.2f}\n
All knowledge: {stats['all_possible_knowledge']:.2f}"""
        proficiency_stats = f"""Avg proficiency: {stats['average_proficiency']:.2f}\n
Avg volatility: {stats['average_volatility']:.2f}\n
Avg effective: {stats['average_effective_proficiency']:.2f}"""
    else:
        knowledge_stats, proficiency_stats = "", ""
    st.sidebar.write(f"""### {f"Vocabulary Statistics: {target_language}"}\n
Total vocabulary: {stats['total_words']} words\n
{knowledge_stats}\n
{f"Familiar: {stats['familiar']} words"}\n
{f"Still learning: {stats['learning']} words"}\n
{f"Stable knowledge: {stats['stable']} words"}\n
{f"Semi-stable: {stats['semi_stable']} words"}\n
{f"Volatile: {stats['volatile']} words"}\n
{proficiency_stats}""")

def _commit_queue(level, item_info, current_iid, lang):
    st.session_state.learning_queue_obj.process_answer(current_iid, level)
    stem = item_info["filename"]
    idx = item_info["index"]
    
    revealed, visible, reviewed, target = load_log_file(lang, stem)
    revealed.add(idx)
    reviewed.add(idx)
    visible[idx] = False
    target.add(idx)
    save_session_state(lang, stem, revealed, target, visible, reviewed)

    st.session_state.queue_source_revealed = False
    st.rerun()
# -------------------------------------------------------------------
#  QUEUE VIEW 
# -------------------------------------------------------------------
def queue_view(model_service, lang):
    st.subheader("Study Queue")
    display_vocabulary_stats(model_service, lang)

    # (re)build queue if missing or language changed
    if "learning_queue_obj" not in st.session_state or st.session_state.learning_queue_target != lang:
        # Get the specific model instance for the language
        model_instance = model_service.get_or_create_model(lang)
        
        # Create the queue and pass the model instance to it
        q = LearningQueue(model_instance)
        
        all_items = scan_input(lang.lower())
        
        # Build an iid -> item map for Python-side lookups
        iid_map = {item["index"]: item for item in all_items}

        q.build_from_input(all_items)
        
        st.session_state.update({
            "learning_queue_obj": q,
            "learning_queue_target": lang,
            "iid_to_item_map": iid_map, # Store the map
        })

    queue: LearningQueue = st.session_state.learning_queue_obj
    total = queue.size()
    if not total:
        st.info("Queue empty.  Click rebuild if you added new material.")
        if st.button("Rebuild Queue"):
            st.cache_data.clear()
            st.session_state.pop("learning_queue_obj", None)
            st.rerun()
        return
    
    q0_size = queue.size(0)
    q1_size = queue.size(1)
    q2_size = queue.size(2)

    st.info(f'Queue contains {total} items. \n\n"Didn\'t understand" ({q0_size}), "Partially understood" ({q1_size}), "Fully understood" ({q2_size}).')

    current_iid = queue.pop_next()
    if current_iid is None:
        st.warning("Queue is empty or contains only invalid items.")
        return
    
    current_item = st.session_state.iid_to_item_map.get(current_iid)
    if not current_item:
        st.error(f"Could not find item for iid {current_iid}. Rebuilding might be necessary.")
        return

    unit = current_item["unit"]

    if "queue_source_revealed" not in st.session_state:
        st.session_state.queue_source_revealed = False

    st.markdown("### Current Unit")
    if st.button(unit["target"], key="unit_target_btn"):
        st.session_state.queue_source_revealed = True
        st.rerun()

    if st.session_state.queue_source_revealed:
        st.info(unit["source"])
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("❌😔❌", use_container_width=True):
                _commit_queue(0, current_item, current_iid, lang)
        with c2:
            if st.button("🔶🤔🔶", use_container_width=True):
                _commit_queue(1, current_item, current_iid, lang)
        with c3:
            if st.button("✅🧐✅", use_container_width=True):
                _commit_queue(2, current_item, current_iid, lang)

# -------------------------------------------------------------------
#  NUMERICAL FILE-NAV HELPERS 
# -------------------------------------------------------------------
def next_number(number_list):
    st.session_state.current_num_index = (st.session_state.current_num_index + 1) % len(
        number_list
    )
    st.session_state.force_state_reset = True


def prev_number(number_list): 
    st.session_state.current_num_index = (st.session_state.current_num_index - 1) % len(
        number_list
    )
    st.session_state.force_state_reset = True




@lru_cache(maxsize=2)
def get_input_dict(input_files):
    input_dict = defaultdict(list)
    for file_name in input_files:
        last_space_index = file_name.rfind(" ")
        if last_space_index != -1:
            key = file_name[:last_space_index]
            num = file_name[last_space_index + 1:]
            try:
                num = int(num)
            except ValueError:
                num = None
            input_dict[key].append(num)
        else:
            input_dict[file_name].append(None)
    input_dict = {k: sorted(v) for k, v in input_dict.items()}
    idx_map = {k: {v: i for i, v in enumerate(nums)} for k, nums in input_dict.items()}
    return input_dict, idx_map


def add_columns_style():
    st.markdown("""
<style>
    [data-testid="stColumn"] {
    min-width: max-content !important;
    flex 1 1 max-content !important;
    }

    [data-testid="stSelectbox"] > [data-testid="stWidgetLabel"] {
    justify-content: center !important;
    }
    .st-key-number-select-box > [data-testid="stSelectbox"] > [data-testid="stWidgetLabel"] {
    display: none;
    }
    
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
#  STREAMLIT MAIN APP
# -------------------------------------------------------------------
def main():
    st.title("i+1 Language Learning Tool")
    add_columns_style()

    if "language_model" not in st.session_state:
        st.session_state.language_model = LanguageLearningModel()
    model = st.session_state.language_model

    lang_choices = ["Latin"]
    
    target_language = st.selectbox("Target language", lang_choices).lower()

    if "show_queue_view" not in st.session_state:
        st.session_state.show_queue_view = False

    if st.button("Study Queue" if not st.session_state.show_queue_view else "Back to Text"):
        st.session_state.update({"show_queue_view": not st.session_state.show_queue_view, "force_state_reset": True})
        if st.session_state.show_queue_view:
            st.session_state.queue_source_revealed = False
        else:
            st.session_state.pop("iid_to_item_map", None)
            st.session_state.pop("learning_queue_obj", None)
            st.session_state.pop("learning_queue_target", None)
        st.rerun()

    if st.session_state.show_queue_view:
        queue_view(model, target_language)
        return

    
    
    # build an index of the NDJSON
    input_nd = Path("input") / f"{target_language}.ndjson"
    if not input_nd.exists():
        st.error(f"No {target_language}.ndjson.")
        return

    # build per‐file index
    files = tuple({ orjson.loads(line)["filename"]
                    for line in input_nd.read_bytes().splitlines() })
    input_dict, num_dict = get_input_dict(files)

    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "force_state_reset" not in st.session_state:
        st.session_state.force_state_reset = False

    # ---------------- file selector ----------------
    colA, colB = st.columns([5, 2])
    with colA:
        selected_file = st.selectbox("Select a file to read", input_dict.keys())
    with colB:
        nums = input_dict[selected_file]
        unnumbered = nums == [None]
        if unnumbered:
            st.session_state.current_num_index = 0
        else:
            idx_map = num_dict[selected_file]
            if (
                "current_num_index" not in st.session_state
                or st.session_state.current_file != selected_file
            ):
                st.session_state.current_num_index = 0
            colL, colM, colR = st.columns([1, 3, 1], vertical_alignment="bottom")
            with colL:
                st.button("◀", on_click=partial(prev_number, nums), use_container_width=True)
            with colM:
                st.session_state.current_num_index = idx_map[
                    st.selectbox(
                        "Select number",
                        nums,
                        index=st.session_state.current_num_index,
                        on_change=lambda: st.session_state.__setitem__("force_state_reset", True),
                        key="number-select-box"
                    )
                ]
            with colR:
                st.button("▶", on_click=partial(next_number, nums), use_container_width=True)

    selected_stem = (
        selected_file
        if unnumbered
        else f"{selected_file} {nums[st.session_state.current_num_index]}"
    )

    # -------------- reset session on file change --------------
    if st.session_state.current_file != selected_file or st.session_state.force_state_reset:
        for k in list(st.session_state.keys()):
            if k not in [
                "language_model",
                "show_queue_view",
                "current_file",
                "current_num_index",
                "force_state_reset",
            ]:
                del st.session_state[k]
        st.session_state.update({
            "current_file": selected_file,
            "force_state_reset": False,
        })
        st.rerun()


    # ----------------------------------------------------------------
    # text-view logic
    # ----------------------------------------------------------------
    
    if "aligned_text" not in st.session_state:
        st.session_state.aligned_text = [
            orjson.loads(line)["unit"]
            for line in input_nd.read_bytes().splitlines()
            if orjson.loads(line)["filename"] == selected_stem
        ]
    total_units = len(st.session_state.aligned_text)

    # load existing log‐state via our new SQLite loader:
    if "revealed" not in st.session_state:
        rev, vis, revd, tgt = load_log_file(target_language, selected_stem)
        st.session_state.update({
            "revealed": rev,
            "visible": vis,
            "reviewed": revd,
            "target_idx": tgt,
        })
    
    if "to_replace_indices" not in st.session_state:
        natural = get_natural_candidates(
            st.session_state.aligned_text, st.session_state.target_idx, model.get_or_create_model(target_language)
        )
        st.session_state.to_replace_indices = st.session_state.target_idx.union(natural)
    
    if "potential_natural_recount" not in st.session_state:
        st.session_state.update({
            "potential_natural_recount": False,
            "potential_natural": 0,
        })
    
    if st.session_state.potential_natural_recount:
        st.session_state.update({
            "potential_natural": len(
                get_natural_candidates(
                    st.session_state.aligned_text,
                    st.session_state.to_replace_indices,
                    model.get_or_create_model(target_language),
                )
            ),
            "potential_natural_recount": False,
        })
    
    display_vocabulary_stats(model, target_language)

    # --- sidebar reset vocabulary ---
    reset_options = ["Current language", "All languages"]
    reset_choice = st.sidebar.radio("Reset vocabulary:", reset_options, index=0)
    if st.sidebar.button("Reset Vocabulary"):
        if st.session_state.get("confirm_reset", False):
            msg = (
                model.reset_vocabulary(target_language)
                if reset_choice == "Current language"
                else model.reset_vocabulary()
            )
            st.sidebar.success(f"{msg}!")
            st.session_state.confirm_reset = False
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.sidebar.warning(
                f"Click again to confirm reset of {reset_choice.lower()}."
            )
    if st.session_state.get("confirm_reset", False):
        if st.sidebar.button("Cancel Reset"):
            st.session_state.confirm_reset = False
            st.rerun()
    
    if st.sidebar.button("Add files"):
        prepare(target_language.lower())
        st.rerun()
        
    def save_current_session_state():
        save_session_state(
                target_language,
                selected_stem,
                st.session_state.revealed,
                st.session_state.to_replace_indices,
                st.session_state.visible,
                st.session_state.reviewed,
            )

    # --- update target sentences button ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Target Sentences"):
            naturals = get_natural_candidates(
                st.session_state.aligned_text,
                st.session_state.to_replace_indices,
                model.get_or_create_model(target_language),
            )
            st.session_state.to_replace_indices.update(naturals)
            st.session_state.potential_natural_recount = True
            
            st.success(f"Added {len(naturals)} natural target sentences!")
            st.rerun()

    with col2:
        tri = st.session_state.to_replace_indices
        transformed = len(tri)
        revealed_cnt = len(tri.intersection(st.session_state.revealed))
        st.info(
            f"Transformed: {transformed}/{total_units} | "
            f"Revealed: {revealed_cnt}/{transformed} | "
            f"New: {st.session_state.potential_natural}"
        )

    # --- reading text ---
    st.subheader("Reading Text")
    st.markdown(
        """
        <style>
            .reviewed {color:#28a745;font-weight:bold;}
            .unrevealed {color:#007bff;}
            .revealed {color:#fd7e14;font-style:italic;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    tri = st.session_state.to_replace_indices
    rev = st.session_state.revealed
    revd = st.session_state.reviewed
    vis = st.session_state.visible
    at = st.session_state.aligned_text
    
    indexes_to_pass = set()
    
    for i, unit in enumerate(at):
        if i in indexes_to_pass:
            continue
        if i in tri:
            is_rev = i in rev
            is_done = i in revd
            show_src = vis.get(i, True) if is_rev else False

            prefix, css = (
                ("✓ ", "reviewed")
                if is_done
                else ("👁️ ", "revealed") if is_rev
                else ("🔍 ", "unrevealed")
            )

            if not is_rev:
                if st.button(f"{prefix}{unit['target']}", key=f"sent_{i}"):
                    st.session_state.revealed.add(i)
                    st.session_state.visible[i] = True
                    save_current_session_state()
                    st.session_state[f"feedback_{i}"] = True
                    st.rerun()
            else:
                if show_src:
                    st.markdown(f"<div class='{css}'>{prefix}{unit['target']}</div>", unsafe_allow_html=True)
                    st.info(unit["source"])
                    if st.button("Hide source", key=f"hide_{i}"):
                        st.session_state.visible[i] = False
                        save_current_session_state()
                        st.rerun()
                else:
                    if st.button(f"{prefix}{unit['target']}", key=f"show_{i}"):
                        st.session_state.visible[i] = True
                        save_current_session_state()
                        st.rerun()

                fb_key = f"feedback_{i}"
                if show_src and not is_done and st.session_state.get(fb_key, False):
                    def _commit(level):
                        model.update_knowledge(unit["words"], target_language, level)
                        st.session_state[fb_key] = False
                        st.session_state.reviewed.add(i)
                        save_current_session_state()
                        st.session_state.potential_natural_recount = True
                        st.rerun()

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("❌😔❌", key=f"fb1_{i}", use_container_width=True):
                            _commit(0)
                    with c2:
                        if st.button("🔶🤔🔶", key=f"fb2_{i}", use_container_width=True):
                            _commit(1)
                    with c3:
                        if st.button("✅🧐✅", key=f"fb3_{i}", use_container_width=True):
                            _commit(2)
                    st.divider()
                    
        else:
            to_write = unit["source"]
            indexes_to_pass = set()
            for check_index, unit in enumerate(at[i+1:], i+1):
                if check_index not in tri:
                    to_write += f"\n\n{unit['source']}"
                    indexes_to_pass.add(check_index)
                else:
                    break
            st.write(to_write)

    if st.button("I understood all other target sentences"):
        mark = 0
        for i in st.session_state.to_replace_indices:
            if i not in st.session_state.revealed:
                model.update_knowledge(st.session_state.aligned_text[i]["words"], target_language, 3)
                st.session_state.revealed.add(i)
                st.session_state.reviewed.add(i)
                st.session_state.visible[i] = False
                mark += 1
        save_current_session_state()
        st.success(f"Marked {mark} sentences as understood!")
        st.rerun()


# -------------------------------------------------------------------
#  BOOTSTRAP
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()

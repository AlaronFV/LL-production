# main.py
# -------------------------------------------------------------------
# i+1 LANGUAGE-LEARNING TOOL
# -------------------------------------------------------------------

import streamlit as st
from pathlib import Path
from functools import partial

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

# --- DATABASE SETUP ---
DB_PATH = Path("data") / "i_plus_one.db"

def get_db_conn():
    """Creates and returns a database connection."""
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)

# Check if the database exists on startup
if not DB_PATH.exists():
    st.error(f"Database not found at '{DB_PATH}'.")
    st.stop()

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
#  NEW DATABASE HELPERS
# -------------------------------------------------------------------

@st.cache_data
def get_available_languages():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT lang FROM texts ORDER BY lang")
        return [row[0] for row in cur.fetchall()] or ["latin"]

@st.cache_data
def get_files_for_language(language: str):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT filename FROM texts WHERE lang = ? ORDER BY filename", (language,))
        return [row[0] for row in cur.fetchall()]

@st.cache_data
def get_chapters_for_file(language: str, filename: str):
    with get_db_conn() as conn:
        cur = conn.cursor()
        # Sorting chapters numerically if they are digits, otherwise alphabetically
        cur.execute("SELECT DISTINCT chapter FROM texts WHERE lang = ? AND filename = ? ORDER BY CAST(chapter AS INTEGER), chapter", (language, filename))
        return [row[0] for row in cur.fetchall()]

def load_text_and_progress(language: str, filename: str, chapter: str):
    query = """
    SELECT t.idx, t.source, t.target, t.words_json,
           COALESCE(p.is_target, 0) as is_target,
           COALESCE(p.is_revealed, 0) as is_revealed,
           COALESCE(p.is_reviewed, 0) as is_reviewed
    FROM texts t
    LEFT JOIN user_progress p ON t.lang = p.lang AND t.filename = p.filename AND t.chapter = p.chapter AND t.idx = p.text_idx
    WHERE t.lang = ? AND t.filename = ? AND t.chapter = ?
    ORDER BY t.idx
    """
    with get_db_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, (language, filename, chapter))
        rows = [dict(row) for row in cur.fetchall()]
        for row in rows:
            row['words'] = orjson.loads(row['words_json'])
        return rows

def save_progress_for_sentence(lang, fname, chap, idx, is_target, is_revealed, is_reviewed):
    query = "INSERT OR REPLACE INTO user_progress VALUES (?, ?, ?, ?, ?, ?, ?)"
    with get_db_conn() as conn:
        conn.execute(query, (lang, fname, chap, idx, is_target, is_revealed, is_reviewed))
        conn.commit()

def save_multiple_as_target(lang, fname, chap, indices):
    records = [(lang, fname, chap, idx, 1, 0, 0) for idx in indices]
    query = "INSERT OR IGNORE INTO user_progress (lang, filename, chapter, text_idx, is_target, is_revealed, is_reviewed) VALUES (?, ?, ?, ?, ?, ?, ?)"
    with get_db_conn() as conn:
        conn.executemany(query, records)
        conn.commit()


# -------------------------------------------------------------------
#  SIDEBAR VOCAB STATISTICS 
# -------------------------------------------------------------------
def display_vocabulary_stats(model, target_language):
    stats = get_vocabulary_statistics(model.get_or_create_model(target_language))
    knowledge_stats = f"""Current knowledge: {stats['all_known_knowledge']:.2f}\n
Learning potential: {stats['all_possible_knowledge'] - stats['all_known_knowledge']:.2f}\n
All knowledge: {stats['all_possible_knowledge']:.2f}"""
    proficiency_stats = f"""Avg proficiency: {stats['average_proficiency']:.2f}\n
Avg volatility: {stats['average_volatility']:.2f}\n
Avg effective: {stats['average_effective_proficiency']:.2f}"""
    st.sidebar.write(f"""### {f"Vocabulary Statistics: {target_language}"}\n
Seen vocabulary: {stats['total_seen_words']:.0f} words\n
Tracked vocabulary: {stats['processed_words']:.0f} words\n
{knowledge_stats}\n
{f"Well known: {stats['well_known']:.0f} words"}\n
{f"Familiar: {stats['familiar']:.0f} words"}\n
{f"Still learning: {stats['learning']:.0f} words"}\n
{f"Stable knowledge: {stats['stable']:.0f} words"}\n
{f"Semi-stable: {stats['semi_stable']:.0f} words"}\n
{f"Volatile: {stats['volatile']:.0f} words"}\n
{proficiency_stats}""")

# -------------------------------------------------------------------
#  QUEUE VIEW 
# -------------------------------------------------------------------
def queue_view(model_service, lang):
    st.subheader("Study Queue")

    if "learning_queue_obj" not in st.session_state or st.session_state.learning_queue_target != lang:
        with st.spinner("Building study queue from database..."):
            # Get all non-reviewed items for the language from the DB
            query = """
            SELECT t.filename, t.chapter, t.idx, t.source, t.target, t.words_json
            FROM texts t
            LEFT JOIN user_progress p ON t.lang = p.lang AND t.filename = p.filename AND t.chapter = p.chapter AND t.idx = p.text_idx
            WHERE t.lang = ? AND (p.is_reviewed IS NULL OR p.is_reviewed = 0) AND t.words_json != '[]'
            """
            with get_db_conn() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(query, (lang,))
                all_item_records = [dict(row) for row in cur.fetchall()]

            if not all_item_records:
                st.info("No items to study. All content has been reviewed!")
                return

            # Prepare data for the LearningQueue
            only_words = [orjson.loads(rec['words_json']) for rec in all_item_records]
            # Map internal queue ID (iid) to our database record
            iid_to_item_map = {i: rec for i, rec in enumerate(all_item_records)}

            model_instance = model_service.get_or_create_model(lang)
            q = LearningQueue(model_instance)
            q.build_from_input(only_words)
            
            st.session_state.update({
                "learning_queue_obj": q,
                "learning_queue_target": lang,
                "iid_to_item_map": iid_to_item_map,
                "queue_source_revealed": False,
            })
    
    display_vocabulary_stats(model_service, lang)

    queue: LearningQueue = st.session_state.learning_queue_obj
    total = queue.size()
    if not total:
        st.info("Queue empty.  Click 'Back to Text' to add new material.")
        return
    
    q0_size = queue.size(0)
    q1_size = queue.size(1)
    q2_size = queue.size(2)

    st.info(f'Queue contains {total} items. \n\n"Didn\'t understand" ({q0_size}), "Partially understood" ({q1_size}), "Fully understood" ({q2_size}).')

    current_iid = queue.peek_next()
    if current_iid is None:
        st.warning("Queue is empty or contains only invalid items.")
        return
    
    current_item = st.session_state.iid_to_item_map.get(current_iid)
    if not current_item:
        st.error(f"Could not find item for iid {current_iid}. Rebuilding might be necessary.")
        return

    st.markdown("### Current Unit")
    if st.button(current_item["target"], key="unit_target_btn"):
        st.session_state.queue_source_revealed = True
        st.rerun()

    if st.session_state.queue_source_revealed:
        st.info(current_item["source"])
        
        def _commit_queue(level):
            # Update learning queue
            st.session_state.learning_queue_obj.process_answer(current_iid, level)
            
            # Mark as reviewed in the database
            save_progress_for_sentence(
                lang, current_item['filename'], current_item['idx'],
                is_target=1, is_revealed=1, is_reviewed=1
            )
            
            st.session_state.queue_source_revealed = False
            st.rerun()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("❌😔❌", use_container_width=True):
                _commit_queue(0)
        with c2:
            if st.button("🔶🤔🔶", use_container_width=True):
                _commit_queue(1)
        with c3:
            if st.button("✅🧐✅", use_container_width=True):
                _commit_queue(2)

def build_display_blocks(aligned_text):
    """Batches consecutive non-target sentences for optimized rendering."""
    if not aligned_text:
        return []

    display_blocks = []
    source_batch = []

    for unit in aligned_text:
        if not unit['is_target']:
            source_batch.append(unit['source'])
        else:
            if source_batch:
                display_blocks.append({'type': 'batch', 'content': "\n\n".join(source_batch)})
                source_batch = []
            display_blocks.append({'type': 'unit', 'data': unit})
    
    if source_batch:
        display_blocks.append({'type': 'batch', 'content': "\n\n".join(source_batch)})

    return display_blocks

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

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        lang_choices = get_available_languages()
        target_language = st.selectbox("Target language", lang_choices).lower()
        
        if st.button("Add/Refresh Files"):
            with st.spinner(f"Processing files for {target_language}..."):
                prepare(target_language)
            st.cache_data.clear()
            st.rerun()
        
        st.header("Vocabulary")
        display_vocabulary_stats(model, target_language)

        st.header("Reset")
        reset_choice = st.radio("Reset vocabulary for:", ["Current language", "All languages"], key="reset_radio")
        if st.button("Reset Vocabulary"):
            msg = model.reset_vocabulary(target_language if reset_choice == "Current language" else None)
            st.sidebar.success(f"{msg}!")
            st.rerun()

    # --- View Toggle ---
    if "show_queue_view" not in st.session_state:
        st.session_state.show_queue_view = False
    if st.button("Study Queue" if not st.session_state.show_queue_view else "Back to Text"):
        st.session_state.show_queue_view = not st.session_state.show_queue_view
        # Clean up queue state when switching away
        if not st.session_state.show_queue_view:
            for k in ["learning_queue_obj", "learning_queue_target", "iid_to_item_map"]:
                st.session_state.pop(k, None)
        st.rerun()

    if st.session_state.show_queue_view:
        queue_view(model, target_language)
        return

    # --- TEXT VIEW ---
    file_choices = get_files_for_language(target_language)
    if not file_choices:
        st.warning(f"No text files found for '{target_language}'. Add some via the sidebar.")
        return

    # --- File and Chapter Selection (Preserving original UI) ---
    colA, colB = st.columns([5, 2])
    with colA:
        selected_file = st.selectbox("Select a file to read", file_choices, key=f"file_sel_{target_language}")
    
    with colB:
        chapters = get_chapters_for_file(target_language, selected_file)
        if "current_num_index" not in st.session_state:
            st.session_state.current_num_index = 0
        
        colL, colM, colR = st.columns([1, 3, 1], vertical_alignment="bottom")
        with colL:
            st.button("◀", on_click=partial(prev_number, chapters), use_container_width=True)
        with colM:
            # This logic ensures the index is valid if the list of chapters changes
            if st.session_state.current_num_index >= len(chapters):
                st.session_state.current_num_index = 0
            
            selected_chapter_index = st.selectbox(
                "Select number", range(len(chapters)),
                index=st.session_state.current_num_index,
                format_func=lambda i: chapters[i],
                on_change=lambda: st.session_state.__setitem__("force_state_reset", True),
                key="number-select-box"
            )
            st.session_state.current_num_index = selected_chapter_index
        with colR:
            st.button("▶", on_click=partial(next_number, chapters), use_container_width=True)
    
    selected_chapter = chapters[st.session_state.current_num_index]

    # --- State Reset and Data Loading Logic ---
    if (st.session_state.get("current_file") != selected_file or 
        st.session_state.get("current_chapter") != selected_chapter or 
        st.session_state.get("force_state_reset", False)):
        
        with st.spinner(f"Loading '{selected_file}' Chapter {selected_chapter}..."):
            # Clear old state
            for k in list(st.session_state.keys()):
                if k.startswith(("visible_", "feedback_")):
                    del st.session_state[k]

            # Load fresh data from DB
            aligned_text = load_text_and_progress(target_language, selected_file, selected_chapter)
            
            # Find natural candidates and update DB
            all_words = [u['words'] for u in aligned_text]
            current_targets = {u['idx'] for u in aligned_text if u['is_target']}
            natural_candidates = get_natural_candidates(model.get_or_create_model(target_language), all_words, current_targets)
            
            if natural_candidates:
                save_multiple_as_target(target_language, selected_file, selected_chapter, natural_candidates)
                # Reload data to include newly marked targets
                aligned_text = load_text_and_progress(target_language, selected_file, selected_chapter)

            # Set session state for the new view
            st.session_state.update({
                "current_file": selected_file,
                "current_chapter": selected_chapter,
                "aligned_text": aligned_text,
                "display_blocks": build_display_blocks(aligned_text),
                "to_replace_indices": {u['idx'] for u in aligned_text if u['is_target']},
                "revealed": {u['idx'] for u in aligned_text if u['is_revealed']},
                "reviewed": {u['idx'] for u in aligned_text if u['is_reviewed']},
                "visible": {},
                "force_state_reset": False
            })
        st.rerun()

    # --- Display Stats ---
    total_units = len(st.session_state.aligned_text)
    transformed = len(st.session_state.to_replace_indices)
    revealed_cnt = len(st.session_state.revealed.intersection(st.session_state.to_replace_indices))
    st.info(f"Transformed: {transformed}/{total_units} | Revealed: {revealed_cnt}/{transformed}")

    # --- Reading Text (Optimized Rendering) ---
    st.subheader(f"Reading: {selected_file} - Chapter {selected_chapter}")
    st.markdown("""<style>.reviewed{color:#28a745;font-weight:bold;} .unrevealed{color:#007bff;} .revealed{color:#fd7e14;font-style:italic;}</style>""", unsafe_allow_html=True)
    
    for block in st.session_state.get("display_blocks", []):
        if block['type'] == 'batch':
            st.write(block['content'])
        elif block['type'] == 'unit':
            unit = block['data']
            idx = unit['idx']
            
            is_rev = idx in st.session_state.revealed
            is_done = idx in st.session_state.reviewed
            show_src = st.session_state.visible.get(idx, False)

            prefix, css = ("✓ ", "reviewed") if is_done else ("👁️ ", "revealed") if is_rev else ("🔍 ", "unrevealed")

            if not is_rev:
                if st.button(f"{prefix}{unit['target']}", key=f"sent_{idx}"):
                    st.session_state.revealed.add(idx)
                    st.session_state.visible[idx] = True
                    save_progress_for_sentence(target_language, selected_file, selected_chapter, idx, 1, 1, 0)
                    st.rerun()
            else:
                if show_src:
                    st.markdown(f"<div class='{css}'>{prefix}{unit['target']}</div>", unsafe_allow_html=True)
                    st.info(unit["source"])
                    if st.button("Hide source", key=f"hide_{idx}"):
                        st.session_state.visible[idx] = False
                        st.rerun()
                else:
                    if st.button(f"{prefix}{unit['target']}", key=f"show_{idx}"):
                        st.session_state.visible[idx] = True
                        st.rerun()
                
                if show_src and not is_done:
                    def _commit(level, u=unit):
                        model.update_knowledge(u["words"], target_language, level)
                        st.session_state.reviewed.add(u['idx'])
                        save_progress_for_sentence(target_language, selected_file, selected_chapter, u['idx'], 1, 1, 1)
                        st.rerun()

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("❌😔❌", key=f"fb1_{idx}", use_container_width=True):
                            _commit(0)
                    with c2:
                        if st.button("🔶🤔🔶", key=f"fb2_{idx}", use_container_width=True):
                            _commit(1)
                    with c3:
                        if st.button("✅🧐✅", key=f"fb3_{idx}", use_container_width=True):
                            _commit(2)
                    st.divider()

# -------------------------------------------------------------------
#  BOOTSTRAP
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()

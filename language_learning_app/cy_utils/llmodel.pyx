# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
#distutils: language = c++

from cy_utils.vocab_model cimport VocabularyModel, promotion_times
cimport numpy as cnp
from collections import defaultdict
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string
from cython.operator cimport dereference as deref

# Tell Cython about our numpy dtypes
cnp.import_array()

# ADAPTED: predict_answer_for_queue now returns int group
cpdef tuple predict_answer_for_queue(VocabularyModel model,
                                     list words_py,
                                     unordered_map[string, int] &words_map_v_cpp,
                                     unordered_map[string, unordered_set[int]] &words_map_i_cpp,
                                     unordered_map[int, float] &sent_map_cpp,
                                     int iid):
    """
    Returns int group (0/1/2).
    """
    cdef Py_ssize_t original_len = len(words_py)
    # dedupe & preserve order
    cdef set seen = set()
    cdef list words = []
    for w in words_py:
        if w not in seen:
            seen.add(w)
            words.append(w)

    cdef Py_ssize_t n = len(words)
    cdef float thr = 0.3
    cdef Py_ssize_t half = n // 2
    cdef Py_ssize_t familiar

    # 1) effective profs
    cdef float[::1] effs = model.get_effective_proficiency(words)
    cdef Py_ssize_t i

    # collect indices of low‐eff words
    unknown_idx = []
    for i in range(n):
        if effs[i] <= thr:
            unknown_idx.append(i)

    unknown_cnt = len(unknown_idx)
    familiar = n - unknown_cnt

    cdef string w_cpp
    cdef unordered_map[string, int].iterator words_map_v_it
    cdef float diff

    if unknown_cnt > half:
        diff = <float>(unknown_cnt - half)
        sent_map_cpp[iid] = original_len / diff

        for i in unknown_idx:
            w = words[i]
            w_cpp = w.encode('utf-8')
                
            words_map_v_it = words_map_v_cpp.find(w_cpp)
                
            if words_map_v_it == words_map_v_cpp.end():
                # fetch or fallback
                if model.idx.has_word(w):
                    wid = model.idx.get_id(w)
                    b = model.prof[wid]
                    v = model.vol[wid]
                else:
                    b = 0.1
                    v = 0.9
                
                words_map_v_cpp[w_cpp] = promotion_times(b, v)
                words_map_i_cpp[w_cpp].insert(iid)
            else:
                words_map_i_cpp[w_cpp].insert(iid)

    # return (group, effs) as (int, ndarray)
    if familiar < half:
        return (0, effs) # Group 0 (was 0.0)
    elif familiar < n:
        return (1, effs) # Group 1 (was 0.5)
    else:
        return (2, effs) # Group 2 (was 1.0)

# NEW: predict_answer_for_natural_candidates
cpdef int predict_answer_for_natural_candidates(VocabularyModel model,
                                                list words_py):
    """
    Returns int group (0/1/2) without building promotion data.
    """
    cdef Py_ssize_t n = len(words_py)
    cdef float thr = 0.3
    cdef Py_ssize_t half = n // 2
    cdef Py_ssize_t familiar = 0

    cdef float[::1] effs = model.get_effective_proficiency(words_py)
    cdef Py_ssize_t i

    for i in range(n):
        if effs[i] > thr:
            familiar += 1

    if familiar < half:
        return 0 # Group 0 (was 0.0)
    elif familiar < n:
        return 1 # Group 1 (was 0.5)
    else:
        return 2 # Group 2 (was 1.0)


cpdef set get_natural_candidates(list aligned_text,
                                 set current_indices,
                                 VocabularyModel model):
    """
    aligned_text: list of dicts with key "words":List[str]
    current_indices: set of ints
    """
    cdef set natural = set()
    cdef Py_ssize_t L = len(aligned_text)
    cdef Py_ssize_t i
    cdef dict unit
    cdef list words
    cdef int ans # Now an integer
    cdef int len_words = 5

    for i in range(L):
        if i in current_indices:
            continue
        unit = aligned_text[i]
        words = unit.get("words", [])
        if not words:
            continue
        # Call the new specialized function
        ans = predict_answer_for_natural_candidates(model, words)
        # Check against integer group values
        if ans >= 1 or len(words) < len_words: # ans >= 1 means groups 0.5 (1) or 1.0 (2)
            natural.add(i)
    return natural

cpdef float calculate_unknownness(float[::1] effs):
    """
    Return len(effs) - sum(effs), i.e. total ‘unknown mass’.
    """
    cdef Py_ssize_t n = len(effs)
    cdef double s = 0.0
    cdef Py_ssize_t i
    for i in range(n):
        s += effs[i]
    return n - s


cpdef dict get_vocabulary_statistics(VocabularyModel model):
    """
    Mirror of your Python version’s stats.
    """
    cdef dict stats = {}
    cdef Py_ssize_t total = model.idx.size()

    stats["total_words"]               = total
    stats["all_possible_knowledge"]    = total * 0.97
    stats["all_known_knowledge"]       = float((<cnp.ndarray>model.eff_prof).sum())
    stats["well_known"]                = int((model.eff_prof > 0.7).sum())
    stats["familiar"]                  = int(((model.eff_prof >= 0.3) & (model.eff_prof <= 0.7)).sum())
    stats["learning"]                  = int(((model.eff_prof < 0.3) & (model.eff_prof > 0)).sum())
    stats["stable"]                    = int((model.vol < 0.3).sum())
    stats["semi_stable"]               = int(((model.vol >= 0.3) & (model.vol <= 0.6)).sum())
    stats["volatile"]                  = int(((model.vol > 0.6) & (model.eff_prof > 0)).sum())
    
    if (model.eff_prof > 0).sum() > 0:
        stats["average_effective_proficiency"] = float(model.eff_prof[model.eff_prof > 0].mean())
    else:
        stats["average_effective_proficiency"] = 0.0
    
    if (model.prof > model.proficiency_min).sum() > 0:
        stats["average_proficiency"]           = float(model.prof[model.prof > model.proficiency_min].mean())
        stats["average_volatility"]            = float(model.vol[model.prof > model.proficiency_min].mean())
    else:
        stats["average_proficiency"]           = 0.0
        stats["average_volatility"]            = 0.0

    return stats

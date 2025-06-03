# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from cy_utils.vocab_model cimport VocabularyModel, promotion_times
cimport numpy as np
from collections import defaultdict

# Tell Cython about our numpy dtypes
np.import_array()

cpdef tuple predict_answer(VocabularyModel model,
                            list words_py,
                            bint build_promotion_data,
                            object promotion_data):
    """
    If build_promotion_data=False, returns float group (0.0/0.5/1.0).
    If build_promotion_data=True, returns (group:float, effs:ndarray).
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
    cdef Py_ssize_t familiar = 0

    # 1) effective profs
    cdef float[::1] effs = model.get_effective_proficiency(words)
    cdef Py_ssize_t i

    if build_promotion_data:
        # collect indices of low‐eff words
        unknown_idx = []
        for i in range(n):
            if effs[i] <= thr:
                unknown_idx.append(i)
            else:
                familiar += 1

        unknown_cnt = len(unknown_idx)
        familiar = n - unknown_cnt

        if unknown_cnt > half:
            words_map, sent_map, iid = promotion_data
            diff = unknown_cnt - half
            sent_map[iid] = original_len / diff

            for i in unknown_idx:
                w = words[i]
                if w not in words_map:
                    # fetch or fallback
                    if w in model.idx._w2i:
                        wid = model.idx._w2i[w]
                        b = model.prof[wid]
                        v = model.vol[wid]
                    else:
                        b = 0.1
                        v = 0.9
                    words_map[w] = {
                        "v": promotion_times(b, v),
                        "i": {iid},
                    }
                else:
                    words_map[w]["i"].add(iid)

        # return (group, effs)
        if familiar < half:
            return (0.0, effs)
        elif familiar < n:
            return (0.5, effs)
        else:
            return (1.0, effs)

    # --- simple‐mode: just count familiar words ---
    for i in range(n):
        if effs[i] > thr:
            familiar += 1

    if familiar < half:
        return (0.0,)
    elif familiar < n:
        return (0.5,)
    else:
        return (1.0,)


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
    cdef float ans
    cdef float thr = 0.5
    cdef int len_words = 5
    for i in range(L):
        if i in current_indices:
            continue
        unit = aligned_text[i]
        words = unit.get("words", [])
        if not words:
            continue
        # predict_answer in simple mode → float
        ans = predict_answer(model, words, False, None)[0]
        if ans >= thr or len(words) < len_words:
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
    cdef Py_ssize_t total = model.idx.size

    stats["total_words"]               = total
    stats["all_possible_knowledge"]    = total * 0.97
    stats["all_known_knowledge"]       = float(model.eff_prof.sum())
    stats["well_known"]                = int((model.eff_prof > 0.7).sum())
    stats["familiar"]                  = int(((model.eff_prof >= 0.3) & (model.eff_prof <= 0.7)).sum())
    stats["learning"]                  = int(((model.eff_prof < 0.3) & (model.eff_prof > 0)).sum())
    stats["stable"]                    = int((model.vol < 0.3).sum())
    stats["semi_stable"]               = int(((model.vol >= 0.3) & (model.vol <= 0.6)).sum())
    stats["volatile"]                  = int(((model.vol > 0.6) & (model.eff_prof > 0)).sum())
    stats["average_effective_proficiency"] = float(model.eff_prof[model.eff_prof > 0].mean())
    stats["average_proficiency"]           = float(model.prof[model.prof > model.proficiency_min].mean())
    stats["average_volatility"]            = float(model.vol[model.prof > model.proficiency_min].mean())

    return stats
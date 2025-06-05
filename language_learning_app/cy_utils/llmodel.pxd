# vocab_predictor.pxd

# C-level import for NumPy declarations
cimport numpy as np

# Import specific C-level declarations from the 'cy_utils.vocab_model' Cython module.
from cy_utils.vocab_model cimport VocabularyModel, promotion_times

# NEW: Imports for C++ containers
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string

# ADAPTED: predict_answer_for_queue now returns int group
cpdef tuple predict_answer_for_queue(VocabularyModel model,
                                     list words_py,
                                     unordered_map[string, int] &words_map_v_cpp, # Reference to C++ map for 'v'
                                     unordered_map[string, unordered_set[int]] &words_map_i_cpp, # Reference to C++ map for 'i'
                                     unordered_map[int, float] &sent_map_cpp, # Reference to C++ map for sentence map
                                     int iid) # Item ID now passed directly
"""
returns (group:int, effs:ndarray).
"""

# NEW: predict_answer_for_natural_candidates, returns int group
cpdef int predict_answer_for_natural_candidates(VocabularyModel model,
                                                 list words_py)
"""
Returns int group (0/1/2) without building promotion data.
"""

cpdef set get_natural_candidates(list aligned_text,
                                 set current_indices,
                                 VocabularyModel model)
"""
aligned_text: list of dicts with key "words":List[str]
current_indices: set of ints
"""

# Declares a C-contiguous 1D NumPy array of floats as the input type for 'effs'.
cpdef float calculate_unknownness(float[::1] effs)
"""
Return len(effs) - sum(effs), i.e. total ‘unknown mass’.
"""

cpdef dict get_vocabulary_statistics(VocabularyModel model)
"""
Mirror of your Python version’s stats.
"""

# vocab_predictor.pxd

# C-level import for NumPy declarations
cimport numpy as np

# Import specific C-level declarations from the 'cy_utils.vocab_model' Cython module.
# This assumes 'cy_utils.vocab_model' is itself a Cython module with a .pxd file
# that defines 'VocabularyModel' as a cdef class and 'promotion_times' as a cpdef function.
from cy_utils.vocab_model cimport VocabularyModel, promotion_times

# Declare the 'cpdef' functions exposed by this Cython module.
# These functions are accessible from both Python and other Cython modules at the C-level.

cpdef tuple predict_answer(VocabularyModel model,
                            list words_py,
                            bint build_promotion_data,
                            object promotion_data)
"""
If build_promotion_data=False, returns float group (0.0/0.5/1.0).
If build_promotion_data=True, returns (group:float, effs:ndarray).
"""

cpdef set get_natural_candidates(list aligned_text,
                                 set current_indices,
                                 VocabularyModel model)
"""
aligned_text: list of dicts with key "words":List[str]
current_indices: set of ints
"""

# Declares a C-contiguous 1D NumPy array of floats as the input type for 'effs'.
# This corresponds to the 'float[::1]' memoryview used in the .pyx file.
cpdef float calculate_unknownness(float[::1] effs)
"""
Return len(effs) - sum(effs), i.e. total ‘unknown mass’.
"""

cpdef dict get_vocabulary_statistics(VocabularyModel model)
"""
Mirror of your Python version’s stats.
"""

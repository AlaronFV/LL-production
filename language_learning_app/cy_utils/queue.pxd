# cy_utils/queue.pxd

# C-level import for NumPy declarations
cimport numpy as np

# C-level imports for custom Cython modules
# Assuming 'cy_utils.vocab_model' defines 'VocabularyModel' as a cdef class.
from cy_utils.vocab_model cimport VocabularyModel
# Assuming 'cy_utils.llmodel' defines 'predict_answer' and 'calculate_unknownness' as cpdef functions.
from cy_utils.llmodel cimport predict_answer, calculate_unknownness

# Declare the cdef class 'LearningQueue'.
# All cdef attributes and cpdef methods are declared here.
cdef class LearningQueue:
    """
    An incremental‐learning queue of sentence‐units.
    """

    # Declare cdef attributes with their Cython types.
    # Python objects that don't have specific C-level Cython types are declared as 'object'.
    cdef object master             # LanguageLearningModel (pure‐Python)
    cdef VocabularyModel tmodel   # Explicitly typed as VocabularyModel
    cdef str lang

    cdef dict heaps                # group(float) → heapdict (heapdict is a Python object)
    cdef dict items                # iid(int) → item(dict)
    cdef dict i2g                  # iid(int) → group(float)
    cdef dict words_map            # word(str) → {"v":int, "i": set(iid)}
    cdef dict sent_map             # iid(int) → float
    cdef object inverted          # word(str) → set(iid) (defaultdict is a Python object)
    cdef object counter           # itertools.count() (itertools.count is a Python object)

    # Declare cpdef methods with their Cython signatures (arguments and return types).
    # The __init__ method is typically not declared in .pxd files for cdef classes.

    cpdef void add_item(self,
                        object item,
                        int iid,
                        object promo_data,
                        double now_h)

    cpdef tuple build_from_input(self, list items)

    cpdef tuple peek_next(self, float grp)

    cpdef tuple pop_next(self, float grp)

    cpdef void remove_item(self, int iid)

    cpdef void process_answer(self, int iid, int feedback_level)

    cpdef void update_item(self,
                           int iid,
                           object promo_data,
                           double now_h)

    cpdef Py_ssize_t size(self, float grp=*)

    # Private cpdef methods are also declared.
    cpdef tuple _score_and_group(self,
                                 object item,
                                 object promo_data,
                                 double now_h)

    cpdef void _add_to_heap(self, int iid, float grp, float key)

    # The 'eff_prof' argument is declared as a C-contiguous 1D NumPy array of floats,
    # matching its usage as 'float[::1]' in the .pyx file.
    cpdef float _promotion_potential(self, list words, float[::1] eff_prof, double now_h)

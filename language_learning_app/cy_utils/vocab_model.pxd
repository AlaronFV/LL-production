# cy_utils/vocab_model.pxd
# distutils: language = c++
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, int32_t
import numpy as np
cimport numpy as cnp # Use cnp for C-level NumPy types
from libcpp.algorithm cimport sort # For sorting word_ids if needed

# ---------------------------------------------------------------------------
#   utility: promotion_times
# ---------------------------------------------------------------------------
cpdef int promotion_times(float b, float v)

# ---------------------------------------------------------------------------
#   WordIndex: bidirectional str <-> uint32
# ---------------------------------------------------------------------------
cdef class _WordIndex:
    cdef public dict _w2i
    cdef public list _i2w

    cpdef uint32_t get_id(self, str w) # This will still add words if called
    cpdef bint has_word(self, str w)
    cpdef str get_word(self, uint32_t idx)
    # No explicit size property declaration here, it's a Python property.

# ---------------------------------------------------------------------------
#   VocabularyModel: per-word arrays + traces + activation & decay
# ---------------------------------------------------------------------------
cdef class VocabularyModel:
    # 1) index
    cdef public _WordIndex         idx

    # 2) per-word NumPy arrays (declared as object as in .pyx, but Cython knows their actual type)
    cdef public object prof, vol, eff_prof, encounters, last_decayed

    # 3) traces and reverse index (NOW C++ VECTORS)
    cdef vector[vector[uint32_t]] _trace_word_ids # Store all word_ids for all traces
    cdef vector[double] _trace_timestamps_h      # timestamp_h for each trace
    cdef vector[float] _trace_activations        # activation for each trace
    cdef vector[float] _trace_decay_factors      # decay_factor for each trace

    # Reverse index: word_id -> list of trace indices (int)
    cdef public dict word_to_traces # Maps int (word_id) to Python list of int (trace indices)

    # 4) parameters
    cdef float learning_rate
    cdef float context_influence
    cdef float activation_threshold
    cdef float base_decay_rate
    cdef float proficiency_min
    cdef float proficiency_max
    cdef double min_elapsed_h # NEW: Minimum elapsed time for decay to apply

    # 5) propagation params
    cdef float propagation_threshold
    cdef float trace_delete_threshold

    # New: for optimized decay scheduling
    cdef public double _next_word_decay_appointment_h # Next time to run full word decay
    cdef public double _next_trace_decay_appointment_h # Next time to run full trace decay


    # cdef methods
    cdef void _resize_arrays(self, Py_ssize_t new_n)
    cdef set _ensure_words(self, list words)
    cdef float _eff_prof_formula(self, float p, float v) nogil

    # cpdef methods
    cpdef float[::1] get_effective_proficiency(self, object words)

    # 1) Word-level decay & activation
    cpdef float get_word_activation(self, uint32_t word_id, double now_h)
    cdef void apply_decay_to_word_id(self, uint32_t wid, double now_h)
    cpdef void apply_decay_to_all_words(self, double now_h) # Logic will change internally
    cdef void apply_decay_to_all_traces(self, double now_h) # New method for traces

    # 2) Full sentence-trace propagation
    cdef void _propagate(self, int source_trace_idx) # Source is now an index
    cdef void _prune_traces(self)

    # These methods directly manipulate the C++ trace data
    cpdef int add_trace_cpp(self, set word_ids_py, double timestamp_h, float activation, float decay_factor)
    cpdef float decay_trace_activation_cpp(self, int trace_idx, double now_h)

    # 3) Context Support
    cpdef float calculate_context_support(self, object id_words, Py_ssize_t n)

    # 4) Predict Understanding
    cpdef float predict_understanding(self, object words, double current_time_h=*)

    # 5) Update Proficiency + add & propagate new trace
    cpdef void update_proficiency(self, object words, float u_val, double now_h)

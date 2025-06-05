# cy_utils/vocab_model.pxd
# distutils: language = c++
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, int32_t
import numpy as np
cimport numpy as cnp # Use cnp for C-level NumPy types
from libcpp.algorithm cimport sort # For sorting word_ids if needed
from libcpp.unordered_map cimport unordered_map # NEW: For C++ unordered_map
from libcpp.string cimport string # NEW: For C++ strings
from libcpp.queue cimport priority_queue # NEW: For C++ priority_queue
from libcpp.functional cimport greater # NEW: For priority_queue comparator

# ---------------------------------------------------------------------------
#   utility: promotion_times
# ---------------------------------------------------------------------------
cpdef int promotion_times(float b, float v)

# NEW: Structure for priority queue items (decay scheduling)
cdef extern from "<utility>" namespace "std":
    cdef cppclass pair[T1, T2]:
        T1 first
        T2 second

# NEW: Custom comparator for decay scheduling priority queue
cdef cppclass DecayItem:
    double next_decay_time
    uint32_t id_or_idx # Can be word_id or trace_idx

    bint operator<(const DecayItem& other) const:
        return next_decay_time < other.next_decay_time # Min-heap based on time

# ---------------------------------------------------------------------------
#   WordIndex: bidirectional str <-> uint32 (ADAPTED for C++ internals)
# ---------------------------------------------------------------------------
cdef class _WordIndex:
    # Internal C++ maps for efficiency
    cdef public unordered_map[string, uint32_t] _w2i_cpp
    cdef public vector[string] _i2w_cpp

    # Python-facing methods will handle str <-> string conversion
    cpdef uint32_t get_id(self, str w)
    cpdef bint has_word(self, str w)
    cpdef str get_word(self, uint32_t idx)
    cpdef Py_ssize_t size(self) # Expose size as a method for consistency

# ---------------------------------------------------------------------------
#   VocabularyModel: per-word arrays + traces + activation & decay
# ---------------------------------------------------------------------------
cdef class VocabularyModel:
    # 1) index
    cdef public _WordIndex         idx

    # 2) per-word NumPy arrays (declared as object as in .pyx, but Cython knows their actual type)
    cdef public object prof, vol, eff_prof, encounters

    # C++ vector for last_decayed timestamps for words (already done)
    cdef vector[double] _word_last_decay_h

    # 3) traces and reverse index (C++ VECTORS and UNORDERED_MAP)
    cdef vector[vector[uint32_t]] _trace_word_ids # Store all word_ids for all traces
    cdef vector[double] _trace_timestamps_h      # timestamp_h for each trace
    cdef vector[float] _trace_activations        # activation for each trace
    cdef vector[float] _trace_decay_factors      # decay_factor for each trace

    # Reverse index: word_id -> list of trace indices (int) - C++ UNORDERED_MAP (already done)
    cdef public unordered_map[uint32_t, vector[int]] word_to_traces

    # 4) parameters
    cdef float learning_rate
    cdef float context_influence
    cdef float activation_threshold
    cdef float base_decay_rate
    cdef float proficiency_min
    cdef float proficiency_max
    cdef double min_elapsed_h # Minimum elapsed time for decay to apply

    # 5) propagation params
    cdef float propagation_threshold
    cdef float trace_delete_threshold

    # NEW: Priority queues for intelligent decay scheduling
    cdef priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]] _word_decay_pq
    cdef priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]] _trace_decay_pq


    # cdef methods
    cdef void _resize_arrays(self, Py_ssize_t new_n)
    cdef set _ensure_words(self, list words)
    cdef float _eff_prof_formula(self, float p, float v) nogil

    # cpdef methods
    cpdef float[::1] get_effective_proficiency(self, object words)

    # 1) Word-level decay & activation
    cpdef float get_word_activation(self, uint32_t word_id, double now_h)
    cdef void apply_decay_to_word_id(self, uint32_t wid, double now_h)
    # ADAPTED: Now processes only due items from PQ
    cpdef void _process_due_word_decays(self, double now_h)
    # ADAPTED: Now processes only due items from PQ
    cpdef void _process_due_trace_decays(self, double now_h)

    # 2) Full sentence-trace propagation
    cdef void _propagate(self, int source_trace_idx) # Source is now an index
    # ADAPTED: For incremental pruning
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

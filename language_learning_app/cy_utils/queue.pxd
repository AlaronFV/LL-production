# cy_utils/queue.pxd
# distutils: language = c++

# C-level import for NumPy declarations
cimport numpy as np

# C-level imports for custom Cython modules
from cy_utils.vocab_model cimport VocabularyModel
from cy_utils.llmodel cimport predict_answer_for_queue, calculate_unknownness

from libc.stdint cimport uint32_t

# NEW: C++ containers and threading primitives
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.queue cimport priority_queue
from libcpp.functional cimport greater
from libcpp.utility cimport pair

# Threading specific imports
from libcpp.thread cimport thread
from libcpp.mutex cimport mutex
from libcpp.condition_variable cimport condition_variable

# Define HeapItem in a separate header and declare it here
cdef extern from "cy_utils/heap_item.h":
    cdef cppclass HeapItem:
        float key
        long long insertion_order
        int iid
        bint operator<(const HeapItem& other) const

# Declare HeapComparator from its own C++ header
# The actual implementation of operator() is in cy_utils/heap_comparator.h
cdef extern from "cy_utils/heap_comparator.h":
    cdef cppclass HeapComparator:
        # No need to redeclare operator() here; just the class itself
        pass

# Declare a specific Cython-visible type for std::priority_queue<HeapItem, std::vector<HeapItem>, HeapComparator>
# This avoids the "priority_queue templated type receives 1 arguments, got 3" error
cdef extern from "<queue>" namespace "std":
    # Use the full C++ qualified name for the specialized type
    cdef cppclass priority_queue_HeapItem "std::priority_queue<HeapItem, std::vector<HeapItem>, HeapComparator>":
        # No members are needed here; this is just a type alias.
        pass

# Declare the cdef class 'LearningQueue'.
cdef class LearningQueue:
    """
    An incremental‐learning queue of sentence‐units.
    """

    cdef object master
    cdef VocabularyModel tmodel
    cdef str lang

    # ADAPTED: Heaps use int keys for groups
    # Use the specialized type defined above
    cdef unordered_map[int, priority_queue_HeapItem] _heaps_cpp

    cdef dict items
    cdef dict i2g                         # ADAPTED: i2g maps iid(int) → group(int)

    cdef unordered_map[string, int] _words_map_v_cpp
    cdef unordered_map[string, unordered_set[int]] _words_map_i_cpp
    cdef unordered_map[int, float] _sent_map_cpp

    cdef unordered_map[uint32_t, vector[int]] inverted_cpp
    
    cdef object counter
    
    cdef deque[int] _dirty_items

    # NEW: Threading related members
    cdef thread *_worker_thread # Pointer to C++ thread object
    cdef mutex _dirty_queue_mutex # Mutex to protect _dirty_items and related flags
    cdef condition_variable _dirty_queue_cv # Condition variable to signal worker
    cdef bint _worker_running # Flag to indicate if worker thread should run
    cdef bint _worker_paused # Flag to indicate if worker thread should pause
    cdef bint _is_processing_dirty # Flag to indicate if worker is actively processing dirty items

    # New cdef methods for thread management
    cdef void _start_worker_thread(self)
    cdef void _stop_worker_thread(self)
    # The 'self' argument is crucial for C++ thread function pointers to call methods on the object
    cdef void _background_dirty_processor_thread_func(self, LearningQueue self)
    cdef void _signal_worker_pause(self) # For main thread to signal pause
    cdef void _signal_worker_resume(self) # For main thread to signal resume
    cdef void _wait_for_worker_to_pause(self) # For main thread to wait for worker to pause

    cpdef void add_item(self,
                        object item,
                        int iid,
                        double now_h)

    cpdef void build_from_input(self, list items)

    cpdef tuple peek_next(self, int grp)

    cpdef tuple pop_next(self, int grp)

    cpdef void remove_item(self, int iid)

    cpdef void process_answer(self, int iid, int feedback_level)

    cpdef void update_item(self,
                           int iid,
                           double now_h)

    cpdef Py_ssize_t size(self, int grp=*)

    cpdef tuple _score_and_group(self,
                                 object item,
                                 int iid, # Add iid to signature
                                 double now_h)

    cpdef void _add_to_heap(self, int iid, int grp, float key)

    cpdef float _promotion_potential(self, list words, float[::1] eff_prof, double now_h)

    # _process_dirty_items is now internal to the background thread, no longer cpdef
    # cpdef void _process_dirty_items(self, int max_items_to_process, double now_h)
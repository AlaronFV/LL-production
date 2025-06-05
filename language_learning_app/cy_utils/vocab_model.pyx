# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
#distutils: language = c++

import struct
from libc.math cimport exp, sqrt, fmax, fmin
from libc.stdint cimport uint32_t, int32_t
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libcpp.algorithm cimport sort
from cpython.bytes cimport PyBytes_FromStringAndSize
from libcpp.unordered_map cimport unordered_map # NEW
from libcpp.string cimport string # NEW
from libcpp.queue cimport priority_queue # NEW
from libcpp.functional cimport greater # NEW
from libcpp.utility cimport pair # NEW: For std::pair

import numpy as np
cimport numpy as cnp

from collections import deque # Keep for BFS in _propagate
from datetime import datetime

# Import the declarations from your .pxd file
from cy_utils.vocab_model cimport _WordIndex, VocabularyModel, promotion_times, DecayItem

# Helper to find intersection size of two sorted vectors (like sets)
cdef inline Py_ssize_t _intersection_size_vec_vec(vector[uint32_t] &vec1, vector[uint32_t] &vec2):
    cdef Py_ssize_t count = 0
    cdef vector[uint32_t].iterator it1 = vec1.begin()
    cdef vector[uint32_t].iterator it2 = vec2.begin()

    while it1 != vec1.end() and it2 != vec2.end():
        if it1[0] < it2[0]:
            it1 += 1
        elif it2[0] < it1[0]:
            it2 += 1
        else: # Elements are equal
            count += 1
            it1 += 1
            it2 += 1
    return count

# Helper to find intersection size of Python set and C++ vector
cdef inline Py_ssize_t _intersection_size_set_vec(set py_set, vector[uint32_t] &cpp_vec):
    cdef Py_ssize_t count = 0
    cdef uint32_t val
    for val in cpp_vec:
        if val in py_set:
            count += 1
    return count

# ---------------------------------------------------------------------------
#   utility: promotion_times (UNCHANGED)
# ---------------------------------------------------------------------------
cpdef int promotion_times(float b, float v):
    """
    Given b,v return t such that
      b*(1-0.3*v)>0.3
    never holds after t iterations of the update rules.
    """
    cdef int t = 0
    cdef float d, C
    while True:
        if b * (1.0 - 0.3 * v) > 0.3:
            return t
        d = b - 0.5
        if d < 0.0:
            d = -d
        C = 0.15 * d * v
        b = b * (1.0 - C) + C
        v = v * (0.95 + 0.025 * d)
        if v < 0.1:
            v = 0.1
        t += 1

# ---------------------------------------------------------------------------
#   WordIndex: bidirectional str <-> uint32 (ADAPTED for C++ internals)
# ---------------------------------------------------------------------------
cdef class _WordIndex:

    def __cinit__(self):
        # Initialize C++ maps directly
        self._w2i_cpp = unordered_map[string, uint32_t]()
        self._i2w_cpp = vector[string]()

    cpdef uint32_t get_id(self, str w):
        cdef string w_cpp = w.encode('utf-8') # Convert Python str to C++ string
        cdef uint32_t idx
        
        if self._w2i_cpp.count(w_cpp): # Use C++ map's count method
            return self._w2i_cpp[w_cpp]
        
        idx = <uint32_t>self._i2w_cpp.size() # Use C++ vector's size
        self._w2i_cpp[w_cpp] = idx
        self._i2w_cpp.push_back(w_cpp)
        return idx

    cpdef bint has_word(self, str w):
        cdef string w_cpp = w.encode('utf-8')
        return self._w2i_cpp.count(w_cpp) > 0

    cpdef str get_word(self, uint32_t idx):
        # Basic bounds check
        if idx >= self._i2w_cpp.size():
            raise IndexError("Word ID out of bounds")
        return self._i2w_cpp[idx].decode('utf-8') # Convert C++ string to Python str

    cpdef Py_ssize_t size(self):
        return self._i2w_cpp.size()


# ---------------------------------------------------------------------------
#   VocabularyModel: per-word arrays + traces + activation & decay
# ---------------------------------------------------------------------------
cdef class VocabularyModel:

    def __cinit__(self,
                  float learning_rate          = 0.1,
                  float context_influence      = 0.2,
                  float activation_threshold   = 0.05,
                  float base_decay_rate        = 0.05,
                  float proficiency_min        = 0.01,
                  float proficiency_max        = 0.99,
                  double min_elapsed_h         = 1.0,
                  float propagation_threshold = 0.01,
                  float trace_delete_threshold= 0.001):
        self.idx                   = _WordIndex()
        self.learning_rate         = learning_rate
        self.context_influence     = context_influence
        self.activation_threshold  = activation_threshold
        self.base_decay_rate       = base_decay_rate
        self.proficiency_min       = proficiency_min
        self.proficiency_max       = proficiency_max
        self.min_elapsed_h         = min_elapsed_h
        self.propagation_threshold = propagation_threshold
        self.trace_delete_threshold= trace_delete_threshold

        self.prof           = None
        self.vol            = None
        self.eff_prof       = None
        self.encounters     = None
        self._resize_arrays(0) # Will initialize prof, vol, eff_prof, encounters and _word_last_decay_h

        # C++ vectors are default-constructed as empty
        self._trace_word_ids = vector[vector[uint32_t]]()
        self._trace_timestamps_h = vector[double]()
        self._trace_activations = vector[float]()
        self._trace_decay_factors = vector[float]()

        # word_to_traces is now a C++ unordered_map
        self.word_to_traces = unordered_map[uint32_t, vector[int]]()

        # NEW: Initialize priority queues for decay scheduling
        # They are default-constructed as empty
        self._word_decay_pq = priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]]()
        self._trace_decay_pq = priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]]()


    # -----------------------------------------------------------------------
    #   grow arrays to size ≥ new_n (ADAPTED for _word_last_decay_h and PQ init)
    # -----------------------------------------------------------------------
    cdef void _resize_arrays(self, Py_ssize_t new_n):
        cdef Py_ssize_t cur
        if self.prof is None:
            cur = 0
        else:
            cur = (<cnp.ndarray> self.prof).shape[0]

            if new_n <= cur:
                return

        cdef Py_ssize_t old_size = cur
        cdef Py_ssize_t size = 1 if cur == 0 else cur
        while size < new_n:
            size <<= 1

        def _grow(old, dtype, fill):
            if old is None:
                return np.full(size, fill, dtype=dtype)
            a = np.empty(size, dtype=dtype)
            a[:cur] = old
            if size > cur:
                a[cur:] = fill
            return a

        self.prof          = _grow(self.prof,         np.float32,   self.proficiency_min)
        self.vol           = _grow(self.vol,          np.float32,   0.9)
        self.eff_prof      = _grow(self.eff_prof,     np.float32,   0.0)
        self.encounters    = _grow(self.encounters,   np.uint32,    0)

        # NEW: Resize C++ vector for last_decayed and populate PQ
        # If current size is 0, initialize to size, otherwise grow
        if old_size == 0:
            self._word_last_decay_h.resize(size, 0.0)
            # Populate PQ for all new words
            cdef uint32_t i
            cdef DecayItem item
            for i in range(size):
                item.id_or_idx = i
                item.next_decay_time = 0.0 # Initial decay time (effectively now)
                self._word_decay_pq.push(item)
        else:
            # Create a temporary vector with new size and copy old data
            cdef vector[double] temp_last_decayed
            temp_last_decayed.resize(size)
            for i in range(old_size):
                temp_last_decayed[i] = self._word_last_decay_h[i]
            for i in range(old_size, size):
                temp_last_decayed[i] = 0.0 # Fill new elements
                # Add new words to the priority queue
                cdef DecayItem item
                item.id_or_idx = i
                item.next_decay_time = 0.0 # Initial decay time (effectively now)
                self._word_decay_pq.push(item)
            self._word_last_decay_h = temp_last_decayed


    # -----------------------------------------------------------------------
    #   ensure any new words are indexed & arrays grown
    #   (ADAPTED to use _WordIndex.size() and _WordIndex.get_id)
    # -----------------------------------------------------------------------
    cdef set _ensure_words(self, list words):
        cdef set ids = set()
        cdef str w
        for w in words:
            ids.add(self.idx.get_id(w)) # This will add word to _WordIndex and grow arrays if needed
        self._resize_arrays(self.idx.size()) # Ensure arrays match current _WordIndex size
        return ids

    cdef float _eff_prof_formula(self, float p, float v) nogil:
        return p * (1.0 - 0.3*v)

    cpdef float[::1] get_effective_proficiency(self, object words):
        cdef Py_ssize_t n = len(words)
        cdef float[::1] out = np.zeros(n, dtype=np.float32)
        cdef Py_ssize_t i
        cdef str w
        cdef uint32_t wid
        for i, w in enumerate(words):
            if self.idx.has_word(w):
                wid = self.idx.get_id(w) # Get ID using the optimized _WordIndex
                out[i] = self.eff_prof[wid]
            else:
                out[i] = 0.0365 # Default value for unknown words
        return out

    # -----------------------------------------------------------------------
    #   Trace management with C++ vectors and unordered_map
    # -----------------------------------------------------------------------
    cpdef int add_trace_cpp(self, set word_ids_py, double timestamp_h, float activation, float decay_factor):
        cdef vector[uint32_t] current_word_ids_cpp
        cdef uint32_t wid
        cdef int trace_idx = self._trace_timestamps_h.size() # Current number of traces

        # Convert Python set to C++ vector and sort it to maintain set-like behavior
        for wid in word_ids_py:
            current_word_ids_cpp.push_back(wid)
        sort(current_word_ids_cpp.begin(), current_word_ids_cpp.end()) # Ensures consistency for intersection logic

        # Add to main C++ vectors
        self._trace_word_ids.push_back(current_word_ids_cpp)
        self._trace_timestamps_h.push_back(timestamp_h)
        self._trace_activations.push_back(activation)
        self._trace_decay_factors.push_back(decay_factor)

        # Update reverse index (word_to_traces) with trace index - NOW C++ UNORDERED_MAP
        cdef vector[int]* trace_list_ptr # Pointer to the vector in the map
        for wid in word_ids_py:
            # Access or create the vector for this word_id and append the trace_idx
            trace_list_ptr = &self.word_to_traces[wid] # This creates if not exists, or gets reference
            trace_list_ptr.push_back(trace_idx)
        
        # NEW: Add trace to the trace decay priority queue
        cdef DecayItem trace_item
        trace_item.id_or_idx = trace_idx
        trace_item.next_decay_time = timestamp_h + self.min_elapsed_h # Schedule next decay check
        self._trace_decay_pq.push(trace_item)

        return trace_idx # Return the index of the newly added trace

    cpdef float decay_trace_activation_cpp(self, int trace_idx, double now_h):
        cdef double elapsed
        cdef float current_activation
        cdef float current_decay_factor
        cdef double current_timestamp_h

        if trace_idx < 0 or trace_idx >= self._trace_activations.size():
            # This should ideally not happen if called correctly from PQ, but for safety
            return 0.0

        current_activation = self._trace_activations[trace_idx]
        current_decay_factor = self._trace_decay_factors[trace_idx]
        current_timestamp_h = self._trace_timestamps_h[trace_idx]

        elapsed = now_h - current_timestamp_h
        if elapsed <= self.min_elapsed_h:
            return current_activation

        current_activation = <float>(
            current_activation / (1.0 + current_decay_factor * elapsed)
        )
        self._trace_activations[trace_idx] = current_activation
        self._trace_timestamps_h[trace_idx] = now_h # Update timestamp when decayed
        return current_activation


    # -----------------------------------------------------------------------
    #   1) Word-level decay & activation (ADAPTED)
    # -----------------------------------------------------------------------
    cpdef float get_word_activation(self, uint32_t word_id, double now_h):
        cdef float A = 0.0
        cdef int cnt = 0
        cdef int trace_idx # Now iterating over indices
        cdef float tA
        cdef vector[int]* trace_indices_ptr # Pointer to the vector of trace indices

        # Check if word_id exists in the unordered_map
        if self.word_to_traces.count(word_id) == 0:
            return 0.0 # No traces for this word

        trace_indices_ptr = &self.word_to_traces.at(word_id) # Use .at() for bounds checking
        
        # Iterate through trace indices associated with this word_id
        for trace_idx in trace_indices_ptr[0]: # Dereference pointer to iterate
            tA = self.decay_trace_activation_cpp(trace_idx, now_h) # Call new decay function
            if tA > self.activation_threshold:
                A   += tA
                cnt += 1
        if cnt == 0:
            return 0.0
        return fmin(1.0, A / sqrt(cnt * 2.0)) # Using fmin from libc.math


    cdef void apply_decay_to_word_id(self, uint32_t wid, double now_h):
        # Use _word_last_decay_h instead of last_decayed NumPy array
        cdef double elapsed = now_h - self._word_last_decay_h[wid]
        if elapsed <= self.min_elapsed_h:
            return
        cdef float p = self.prof[wid]
        cdef float v = self.vol[wid]
        cdef float encf = 1.0 / (1.0 + 0.2 * self.encounters[wid])
        if encf < 0.1:
            encf = 0.1
        cdef float dr = self.base_decay_rate * v * encf
        cdef double damt = 1.0 - exp(-dr * elapsed / 24.0)
        p *= (1.0 - damt)
        if p < self.proficiency_min:
            p = self.proficiency_min
        v += fmin(0.1, 0.01 * damt * elapsed / 24.0) # Using fmin
        if v > 0.9:
            v = 0.9
        self.prof[wid]        = p
        self.vol[wid]         = v
        self.eff_prof[wid]    = self._eff_prof_formula(p, v)
        self._word_last_decay_h[wid]= now_h # Update C++ vector timestamp

    cpdef void _process_due_word_decays(self, double now_h):
        """
        NEW: Processes words from the priority queue whose decay time is due.
        """
        cdef DecayItem current_item
        cdef uint32_t wid

        while not self._word_decay_pq.empty():
            current_item = self._word_decay_pq.top()
            if current_item.next_decay_time > now_h:
                break # No more items are due yet
            
            self._word_decay_pq.pop() # Remove from PQ

            wid = current_item.id_or_idx
            # Ensure word ID is still valid (e.g., if vocabulary size shrunk, though unlikely)
            if wid < self.idx.size():
                self.apply_decay_to_word_id(wid, now_h)
                # Re-schedule the word for its next decay
                current_item.next_decay_time = now_h + self.min_elapsed_h # Schedule next check for min_elapsed_h later
                self._word_decay_pq.push(current_item)
            # else: word no longer exists, just discard from PQ

    cpdef void _process_due_trace_decays(self, double now_h):
        """
        NEW: Processes traces from the priority queue whose decay time is due.
        """
        cdef DecayItem current_item
        cdef int trace_idx

        while not self._trace_decay_pq.empty():
            current_item = self._trace_decay_pq.top()
            if current_item.next_decay_time > now_h:
                break # No more items are due yet
            
            self._trace_decay_pq.pop() # Remove from PQ

            trace_idx = current_item.id_or_idx
            # Ensure trace index is still valid (e.g., if trace was pruned)
            if trace_idx < self._trace_activations.size() and self._trace_activations[trace_idx] >= self.trace_delete_threshold:
                self.decay_trace_activation_cpp(trace_idx, now_h)
                # Re-schedule the trace for its next decay
                current_item.next_decay_time = now_h + self.min_elapsed_h # Schedule next check for min_elapsed_h later
                self._trace_decay_pq.push(current_item)
            # else: trace no longer exists or was pruned, just discard from PQ


    # -----------------------------------------------------------------------
    #   2) Full sentence-trace propagation (ADAPTED for C++ unordered_map)
    # -----------------------------------------------------------------------
    cdef void _propagate(self, int source_trace_idx):
        queue = deque() # Stores (trace_idx, delta_activation)
        # Basic bounds check for safety
        if source_trace_idx < 0 or source_trace_idx >= self._trace_activations.size():
            return

        queue.append((source_trace_idx, self._trace_activations[source_trace_idx]))

        cdef float delta
        cdef int current_trace_idx, neighbor_trace_idx
        cdef Py_ssize_t ov
        cdef float step
        cdef float tj_part

        cdef vector[uint32_t]* current_word_ids_ptr
        cdef vector[uint32_t]* neighbor_word_ids_ptr
        cdef vector[int]* trace_indices_for_word_ptr

        # Use a C++ unordered_set for `current_trace_neighbors` for efficiency
        cdef unordered_map[int, bint] visited_neighbors # Use map as a set to track visited neighbors for current propagation cycle

        cdef uint32_t word_id_in_current_trace

        while not queue.empty(): # Use C++ deque empty()
            current_trace_idx, delta = queue.front() # Use C++ deque front()
            queue.pop_front() # Use C++ deque pop_front()

            # Clear visited_neighbors for each new trace in the BFS
            visited_neighbors.clear()

            current_word_ids_ptr = &self._trace_word_ids[current_trace_idx]

            for word_id_in_current_trace in current_word_ids_ptr[0]:
                if self.word_to_traces.count(word_id_in_current_trace) > 0:
                    trace_indices_for_word_ptr = &self.word_to_traces.at(word_id_in_current_trace)
                    for neighbor_trace_idx in trace_indices_for_word_ptr[0]:
                        if neighbor_trace_idx != current_trace_idx and visited_neighbors.count(neighbor_trace_idx) == 0:
                            visited_neighbors[neighbor_trace_idx] = True # Mark as visited for this BFS step

            for neighbor_trace_idx in visited_neighbors: # Iterate over C++ map keys
                neighbor_word_ids_ptr = &self._trace_word_ids[neighbor_trace_idx]

                ov = _intersection_size_vec_vec(current_word_ids_ptr[0], neighbor_word_ids_ptr[0])

                if ov == 0:
                    continue

                step = delta / current_word_ids_ptr[0].size() * ov

                tj_part = self._trace_activations[neighbor_trace_idx] / neighbor_word_ids_ptr[0].size() * ov

                if tj_part < step:
                    self._trace_activations[neighbor_trace_idx] += step - tj_part
                    if self._trace_activations[neighbor_trace_idx] > 1.0:
                        self._trace_activations[neighbor_trace_idx] = 1.0

                    if self.propagation_threshold < step:
                        queue.push_back((neighbor_trace_idx, step)) # Use C++ deque push_back()

    cdef void _prune_traces(self):
        """
        ADAPTED for incremental pruning concept (marking and rebuilding).
        This version still rebuilds, but the next step would be true mark-and-sweep.
        """
        cdef vector[vector[uint32_t]] new_trace_word_ids
        cdef vector[double] new_trace_timestamps_h
        cdef vector[float] new_trace_activations
        cdef vector[float] new_trace_decay_factors

        cdef uint32_t old_trace_idx = 0
        cdef uint32_t wid
        cdef int trace_count = self._trace_timestamps_h.size()

        cdef vector[int] traces_to_keep_indices
        for old_trace_idx in range(trace_count):
            if self._trace_activations[old_trace_idx] >= self.trace_delete_threshold:
                traces_to_keep_indices.push_back(old_trace_idx)
        
        cdef Py_ssize_t num_kept_traces = traces_to_keep_indices.size()
        new_trace_word_ids.resize(num_kept_traces)
        new_trace_timestamps_h.resize(num_kept_traces)
        new_trace_activations.resize(num_kept_traces)
        new_trace_decay_factors.resize(num_kept_traces)

        cdef Py_ssize_t new_idx = 0
        for old_idx in traces_to_keep_indices:
            new_trace_word_ids[new_idx] = self._trace_word_ids[old_idx]
            new_trace_timestamps_h[new_idx] = self._trace_timestamps_h[old_idx]
            new_trace_activations[new_idx] = self._trace_activations[old_idx]
            new_trace_decay_factors[new_idx] = self._trace_decay_factors[old_idx]
            new_idx += 1

        self._trace_word_ids = new_trace_word_ids
        self._trace_timestamps_h = new_trace_timestamps_h
        self._trace_activations = new_trace_activations
        self._trace_decay_factors = new_trace_decay_factors

        self.word_to_traces.clear()
        cdef vector[uint32_t]* word_ids_for_rebuild_ptr

        for new_idx in range(num_kept_traces):
            word_ids_for_rebuild_ptr = &self._trace_word_ids[new_idx]
            for wid in word_ids_for_rebuild_ptr[0]:
                self.word_to_traces[wid].push_back(new_idx)


    # -----------------------------------------------------------------------
    #   3) Context Support (ADAPTED for C++ unordered_map)
    # -----------------------------------------------------------------------
    cpdef float calculate_context_support(self, object id_words, Py_ssize_t n):
        if not id_words or self._trace_timestamps_h.empty():
            return 0.0
        cdef float mx = 0.0
        cdef int trace_idx
        cdef Py_ssize_t ov
        cdef float r
        cdef vector[uint32_t]* trace_word_ids_vec_ptr

        cdef int num_traces = self._trace_timestamps_h.size()
        for trace_idx in range(num_traces):
            trace_word_ids_vec_ptr = &self._trace_word_ids[trace_idx]
            ov = _intersection_size_set_vec(id_words, trace_word_ids_vec_ptr[0])
            
            if ov > 0:
                r = (ov / fmax(n, trace_word_ids_vec_ptr[0].size())) * self._trace_activations[trace_idx]
                if r > mx:
                    mx = r
        return mx * 0.5


    # -----------------------------------------------------------------------
    #   4) Predict Understanding (ADAPTED for new decay scheduling)
    # -----------------------------------------------------------------------
    cpdef float predict_understanding(self, object words, double current_time_h=-1):
        if not words:
            return 0.0
        if current_time_h < 0:
            current_time_h = datetime.now().timestamp() / 3600.0
        
        # NEW: Process due word decays using the priority queue
        self._process_due_word_decays(current_time_h)
        
        # NEW: Process due trace decays using the priority queue
        self._process_due_trace_decays(current_time_h)

        cdef float sum_p = 0.0, min_p = 1.0
        cdef set non_zero = set()
        cdef str w
        cdef uint32_t wid
        cdef float ep
        for w in words:
            if self.idx.has_word(w):
                wid = self.idx.get_id(w)
                if wid < self.idx.size():
                    ep = self.eff_prof[wid]
                    non_zero.add(wid)
                else:
                    ep = 0.0365
            else:
                ep = 0.0365
            
            if ep < min_p:
                min_p = ep
            sum_p += ep
        
        cdef Py_ssize_t n = len(words)
        cdef float avg_p = sum_p / n
        cdef float ctx   = self.calculate_context_support(non_zero, n)

        cdef float lf = 1.0 / (1 + 0.1 * n)
        if lf > 0.5:
            lf = 0.5
        cdef float mw = 0.3 + 0.2 * lf
        cdef float aw = 0.5 - 0.2 * lf
        cdef float cw = 0.2

        cdef float u = mw * min_p + aw * avg_p + cw * ctx
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        return u


    # -----------------------------------------------------------------------
    #   5) Update Proficiency + add & propagate new trace (ADAPTED)
    # -----------------------------------------------------------------------
    cpdef void update_proficiency(self, object words, float u_val, double now_h):
        cdef list clean = [w.lower() for w in words if w.strip()]
        
        cdef float expected = self.predict_understanding(clean, now_h) # This also calls decay if needed

        cdef set wids_py = self._ensure_words(clean) # This will still add words if they don't exist

        # Add the new trace using the C++ vector function
        cdef int new_trace_idx = self.add_trace_cpp(wids_py, now_h, 1.0, (0.1 - 0.05 * u_val))

        # Propagate using the new trace index
        self._propagate(new_trace_idx)
        self._prune_traces()

        cdef float err = u_val - expected
        cdef float lr  = self.learning_rate
        cdef float pmn = self.proficiency_min
        cdef float pmx = self.proficiency_max

        cdef float cp
        cdef float cv
        cdef float act
        cdef float lop
        cdef bint better
        cdef float bu
        cdef float um
        cdef float dv
        cdef float upd
        cdef float vd
        cdef uint32_t wid # Use uint32_t for word ID

        for wid in wids_py: # Iterate over Python set of word IDs
            self.encounters[wid]   += 1
            self._word_last_decay_h[wid]  = now_h # Update C++ vector timestamp
            cp = self.prof[wid]
            cv = self.vol[wid]
            act= self.get_word_activation(wid, now_h) # This call now uses C++ trace data
            lop= abs(u_val - cp)
            better = err > 0
            bu  = lr * (abs(err) if abs(err) < lop else lop)
            um  = bu * cv * act
            dv  = (pmx - cp) if better else (cp - pmn)
            upd = um * dv * (1 if better else -1)

            cp += upd
            if cp < pmn:   cp = pmn
            if cp > pmx:   cp = pmx

            vd = 0.1 * cv * u_val * act * (1.0 - 0.5 * fmin(1.0, abs(err))) # Using fmin
            cv -= vd
            if cv < 0.1:   cv = 0.1
            if self.encounters[wid] > 3:
                cv = fmax(0.1, cv * 0.9) # Using fmax

            self.prof[wid]        = cp
            self.vol[wid]         = cv
            self.eff_prof[wid]    = self._eff_prof_formula(cp, cv)


    # -----------------------------------------------------------------------
    #   6) Fast binary save/load (ADAPTED FOR C++ VECTORS AND UNORDERED_MAP AND PQs)
    # -----------------------------------------------------------------------
    def save_fast(self, str path):
        cdef uint32_t n_words  = self.idx.size() # Use idx.size()
        cdef uint32_t n_traces = self._trace_timestamps_h.size()
        cdef bytes b
        cdef uint32_t i
        cdef vector[uint32_t]* current_trace_word_ids_cpp_ptr

        with open(path, "wb") as f:
            # 1) header: n_words, n_traces
            # No longer saving _next_word_decay_appointment_h, _next_trace_decay_appointment_h directly
            # as they are managed by the PQs.
            f.write(struct.pack("<II", n_words, n_traces))

            # 2) pack your 8 floats as little‐endian doubles (UNCHANGED)
            f.write(struct.pack(
                "<8d",
                self.learning_rate,
                self.context_influence,
                self.activation_threshold,
                self.base_decay_rate,
                self.proficiency_min,
                self.proficiency_max,
                self.propagation_threshold,
                self.trace_delete_threshold
            ))

            # 3) numeric arrays (only the first n_words elements)
            f.write(np.ascontiguousarray(self.prof[:n_words],          dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.vol[:n_words],           dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.eff_prof[:n_words],      dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.encounters[:n_words],    dtype='<u4').tobytes())
            # Save _word_last_decay_h (C++ vector) directly
            if n_words > 0:
                f.write(PyBytes_FromStringAndSize(<char*> self._word_last_decay_h.data(), self._word_last_decay_h.size() * sizeof(double)))

            # 4) vocabulary (ADAPTED to use _WordIndex's C++ internals)
            f.write(struct.pack("<I", self.idx._i2w_cpp.size())) # Write number of words
            cdef string word_cpp
            for i in range(self.idx._i2w_cpp.size()):
                word_cpp = self.idx._i2w_cpp[i]
                f.write(struct.pack("<I", word_cpp.size())) # Write length of C++ string
                f.write(PyBytes_FromStringAndSize(word_cpp.data(), word_cpp.size())) # Write C++ string data

            # 5) traces - Now directly save the C++ vector contents
            # Fixed-size data for each trace
            if n_traces > 0:
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_timestamps_h.data(), self._trace_timestamps_h.size() * sizeof(double)))
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_activations.data(), self._trace_activations.size() * sizeof(float)))
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_decay_factors.data(), self._trace_decay_factors.size() * sizeof(float)))

            # Variable-length word_ids: write length then data for each inner vector
            for i in range(n_traces):
                current_trace_word_ids_cpp_ptr = &self._trace_word_ids[i]
                f.write(struct.pack("<I", current_trace_word_ids_cpp_ptr[0].size()))
                if current_trace_word_ids_cpp_ptr[0].size() > 0:
                    f.write(PyBytes_FromStringAndSize(<char*> current_trace_word_ids_cpp_ptr[0].data(), current_trace_word_ids_cpp_ptr[0].size() * sizeof(uint32_t)))
            
            # 6) word_to_traces (C++ unordered_map)
            f.write(struct.pack("<I", self.word_to_traces.size()))
            cdef pair[uint32_t, vector[int]] item
            for item in self.word_to_traces:
                f.write(struct.pack("<I", item.first))
                f.write(struct.pack("<I", item.second.size()))
                if item.second.size() > 0:
                    f.write(PyBytes_FromStringAndSize(<char*> item.second.data(), item.second.size() * sizeof(int)))

            # 7) Save priority queues (word_decay_pq, trace_decay_pq)
            # Save size, then elements one by one (top to bottom)
            f.write(struct.pack("<I", self._word_decay_pq.size()))
            cdef priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]] temp_word_pq = self._word_decay_pq
            while not temp_word_pq.empty():
                item = temp_word_pq.top()
                f.write(struct.pack("<dI", item.next_decay_time, item.id_or_idx))
                temp_word_pq.pop()

            f.write(struct.pack("<I", self._trace_decay_pq.size()))
            cdef priority_queue[DecayItem, vector[DecayItem], greater[DecayItem]] temp_trace_pq = self._trace_decay_pq
            while not temp_trace_pq.empty():
                item = temp_trace_pq.top()
                f.write(struct.pack("<dI", item.next_decay_time, item.id_or_idx))
                temp_trace_pq.pop()


    @classmethod
    def load_fast(cls, str path):
        cdef uint32_t n_words, n_traces, i, cnt, word_id_count
        cdef bytes buf
        cdef tuple tpl
        cdef VocabularyModel m # Declare m as VocabularyModel type

        cdef vector[uint32_t]* current_trace_word_ids_cpp_ref
        cdef uint32_t word_id_val
        cdef int trace_idx_val
        cdef string word_cpp_buf
        cdef double decay_time_val
        cdef DecayItem loaded_decay_item

        # Temporary bytearrays for reading
        cdef bytearray temp_bytes_timestamps
        cdef bytearray temp_bytes_activations
        cdef bytearray temp_bytes_decay_factors
        cdef bytearray temp_bytes_word_ids
        cdef bytearray temp_bytes_word_last_decayed
        cdef bytearray temp_bytes_trace_indices

        with open(path, "rb") as f:
            # read header (ADAPTED: 2 uint32_t)
            buf = f.read(8) # 2*4 = 8 bytes
            n_words, n_traces = struct.unpack("<II", buf)

            # read your eight floats back
            buf = f.read(8 * 8)
            tpl = struct.unpack("<8d", buf)
            m = cls(learning_rate=tpl[0], context_influence=tpl[1], activation_threshold=tpl[2],
                    base_decay_rate=tpl[3], proficiency_min=tpl[4], proficiency_max=tpl[5],
                    propagation_threshold=tpl[6], trace_delete_threshold=tpl[7])

            # numeric arrays
            m.prof          = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.vol           = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.eff_prof      = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.encounters    = np.frombuffer(f.read(n_words * 4), dtype='<u4').copy()
            
            # Read _word_last_decay_h (C++ vector)
            m._word_last_decay_h.resize(n_words)
            if n_words > 0:
                temp_bytes_word_last_decayed = bytearray(n_words * sizeof(double))
                f.readinto(temp_bytes_word_last_decayed)
                memcpy(<char*> m._word_last_decay_h.data(), <char*> temp_bytes_word_last_decayed, n_words * sizeof(double))


            # rebuild index (ADAPTED to use _WordIndex's C++ internals)
            m.idx = _WordIndex.__new__(_WordIndex) # Create new instance
            m.idx._w2i_cpp.clear() # Clear internal C++ map
            m.idx._i2w_cpp.clear() # Clear internal C++ vector
            
            buf = f.read(4)
            cdef uint32_t num_words_in_idx, word_len
            num_words_in_idx, = struct.unpack("<I", buf)

            for i in range(num_words_in_idx):
                buf = f.read(4)
                word_len, = struct.unpack("<I", buf)
                word_cpp_buf = string(f.read(word_len)) # Read directly into C++ string
                m.idx._i2w_cpp.push_back(word_cpp_buf)
                m.idx._w2i_cpp[word_cpp_buf] = i


            # === REBUILD TRACE DATA FROM C++ VECTORS ===
            m._trace_timestamps_h.resize(n_traces)
            m._trace_activations.resize(n_traces)
            m._trace_decay_factors.resize(n_traces)
            m._trace_word_ids.resize(n_traces)

            if n_traces > 0:
                temp_bytes_timestamps = bytearray(n_traces * sizeof(double))
                f.readinto(temp_bytes_timestamps)
                memcpy(<char*> m._trace_timestamps_h.data(), <char*> temp_bytes_timestamps, n_traces * sizeof(double))

                temp_bytes_activations = bytearray(n_traces * sizeof(float))
                f.readinto(temp_bytes_activations)
                memcpy(<char*> m._trace_activations.data(), <char*> temp_bytes_activations, n_traces * sizeof(float))

                temp_bytes_decay_factors = bytearray(n_traces * sizeof(float))
                f.readinto(temp_bytes_decay_factors)
                memcpy(<char*> m._trace_decay_factors.data(), <char*> temp_bytes_decay_factors, n_traces * sizeof(float))


            for i in range(n_traces):
                buf = f.read(4)
                word_id_count, = struct.unpack("<I", buf)

                current_trace_word_ids_cpp_ref = &m._trace_word_ids[i]
                current_trace_word_ids_cpp_ref[0].resize(word_id_count)

                if word_id_count > 0:
                    temp_bytes_word_ids = bytearray(word_id_count * sizeof(uint32_t))
                    f.readinto(temp_bytes_word_ids)
                    memcpy(<char*> current_trace_word_ids_cpp_ref[0].data(), <char*> temp_bytes_word_ids, word_id_count * sizeof(uint32_t))

            # 6) Read word_to_traces (C++ unordered_map)
            m.word_to_traces.clear()
            buf = f.read(4)
            cdef uint32_t map_size, vector_size, key_val
            map_size, = struct.unpack("<I", buf)

            for i in range(map_size):
                buf = f.read(4)
                key_val, = struct.unpack("<I", buf)

                buf = f.read(4)
                vector_size, = struct.unpack("<I", buf)

                cdef vector[int] trace_indices_vec
                trace_indices_vec.resize(vector_size)

                if vector_size > 0:
                    temp_bytes_trace_indices = bytearray(vector_size * sizeof(int))
                    f.readinto(temp_bytes_trace_indices)
                    memcpy(<char*> trace_indices_vec.data(), <char*> temp_bytes_trace_indices, vector_size * sizeof(int))
                
                m.word_to_traces[key_val] = trace_indices_vec

            # 7) Load priority queues (word_decay_pq, trace_decay_pq)
            cdef uint32_t pq_size
            buf = f.read(4)
            pq_size, = struct.unpack("<I", buf)
            for i in range(pq_size):
                buf = f.read(sizeof(double) + sizeof(uint32_t)) # double + uint32_t
                decay_time_val, id_val = struct.unpack("<dI", buf)
                loaded_decay_item.next_decay_time = decay_time_val
                loaded_decay_item.id_or_idx = id_val
                m._word_decay_pq.push(loaded_decay_item)

            buf = f.read(4)
            pq_size, = struct.unpack("<I", buf)
            for i in range(pq_size):
                buf = f.read(sizeof(double) + sizeof(uint32_t))
                decay_time_val, id_val = struct.unpack("<dI", buf)
                loaded_decay_item.next_decay_time = decay_time_val
                loaded_decay_item.id_or_idx = id_val
                m._trace_decay_pq.push(loaded_decay_item)

        return m

# Expose the API
__all__ = ["VocabularyModel", "promotion_times"]

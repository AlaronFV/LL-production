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

import numpy as np
cimport numpy as cnp

from collections import deque, defaultdict
from datetime import datetime

# Import the declarations from your .pxd file
from cy_utils.vocab_model cimport _WordIndex, VocabularyModel, promotion_times

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
#   WordIndex: bidirectional str <-> uint32 (UNCHANGED, except for size property)
# ---------------------------------------------------------------------------
cdef class _WordIndex:

    def __cinit__(self):
        self._w2i = {}
        self._i2w = []

    cpdef uint32_t get_id(self, str w):
        cdef uint32_t idx
        if w in self._w2i:
            return <uint32_t>self._w2i[w]
        idx = <uint32_t>len(self._i2w)
        self._w2i[w] = idx
        self._i2w.append(w)
        return idx

    cpdef bint has_word(self, str w):
        return w in self._w2i

    cpdef str get_word(self, uint32_t idx):
        return <str>self._i2w[idx]

    @property
    def size(self):
        return len(self._i2w)

    @property
    def id2word(self):
        return self._i2w


# ---------------------------------------------------------------------------
#   MemoryTrace: REMOVED COMPLETELY
# ---------------------------------------------------------------------------

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
                  double min_elapsed_h         = 1.0, # NEW: Default to 1 hour
                  float propagation_threshold = 0.01,
                  float trace_delete_threshold= 0.001):
        self.idx                   = _WordIndex()
        self.learning_rate         = learning_rate
        self.context_influence     = context_influence
        self.activation_threshold  = activation_threshold
        self.base_decay_rate       = base_decay_rate
        self.proficiency_min       = proficiency_min
        self.proficiency_max       = proficiency_max
        self.min_elapsed_h         = min_elapsed_h # NEW: Initialize
        self.propagation_threshold = propagation_threshold
        self.trace_delete_threshold= trace_delete_threshold

        self.prof           = None
        self.vol            = None
        self.eff_prof       = None
        self.encounters     = None
        self.last_decayed   = None
        self._resize_arrays(0)

        # C++ vectors are default-constructed as empty
        self.word_to_traces = {}

        # Initialize appointment times to a very old timestamp to ensure first run triggers full decay
        self._next_word_decay_appointment_h = -1.0
        self._next_trace_decay_appointment_h = -1.0


    # -----------------------------------------------------------------------
    #   grow arrays to size ≥ new_n (UNCHANGED)
    # -----------------------------------------------------------------------
    cdef void _resize_arrays(self, Py_ssize_t new_n):
        cdef Py_ssize_t cur
        if self.prof is None:
            cur = 0
        else:
            cur = (<cnp.ndarray> self.prof).shape[0]

            if new_n <= cur:
                return

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
        self.last_decayed  = _grow(self.last_decayed, np.float64, 0.0)

    # -----------------------------------------------------------------------
    #   ensure any new words are indexed & arrays grown (UNCHANGED)
    # -----------------------------------------------------------------------
    cdef set _ensure_words(self, list words):
        cdef set ids = set()
        for w in words:
            ids.add(self.idx.get_id(w))
        self._resize_arrays(self.idx.size)
        return ids

    cdef float _eff_prof_formula(self, float p, float v) nogil:
        return p * (1.0 - 0.3*v)

    cpdef float[::1] get_effective_proficiency(self, object words):
        cdef Py_ssize_t n = len(words)
        cdef float[::1] out = np.zeros(n, dtype=np.float32)
        cdef Py_ssize_t i
        cdef str w
        for i, w in enumerate(words):
            # ADAPTED: Check for word existence without adding it
            if self.idx.has_word(w):
                out[i] = self.eff_prof[self.idx._w2i[w]]
            else:
                out[i] = 0.0365 # Default value for unknown words
        return out

    # -----------------------------------------------------------------------
    #   Trace management with C++ vectors
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

        # Update reverse index (word_to_traces) with trace index
        for wid in word_ids_py:
            self.word_to_traces.setdefault(wid, []).append(trace_idx)
        
        return trace_idx # Return the index of the newly added trace

    cpdef float decay_trace_activation_cpp(self, int trace_idx, double now_h):
        cdef double elapsed
        cdef float current_activation
        cdef float current_decay_factor
        cdef double current_timestamp_h

        if trace_idx < 0 or trace_idx >= self._trace_activations.size():
            raise IndexError("Trace index out of bounds in decay_trace_activation_cpp")

        current_activation = self._trace_activations[trace_idx]
        current_decay_factor = self._trace_decay_factors[trace_idx]
        current_timestamp_h = self._trace_timestamps_h[trace_idx]

        # ADAPTED: Check against min_elapsed_h
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

        # Iterate through trace indices associated with this word_id
        for trace_idx in self.word_to_traces.get(word_id, []):
            tA = self.decay_trace_activation_cpp(trace_idx, now_h) # Call new decay function
            if tA > self.activation_threshold:
                A   += tA
                cnt += 1
        if cnt == 0:
            return 0.0
        return fmin(1.0, A / sqrt(cnt * 2.0)) # Using fmin from libc.math


    cdef void apply_decay_to_word_id(self, uint32_t wid, double now_h):
        cdef double elapsed = now_h - self.last_decayed[wid]
        # ADAPTED: Check against min_elapsed_h
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
        self.last_decayed[wid]= now_h

    cpdef void apply_decay_to_all_words(self, double now_h):
        # ADAPTED: Always perform full scan, then update appointment time
        cdef uint32_t wid
        cdef double min_last_decayed_h = now_h # Initialize with current time, or a very large value

        for wid in range(self.idx.size):
            self.apply_decay_to_word_id(wid, now_h)
            # Find the minimum last_decayed among all words *after* decay
            # Only consider words that were actually decayed in this run (elapsed > min_elapsed_h)
            # or if the timestamp was already very recent.
            if self.last_decayed[wid] < min_last_decayed_h:
                min_last_decayed_h = self.last_decayed[wid]
        
        # Calculate the next word decay appointment: min_last_decayed_h + self.min_elapsed_h
        self._next_word_decay_appointment_h = min_last_decayed_h + self.min_elapsed_h

    # NEW: Apply decay to all traces
    cdef void apply_decay_to_all_traces(self, double now_h):
        # ADAPTED: Always perform full scan, then update appointment time
        cdef int trace_idx
        cdef double min_trace_timestamp_h = now_h # Initialize with current time, or a very large value
        cdef int num_traces = self._trace_timestamps_h.size()

        for trace_idx in range(num_traces):
            self.decay_trace_activation_cpp(trace_idx, now_h)
            # Find the minimum timestamp among all traces *after* decay
            # Only consider traces that were actually decayed in this run (elapsed > min_elapsed_h)
            # or if the timestamp was already very recent.
            if self._trace_timestamps_h[trace_idx] < min_trace_timestamp_h:
                min_trace_timestamp_h = self._trace_timestamps_h[trace_idx]
        
        # Calculate the next trace decay appointment: min_trace_timestamp_h + self.min_elapsed_h
        self._next_trace_decay_appointment_h = min_trace_timestamp_h + self.min_elapsed_h


    # -----------------------------------------------------------------------
    #   2) Full sentence-trace propagation (ADAPTED)
    # -----------------------------------------------------------------------
    cdef void _propagate(self, int source_trace_idx):
        queue = deque() # Stores (trace_idx, delta_activation)
        # Basic bounds check for safety
        if source_trace_idx < 0 or source_trace_idx >= self._trace_activations.size():
            return # Or raise an appropriate error

        queue.append((source_trace_idx, self._trace_activations[source_trace_idx])) # Use activation from C++ vector

        cdef float delta
        cdef int current_trace_idx, neighbor_trace_idx
        cdef Py_ssize_t ov
        cdef float step
        cdef float tj_part

        # Declare pointers here (outside the loop), without immediate initialization.
        # They will point to the vector objects inside _trace_word_ids.
        cdef vector[uint32_t] current_word_ids_ptr
        cdef vector[uint32_t] neighbor_word_ids_ptr

        cdef set current_trace_neighbors

        cdef uint32_t word_id_in_current_trace

        while queue:
            current_trace_idx, delta = queue.popleft() # This line should be fine now

            # This set will collect *unique* neighbor trace indices
            current_trace_neighbors = set() # This is a Python set, which is fine

            # Assign the pointer to the address of the vector at the current index.
            # This is where the problematic line was.
            current_word_ids_ptr = self._trace_word_ids[current_trace_idx]

            # Find neighbors through shared words
            
            # Iterate over the dereferenced vector using the pointer
            for word_id_in_current_trace in current_word_ids_ptr:
                # word_to_traces.get returns a Python list, so this part is Python-level iteration
                for neighbor_trace_idx in self.word_to_traces.get(word_id_in_current_trace, []):
                    if neighbor_trace_idx != current_trace_idx: # Exclude self
                        current_trace_neighbors.add(neighbor_trace_idx)

            # Process each unique neighbor
            for neighbor_trace_idx in current_trace_neighbors:
                # Assign the neighbor pointer similarly
                neighbor_word_ids_ptr = self._trace_word_ids[neighbor_trace_idx]

                # Calculate overlap between current trace and neighbor trace word_ids
                # Pass dereferenced pointers to _intersection_size_vec_vec
                ov = _intersection_size_vec_vec(current_word_ids_ptr, neighbor_word_ids_ptr)

                if ov == 0:
                    continue

                # Access size using the dereferenced pointer
                step = delta / current_word_ids_ptr.size() * ov

                # Access size using the dereferenced pointer
                tj_part = self._trace_activations[neighbor_trace_idx] / neighbor_word_ids_ptr.size() * ov

                if tj_part < step:
                    self._trace_activations[neighbor_trace_idx] += step - tj_part
                    if self._trace_activations[neighbor_trace_idx] > 1.0:
                        self._trace_activations[neighbor_trace_idx] = 1.0

                    if self.propagation_threshold < step:
                        queue.append((neighbor_trace_idx, step))

    cdef void _prune_traces(self):
        cdef vector[vector[uint32_t]] new_trace_word_ids
        cdef vector[double] new_trace_timestamps_h
        cdef vector[float] new_trace_activations
        cdef vector[float] new_trace_decay_factors

        cdef uint32_t old_trace_idx = 0
        cdef uint32_t wid
        cdef int trace_count = self._trace_timestamps_h.size()

        # First pass: identify traces to keep and populate new C++ vectors
        for old_trace_idx in range(trace_count):
            if self._trace_activations[old_trace_idx] >= self.trace_delete_threshold:
                new_trace_word_ids.push_back(self._trace_word_ids[old_trace_idx])
                new_trace_timestamps_h.push_back(self._trace_timestamps_h[old_trace_idx])
                new_trace_activations.push_back(self._trace_activations[old_trace_idx])
                new_trace_decay_factors.push_back(self._trace_decay_factors[old_trace_idx])

        # Replace old C++ vectors with new ones
        self._trace_word_ids = new_trace_word_ids
        self._trace_timestamps_h = new_trace_timestamps_h
        self._trace_activations = new_trace_activations
        self._trace_decay_factors = new_trace_decay_factors

        # Rebuild word_to_traces (this part will be Python-heavy, but necessary)
        self.word_to_traces.clear() # Clear the old dictionary
        cdef uint32_t current_new_idx
        cdef vector[uint32_t] word_ids_for_rebuild
        
        for current_new_idx in range(self._trace_timestamps_h.size()):
            word_ids_for_rebuild = self._trace_word_ids[current_new_idx]
            for wid in word_ids_for_rebuild:
                self.word_to_traces.setdefault(wid, []).append(current_new_idx)


    # -----------------------------------------------------------------------
    #   3) Context Support (ADAPTED)
    # -----------------------------------------------------------------------
    cpdef float calculate_context_support(self, object id_words, Py_ssize_t n):
        if not id_words or self._trace_timestamps_h.empty(): # Check C++ vector size
            return 0.0
        cdef float mx = 0.0
        cdef int trace_idx
        cdef Py_ssize_t ov
        cdef float r
        cdef vector[uint32_t] trace_word_ids_vec # To hold a reference to the C++ vector

        cdef int num_traces = self._trace_timestamps_h.size()
        for trace_idx in range(num_traces):
            trace_word_ids_vec = self._trace_word_ids[trace_idx]
            ov = _intersection_size_set_vec(id_words, trace_word_ids_vec) # Use helper
            
            if ov > 0:
                # Direct access to activation and word_ids size
                r = (ov / fmax(n, trace_word_ids_vec.size())) * self._trace_activations[trace_idx]
                if r > mx:
                    mx = r
        return mx * 0.5


    # -----------------------------------------------------------------------
    #   4) Predict Understanding (ADAPTED for robust word check and optimized decay)
    # -----------------------------------------------------------------------
    cpdef float predict_understanding(self, object words, double current_time_h=-1):
        if not words:
            return 0.0
        if current_time_h < 0:
            current_time_h = datetime.now().timestamp() / 3600.0
        
        # ADAPTED: Conditionally apply decay to all words
        if current_time_h >= self._next_word_decay_appointment_h:
            self.apply_decay_to_all_words(current_time_h)
        
        # ADAPTED: Conditionally apply decay to all traces
        if current_time_h >= self._next_trace_decay_appointment_h:
            self.apply_decay_to_all_traces(current_time_h)

        cdef float sum_p = 0.0, min_p = 1.0
        cdef set non_zero = set()
        cdef str w
        cdef uint32_t wid
        cdef float ep
        for w in words:
            # ADAPTED: Check for word existence without adding it
            if self.idx.has_word(w):
                wid = self.idx._w2i[w] # Get ID only if it exists
                # Ensure wid is within the bounds of the NumPy arrays
                if wid < self.idx.size: # Check against actual vocabulary size
                    ep = self.eff_prof[wid]
                    non_zero.add(wid)
                else:
                    ep = 0.0365 # Should not happen if has_word is true and idx.size is correct
            else:
                ep = 0.0365 # Default value for unknown words
            
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

        # ADAPTED: Removed the 'updates' dictionary creation
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
            self.last_decayed[wid]  = now_h
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
    #   6) Fast binary save/load (ADAPTED FOR C++ VECTORS AND NEW TIMESTAMPS)
    # -----------------------------------------------------------------------
    def save_fast(self, str path):
        cdef uint32_t n_words  = self.idx.size
        cdef uint32_t n_traces = self._trace_timestamps_h.size() # Number of traces from C++ vector size
        cdef bytes b
        cdef uint32_t i
        cdef vector[uint32_t] current_trace_word_ids_cpp

        with open(path, "wb") as f:
            # 1) header: n_words, n_traces, _next_word_decay_appointment_h, _next_trace_decay_appointment_h
            # Two uint32_t and two double values (2*4 + 2*8 = 24 bytes)
            f.write(struct.pack("<IIdd",
                                n_words,
                                n_traces,
                                self._next_word_decay_appointment_h,
                                self._next_trace_decay_appointment_h))

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

            # 3) numeric arrays (only the first n_words elements) (UNCHANGED)
            f.write(np.ascontiguousarray(self.prof[:n_words],          dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.vol[:n_words],           dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.eff_prof[:n_words],      dtype='<f4').tobytes())
            f.write(np.ascontiguousarray(self.encounters[:n_words],    dtype='<u4').tobytes())
            f.write(np.ascontiguousarray(self.last_decayed[:n_words],  dtype='<f8').tobytes())

            # 4) vocabulary (UNCHANGED)
            for w in self.idx._i2w:
                b = w.encode('utf-8')
                f.write(struct.pack("<I", len(b)))
                f.write(b)

            # 5) traces - Now directly save the C++ vector contents
            # Fixed-size data for each trace
            if n_traces > 0:
                # Correctly construct bytes objects from the C++ vector data using PyBytes_FromStringAndSize
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_timestamps_h.data(), self._trace_timestamps_h.size() * sizeof(double)))
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_activations.data(), self._trace_activations.size() * sizeof(float)))
                f.write(PyBytes_FromStringAndSize(<char*> self._trace_decay_factors.data(), self._trace_decay_factors.size() * sizeof(float)))

            # Variable-length word_ids: write length then data for each inner vector
            for i in range(n_traces):
                current_trace_word_ids_cpp = self._trace_word_ids[i]
                f.write(struct.pack("<I", current_trace_word_ids_cpp.size())) # Write length of this trace's word_ids
                if current_trace_word_ids_cpp.size() > 0:
                    # Correctly construct bytes object from the C++ vector data
                    f.write(PyBytes_FromStringAndSize(<char*> current_trace_word_ids_cpp.data(), current_trace_word_ids_cpp.size() * sizeof(uint32_t)))

    @classmethod
    def load_fast(cls, str path):
        cdef uint32_t n_words, n_traces, i, cnt, word_id_count
        cdef double next_word_decay_appointment_h, next_trace_decay_appointment_h # New variables
        cdef bytes buf
        cdef tuple tpl
        cdef VocabularyModel m # Declare m as VocabularyModel type

        cdef vector[uint32_t] current_trace_word_ids_cpp_ref # Reference to the inner vector
        cdef uint32_t word_id_val

        # Temporary bytearrays for reading
        cdef bytearray temp_bytes_timestamps
        cdef bytearray temp_bytes_activations
        cdef bytearray temp_bytes_decay_factors
        cdef bytearray temp_bytes_word_ids


        with open(path, "rb") as f:
            # read header (ADAPTED: 2 uint32_t and 2 double values)
            buf = f.read(24) # 2*4 + 2*8 = 24 bytes
            n_words, n_traces, next_word_decay_appointment_h, next_trace_decay_appointment_h = struct.unpack("<IIdd", buf)

            # read your eight floats back
            buf = f.read(8 * 8)
            tpl = struct.unpack("<8d", buf)
            m = cls(tpl[0], tpl[1], tpl[2], tpl[3],
                            tpl[4], tpl[5], tpl[6], tpl[7])

            # Set the loaded appointment times
            m._next_word_decay_appointment_h = next_word_decay_appointment_h
            m._next_trace_decay_appointment_h = next_trace_decay_appointment_h


            # numeric arrays
            m.prof          = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.vol           = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.eff_prof      = np.frombuffer(f.read(n_words * 4), dtype='<f4').copy()
            m.encounters    = np.frombuffer(f.read(n_words * 4), dtype='<u4').copy()
            m.last_decayed  = np.frombuffer(f.read(n_words * 8), dtype='<f8').copy()

            # rebuild index
            m.idx = _WordIndex.__new__(_WordIndex)
            m.idx._i2w = []
            m.idx._w2i = {}
            for i in range(n_words):
                buf = f.read(4)
                cnt, = struct.unpack("<I", buf)
                w = f.read(cnt).decode('utf-8')
                m.idx._i2w.append(w)
                m.idx._w2i[w] = i

            # === REBUILD TRACE DATA FROM C++ VECTORS ===
            # Resize C++ vectors to hold data
            m._trace_timestamps_h.resize(n_traces)
            m._trace_activations.resize(n_traces)
            m._trace_decay_factors.resize(n_traces)
            m._trace_word_ids.resize(n_traces) # Resize outer vector

            # Read fixed-size trace data directly into C++ vector data pointers
            if n_traces > 0:
                # Create a bytearray of the correct size
                temp_bytes_timestamps = bytearray(n_traces * sizeof(double))
                f.readinto(temp_bytes_timestamps)
                # Copy data from the bytearray to the C++ vector
                memcpy(<char*> m._trace_timestamps_h.data(), <char*> temp_bytes_timestamps, n_traces * sizeof(double))

                temp_bytes_activations = bytearray(n_traces * sizeof(float))
                f.readinto(temp_bytes_activations)
                memcpy(<char*> m._trace_activations.data(), <char*> temp_bytes_activations, n_traces * sizeof(float))

                temp_bytes_decay_factors = bytearray(n_traces * sizeof(float))
                f.readinto(temp_bytes_decay_factors)
                memcpy(<char*> m._trace_decay_factors.data(), <char*> temp_bytes_decay_factors, n_traces * sizeof(float))


            m.word_to_traces = {} # Clear for rebuild
            for i in range(n_traces):
                buf = f.read(4)
                word_id_count, = struct.unpack("<I", buf)

                # Access the i-th inner vector and resize it
                current_trace_word_ids_cpp_ref = m._trace_word_ids[i]
                current_trace_word_ids_cpp_ref.resize(word_id_count)

                if word_id_count > 0:
                    # Create a bytearray for this inner vector
                    temp_bytes_word_ids = bytearray(word_id_count * sizeof(uint32_t))
                    f.readinto(temp_bytes_word_ids)
                    # Copy data from the bytearray to the C++ inner vector
                    memcpy(<char*> current_trace_word_ids_cpp_ref.data(), <char*> temp_bytes_word_ids, word_id_count * sizeof(uint32_t))


                # Rebuild reverse index (word_to_traces)
                # Iterate through the C++ vector directly for efficiency
                for word_id_val in current_trace_word_ids_cpp_ref:
                    m.word_to_traces.setdefault(word_id_val, []).append(i)

        return m

# Expose the API
__all__ = ["VocabularyModel", "promotion_times"]
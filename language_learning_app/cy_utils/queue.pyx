# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
#distutils: language = c++

from cy_utils.vocab_model cimport VocabularyModel
from cy_utils.llmodel cimport predict_answer_for_queue, calculate_unknownness
import itertools
cimport numpy as np
from collections import defaultdict
from datetime import datetime

# NEW: C++ containers and types
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
from libcpp.mutex cimport mutex, unique_lock # Import unique_lock here
from libcpp.condition_variable cimport condition_variable
from libcpp.chrono cimport milliseconds # For potential sleep/timeout

from cython.operator cimport dereference as deref

# Import the specialized priority queue type and HeapComparator from pxd
# HeapComparator is now imported from pxd which declares it from the .h file
from cy_utils.queue cimport LearningQueue, HeapItem, priority_queue_HeapItem, HeapComparator


cdef class LearningQueue:
    """
    An incremental‐learning queue of sentence‐units.
    """

    def __init__(self, master, lang):
        self.master    = master
        self.lang      = lang
        self.tmodel    = master.get_or_create_model(lang)
        
        # Initialize with the specialized type defined in pxd
        # Pass the HeapComparator instance to the constructor
        self._heaps_cpp = unordered_map[int, priority_queue_HeapItem]()
        self._heaps_cpp[0] = priority_queue_HeapItem(HeapComparator())
        self._heaps_cpp[1] = priority_queue_HeapItem(HeapComparator())
        self._heaps_cpp[2] = priority_queue_HeapItem(HeapComparator())

        self.items     = {}
        self.i2g       = {}

        self._words_map_v_cpp = unordered_map[string, int]()
        self._words_map_i_cpp = unordered_map[string, unordered_set[int]]()
        self._sent_map_cpp = unordered_map[int, float]()

        self.inverted_cpp = unordered_map[uint32_t, vector[int]]()
        
        self.counter   = itertools.count()
        
        self._dirty_items = deque[int]()

        # NEW: Threading related initializations
        self._worker_thread = NULL # Initialize pointer to NULL
        self._worker_running = False
        self._worker_paused = False
        self._is_processing_dirty = False # Flag to indicate if worker is actively processing dirty items

        # Start the worker thread immediately
        self._start_worker_thread()

    def __dealloc__(self):
        # Ensure worker thread is stopped and joined when object is deallocated
        self._stop_worker_thread()
        # Clean up dynamically allocated thread object
        if self._worker_thread != NULL:
            del self._worker_thread
            self._worker_thread = NULL

    cdef void _start_worker_thread(self):
        if not self._worker_running:
            self._worker_running = True
            # Pass 'self' to the C++ thread function to allow it to call methods
            # Use 'new' to dynamically allocate the thread object
            self._worker_thread = new thread(self._background_dirty_processor_thread_func, self)

    cdef void _stop_worker_thread(self):
        if self._worker_running:
            # Acquire mutex to safely signal stop and notify worker
            with self._dirty_queue_mutex:
                self._worker_running = False # Signal worker to stop
                self._dirty_queue_cv.notify_all() # Wake up worker if it's waiting
            
            # Join the thread outside the mutex lock to avoid deadlock if worker tries to acquire it
            if self._worker_thread != NULL and self._worker_thread.joinable(): # Check if joinable before joining
                self._worker_thread.join() # Wait for worker to finish
            
            # The 'del self._worker_thread' is moved to __dealloc__
            self._worker_running = False # Reset flag

    cdef void _signal_worker_pause(self):
        with self._dirty_queue_mutex: # Protect the flag
            self._worker_paused = True
            # No need to notify here, worker will check _worker_paused in its loop or on next wait

    cdef void _signal_worker_resume(self):
        with self._dirty_queue_mutex: # Protect the flag
            self._worker_paused = False
            self._dirty_queue_cv.notify_one() # Wake up worker

    cdef void _wait_for_worker_to_pause(self):
        # This is called by the main thread to ensure the worker is not processing
        # when the main thread needs to modify shared data.
        with self._dirty_queue_mutex:
            # Wait while worker is actively processing or if it's paused but not yet idle
            # Note: The condition here is critical. Wait if the worker is *processing* dirty items.
            # If it's merely paused and not processing, you might not need to wait.
            # The `_is_processing_dirty` flag is key.
            self._dirty_queue_cv.wait(self._dirty_queue_mutex, lambda: not self._is_processing_dirty)

    cdef void _background_dirty_processor_thread_func(self, LearningQueue self):
        # This function runs in a C++ thread, so it starts without the GIL
        cdef int items_to_process_per_batch = 20 # Configurable batch size
        cdef int processed_count
        cdef int iid_to_process
        cdef bint item_exists
        cdef unique_lock[mutex] lock

        while True:
            # 1. Acquire the unique_lock for the mutex
            lock(self._dirty_queue_mutex)
            
            # 2. Wait until conditions are met to proceed (not paused, not empty, or stopping)
            # The 'wait' method releases the lock before waiting and re-acquires it on notification.
            while (self._worker_paused or self._dirty_items.empty()) and self._worker_running:
                self._dirty_queue_cv.wait(lock)
            
            # 3. Check stop signal immediately after waking up
            if not self._worker_running:
                break # Exit thread loop

            # 4. Indicate that the worker is actively processing (while holding the lock)
            self._is_processing_dirty = True

            # 5. Process a batch of items
            processed_count = 0
            while processed_count < items_to_process_per_batch:
                item_exists = False

                # Check if there are items to process or if we should pause (still under mutex)
                if self._dirty_items.empty() or self._worker_paused:
                    break # No more items or paused, break from batch processing

                # Get item from queue (still under mutex)
                iid_to_process = self._dirty_items.front()
                self._dirty_items.pop_front()
                
                # Release C++ lock to acquire GIL for Python object access and method calls
                with nogil: # Ensures C++ lock is released
                    lock.unlock() # Explicitly unlock C++ mutex before acquiring GIL
                
                with gil: # Acquire GIL
                    # Inside GIL, the C++ mutex is NOT held.
                    if iid_to_process in self.items:
                        # Call update_item directly, passing iid_to_process
                        self.update_item(iid_to_process, datetime.now().timestamp() / 3600.0)
                        item_exists = True
                    else:
                        item_exists = False # Item was removed by other means

                # Re-acquire C++ lock after releasing GIL (for the next iteration or loop exit)
                with nogil: # Ensures GIL is released
                    lock.lock() # Re-acquire C++ mutex

                if item_exists:
                    processed_count += 1
                
                # Small pause to yield control, if needed (optional)
                # std::this_thread::sleep_for(milliseconds(1)); # Requires <chrono> and <thread>

            # 6. After processing the batch, update status while holding the lock
            self._is_processing_dirty = False # Indicate processing is done for this batch
            # 7. Notify main thread if queue is empty or processing is done
            if self._dirty_items.empty():
                self._dirty_queue_cv.notify_all() # Notify any threads waiting for _is_processing_dirty to be false or queue empty

            # 'lock' automatically releases the mutex when it goes out of scope at the end of this `while True` loop iteration.

        # Thread is stopping, 'lock' will be destroyed and mutex released.
        pass

    cpdef void add_item(self,
                        object item,
                        int iid,
                        double now_h):
        """
        Score a single item and push it into the appropriate heap.
        """
        cdef int grp
        cdef float key

        # Call _score_and_group directly, accessing maps from self
        grp, key = self._score_and_group(item, iid, now_h)

        self.items[iid] = item
        cdef str w_str
        cdef uint32_t wid
        for w_str in item["unit"]["words"]:
            if self.tmodel.idx.has_word(w_str):
                wid = self.tmodel.idx.get_id(w_str)
                # Protect inverted_cpp if accessed by worker (it is)
                with self._dirty_queue_mutex:
                    self.inverted_cpp[wid].push_back(iid)

        self._add_to_heap(iid, grp, key)

    cpdef void build_from_input(self, list items):
        """
        items: list of {"filename", "unit"}
        """
        # For bulk loading, it's better to pause the worker thread, load, then resume.
        # This prevents the worker from trying to process incomplete data during build.
        self._signal_worker_pause()
        self._wait_for_worker_to_pause() # Wait for worker to truly pause

        cdef int i
        cdef Py_ssize_t n = len(items)
        cdef double now_h = datetime.now().timestamp() / 3600.0
        for i in range(n):
            # No need to protect _dirty_items here as worker is paused and _add_to_heap takes care of its lock
            # Call add_item directly, accessing maps from self
            self.add_item(items[i], i, now_h)
        
        self._signal_worker_resume() # Resume worker after build

    cpdef tuple peek_next(self, int grp):
        """
        Look at the top of the heap for group=grp.
        No longer directly processes dirty items.
        """
        # The background thread handles dirty item processing.
        # For peek, we don't necessarily need to wait for the worker to finish.
        # Just need to protect the heap access itself.
        
        cdef priority_queue_HeapItem *heap_ptr
        if self._heaps_cpp.count(grp) == 0:
            return None, None
        
        cdef HeapItem top_item

        # Protect heap access
        with self._dirty_queue_mutex:
            heap_ptr = &self._heaps_cpp.at(grp)
            if heap_ptr[0].empty():
                return None, None
            
            top_item = heap_ptr[0].top()
            # Return a copy of the item; actual removal happens in pop_next
            # Use self.items[top_item.iid] to get the Python object
            return self.items[top_item.iid], top_item.iid

    cpdef tuple pop_next(self, int grp):
        """
        Pop the top element from group=grp.
        No longer directly processes dirty items.
        """
        # Similar to peek_next, rely on background thread for processing.
        
        cdef priority_queue_HeapItem *heap_ptr
        if self._heaps_cpp.count(grp) == 0:
            return None, None
        
        cdef HeapItem top_item

        # Protect heap modification
        with self._dirty_queue_mutex:
            heap_ptr = &self._heaps_cpp.at(grp)
            if heap_ptr[0].empty():
                return None, None
            
            top_item = heap_ptr[0].top()
            heap_ptr[0].pop() # Remove from C++ heap
        
            # Remove from Python dicts
            self.i2g.pop(top_item.iid, None)
            return self.items.pop(top_item.iid), top_item.iid

    cpdef void remove_item(self, int iid):
        """
        Remove an item entirely from all queues and indices.
        """
        # ADAPTED: grp is int
        grp = self.i2g.pop(iid, -1)
        # Note: If an item is removed from a heap (by popping), it's truly gone.
        # If it's removed by this method, it's just removed from i2g and items.
        # For the C++ heap, a "soft delete" (pushing with high key) was done in _add_to_heap.
        # The actual removal from the heap happens when it's popped.
        
        cdef str w_str
        cdef uint32_t wid
        cdef vector[int]* iids_vec_ptr

        if iid not in self.items:
            return
        
        cdef Py_ssize_t k

        # Remove from inverted_cpp. This section must be mutex-protected.
        # The worker thread also reads/modifies inverted_cpp.
        for w_str in self.items[iid]["unit"]["words"]:
            if self.tmodel.idx.has_word(w_str):
                wid = self.tmodel.idx.get_id(w_str)
                with self._dirty_queue_mutex: # Protect inverted_cpp
                    if self.inverted_cpp.count(wid) > 0:
                        iids_vec_ptr = &self.inverted_cpp.at(wid)
                        
                        # Find and erase the iid from the vector
                        # This loop iterates through the vector.
                        # It's O(N) for the vector, but safe within the lock.
                        for k in range(iids_vec_ptr[0].size()):
                            if iids_vec_ptr[0][k] == iid:
                                iids_vec_ptr[0].erase(iids_vec_ptr[0].begin() + k)
                                break
                        # If vector becomes empty, remove the word_id entry from the map
                        if iids_vec_ptr[0].empty():
                            self.inverted_cpp.erase(wid)

        self.items.pop(iid, None) # Remove from Python dict

    cpdef void process_answer(self, int iid, int feedback_level):
        """
        Called after the user answers a question. Updates the master model,
        then removes & re-scores any affected dependents.
        """
        # 1. Signal worker to pause and wait for it to finish current processing
        self._signal_worker_pause()
        self._wait_for_worker_to_pause() # Ensure worker is truly paused and not touching shared data

        # 2. Main thread proceeds with its immediate, critical work safely
        item = self.items[iid]
        self.master.update_knowledge(item["unit"]["words"],
                                     self.lang,
                                     feedback_level)
        self.remove_item(iid) # This will also update inverted_cpp

        cdef set changed_words_py = set(item["unit"]["words"])
        cdef set deps_to_add_to_dirty_queue = set()

        cdef str w_str
        cdef uint32_t wid
        cdef vector[int]* iids_vec_ptr

        cdef string w_cpp_key

        for w_str in changed_words_py:
            w_cpp_key = w_str.encode('utf-8')
            # Protect these maps during modification by the main thread
            with self._dirty_queue_mutex:
                self._words_map_v_cpp.erase(w_cpp_key) # Clear cached values
                self._words_map_i_cpp.erase(w_cpp_key) # Clear cached values
            
            if self.tmodel.idx.has_word(w_str):
                wid = self.tmodel.idx.get_id(w_str)
                # inverted_cpp is protected by remove_item via its mutex
                # No extra mutex needed here *if* remove_item handles all necessary locks for inverted_cpp.
                # Since inverted_cpp is protected by self._dirty_queue_mutex in remove_item,
                # and this `process_answer` function pauses the worker and takes over,
                # it's safe to directly access inverted_cpp here *after* remove_item.
                # However, adding a `with self._dirty_queue_mutex:` here too is safer for consistency.
                with self._dirty_queue_mutex:
                    if self.inverted_cpp.count(wid) > 0:
                        iids_vec_ptr = &self.inverted_cpp.at(wid)
                        for d_iid in iids_vec_ptr[0]:
                            deps_to_add_to_dirty_queue.add(d_iid)

        for d_iid in deps_to_add_to_dirty_queue:
            if d_iid in self.items: # Only re-add if item still exists
                with self._dirty_queue_mutex: # Protect _sent_map_cpp and _dirty_items when adding
                    self._sent_map_cpp.erase(d_iid) # Invalidate cached sentence score
                    self._dirty_items.push_back(d_iid) # Add to dirty queue for re-scoring
                    self._dirty_queue_cv.notify_one() # Notify worker that there's a new item

        # 3. Signal worker to resume after adding new items
        self._signal_worker_resume()

    cpdef void update_item(self,
                           int iid,
                           double now_h):
        """
        Re-score an existing item (after some words changed).
        This is called by both main thread (e.g., build_from_input) and worker thread.
        """
        cdef int grp
        cdef float key
        # Call _score_and_group directly, accessing maps from self
        # This function internally handles locking for accesses to _words_map_v_cpp etc.
        grp, key = self._score_and_group(self.items[iid], iid, now_h)
        self._add_to_heap(iid, grp, key)

    cpdef Py_ssize_t size(self, int grp=-1):
        """
        Total size of all queues, or size of one group.
        """
        cdef Py_ssize_t tot
        cdef unordered_map[int, priority_queue_HeapItem].iterator it # Use specialized type here
        if grp == -1:
            tot = 0
            # Need to protect _heaps_cpp as worker might modify it (though group keys are stable).
            # Safer to assume it's shared.
            with self._dirty_queue_mutex: # Reusing the mutex for general shared data access
                for it in self._heaps_cpp.begin():
                    tot += deref(it).second.size()
            return tot
        elif self._heaps_cpp.count(grp) > 0:
            with self._dirty_queue_mutex: # Protect specific heap access
                return self._heaps_cpp.at(grp).size()
        return 0

    cpdef tuple _score_and_group(self,
                                 object item,
                                 int iid,
                                 double now_h):
        """
        Returns (grp,key). Always calls predict_answer in promotion-data mode.
        """
        cdef list words = item["unit"]["words"]
        cdef int grp
        cdef float key
        cdef float[::1] effs

        # Acquire mutex before accessing/modifying shared C++ maps
        with self._dirty_queue_mutex:
            (grp, effs) = predict_answer_for_queue(self.tmodel, words,
                                                   self._words_map_v_cpp, self._words_map_i_cpp, self._sent_map_cpp, iid)

            if grp == 0:
                key = calculate_unknownness(effs)
            elif grp == 2:
                key = self.tmodel.predict_understanding(words, now_h)
            else: # grp == 1
                key = self._promotion_potential(words, effs, now_h)

        return grp, key

    cpdef void _add_to_heap(self, int iid, int grp, float key):
        """
        Maintains a stable heap per group using C++ priority_queue.
        """
        cdef int old_grp = self.i2g.get(iid, -1)
        
        # This section needs to be protected as it modifies _heaps_cpp and i2g,
        # which are shared with other methods (like pop_next, size, process_answer).
        cdef HeapItem dummy_item
        cdef HeapItem new_heap_item
        with self._dirty_queue_mutex: # Protect heap modifications and i2g
            if old_grp != -1 and old_grp != grp:
                if self._heaps_cpp.count(old_grp) > 0:
                    # Push a dummy item to effectively "remove" the old one from the heap.
                    # It will eventually be popped when it reaches the top.
                    dummy_item.key = 1e9 # Very high key to send it to bottom (effectively delete)
                    dummy_item.insertion_order = next(self.counter) # Needs a unique ID
                    dummy_item.iid = iid
                    self._heaps_cpp.at(old_grp).push(dummy_item)

            
            new_heap_item.key = key
            new_heap_item.insertion_order = next(self.counter) # Get a new unique insertion order
            new_heap_item.iid = iid
            
            # Ensure the heap for this group exists before pushing
            if self._heaps_cpp.count(grp) == 0:
                self._heaps_cpp[grp] = priority_queue_HeapItem(HeapComparator()) # Initialize with comparator
            
            self._heaps_cpp.at(grp).push(new_heap_item)
            self.i2g[iid] = grp # Update i2g to the new group

    cpdef float _promotion_potential(self, list words, float[::1] eff_prof, double now_h):
        """
        How strongly these words “pull” promotion of partially‐known sentences.
        ADAPTED to use C++ words_map and sent_map.
        """
        cdef float pot = 0.0
        cdef float s
        cdef int iid_val
        cdef float pred
        cdef float total
        cdef str w_str
        cdef string w_cpp_key
        
        cdef unordered_map[string, int].iterator words_map_v_it
        cdef unordered_map[string, unordered_set[int]].iterator words_map_i_it
        cdef unordered_map[int, float].iterator sent_map_it
        cdef unordered_set[int].iterator iid_set_it

        for w_str in words:
            w_cpp_key = w_str.encode('utf-8')
            # Protect these lookups as they are shared with the worker thread
            with self._dirty_queue_mutex:
                words_map_v_it = self._words_map_v_cpp.find(w_cpp_key)
                words_map_i_it = self._words_map_i_cpp.find(w_cpp_key)
                
                if words_map_v_it != self._words_map_v_cpp.end() and words_map_i_it != self._words_map_i_cpp.end():
                    s = 0.0
                    for iid_set_it in deref(words_map_i_it).second.begin():
                        sent_map_it = self._sent_map_cpp.find(deref(iid_set_it))
                        if sent_map_it != self._sent_map_cpp.end():
                            s += deref(sent_map_it).second
                    pot += s / max(1, deref(words_map_v_it).second)

        if pot == 0.0:
            pred = self.tmodel.predict_understanding(words, now_h)
            total = 0.0
            for val in eff_prof:
                total += (val if val > pred else pred)
            return total / max(1, len(words))

        return -pot
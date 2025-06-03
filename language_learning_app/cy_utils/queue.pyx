# cy_utils/queue.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from cy_utils.vocab_model cimport VocabularyModel
from cy_utils.llmodel cimport predict_answer, calculate_unknownness
import heapdict
import itertools
cimport numpy as np
from collections import defaultdict
from datetime import datetime

from cy_utils.queue cimport LearningQueue

cdef class LearningQueue:
    """
    An incremental‐learning queue of sentence‐units.
    """

    def __init__(self, master, lang):
        self.master    = master
        self.lang      = lang
        self.tmodel    = master.get_or_create_model(lang)
        self.heaps     = {
            0.0: heapdict.heapdict(),
            0.5: heapdict.heapdict(),
            1.0: heapdict.heapdict()
        }
        self.items     = {}
        self.i2g       = {}
        self.words_map = {}
        self.sent_map  = {}
        self.inverted  = defaultdict(set)
        self.counter   = itertools.count()

    cpdef void add_item(self,
                        object item,
                        int iid,
                        object promo_data,
                        double now_h):
        """
        Score a single item and push it into the appropriate heap.
        """
        cdef float grp, key
        grp, key = self._score_and_group(item, promo_data, now_h)

        # store
        self.items[iid] = item
        # invert index by word
        for w in item["unit"]["words"]:
            self.inverted[w].add(iid)
        # push into heap
        self._add_to_heap(iid, grp, key)

    cpdef tuple build_from_input(self, list items):
        """
        items: list of {"filename", "index", "unit"}
        returns (words_map, sent_map) after building.
        """
        cdef int i
        cdef Py_ssize_t n = len(items)
        cdef double now_h = datetime.now().timestamp() / 3600.0
        for i in range(n):
            # always pass promo_data tuple
            self.add_item(items[i], i, (self.words_map, self.sent_map, i), now_h)
        return self.words_map, self.sent_map

    cpdef tuple peek_next(self, float grp):
        """
        Look at the top of the heap for group=grp.
        """
        if not self.heaps[grp]:
            return None, None
        iid, _ = self.heaps[grp].peekitem()
        return self.items[iid], iid

    cpdef tuple pop_next(self, float grp):
        """
        Pop the top element from group=grp.
        """
        if not self.heaps[grp]:
            return None, None
        iid, _ = self.heaps[grp].popitem()
        self.i2g.pop(iid, None)
        return self.items.pop(iid), iid

    cpdef void remove_item(self, int iid):
        """
        Remove an item entirely from all queues and indices.
        """
        # remove from heaps
        grp = self.i2g.pop(iid, None)
        if grp is not None:
            self.heaps[grp].pop(iid, None)

        # remove from inverted index
        for w in self.items[iid]["unit"]["words"]:
            self.inverted[w].discard(iid)

        # drop it
        self.items.pop(iid, None)

    cpdef void process_answer(self, int iid, int feedback_level):
        """
        Called after the user answers a question.  Updates the master model,
        then removes & re-scores any affected dependents.
        """
        item = self.items[iid]
        # update underlying model
        self.master.update_knowledge(item["unit"]["words"],
                                     self.lang,
                                     feedback_level)
        # remove answered item
        self.remove_item(iid)
        cdef double now_h = datetime.now().timestamp() / 3600.0

        # any sentence sharing a word needs re-scoring
        cdef set changed = set(item["unit"]["words"])
        cdef set deps = set()
        for w in changed:
            # clear old promotion data for word
            self.words_map.pop(w, None)
            deps |= self.inverted[w]

        for d in deps:
            if d in self.items:
                self.sent_map.pop(d, None)
                self.update_item(d, (self.words_map, self.sent_map, d), now_h)

    cpdef void update_item(self,
                           int iid,
                           object promo_data,
                           double now_h):
        """
        Re-score an existing item (after some words changed).
        """
        cdef float grp, key
        grp, key = self._score_and_group(self.items[iid],
                                         promo_data,
                                         now_h)
        self._add_to_heap(iid, grp, key)

    cpdef Py_ssize_t size(self, float grp=-1.0):
        """
        Total size of all queues, or size of one group.
        """
        cdef Py_ssize_t tot
        if grp < 0.0:
            tot = 0
            for h in self.heaps.values():
                tot += len(h)
            return tot
        return len(self.heaps[grp])

    cpdef tuple _score_and_group(self,
                                               object item,
                                               object promo_data,
                                               double now_h):
        """
        Returns (grp,key).  Always calls predict_answer in promotion-data mode.
        """
        cdef list words = item["unit"]["words"]
        cdef float grp, key
        cdef float[::1] effs
        # always build promotion‐data
        (grp, effs) = predict_answer(self.tmodel, words, True, promo_data)

        if grp == 0.0:
            key = calculate_unknownness(effs)
        elif grp == 1.0:
            # fallback to Python model’s predict
            key = self.tmodel.predict_understanding(words, now_h)
        else:
            # partial group, use promotion potential
            key = self._promotion_potential(words, effs, now_h)

        return grp, key

    cpdef void _add_to_heap(self, int iid, float grp, float key):
        """
        Maintains a stable heapdict per group.
        """
        cdef float old = self.i2g.get(iid, -1.0)
        if old >= 0.0:
            self.heaps[old].pop(iid, None)
        # tie‐break by insertion order
        self.heaps[grp][iid] = (key, next(self.counter))
        self.i2g[iid] = grp

    cpdef float _promotion_potential(self, list words, float[::1] eff_prof, double now_h):
        """
        How strongly these words “pull” promotion of partially‐known sentences.
        """
        cdef float pot = 0.0
        cdef float s
        cdef int iid
        cdef float pred
        cdef float total
        for w in words:
            data = self.words_map.get(w)
            if not data:
                continue
            s = 0.0
            for iid in data["i"]:
                s += self.sent_map.get(iid, 0.0)
            pot += s / max(1, data["v"])

        if pot == 0.0:
            # fallback: average above‐threshold activation
            pred = self.tmodel.predict_understanding(words, now_h)
            total = 0.0
            for val in eff_prof:
                total += (val if val > pred else pred)
            return total / len(words)

        return -pot
// cy_utils/heap_comparator.h
#ifndef CY_UTILS_HEAP_COMPARATOR_H
#define CY_UTILS_HEAP_COMPARATOR_H

#include "heap_item.h" // Include HeapItem definition

struct HeapComparator {
    // C++ operator() implementation for min-heap behavior
    // For a min-heap using std::priority_queue (which is a max-heap by default),
    // this comparator must return true if 'a' is "less preferred" than 'b',
    // meaning 'a' has a larger key or larger insertion_order (for tie-breaking).
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.key != b.key) {
            return a.key > b.key; // For min-heap, smaller key means higher priority, so return true if a.key is GREATER
        }
        // Tie-breaker: Smaller insertion_order means higher priority for items with same key
        // So, if a.insertion_order is GREATER, it has lower priority.
        return a.insertion_order > b.insertion_order;
    }
};

#endif // CY_UTILS_HEAP_COMPARATOR_H

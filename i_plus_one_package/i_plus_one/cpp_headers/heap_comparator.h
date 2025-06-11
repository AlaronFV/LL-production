#ifndef I_PLUS_ONE_HEAP_COMPARATOR_H
#define I_PLUS_ONE_HEAP_COMPARATOR_H

#include "heap_item.h"

namespace i_plus_one {

// The comparator for the std::set to make it act like a max-priority queue.
// It sorts items with higher keys first.
struct HeapComparator {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.key != b.key) {
            return a.key > b.key; // Higher key comes first
        }
        // If keys are equal, older items (smaller insertion_order) come first.
        return a.insertion_order < b.insertion_order;
    }
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_HEAP_COMPARATOR_H
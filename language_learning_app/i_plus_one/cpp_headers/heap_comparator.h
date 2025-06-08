#ifndef I_PLUS_ONE_HEAP_COMPARATOR_H
#define I_PLUS_ONE_HEAP_COMPARATOR_H

#include "heap_item.h"

namespace i_plus_one {

struct HeapComparator {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        if (a.key != b.key) {
            return a.key > b.key;
        }
        return a.insertion_order > b.insertion_order;
    }
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_HEAP_COMPARATOR_H
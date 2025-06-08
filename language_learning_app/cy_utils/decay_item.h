// cy_utils/decay_item.h
#ifndef CY_UTILS_DECAY_ITEM_H
#define CY_UTILS_DECAY_ITEM_H

#include <cstdint> // Required for std::uint32_t

struct DecayItem {
    double next_decay_time;
    std::uint32_t id_or_idx; // Using std::uint32_t for C++ standard integer types

    // C++ operator< implementation
    // For a min-heap (which your priority_queue with `greater` will be),
    // a standard less-than operator is usually defined.
    // `greater<DecayItem>` will then reverse the order, making it a min-heap.
    bool operator<(const DecayItem& other) const {
        return this->next_decay_time < other.next_decay_time;
        // If next_decay_time can be equal and you need a stable sort,
        // you might add a tie-breaker, e.g.:
        // if (this->next_decay_time != other.next_decay_time) {
        //     return this->next_decay_time < other.next_decay_time;
        // }
        // return this->id_or_idx < other.id_or_idx;
    }
};

#endif // CY_UTILS_DECAY_ITEM_H
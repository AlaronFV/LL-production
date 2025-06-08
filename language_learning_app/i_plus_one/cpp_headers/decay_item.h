#ifndef I_PLUS_ONE_DECAY_ITEM_H
#define I_PLUS_ONE_DECAY_ITEM_H

#include <cstdint>

namespace i_plus_one {

struct DecayItem {
    double next_decay_time;
    uint32_t id_or_idx;

    bool operator<(const DecayItem& other) const {
        return this->next_decay_time < other.next_decay_time;
    }
};

struct DecayItemComparator {
    bool operator()(const DecayItem& a, const DecayItem& b) const {
        return a.next_decay_time > b.next_decay_time;
    }
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_DECAY_ITEM_H
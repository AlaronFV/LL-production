#ifndef I_PLUS_ONE_HEAP_ITEM_H
#define I_PLUS_ONE_HEAP_ITEM_H

namespace i_plus_one {

struct HeapItem {
    float key;
    long long insertion_order;
    int iid;

    // Required for std::set to find and erase items.
    bool operator<(const HeapItem& other) const {
        if (key != other.key) {
            return key < other.key;
        }
        return insertion_order > other.insertion_order;
    }
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_HEAP_ITEM_H
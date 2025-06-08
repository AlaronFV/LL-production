// cy_utils/heap_item.h
#ifndef CY_UTILS_HEAP_ITEM_H
#define CY_UTILS_HEAP_ITEM_H


// Define HeapItem as a C++ struct
struct HeapItem {
    float key;
    long long insertion_order;
    int iid; // Assuming 'int', adjust to uint32_t if necessary

    // C++ operator< implementation
    // For a min-heap using std::priority_queue with std::greater<HeapItem>,
    // this operator should define a standard "less than" comparison.
    bool operator<(const HeapItem& other) const {
        if (this->key != other.key) {
            return this->key < other.key;
        }
        // Tie-breaker: Smaller insertion_order means higher priority for items with same key
        return this->insertion_order < other.insertion_order;
    }
};

#endif // CY_UTILS_HEAP_ITEM_H
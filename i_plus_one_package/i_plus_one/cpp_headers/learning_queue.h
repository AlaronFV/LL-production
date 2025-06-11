#ifndef I_PLUS_ONE_LEARNING_QUEUE_H
#define I_PLUS_ONE_LEARNING_QUEUE_H

#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <array>
#include <pybind11/pybind11.h>

#include "vocab_model.h"
#include "heap_item.h"
#include "heap_comparator.h"

namespace py = pybind11;

namespace i_plus_one {

class LearningQueue {
public:
    LearningQueue(std::shared_ptr<VocabularyModel> model);

    void build_from_input(const py::list& items);
    int peek_next();
    void process_answer(int iid, int feedback_level);
    size_t size(int grp = -1) const;

private:
    // --- Core Data Structures ---
    using SearchableHeap = std::set<HeapItem, HeapComparator>;

    struct ItemData {
        int iid;
        std::vector<uint32_t> word_ids;
        std::vector<uint32_t> unknown_word_ids_cache;
        int group;
    };

    std::shared_ptr<VocabularyModel> tmodel;
    
    std::unordered_map<int, ItemData> _item_storage;
    std::unordered_set<int> _active_iids;

    std::array<SearchableHeap, 3> _heaps;
    std::unordered_map<int, HeapItem> _iid_to_heap_item;

    std::unordered_map<uint32_t, std::vector<int>> _word_to_iids;
    std::unordered_map<uint32_t, std::vector<int>> _unknown_word_to_group1_iids;

    std::unordered_map<int, float> _sent_map;
    std::unordered_map<uint32_t, int> _words_map_v;
    
    long long _counter = 0;

    // --- Private Methods ---

    // Build process
    void _pass1_ingest_and_group(const py::list& items);
    void _pass2_build_caches_and_dependency_graph();
    void _pass3_final_score_and_insert(double now_h);

    // Scoring
    float _calculate_key(const ItemData& item, const std::vector<float>& eff_profs, double now_h);
    float _get_promotion_potential(const ItemData& item);
    int _get_group(const std::vector<uint32_t>& word_ids, std::vector<float>& out_eff_profs, std::vector<uint32_t>& out_unknown_word_ids);

    // Heap operations
    void _update_heap(int iid, int new_group, float new_key);
    void _remove_from_heap(int iid);

    // Transactional update helpers
    void _maintain_caches_and_graph(const std::vector<std::pair<int, int>>& group_changes);
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_LEARNING_QUEUE_H
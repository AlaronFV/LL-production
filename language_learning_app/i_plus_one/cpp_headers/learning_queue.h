#ifndef I_PLUS_ONE_LEARNING_QUEUE_H
#define I_PLUS_ONE_LEARNING_QUEUE_H

#include <string>
#include <vector>
#include <queue>
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
    py::tuple pop_next();
    void process_answer(int iid, int feedback_level);
    
    size_t size(int grp = -1) const;

private:
    using Heap = std::priority_queue<HeapItem, std::vector<HeapItem>, HeapComparator>;

    // CHANGED: Now takes vector of strings
    void _add_item_internal(int iid, const std::vector<std::string>& words);
    // CHANGED: Now takes vector of strings
    std::pair<int, float> _score_and_group(int iid, const std::vector<std::string>& words);
    void _add_to_heap(int iid, int grp, float key);
    // CHANGED: Now takes vector of strings
    float _promotion_potential(const std::vector<std::string>& words, const std::vector<float>& eff_prof);
    void _rescore_items(const std::unordered_set<int>& iids_to_rescore);

    std::shared_ptr<VocabularyModel> tmodel;

    std::unordered_map<int, Heap> _heaps;
    // CHANGED: Stores the original words as strings, not pre-resolved IDs
    std::unordered_map<int, std::vector<std::string>> item_words;
    std::unordered_map<int, int> iid_to_group;
    std::unordered_set<int> active_iids;

    std::array<size_t, 3> _active_heap_sizes = {0, 0, 0};

    // For promotion potential calculation - keys are now strings
    std::unordered_map<std::string, int> _words_map_v;
    std::unordered_map<std::string, std::unordered_set<int>> _words_map_i;
    std::unordered_map<int, float> _sent_map;

    // Inverted index from word ID to item IDs
    std::unordered_map<uint32_t, std::vector<int>> word_id_to_iids;
    
    long long counter = 0;
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_LEARNING_QUEUE_H
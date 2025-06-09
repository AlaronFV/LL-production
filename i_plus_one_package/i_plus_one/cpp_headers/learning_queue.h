#ifndef I_PLUS_ONE_LEARNING_QUEUE_H
#define I_PLUS_ONE_LEARNING_QUEUE_H

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
    int peek_next();
    void process_answer(int iid, int feedback_level);
    
    size_t size(int grp = -1) const;

private:
    using Heap = std::priority_queue<HeapItem, std::vector<HeapItem>, HeapComparator>;

    void _add_item_internal(int iid, const py::list& words, double now_h);
    std::pair<int, float> _score_and_group(int iid, const std::vector<uint32_t>& word_ids, double now_h);
    void _add_to_heap(int iid, int grp, float key);
    float _promotion_potential(const std::vector<uint32_t>& word_ids, const std::vector<float>& eff_prof, double now_h);
    void _rescore_items(const std::unordered_set<int>& iids_to_rescore, double now_h);
    void _pop_next();
    std::shared_ptr<VocabularyModel> tmodel;

    std::unordered_map<int, Heap> _heaps;
    std::unordered_map<int, std::vector<uint32_t>> item_word_ids;
    std::unordered_map<int, int> iid_to_group;
    std::unordered_set<int> active_iids;

    std::array<size_t, 3> _active_heap_sizes = {0, 0, 0};

    std::unordered_map<uint32_t, int> _words_map_v;
    std::unordered_map<uint32_t, std::unordered_set<int>> _words_map_i;
    std::unordered_map<int, float> _sent_map;

    std::unordered_map<uint32_t, std::vector<int>> word_id_to_iids;
    
    long long counter = 0;
};

} // namespace i_plus_one

#endif // I_PLUS_ONE_LEARNING_QUEUE_H
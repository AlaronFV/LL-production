#include "cpp_headers/learning_queue.h"
#include "cpp_headers/llmodel.h"
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace i_plus_one {

LearningQueue::LearningQueue(std::shared_ptr<VocabularyModel> model) : tmodel(model) {
    if (!tmodel) {
        throw std::invalid_argument("LearningQueue must be initialized with a valid VocabularyModel instance.");
    }
    _heaps[0] = Heap(HeapComparator());
    _heaps[1] = Heap(HeapComparator());
    _heaps[2] = Heap(HeapComparator());
}

void LearningQueue::build_from_input(const py::list& items) {
    py::gil_scoped_acquire acquire;
    int iid_counter = 0;
    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    for (const auto& item_handle : items) {
        py::dict item = item_handle.cast<py::dict>();
        
        std::vector<std::string> words = item["unit"]["words"].cast<std::vector<std::string>>();
        _add_item_internal(iid_counter, words, now_h);
        iid_counter++;
    }
}

void LearningQueue::_add_item_internal(int iid, const std::vector<std::string>& words, double now_h) {
    std::vector<uint32_t> word_ids;
    word_ids.reserve(words.size());
    for (const auto& w : words) {
        uint32_t wid = tmodel->get_idx().get_id(w);
        word_ids.push_back(wid);
        word_id_to_iids[wid].push_back(iid);
    }
    // CRITICAL FIX: No longer call _resize_arrays here.
    // The model now handles its own arrays only when a word is processed.

    item_word_ids[iid] = word_ids;
    active_iids.insert(iid);

    auto [grp, key] = _score_and_group(iid, word_ids, now_h);
    _add_to_heap(iid, grp, key);
    
    _active_heap_sizes[grp]++;
}

std::pair<int, float> LearningQueue::_score_and_group(int iid, const std::vector<uint32_t>& word_ids, double now_h) {
    auto [grp, effs] = predict_answer_for_queue(*tmodel, word_ids, _words_map_v, _words_map_i, _sent_map, iid);
    
    float key;
    if (grp == 0) {
        key = calculate_unknownness(effs);
    } else if (grp == 2) {
        key = tmodel->_predict_understanding_by_id(word_ids, now_h);
    } else {
        key = _promotion_potential(word_ids, effs, now_h);
    }
    return {grp, key};
}

void LearningQueue::_add_to_heap(int iid, int grp, float key) {
    _heaps[grp].push({key, counter++, iid});
    iid_to_group[iid] = grp;
}

int LearningQueue::pop_next() {
    int grp_to_pop = -1;
    if (_active_heap_sizes[2] > 0) grp_to_pop = 2;
    else if (_active_heap_sizes[1] > 0) grp_to_pop = 1;
    else if (_active_heap_sizes[0] > 0) grp_to_pop = 0;

    if (grp_to_pop == -1) {
        py::gil_scoped_acquire acquire;
        return -1;
    }
    
    while (!_heaps[grp_to_pop].empty()) {
        HeapItem top_item = _heaps[grp_to_pop].top();
        _heaps[grp_to_pop].pop();

        if (active_iids.count(top_item.iid)) {
            py::gil_scoped_acquire acquire;
            return top_item.iid;
        }
    }
    
    py::gil_scoped_acquire acquire;
    return -1;
}

void LearningQueue::process_answer(int iid, int feedback_level) {
    if (!active_iids.count(iid)) return;

    const auto& word_ids = item_word_ids.at(iid);
    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    tmodel->update_proficiency(word_ids, feedback_level / 2.0f, now_h);
    tmodel->save_fast(tmodel->get_model_path());

    int old_grp = iid_to_group.at(iid);
    _active_heap_sizes[old_grp]--;
    
    active_iids.erase(iid);
    iid_to_group.erase(iid);
    item_word_ids.erase(iid);
    
    for (uint32_t wid : word_ids) {
        if (word_id_to_iids.count(wid)) {
            auto& vec = word_id_to_iids.at(wid);
            vec.erase(std::remove(vec.begin(), vec.end(), iid), vec.end());
            if (vec.empty()) word_id_to_iids.erase(wid);
        }
    }

    std::unordered_set<int> dependents;
    for (uint32_t wid : word_ids) {
        if (word_id_to_iids.count(wid)) {
            for (int dep_iid : word_id_to_iids.at(wid)) {
                dependents.insert(dep_iid);
            }
        }
    }
    _rescore_items(dependents, now_h);
}

void LearningQueue::_rescore_items(const std::unordered_set<int>& iids_to_rescore, double now_h) {
    for (int iid : iids_to_rescore) {
        if (!active_iids.count(iid)) continue;

        int old_grp = iid_to_group.at(iid);
        
        const auto& word_ids = item_word_ids.at(iid);
        auto [new_grp, new_key] = _score_and_group(iid, word_ids, now_h);
        
        _add_to_heap(iid, new_grp, new_key);

        if (old_grp != new_grp) {
            _active_heap_sizes[old_grp]--;
            _active_heap_sizes[new_grp]++;
        }
    }
}

size_t LearningQueue::size(int grp) const {
    if (grp >= 0 && grp < 3) {
        return _active_heap_sizes[grp];
    }
    return std::accumulate(_active_heap_sizes.begin(), _active_heap_sizes.end(), 0);
}

float LearningQueue::_promotion_potential(const std::vector<uint32_t>& word_ids, const std::vector<float>& eff_prof, double now_h) {
    float pot = 0.0f;
    for (uint32_t wid : word_ids) {
        if (_words_map_v.count(wid) && _words_map_i.count(wid)) {
            float s = 0.0f;
            for (int iid_val : _words_map_i.at(wid)) {
                if (_sent_map.count(iid_val)) {
                    s += _sent_map.at(iid_val);
                }
            }
            pot += s / std::max(1, _words_map_v.at(wid));
        }
    }

    if (pot == 0.0f) {
        float pred = tmodel->_predict_understanding_by_id(word_ids, now_h);
        float total = 0.0f;
        for (float val : eff_prof) {
            total += (val > pred) ? val : pred;
        }
        return total / std::max(1.0f, (float)word_ids.size());
    }
    return -pot;
}

} // namespace i_plus_one
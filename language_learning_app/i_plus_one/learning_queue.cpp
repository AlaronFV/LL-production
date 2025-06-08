#include "cpp_headers/learning_queue.h"
#include "cpp_headers/llmodel.h"
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <set>

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
    int iid_counter = 0; // CHANGED: Use a simple counter for iid
    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    for (const auto& item_handle : items) {
        py::dict item = item_handle.cast<py::dict>();
        int iid = iid_counter++; // CHANGED
        
        if (active_iids.count(iid)) continue;
        
        std::vector<std::string> words = item["unit"]["words"].cast<std::vector<std::string>>();
        _add_item_internal(iid, words, now_h);
    }
}

// CHANGED: Operates on vector of strings
void LearningQueue::_add_item_internal(int iid, const std::vector<std::string>& words, double now_h) {
    // We only update the inverted index for words that are *currently* in the model.
    // This index is used for finding dependents, so it must be based on valid IDs.
    for (const auto& w : words) {
        if (tmodel->get_idx().has_word(w)) {
            uint32_t wid = tmodel->get_idx().get_id(w);
            word_id_to_iids[wid].push_back(iid);
        }
    }
    // Note: We do not call _resize_arrays here. The model grows when words are learned, not just seen.

    item_words[iid] = words; // Store the original strings
    active_iids.insert(iid);

    auto [grp, key] = _score_and_group(iid, words, now_h);
    _add_to_heap(iid, grp, key);
    
    _active_heap_sizes[grp]++;
}

void LearningQueue::_add_to_heap(int iid, int grp, float key) {
    _heaps[grp].push({key, counter++, iid});
    iid_to_group[iid] = grp;
}

py::tuple LearningQueue::pop_next() {
    int grp_to_pop = -1;
    if (_active_heap_sizes[2] > 0) grp_to_pop = 2;
    else if (_active_heap_sizes[1] > 0) grp_to_pop = 1;
    else if (_active_heap_sizes[0] > 0) grp_to_pop = 0;

    if (grp_to_pop == -1) {
        py::gil_scoped_acquire acquire;
        return py::make_tuple(py::none(), py::none());
    }
    
    while (!_heaps[grp_to_pop].empty()) {
        HeapItem top_item = _heaps[grp_to_pop].top();
        _heaps[grp_to_pop].pop();

        if (active_iids.count(top_item.iid)) {
            py::gil_scoped_acquire acquire;
            // The python side will fetch the item's content using the iid
            return py::make_tuple(top_item.iid, iid_to_group[top_item.iid]);
        }
    }
    
    py::gil_scoped_acquire acquire;
    return py::make_tuple(py::none(), py::none());
}

void LearningQueue::process_answer(int iid, int feedback_level) {
    if (!active_iids.count(iid)) return;

    // CHANGED: Retrieve the word strings for the answered item
    const auto& words = item_words.at(iid);
    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    // 1. Update model (pass strings, model will handle ensuring words/ids)
    tmodel->update_proficiency(words, feedback_level / 2.0f, now_h);
    if (!tmodel->get_model_path().empty()) {
        tmodel->save_fast(tmodel->get_model_path());
    }

    // 2. Remove answered item
    int old_grp = iid_to_group.at(iid);
    _active_heap_sizes[old_grp]--; 
    
    active_iids.erase(iid);
    iid_to_group.erase(iid);
    item_words.erase(iid);
    
    // Find dependents based on the words that were just updated
    std::unordered_set<int> dependents;
    for (const auto& w : words) {
        // The word is now guaranteed to be in the model due to update_proficiency
        uint32_t wid = tmodel->get_idx().get_id(w);
        if (word_id_to_iids.count(wid)) {
            for (int dep_iid : word_id_to_iids.at(wid)) {
                dependents.insert(dep_iid);
            }
            // Clean up the processed word's entry in the inverted index
            word_id_to_iids.at(wid).erase(
                std::remove(word_id_to_iids.at(wid).begin(), word_id_to_iids.at(wid).end(), iid),
                word_id_to_iids.at(wid).end()
            );
        }
    }

    _rescore_items(dependents, now_h);
}

void LearningQueue::_rescore_items(const std::unordered_set<int>& iids_to_rescore, double now_h) {
    for (int iid : iids_to_rescore) {
        if (!active_iids.count(iid)) continue;

        int old_grp = iid_to_group.at(iid);
        
        const auto& words = item_words.at(iid);
        auto [new_grp, new_key] = _score_and_group(iid, words, now_h);
        
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

// CHANGED: Operates on vector of strings, calls the updated llmodel function
std::pair<int, float> LearningQueue::_score_and_group(int iid, const std::vector<std::string>& words, double now_h) {
    
    auto [grp, effs] = predict_answer_for_queue(*tmodel, words, _words_map_v, _words_map_i, _sent_map, iid);

    float key;
    if (grp == 0) {
        key = calculate_unknownness(effs);
    } else if (grp == 2) {
        key = tmodel->predict_understanding(words, now_h);
    } else { // grp == 1
        key = _promotion_potential(words, effs, now_h);
    }

    return {grp, key};
}

// NEW & CORRECTED LOGIC
float LearningQueue::_promotion_potential(const std::vector<std::string>& words, const std::vector<float>& eff_prof, double now_h) {
    float pot = 0.0;

    // Deduplicate words for this calculation
    std::set<std::string> unique_words(words.begin(), words.end());

    for (const auto& w : unique_words) {
        if (_words_map_v.count(w) && _words_map_i.count(w)) {
            float s = 0.0;
            for (int iid_val : _words_map_i.at(w)) {
                if (_sent_map.count(iid_val)) {
                    s += _sent_map.at(iid_val);
                }
            }
            pot += s / std::max(1, _words_map_v.at(w));
        }
    }

    if (pot == 0.0) {
        // Fallback: calculate a positive score based on proficiency
        float pred = tmodel->predict_understanding(words, now_h);
        float total = 0.0;
        for (float val : eff_prof) {
            total += (val > pred) ? val : pred;
        }
        return total / std::max(1.0f, (float)words.size());
    }

    // Main path: return a negative value so higher potential = lower key
    return -pot;
}

} // namespace i_plus_one
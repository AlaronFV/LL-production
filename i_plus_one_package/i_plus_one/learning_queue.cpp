#include "cpp_headers/learning_queue.h"
#include "cpp_headers/llmodel.h"
#include <string>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <set>

namespace i_plus_one {

// --- Constructor ---
LearningQueue::LearningQueue(std::shared_ptr<VocabularyModel> model) : tmodel(model) {
    if (!tmodel) {
        throw std::invalid_argument("LearningQueue must be initialized with a valid VocabularyModel instance.");
    }
}

// --- Public API ---

void LearningQueue::build_from_input(const py::list& items) {
    py::gil_scoped_acquire acquire;
    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    _pass1_ingest_and_group(items);
    _pass2_build_caches_and_dependency_graph();
    _pass3_final_score_and_insert(now_h);
}

int LearningQueue::peek_next() {
    py::gil_scoped_acquire acquire;
    for (int grp = 2; grp >= 0; --grp) {
        if (!_heaps[grp].empty()) {
            return _heaps[grp].begin()->iid;
        }
    }
    return -1;
}

void LearningQueue::process_answer(int iid, int feedback_level) {
    if (_active_iids.find(iid) == _active_iids.end()) return;

    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    const auto& answered_item_words = _item_storage.at(iid).word_ids;
    tmodel->update_proficiency(answered_item_words, feedback_level / 2.0f, now_h);
    
    _active_iids.erase(iid);
    _remove_from_heap(iid);

    // --- Phase 1: First-Order Impact Analysis ---
    std::unordered_set<int> first_order_dependents;
    for (uint32_t wid : answered_item_words) {
        if (_word_to_iids.count(wid)) {
            first_order_dependents.insert(_word_to_iids.at(wid).begin(), _word_to_iids.at(wid).end());
        }
    }

    std::unordered_map<uint32_t, float> word_delta_map;
    std::vector<std::pair<int, int>> group_changes; // Stores {iid, old_group}

    for (int dep_iid : first_order_dependents) {
        if (_active_iids.find(dep_iid) == _active_iids.end()) continue;

        auto& item = _item_storage.at(dep_iid);
        int old_group = item.group;
        
        std::vector<float> eff_profs;
        int new_group = _get_group(item.word_ids, eff_profs, item.unknown_word_ids_cache);

        if (old_group != new_group) {
            group_changes.push_back({dep_iid, old_group});
            item.group = new_group;
        } else if (new_group == 0) {
            float old_sent_map_val = _sent_map.count(dep_iid) ? _sent_map.at(dep_iid) : 0.0f;
            
            size_t n = std::set<uint32_t>(item.word_ids.begin(), item.word_ids.end()).size();
            size_t unknown_cnt = item.unknown_word_ids_cache.size();
            size_t diff = (unknown_cnt - (n / 2));
            float new_sent_map_val = (diff > 0) ? static_cast<float>(n) / diff : 0.0f;

            float delta = new_sent_map_val - old_sent_map_val;
            if (std::abs(delta) > 1e-6) {
                _sent_map[dep_iid] = new_sent_map_val;
                for (uint32_t wid : item.unknown_word_ids_cache) {
                    word_delta_map[wid] += delta;
                }
            }
        }
    }

    // --- Phase 2: Cascading Delta Calculation ---
    std::unordered_map<int, float> group1_key_deltas;
    for (const auto& pair : word_delta_map) {
        uint32_t wid = pair.first;
        float total_delta = pair.second;
        if (_unknown_word_to_group1_iids.count(wid)) {
            float promotion_times = _words_map_v.count(wid) ? static_cast<float>(_words_map_v.at(wid)) : 1.0f;
            float score_delta = total_delta / std::max(1.0f, promotion_times);
            for (int g1_iid : _unknown_word_to_group1_iids.at(wid)) {
                group1_key_deltas[g1_iid] += score_delta;
            }
        }
    }

    // --- Phase 3: Transactional Commit ---
    for (const auto& pair : group1_key_deltas) {
        int iid_to_update = pair.first;
        float key_delta = pair.second;
        if (_active_iids.count(iid_to_update) && _iid_to_heap_item.count(iid_to_update)) {
            const auto& heap_item = _iid_to_heap_item.at(iid_to_update);
            _update_heap(iid_to_update, 1, heap_item.key + key_delta);
        }
    }
    
    for (const auto& change : group_changes) {
         int iid_to_rescore = change.first;
         if (_active_iids.count(iid_to_rescore)) {
            auto& item = _item_storage.at(iid_to_rescore);
            std::vector<float> eff_profs;
            _get_group(item.word_ids, eff_profs, item.unknown_word_ids_cache);
            float new_key = _calculate_key(item, eff_profs, now_h);
            _update_heap(item.iid, item.group, new_key);
         }
    }

    // --- Phase 4: Self-Heal Caches & Graph ---
    _maintain_caches_and_graph(group_changes);
}

size_t LearningQueue::size(int grp) const {
    if (grp >= 0 && grp < 3) {
        return _heaps[grp].size();
    }
    size_t total = 0;
    for(const auto& heap : _heaps) total += heap.size();
    return total;
}

// --- Private Methods Implementation ---

void LearningQueue::_pass1_ingest_and_group(const py::list& items) {
    _item_storage.clear();
    _word_to_iids.clear();
    _active_iids.clear();

    int iid_counter = 0;
    for (const auto& item_handle : items) {
        int iid = iid_counter++;
        auto words_py = item_handle.cast<py::list>();
        
        ItemData data;
        data.iid = iid;
        data.word_ids.reserve(words_py.size());
        for (const auto& w : words_py) {
            uint32_t wid = tmodel->get_idx().get_id(w.cast<std::string>());
            data.word_ids.push_back(wid);
        }
        // Populate reverse index after all words are added
        for (uint32_t wid : std::set<uint32_t>(data.word_ids.begin(), data.word_ids.end())) {
             _word_to_iids[wid].push_back(iid);
        }

        std::vector<float> dummy_effs;
        data.group = _get_group(data.word_ids, dummy_effs, data.unknown_word_ids_cache);
        _item_storage[iid] = std::move(data);
        _active_iids.insert(iid);
    }
}

void LearningQueue::_pass2_build_caches_and_dependency_graph() {
    _sent_map.clear();
    _words_map_v.clear();
    _unknown_word_to_group1_iids.clear();

    for (auto const& [iid, item] : _item_storage) {
        if (item.group == 0) {
            size_t n = std::set<uint32_t>(item.word_ids.begin(), item.word_ids.end()).size();
            size_t unknown_cnt = item.unknown_word_ids_cache.size();
            size_t diff = (unknown_cnt - (n / 2));
            if (diff > 0) _sent_map[iid] = static_cast<float>(n) / diff;

            for (uint32_t wid : item.unknown_word_ids_cache) {
                if (_words_map_v.find(wid) == _words_map_v.end()) {
                    float b = 0.1f, v = 0.9f;
                    const auto& g2p_map = tmodel->get_global_to_processed_map();
                    auto it = g2p_map.find(wid);
                    if (it != g2p_map.end()) {
                        b = tmodel->get_prof()[it->second];
                        v = tmodel->get_vol()[it->second];
                    }
                    _words_map_v[wid] = promotion_times(b, v);
                }
            }
        }
    }

    for (auto const& [wid, _] : _words_map_v) {
        if (_word_to_iids.count(wid)) {
            for (int iid : _word_to_iids.at(wid)) {
                if (_item_storage.at(iid).group == 1) {
                    _unknown_word_to_group1_iids[wid].push_back(iid);
                }
            }
        }
    }
}

void LearningQueue::_pass3_final_score_and_insert(double now_h) {
    for (int i = 0; i < 3; ++i) _heaps[i].clear();
    _iid_to_heap_item.clear();

    for (auto& [iid, item] : _item_storage) {
        std::vector<float> eff_profs;
        _get_group(item.word_ids, eff_profs, item.unknown_word_ids_cache);
        float key = _calculate_key(item, eff_profs, now_h);
        HeapItem heap_item = {key, _counter++, iid};
        _heaps[item.group].insert(heap_item);
        _iid_to_heap_item[iid] = heap_item;
    }
}

int LearningQueue::_get_group(const std::vector<uint32_t>& word_ids, std::vector<float>& out_eff_profs, std::vector<uint32_t>& out_unknown_word_ids) {
    out_unknown_word_ids.clear();
    if (word_ids.empty()) return 2;

    std::set<uint32_t> unique_ids_set(word_ids.begin(), word_ids.end());
    std::vector<uint32_t> unique_word_ids(unique_ids_set.begin(), unique_ids_set.end());
    
    out_eff_profs = tmodel->get_effective_proficiency_by_id(unique_word_ids);
    
    size_t familiar = 0;
    for(size_t i = 0; i < unique_word_ids.size(); ++i) {
        if (out_eff_profs[i] > 0.3f) {
            familiar++;
        } else {
            out_unknown_word_ids.push_back(unique_word_ids[i]);
        }
    }

    size_t n = unique_word_ids.size();
    if (familiar == n) return 2;
    if (familiar >= n / 2) return 1;
    return 0;
}

float LearningQueue::_calculate_key(const ItemData& item, const std::vector<float>& eff_profs, double now_h) {
    if (item.group == 0) {
        return calculate_unknownness(eff_profs);
    }
    if (item.group == 2) {
        return tmodel->_predict_understanding_by_id(item.word_ids, now_h);
    }
    // Group 1
    return _get_promotion_potential(item);
}

float LearningQueue::_get_promotion_potential(const ItemData& item) {
    float pot = 0.0f;
    for (uint32_t wid : item.unknown_word_ids_cache) {
        if (_words_map_v.count(wid)) {
            float s = 0.0f;
            if (_word_to_iids.count(wid)) {
                for (int iid_val : _word_to_iids.at(wid)) {
                     if (_item_storage.at(iid_val).group == 0 && _sent_map.count(iid_val)) {
                        s += _sent_map.at(iid_val);
                     }
                }
            }
            pot += s / std::max(1, _words_map_v.at(wid));
        }
    }

    if (std::abs(pot) < 1e-6) { // Fallback scoring
        double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        float pred = tmodel->_predict_understanding_by_id(item.word_ids, now_h);
        float total = 0.0f;
        std::vector<float> effs = tmodel->get_effective_proficiency_by_id(item.word_ids);
        for (float val : effs) {
            total += (val > pred) ? val : pred;
        }
        return total / std::max(1.0f, (float)item.word_ids.size());
    }

    return pot;
}

void LearningQueue::_update_heap(int iid, int new_group, float new_key) {
    _remove_from_heap(iid);
    HeapItem new_heap_item = {new_key, _counter++, iid};
    _heaps[new_group].insert(new_heap_item);
    _iid_to_heap_item[iid] = new_heap_item;
    _item_storage.at(iid).group = new_group;
}

void LearningQueue::_remove_from_heap(int iid) {
    if (_iid_to_heap_item.count(iid)) {
        const auto& old_item = _iid_to_heap_item.at(iid);
        int old_group = _item_storage.at(iid).group;
        _heaps[old_group].erase(old_item);
        _iid_to_heap_item.erase(iid);
    }
}

void LearningQueue::_maintain_caches_and_graph(const std::vector<std::pair<int, int>>& group_changes) {
    for (const auto& change : group_changes) {
        int iid = change.first;
        int old_group = change.second;
        const auto& item = _item_storage.at(iid);
        int new_group = item.group;

        // --- Handle leaving old group ---
        if (old_group == 0) {
            _sent_map.erase(iid);
        } else if (old_group == 1) {
            for (uint32_t wid : item.unknown_word_ids_cache) {
                if (_unknown_word_to_group1_iids.count(wid)) {
                    auto& vec = _unknown_word_to_group1_iids.at(wid);
                    vec.erase(std::remove(vec.begin(), vec.end(), iid), vec.end());
                }
            }
        }

        // --- Handle entering new group ---
        if (new_group == 0) {
            size_t n = std::set<uint32_t>(item.word_ids.begin(), item.word_ids.end()).size();
            size_t unknown_cnt = item.unknown_word_ids_cache.size();
            size_t diff = unknown_cnt - (n / 2);
            if (diff > 0) _sent_map[iid] = static_cast<float>(n) / diff;
        } else if (new_group == 1) {
            for (uint32_t wid : item.unknown_word_ids_cache) {
                _unknown_word_to_group1_iids[wid].push_back(iid);
            }
        }
    }
}

} // namespace i_plus_one
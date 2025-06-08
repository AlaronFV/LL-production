#include "cpp_headers/llmodel.h"
#include "cpp_headers/vocab_model.h"
#include <numeric>
#include <set>

namespace i_plus_one {

std::tuple<int, std::vector<float>> predict_answer_for_queue(
    VocabularyModel& model,
    const std::vector<uint32_t>& word_ids,
    std::unordered_map<uint32_t, int>& words_map_v,
    std::unordered_map<uint32_t, std::unordered_set<int>>& words_map_i,
    std::unordered_map<int, float>& sent_map,
    int iid) {

    std::set<uint32_t> unique_ids_set(word_ids.begin(), word_ids.end());
    std::vector<uint32_t> unique_word_ids(unique_ids_set.begin(), unique_ids_set.end());

    size_t n = unique_word_ids.size();
    if (n == 0) {
        return std::make_tuple(2, std::vector<float>{});
    }

    float thr = 0.3f;
    size_t half = n / 2;

    std::vector<float> effs = model.get_effective_proficiency_by_id(unique_word_ids);
    
    std::vector<uint32_t> unknown_word_ids;
    for(size_t i = 0; i < n; ++i) {
        if (effs[i] <= thr) {
            unknown_word_ids.push_back(unique_word_ids[i]);
        }
    }

    size_t unknown_cnt = unknown_word_ids.size();
    size_t familiar = n - unknown_cnt;

    if (unknown_cnt > half) {
        float diff = static_cast<float>(unknown_cnt - half);
        sent_map[iid] = word_ids.size() / diff;

        for (uint32_t wid : unknown_word_ids) {
            if (words_map_v.find(wid) == words_map_v.end()) {
                float b = 0.1f, v = 0.9f;
                // Check if the word is processed to get real values
                if (model.get_processed_word_ids().count(wid)) {
                    b = model.get_prof()[wid];
                    v = model.get_vol()[wid];
                }
                words_map_v[wid] = promotion_times(b, v);
            }
            words_map_i[wid].insert(iid);
        }
    }

    int group;
    if (familiar < half) group = 0;
    else if (familiar < n) group = 1;
    else group = 2;

    return std::make_tuple(group, effs);
}

int predict_answer_for_natural_candidates(
    VocabularyModel& model,
    const std::vector<uint32_t>& word_ids) {

    size_t n = word_ids.size();
    if (n == 0) return 2;
    
    float thr = 0.3f;
    size_t half = n / 2;
    size_t familiar = 0;

    std::vector<float> effs = model.get_effective_proficiency_by_id(word_ids);
    for (float eff : effs) {
        if (eff > thr) {
            familiar++;
        }
    }

    if (familiar < half) return 0;
    if (familiar < n) return 1;
    return 2;
}

std::set<int> get_natural_candidates(
    VocabularyModel& model,
    const py::list& aligned_text_py,
    const std::set<int>& current_indices) {
    
    std::set<int> natural;
    py::gil_scoped_acquire acquire;

    for (const auto& item_handle : aligned_text_py) {
        py::dict unit_dict = item_handle.cast<py::dict>();
        // Assuming the structure from main.py is {"filename":..., "index":..., "unit":...}
        // and aligned_text is a list of these full records.
        int i = unit_dict["index"].cast<int>();

        if (current_indices.count(i)) continue;

        py::dict unit = unit_dict["unit"].cast<py::dict>();
        py::list words_py = unit["words"].cast<py::list>();
        if (words_py.empty()) continue;

        std::vector<std::string> words_str = words_py.cast<std::vector<std::string>>();
        std::vector<uint32_t> word_ids;
        word_ids.reserve(words_str.size());
        for(const auto& w : words_str) {
            // Here we must add the word to the index if it's not present
            word_ids.push_back(model.get_idx().get_id(w));
        }

        int ans = predict_answer_for_natural_candidates(model, word_ids);

        if (ans >= 1 || words_str.size() < 5) {
            natural.insert(i);
        }
    }
    return natural;
}

float calculate_unknownness(const std::vector<float>& effs) {
    if (effs.empty()) return 0.0f;
    double s = std::accumulate(effs.begin(), effs.end(), 0.0);
    return effs.size() - s;
}

std::map<std::string, double> get_vocabulary_statistics(VocabularyModel& model) {
    std::map<std::string, double> stats;
    const auto& eff_prof = model.get_eff_prof();
    const auto& prof = model.get_prof();
    const auto& vol = model.get_vol();
    const auto& processed_ids = model.get_processed_word_ids();
    float prof_min = model.get_proficiency_min();
    
    stats["total_words"] = model.get_idx().size();
    stats["processed_words"] = processed_ids.size();

    double all_known_knowledge = 0;
    double avg_eff_prof_sum = 0;
    double avg_prof_sum = 0;
    double avg_vol_sum = 0;
    size_t well_known_count = 0;
    size_t familiar_count = 0;
    size_t learning_count = 0;
    size_t stable_count = 0;
    size_t semi_stable_count = 0;
    size_t volatile_count = 0;

    for (uint32_t wid : processed_ids) {
        float ep = eff_prof[wid];
        float p = prof[wid];
        float v = vol[wid];

        all_known_knowledge += ep;
        avg_eff_prof_sum += ep;

        if (ep > 0.7f) well_known_count++;
        else if (ep >= 0.3f) familiar_count++;
        else learning_count++;

        if (v < 0.3f) stable_count++;
        else if (v <= 0.6f) semi_stable_count++;
        else volatile_count++;
        
        if (p > prof_min) {
            avg_prof_sum += p;
            avg_vol_sum += v;
        }
    }

    stats["all_known_knowledge"] = all_known_knowledge;
    stats["well_known"] = well_known_count;
    stats["familiar"] = familiar_count;
    stats["learning"] = learning_count;
    stats["stable"] = stable_count;
    stats["semi_stable"] = semi_stable_count;
    stats["volatile"] = volatile_count;

    size_t processed_count = processed_ids.size();
    stats["average_effective_proficiency"] = (processed_count > 0) ? (avg_eff_prof_sum / processed_count) : 0.0;
    stats["average_proficiency"] = (processed_count > 0) ? (avg_prof_sum / processed_count) : 0.0;
    stats["average_volatility"] = (processed_count > 0) ? (avg_vol_sum / processed_count) : 0.0;

    return stats;
}

} // namespace i_plus_one
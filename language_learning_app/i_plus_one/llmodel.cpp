#include "cpp_headers/llmodel.h"
#include <numeric>
#include <algorithm>

namespace i_plus_one {

std::tuple<int, std::vector<float>> predict_answer_for_queue(
    VocabularyModel& model,
    const std::vector<std::string>& words_py,
    std::unordered_map<std::string, int>& words_map_v,
    std::unordered_map<std::string, std::unordered_set<int>>& words_map_i,
    std::unordered_map<int, float>& sent_map,
    int iid) {
    
    // Deduplicate words while preserving order
    std::vector<std::string> words;
    std::set<std::string> seen;
    for(const auto& w : words_py) {
        if(seen.find(w) == seen.end()) {
            seen.insert(w);
            words.push_back(w);
        }
    }

    size_t n = words.size();
    float thr = 0.3f;
    size_t half = n / 2;

    std::vector<float> effs = model.get_effective_proficiency(words);
    
    std::vector<int> unknown_idx;
    for(size_t i = 0; i < n; ++i) {
        if (effs[i] <= thr) {
            unknown_idx.push_back(i);
        }
    }

    size_t unknown_cnt = unknown_idx.size();
    size_t familiar = n - unknown_cnt;

    if (unknown_cnt > half) {
        float diff = static_cast<float>(unknown_cnt - half);
        sent_map[iid] = words_py.size() / diff;

        for (int i : unknown_idx) {
            const std::string& w = words[i];
            if (words_map_v.find(w) == words_map_v.end()) {
                float b = 0.1f, v = 0.9f;
                if (model.get_idx().has_word(w)) {
                    uint32_t wid = model.get_idx().get_id(w);
                    b = model.get_prof()[wid];
                    v = model.get_vol()[wid];
                }
                words_map_v[w] = promotion_times(b, v);
            }
            words_map_i[w].insert(iid);
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
    const std::vector<std::string>& words_py) {

    size_t n = words_py.size();
    if (n == 0) return 2; // Empty is fully known
    float thr = 0.3f;
    size_t half = n / 2;
    size_t familiar = 0;

    std::vector<float> effs = model.get_effective_proficiency(words_py);
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
    for (size_t i = 0; i < py::len(aligned_text_py); ++i) {
        if (current_indices.count(i)) continue;

        py::dict unit = aligned_text_py[i].cast<py::dict>();
        if (!unit.contains("words")) continue;

        std::vector<std::string> words = unit["words"].cast<std::vector<std::string>>();
        if (words.empty()) continue;
        
        int ans = predict_answer_for_natural_candidates(model, words);

        if (ans >= 1 || words.size() < 5) {
            natural.insert(i);
        }
    }
    return natural;
}

float calculate_unknownness(const std::vector<float>& effs) {
    double s = std::accumulate(effs.begin(), effs.end(), 0.0);
    return effs.size() - s;
}

std::map<std::string, double> get_vocabulary_statistics(VocabularyModel& model) {
    std::map<std::string, double> stats;
    const auto& eff_prof = model.get_eff_prof();
    const auto& prof = model.get_prof();
    const auto& vol = model.get_vol();
    float prof_min = model.get_proficiency_min();
    size_t total = model.get_idx().size();

    stats["total_words"] = total;
    stats["all_possible_knowledge"] = total * 0.97;
    
    stats["all_known_knowledge"] = std::accumulate(eff_prof.begin(), eff_prof.end(), 0.0);
    
    stats["well_known"] = std::count_if(eff_prof.begin(), eff_prof.end(), [](float v){ return v > 0.7f; });
    stats["familiar"] = std::count_if(eff_prof.begin(), eff_prof.end(), [](float v){ return v >= 0.3f && v <= 0.7f; });
    stats["learning"] = std::count_if(eff_prof.begin(), eff_prof.end(), [](float v){ return v < 0.3f && v > 0; });
    
    stats["stable"] = std::count_if(vol.begin(), vol.end(), [](float v){ return v < 0.3f; });
    stats["semi_stable"] = std::count_if(vol.begin(), vol.end(), [](float v){ return v >= 0.3f && v <= 0.6f; });
    stats["volatile"] = 0;
    for(size_t i=0; i<total; ++i) {
        if (vol[i] > 0.6f && eff_prof[i] > 0) {
            stats["volatile"]++;
        }
    }

    double eff_prof_sum = 0, prof_sum = 0, vol_sum = 0;
    int eff_prof_count = 0, prof_count = 0;

    for(size_t i=0; i<total; ++i) {
        if (eff_prof[i] > 0) {
            eff_prof_sum += eff_prof[i];
            eff_prof_count++;
        }
        if (prof[i] > prof_min) {
            prof_sum += prof[i];
            vol_sum += vol[i];
            prof_count++;
        }
    }

    stats["average_effective_proficiency"] = (eff_prof_count > 0) ? (eff_prof_sum / eff_prof_count) : 0.0;
    stats["average_proficiency"] = (prof_count > 0) ? (prof_sum / prof_count) : 0.0;
    stats["average_volatility"] = (prof_count > 0) ? (vol_sum / prof_count) : 0.0;

    return stats;
}

} // namespace i_plus_one
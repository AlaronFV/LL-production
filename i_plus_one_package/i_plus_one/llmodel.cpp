#include "cpp_headers/llmodel.h"
#include "cpp_headers/vocab_model.h"
#include <numeric>
#include <set>

namespace i_plus_one {


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
    const py::list& aligned_text_words_py,
    const std::set<int>& current_indices) {
    
    std::set<int> natural;
    py::gil_scoped_acquire acquire;
    int i = 0;

    for (const auto& item_handle : aligned_text_words_py) {
        py::list words_py = item_handle.cast<py::list>();
        if (words_py.empty() || current_indices.count(i)) {
            i++; 
            continue;
        }

        std::vector<uint32_t> word_ids;
        word_ids.reserve(words_py.size());
        for(const auto& w : words_py) {
            word_ids.push_back(model.get_idx().get_id(w.cast<std::string>()));
        }

        int ans = predict_answer_for_natural_candidates(model, word_ids);

        if (ans >= 1 || word_ids.size() < 5) {
            natural.insert(i);
        }
        i++;
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
    size_t processed_count = prof.size();
    
    stats["total_seen_words"] = model.get_idx().size();
    stats["processed_words"] = processed_count;
    stats["all_possible_knowledge"] = processed_count * 0.97;

    double all_known_knowledge = 0;
    double avg_prof_sum = 0;
    double avg_vol_sum = 0;
    size_t well_known_count = 0;
    size_t familiar_count = 0;
    size_t learning_count = 0;
    size_t stable_count = 0;
    size_t semi_stable_count = 0;
    size_t volatile_count = 0;

    for (size_t p_id = 0; p_id < processed_count; ++p_id) {
        float ep = eff_prof[p_id];
        float p = prof[p_id];
        float v = vol[p_id];

        all_known_knowledge += ep;
        avg_prof_sum += p;
        avg_vol_sum += v;

        if (ep > 0.7f) well_known_count++;
        else if (ep >= 0.3f) familiar_count++;
        else learning_count++;

        if (v < 0.3f) stable_count++;
        else if (v <= 0.6f) semi_stable_count++;
        else volatile_count++;
    }

    stats["all_known_knowledge"] = all_known_knowledge;
    stats["well_known"] = well_known_count;
    stats["familiar"] = familiar_count;
    stats["learning"] = learning_count;
    stats["stable"] = stable_count;
    stats["semi_stable"] = semi_stable_count;
    stats["volatile"] = volatile_count;

    stats["average_effective_proficiency"] = (processed_count > 0) ? (all_known_knowledge / processed_count) : 0.0;
    stats["average_proficiency"] = (processed_count > 0) ? (avg_prof_sum / processed_count) : 0.0;
    stats["average_volatility"] = (processed_count > 0) ? (avg_vol_sum / processed_count) : 0.0;

    return stats;
}

} // namespace i_plus_one
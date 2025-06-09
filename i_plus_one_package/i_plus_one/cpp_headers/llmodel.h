#ifndef I_PLUS_ONE_LLMODEL_H
#define I_PLUS_ONE_LLMODEL_H

#include "vocab_model.h"
#include <vector>
#include <string>
#include <set>
#include <map>
#include <unordered_set>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace i_plus_one {

class VocabularyModel;

std::tuple<int, std::vector<float>> predict_answer_for_queue(
    VocabularyModel& model,
    const std::vector<uint32_t>& word_ids,
    std::unordered_map<uint32_t, int>& words_map_v,
    std::unordered_map<uint32_t, std::unordered_set<int>>& words_map_i,
    std::unordered_map<int, float>& sent_map,
    int iid);

int predict_answer_for_natural_candidates(
    VocabularyModel& model,
    const std::vector<uint32_t>& word_ids);

std::set<int> get_natural_candidates(
    VocabularyModel& model,
    const py::list& aligned_text_py,
    const std::set<int>& current_indices);

float calculate_unknownness(const std::vector<float>& effs);

std::map<std::string, double> get_vocabulary_statistics(VocabularyModel& model);

} // namespace i_plus_one

#endif // I_PLUS_ONE_LLMODEL_H
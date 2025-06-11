#ifndef I_PLUS_ONE_LLMODEL_H
#define I_PLUS_ONE_LLMODEL_H

#include "vocab_model.h"
#include <vector>
#include <string>
#include <set>
#include <map>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace i_plus_one {

class VocabularyModel;


int predict_answer_for_natural_candidates(
    VocabularyModel& model,
    const std::vector<uint32_t>& word_ids);

std::set<int> get_natural_candidates(
    VocabularyModel& model,
    const py::list& aligned_text_words_py,
    const std::set<int>& current_indices);

float calculate_unknownness(const std::vector<float>& effs);

std::map<std::string, double> get_vocabulary_statistics(VocabularyModel& model);

} // namespace i_plus_one

#endif // I_PLUS_ONE_LLMODEL_H
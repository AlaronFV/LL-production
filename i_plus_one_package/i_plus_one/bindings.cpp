#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>

#include "cpp_headers/vocab_model.h"
#include "cpp_headers/llmodel.h"
#include "cpp_headers/learning_queue.h"

namespace py = pybind11;
using namespace i_plus_one;

PYBIND11_MODULE(i_plus_one_cpp, m) {
    m.doc() = "C++ core module for i_plus_one";

    // --- VocabularyModel Class ---
    py::class_<VocabularyModel, std::shared_ptr<VocabularyModel>>(m, "VocabularyModel")
        .def(py::init<float, float>(), py::arg("learning_rate") = 0.15f, py::arg("base_decay_rate") = 0.05f)
        .def_static("load_fast", &VocabularyModel::load_fast, "Load model from a binary file", py::arg("path"))
        .def("save_fast", [](std::shared_ptr<VocabularyModel> self, const std::string& path) {
            self->set_model_path(path);
            self->save_fast(path);
        }, "Save model to a binary file", py::arg("path"))
        .def("update_from_words", &VocabularyModel::update_from_words, 
            "Update proficiency from a list of words, used for interactive reading.",
            py::arg("words"), py::arg("u_val"));

    // --- LearningQueue Class ---
    py::class_<LearningQueue>(m, "LearningQueue")
        .def(py::init<std::shared_ptr<VocabularyModel>>(), py::arg("model"))
        .def("build_from_input", &LearningQueue::build_from_input, py::arg("items"))
        .def("peek_next", &LearningQueue::peek_next)
        .def("process_answer", &LearningQueue::process_answer, py::arg("iid"), py::arg("feedback_level"))
        .def("size", &LearningQueue::size, py::arg("grp") = -1);
        
    // --- Standalone functions ---
    m.def("get_vocabulary_statistics", 
        [](std::shared_ptr<VocabularyModel> model) {
            return get_vocabulary_statistics(*model);
        }, "Get vocabulary statistics", py::arg("model"));
        
    m.def("get_natural_candidates", 
        [](std::shared_ptr<VocabularyModel> model, const py::list& aligned_text_words_py, const std::set<int>& current_indices) {
            return get_natural_candidates(*model, aligned_text_words_py, current_indices);
        }, "Get natural sentence candidates",
        py::arg("model"), py::arg("aligned_text_words_py"), py::arg("current_indices"));
}
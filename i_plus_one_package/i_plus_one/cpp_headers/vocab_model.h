#ifndef I_PLUS_ONE_VOCAB_MODEL_H
#define I_PLUS_ONE_VOCAB_MODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <set>
#include <memory>
#include "decay_item.h"

namespace i_plus_one {

class WordIndex {
public:
    uint32_t get_id(const std::string& w);
    bool has_word(const std::string& w) const;
    std::string get_word(uint32_t idx) const;
    size_t size() const;
    void clear();
    void add_word_for_load(const std::string& w, uint32_t idx);
    const std::vector<std::string>& get_i2w() const { return _i2w; }

private:
    std::unordered_map<std::string, uint32_t> _w2i;
    std::vector<std::string> _i2w;
};

class VocabularyModel {
public:
    VocabularyModel(
        float learning_rate = 0.15f,
        float base_decay_rate = 0.05f,
        float context_influence = 0.2f,
        float activation_threshold = 0.05f,
        float proficiency_min = 0.01f,
        float proficiency_max = 0.99f,
        double min_elapsed_h = 1.0,
        float propagation_threshold = 0.01f,
        float trace_delete_threshold = 0.001f);

    // Public API for proficiency updates and predictions
    void update_from_words(const std::vector<std::string>& words, float u_val);
    float predict_understanding(const std::vector<std::string>& words, double current_time_h);
    std::vector<float> get_effective_proficiency_by_str(const std::vector<std::string>& words);

    // Core proficiency logic (operates on global IDs)
    void update_proficiency(const std::vector<uint32_t>& word_ids, float u_val, double now_h);
    std::vector<float> get_effective_proficiency_by_id(const std::vector<uint32_t>& word_ids);

    // Serialization
    void save_fast(const std::string& path);
    static std::shared_ptr<VocabularyModel> load_fast(const std::string& path);

    // Public accessors for internal state
    WordIndex& get_idx() { return idx; }
    const std::string& get_model_path() const { return _model_path; }
    void set_model_path(const std::string& path) { _model_path = path; }
    
    // For statistics
    const std::vector<float>& get_prof() const { return prof; }
    const std::vector<float>& get_vol() const { return vol; }
    const std::vector<float>& get_eff_prof() const { return eff_prof; }
    // CHANGED: This now returns the mapping instead of the set
    const std::unordered_map<uint32_t, uint32_t>& get_global_to_processed_map() const { return _global_to_processed_id; }


private:
    friend class LearningQueue;
    friend std::tuple<int, std::vector<float>> predict_answer_for_queue(VocabularyModel& model, const std::vector<uint32_t>& word_ids, std::unordered_map<uint32_t, int>& words_map_v, std::unordered_map<uint32_t, std::unordered_set<int>>& words_map_i, std::unordered_map<int, float>& sent_map, int iid);
    friend int predict_answer_for_natural_candidates(VocabularyModel& model, const std::vector<uint32_t>& word_ids);

    // CHANGED: No longer needed, as we add to vectors directly.
    // void _resize_arrays(size_t new_n); 
    uint32_t _get_or_create_processed_id(uint32_t global_id, double now_h);
    float _eff_prof_formula(float p, float v) const;
    int _add_trace(const std::set<uint32_t>& word_ids, double timestamp_h, float activation, float decay_factor);
    float _decay_trace_activation(int trace_idx, double now_h);
    float _get_word_activation(uint32_t word_id, double now_h);
    // CHANGED: Now takes a processed_id for direct array access
    void _apply_decay_to_word_id(uint32_t processed_id, double now_h);
    void _process_due_word_decays(double now_h);
    void _process_due_trace_decays(double now_h);
    void _propagate(int source_trace_idx, double now_h);
    void _prune_traces();
    float _calculate_context_support(const std::unordered_set<uint32_t>& id_words, size_t n);
    float _predict_understanding_by_id(const std::vector<uint32_t>& word_ids, double current_time_h);

    // Member variables
    WordIndex idx;
    std::string _model_path;
    
    // REMOVED: Replaced by the mapping below
    // std::unordered_set<uint32_t> _processed_word_ids;

    // ADDED: The new mapping system
    std::unordered_map<uint32_t, uint32_t> _global_to_processed_id;
    std::vector<uint32_t> _processed_to_global_id;

    // Core model arrays - their size is now determined by the number of processed words
    std::vector<float> prof;
    std::vector<float> vol;
    std::vector<float> eff_prof;
    std::vector<uint32_t> encounters;
    std::vector<double> _word_last_decay_h;

    // Trace-related data (still uses global IDs)
    std::vector<std::vector<uint32_t>> _trace_word_ids;
    std::vector<double> _trace_timestamps_h;
    std::vector<float> _trace_activations;
    std::vector<float> _trace_decay_factors;
    std::unordered_map<uint32_t, std::vector<int>> word_to_traces;

    // Decay scheduling queues
    // CHANGED: The word decay queue will now store the DENSE processed_id
    std::priority_queue<DecayItem, std::vector<DecayItem>, DecayItemComparator> _word_decay_pq;
    std::priority_queue<DecayItem, std::vector<DecayItem>, DecayItemComparator> _trace_decay_pq;

    // Parameters
    float learning_rate;
    float context_influence;
    float activation_threshold;
    float base_decay_rate;
    float proficiency_min;
    float proficiency_max;
    double min_elapsed_h;
    float propagation_threshold;
    float trace_delete_threshold;
};

int promotion_times(float b, float v);

} // namespace i_plus_one

#endif // I_PLUS_ONE_VOCAB_MODEL_H
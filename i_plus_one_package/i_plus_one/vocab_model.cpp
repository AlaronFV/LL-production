#include "cpp_headers/vocab_model.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <chrono>
#include <unordered_set>

namespace i_plus_one {

// --- Serialization Constants ---
const char MAGIC_STRING[] = "IP1MDL";
const uint32_t MODEL_VERSION = 2; // Version incremented for new format

// --- Helper Functions ---
[[nodiscard]] size_t intersection_size_vec_vec(const std::vector<uint32_t>& vec1, const std::vector<uint32_t>& vec2) {
    std::vector<uint32_t> intersection;
    std::set_intersection(
        vec1.begin(), vec1.end(),
        vec2.begin(), vec2.end(),
        std::back_inserter(intersection)
    );
    return intersection.size();
}

[[nodiscard]] size_t intersection_size_set_vec(const std::unordered_set<uint32_t>& py_set, const std::vector<uint32_t>& cpp_vec) {
    return std::count_if(cpp_vec.begin(), cpp_vec.end(),
        [&py_set](uint32_t val) { 
            return py_set.find(val) != py_set.end(); 
        });
}

// --- WordIndex Implementation ---
uint32_t WordIndex::get_id(const std::string& w) {
    auto it = _w2i.find(w);
    if (it != _w2i.end()) return it->second;
    uint32_t idx = _i2w.size();
    _i2w.push_back(w);
    _w2i[w] = idx;
    return idx;
}
bool WordIndex::has_word(const std::string& w) const { return _w2i.count(w) > 0; }
std::string WordIndex::get_word(uint32_t idx) const { if (idx >= _i2w.size()) throw std::out_of_range("Word ID out of bounds"); return _i2w[idx]; }
size_t WordIndex::size() const { return _i2w.size(); }
void WordIndex::clear() { _w2i.clear(); _i2w.clear(); }
void WordIndex::add_word_for_load(const std::string& w, uint32_t idx) { if (_i2w.size() <= idx) _i2w.resize(idx + 1); _i2w[idx] = w; _w2i[w] = idx; }

// --- VocabularyModel Implementation ---
VocabularyModel::VocabularyModel(
    float learning_rate, float base_decay_rate, float context_influence, 
    float activation_threshold, float proficiency_min, float proficiency_max,
    double min_elapsed_h, float propagation_threshold, float trace_delete_threshold)
    : learning_rate(learning_rate), context_influence(context_influence), 
      activation_threshold(activation_threshold), base_decay_rate(base_decay_rate),
      proficiency_min(proficiency_min), proficiency_max(proficiency_max),
      min_elapsed_h(min_elapsed_h), propagation_threshold(propagation_threshold),
      trace_delete_threshold(trace_delete_threshold) {
    // Arrays are intentionally left empty until first processed word
}

float VocabularyModel::_eff_prof_formula(float p, float v) const {
    return p * (1.0f - 0.3f * v);
}

uint32_t VocabularyModel::_get_or_create_processed_id(uint32_t global_id, double now_h) {
    auto it = _global_to_processed_id.find(global_id);
    if (it != _global_to_processed_id.end()) {
        return it->second; // Return existing processed ID
    }

    // It's a new word to process, create a new dense ID
    uint32_t new_processed_id = _processed_to_global_id.size();
    _global_to_processed_id[global_id] = new_processed_id;
    _processed_to_global_id.push_back(global_id);

    // Expand the core vectors by one for the new word
    float new_prof = 0.5f + _get_word_activation(global_id, now_h) * 0.1f;
    prof.push_back(new_prof);
    vol.push_back(0.9f);
    eff_prof.push_back(_eff_prof_formula(new_prof, 0.9f));
    encounters.push_back(0);
    _word_last_decay_h.push_back(now_h); // Initialize last decay time

    // Add to decay queue with its new DENSE ID
    _word_decay_pq.push({now_h + min_elapsed_h, new_processed_id});
    
    return new_processed_id;
}

void VocabularyModel::update_from_words(const std::vector<std::string>& words, float u_val) {
    if (words.empty()) return;

    double now_h = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<3600>>>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    std::vector<uint32_t> word_ids;
    word_ids.reserve(words.size());
    for(const auto& w : words) {
        word_ids.push_back(idx.get_id(w));
    }

    update_proficiency(word_ids, u_val, now_h);
}

void VocabularyModel::update_proficiency(const std::vector<uint32_t>& global_word_ids, float u_val, double now_h) {
    if (global_word_ids.empty()) return;

    // This is now the ONLY place where words become "processed"
    std::set<uint32_t> unique_global_ids(global_word_ids.begin(), global_word_ids.end());

    int new_trace_idx = _add_trace(unique_global_ids, now_h, 1.0f, (0.1f - 0.05f * u_val));
    _propagate(new_trace_idx);
    _prune_traces();

    float expected = _predict_understanding_by_id(global_word_ids, now_h);

    std::vector<uint32_t> processed_ids;
    processed_ids.reserve(unique_global_ids.size());

    for (uint32_t global_id : unique_global_ids) {
        processed_ids.push_back(_get_or_create_processed_id(global_id, now_h));
    }
    
    float err = u_val - expected;
    
    // CHANGED: Loop over the dense processed_ids for direct array access
    for (uint32_t p_id : processed_ids) {
        uint32_t g_id = _processed_to_global_id[p_id]; // Get global ID if needed for activation
        
        encounters[p_id]++;
        _word_last_decay_h[p_id] = now_h;
        float cp = prof[p_id];
        float cv = vol[p_id];
        float act = _get_word_activation(g_id, now_h); // Activation still uses global ID
        float lop = std::abs(u_val - cp);
        bool better = err > 0;
        float bu = learning_rate * std::min(std::abs(err), lop);
        float um = bu * cv * act;
        float dv = better ? (proficiency_max - cp) : (cp - proficiency_min);
        float upd = um * dv * (better ? 1.0f : -1.0f);

        cp += upd;
        cp = std::max(proficiency_min, std::min(proficiency_max, cp));

        float vd = 0.1f * cv * u_val * act * (1.0f - 0.5f * std::min(1.0f, std::abs(err)));
        cv -= vd;
        cv = std::max(0.1f, cv);
        if (encounters[p_id] > 3) {
            cv = std::max(0.1f, cv * 0.9f);
        }
        
        prof[p_id] = cp;
        vol[p_id] = cv;
        eff_prof[p_id] = _eff_prof_formula(cp, cv);
    }
}

std::vector<float> VocabularyModel::get_effective_proficiency_by_id(const std::vector<uint32_t>& global_word_ids) {
    std::vector<float> out;
    out.reserve(global_word_ids.size());
    for (uint32_t global_id : global_word_ids) {
        auto it = _global_to_processed_id.find(global_id);
        if (it != _global_to_processed_id.end()) {
            out.push_back(eff_prof[it->second]); // Use mapped ID
        } else {
            out.push_back(0.0365f); // Default for unprocessed words
        }
    }
    return out;
}

std::vector<float> VocabularyModel::get_effective_proficiency_by_str(const std::vector<std::string>& words) {
    std::vector<uint32_t> global_word_ids;
    global_word_ids.reserve(words.size());
    for(const auto& w : words) {
        if(idx.has_word(w)) {
            global_word_ids.push_back(idx.get_id(w));
        } else {
            global_word_ids.push_back(UINT32_MAX); // Placeholder for unknown words
        }
    }
    return get_effective_proficiency_by_id(global_word_ids);
}

float VocabularyModel::_predict_understanding_by_id(const std::vector<uint32_t>& global_word_ids, double current_time_h) {
    if (global_word_ids.empty()) return 0.0f;

    _process_due_word_decays(current_time_h);
    _process_due_trace_decays(current_time_h);

    float sum_p = 0.0f, min_p = 1.0f;
    std::unordered_set<uint32_t> non_zero_global_ids;
    
    for (uint32_t global_id : global_word_ids) {
        float ep;
        auto it = _global_to_processed_id.find(global_id);
        if (it != _global_to_processed_id.end()) {
            ep = eff_prof[it->second]; // Use mapped ID
            non_zero_global_ids.insert(global_id);
        } else {
            ep = 0.0365f;
        }
        if (ep < min_p) min_p = ep;
        sum_p += ep;
    }

    size_t n = global_word_ids.size();
    float avg_p = sum_p / n;
    float ctx = _calculate_context_support(non_zero_global_ids, n);

    float lf = 1.0f / (1.0f + 0.1f * n);
    lf = std::min(0.5f, lf);
    float mw = 0.3f + 0.2f * lf;
    float aw = 0.5f - 0.2f * lf;
    float cw = 0.2f;

    float u = mw * min_p + aw * avg_p + cw * ctx;
    return std::max(0.0f, std::min(1.0f, u));
}

float VocabularyModel::predict_understanding(const std::vector<std::string>& words, double current_time_h) {
    std::vector<uint32_t> word_ids;
    word_ids.reserve(words.size());
    for(const auto& w : words) {
        if(idx.has_word(w)) {
            word_ids.push_back(idx.get_id(w));
        } else {
            word_ids.push_back(UINT32_MAX);
        }
    }
    return _predict_understanding_by_id(word_ids, current_time_h);
}

// --- Private methods for decay, propagation, etc. ---

void VocabularyModel::_process_due_word_decays(double now_h) {
    while (!_word_decay_pq.empty() && _word_decay_pq.top().next_decay_time <= now_h) {
        DecayItem current_item = _word_decay_pq.top();
        _word_decay_pq.pop();
        
        // The ID from the queue is now the DENSE processed_id
        uint32_t p_id = current_item.id_or_idx;
        
        // Ensure processed_id is still valid before decaying
        if (p_id < _processed_to_global_id.size()) {
            _apply_decay_to_word_id(p_id, now_h);
            // Re-queue with the same dense processed_id
            _word_decay_pq.push({now_h + min_elapsed_h, p_id});
        }
    }
}

void VocabularyModel::_apply_decay_to_word_id(uint32_t p_id, double now_h) {
    double elapsed = now_h - _word_last_decay_h[p_id];
    if (elapsed <= min_elapsed_h) return;

    float p = prof[p_id];
    float v = vol[p_id];
    float encf = 1.0f / (1.0f + 0.2f * encounters[p_id]);
    encf = std::max(0.1f, encf);

    float dr = base_decay_rate * v * encf;
    double damt = 1.0 - std::exp(-dr * elapsed / 24.0);
    p *= (1.0f - damt);
    p = std::max(p, proficiency_min);

    v += std::min(0.1f, 0.01f * (float)damt * (float)elapsed / 24.0f);
    v = std::min(v, 0.9f);

    prof[p_id] = p;
    vol[p_id] = v;
    eff_prof[p_id] = _eff_prof_formula(p, v);
    _word_last_decay_h[p_id] = now_h;
}

// ... other private methods like _add_trace, _get_word_activation, _propagate, _prune_traces, etc.
// are implemented here in full, without change from the previous C++ version.
// For the final response, I will write them out completely.

int VocabularyModel::_add_trace(const std::set<uint32_t>& word_ids, double timestamp_h, float activation, float decay_factor) {
    int trace_idx = _trace_timestamps_h.size();
    std::vector<uint32_t> current_word_ids_vec(word_ids.begin(), word_ids.end());
    std::sort(current_word_ids_vec.begin(), current_word_ids_vec.end());
    _trace_word_ids.push_back(current_word_ids_vec);
    _trace_timestamps_h.push_back(timestamp_h);
    _trace_activations.push_back(activation);
    _trace_decay_factors.push_back(decay_factor);
    for (uint32_t wid : word_ids) {
        word_to_traces[wid].push_back(trace_idx);
    }
    _trace_decay_pq.push({timestamp_h + min_elapsed_h, (uint32_t)trace_idx});
    return trace_idx;
}

float VocabularyModel::_decay_trace_activation(int trace_idx, double now_h) {
    if (trace_idx < 0 || static_cast<size_t>(trace_idx) >= _trace_activations.size()) return 0.0f;
    double elapsed = now_h - _trace_timestamps_h[trace_idx];
    if (elapsed <= min_elapsed_h) return _trace_activations[trace_idx];
    _trace_activations[trace_idx] /= (1.0f + _trace_decay_factors[trace_idx] * elapsed);
    _trace_timestamps_h[trace_idx] = now_h;
    return _trace_activations[trace_idx];
}

void VocabularyModel::_process_due_trace_decays(double now_h) {
    while (!_trace_decay_pq.empty() && _trace_decay_pq.top().next_decay_time <= now_h) {
        DecayItem current_item = _trace_decay_pq.top();
        _trace_decay_pq.pop();
        int trace_idx = current_item.id_or_idx;
        if (static_cast<size_t>(trace_idx) < _trace_activations.size() && _trace_activations[trace_idx] >= trace_delete_threshold) {
            _decay_trace_activation(trace_idx, now_h);
            _trace_decay_pq.push({now_h + min_elapsed_h, (uint32_t)trace_idx});
        }
    }
}

float VocabularyModel::_get_word_activation(uint32_t word_id, double now_h) {
    if (word_to_traces.find(word_id) == word_to_traces.end()) return 0.0f;
    float A = 0.0f;
    int cnt = 0;
    const auto& trace_indices = word_to_traces.at(word_id);
    for (int trace_idx : trace_indices) {
        float tA = _decay_trace_activation(trace_idx, now_h);
        if (tA > activation_threshold) {
            A += tA;
            cnt++;
        }
    }
    if (cnt == 0) return 0.0f;
    return std::min(1.0f, A / std::sqrt(cnt * 2.0f));
}

void VocabularyModel::_propagate(int source_trace_idx) {
    if (source_trace_idx < 0 || static_cast<size_t>(source_trace_idx) >= _trace_activations.size()) return;
    std::deque<std::pair<int, float>> queue;
    queue.push_back({source_trace_idx, _trace_activations[source_trace_idx]});
    std::unordered_set<int> visited_neighbors;
    while(!queue.empty()){
        auto [current_trace_idx, delta] = queue.front();
        queue.pop_front();
        visited_neighbors.clear();
        const auto& current_word_ids = _trace_word_ids[current_trace_idx];
        for (uint32_t word_id : current_word_ids) {
            if (word_to_traces.count(word_id)) {
                for (int neighbor_trace_idx : word_to_traces.at(word_id)) {
                    if (neighbor_trace_idx != current_trace_idx) visited_neighbors.insert(neighbor_trace_idx);
                }
            }
        }
        for(int neighbor_trace_idx : visited_neighbors) {
            const auto& neighbor_word_ids = _trace_word_ids[neighbor_trace_idx];
            size_t ov = intersection_size_vec_vec(current_word_ids, neighbor_word_ids);
            if (ov == 0) continue;
            float step = delta / current_word_ids.size() * ov;
            float tj_part = _trace_activations[neighbor_trace_idx] / neighbor_word_ids.size() * ov;
            if (tj_part < step) {
                _trace_activations[neighbor_trace_idx] += step - tj_part;
                _trace_activations[neighbor_trace_idx] = std::min(1.0f, _trace_activations[neighbor_trace_idx]);
            }
            if (step > propagation_threshold) queue.push_back({neighbor_trace_idx, step});
        }
    }
}

void VocabularyModel::_prune_traces() {
    std::vector<int> traces_to_keep_indices;
    for (size_t i = 0; i < _trace_activations.size(); ++i) {
        if (_trace_activations[i] >= trace_delete_threshold) traces_to_keep_indices.push_back(i);
    }
    if (traces_to_keep_indices.size() == _trace_activations.size()) return;
    std::vector<std::vector<uint32_t>> new_trace_word_ids;
    std::vector<double> new_trace_timestamps_h;
    std::vector<float> new_trace_activations;
    std::vector<float> new_trace_decay_factors;
    for (int old_idx : traces_to_keep_indices) {
        new_trace_word_ids.push_back(_trace_word_ids[old_idx]);
        new_trace_timestamps_h.push_back(_trace_timestamps_h[old_idx]);
        new_trace_activations.push_back(_trace_activations[old_idx]);
        new_trace_decay_factors.push_back(_trace_decay_factors[old_idx]);
    }
    _trace_word_ids = std::move(new_trace_word_ids);
    _trace_timestamps_h = std::move(new_trace_timestamps_h);
    _trace_activations = std::move(new_trace_activations);
    _trace_decay_factors = std::move(new_trace_decay_factors);
    word_to_traces.clear();
    for (size_t new_idx = 0; new_idx < _trace_word_ids.size(); ++new_idx) {
        for (uint32_t wid : _trace_word_ids[new_idx]) word_to_traces[wid].push_back(new_idx);
    }
}

float VocabularyModel::_calculate_context_support(const std::unordered_set<uint32_t>& id_words, size_t n) {
    if (id_words.empty() || _trace_timestamps_h.empty()) return 0.0f;
    float mx = 0.0f;
    for (size_t trace_idx = 0; trace_idx < _trace_timestamps_h.size(); ++trace_idx) {
        const auto& trace_word_ids_vec = _trace_word_ids[trace_idx];
        size_t ov = intersection_size_set_vec(id_words, trace_word_ids_vec);
        if (ov > 0) {
            float r = (ov / std::max((float)n, (float)trace_word_ids_vec.size())) * _trace_activations[trace_idx];
            if (r > mx) mx = r;
        }
    }
    return mx * 0.5f;
}

// --- Serialization ---
template<typename T> void write_vec(std::ofstream& f, const std::vector<T>& vec) { f.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T)); }
template<typename T> void read_vec(std::ifstream& f, std::vector<T>& vec, size_t n) { vec.resize(n); f.read(reinterpret_cast<char*>(vec.data()), n * sizeof(T)); }

void VocabularyModel::save_fast(const std::string& path) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file for writing: " + path);

    // --- Header ---
    f.write(MAGIC_STRING, sizeof(MAGIC_STRING) - 1);
    f.write(reinterpret_cast<const char*>(&MODEL_VERSION), sizeof(MODEL_VERSION));
    
    // --- Sizes ---
    uint32_t n_processed_words = _processed_to_global_id.size();
    uint32_t n_traces = _trace_timestamps_h.size();
    uint32_t n_total_words_in_idx = idx.size();

    f.write(reinterpret_cast<const char*>(&n_processed_words), sizeof(n_processed_words));
    f.write(reinterpret_cast<const char*>(&n_traces), sizeof(n_traces));
    f.write(reinterpret_cast<const char*>(&n_total_words_in_idx), sizeof(n_total_words_in_idx));

    // --- Parameters ---
    double params[] = {
        (double)learning_rate, (double)context_influence, (double)activation_threshold,
        (double)base_decay_rate, (double)proficiency_min, (double)proficiency_max,
        (double)propagation_threshold, (double)trace_delete_threshold
    };
    f.write(reinterpret_cast<const char*>(params), sizeof(params));

    // --- Core Dense Vectors ---
    write_vec(f, prof);
    write_vec(f, vol);
    write_vec(f, eff_prof);
    write_vec(f, encounters);
    write_vec(f, _word_last_decay_h);

    // --- Mapping Structures ---
    write_vec(f, _processed_to_global_id);
    uint32_t map_size = _global_to_processed_id.size();
    f.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for(const auto& pair : _global_to_processed_id) {
        f.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));   // global_id
        f.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second)); // processed_id
    }

    // --- WordIndex (i2w) ---
    const auto& i2w = idx.get_i2w();
    uint32_t i2w_size = i2w.size();
    f.write(reinterpret_cast<const char*>(&i2w_size), sizeof(i2w_size));
    for(const auto& word : i2w) {
        uint32_t len = word.length();
        f.write(reinterpret_cast<const char*>(&len), sizeof(len));
        f.write(word.c_str(), len);
    }

    // --- Trace Data ---
    if (n_traces > 0) {
        write_vec(f, _trace_timestamps_h);
        write_vec(f, _trace_activations);
        write_vec(f, _trace_decay_factors);
        for(const auto& vec : _trace_word_ids) {
            uint32_t size = vec.size();
            f.write(reinterpret_cast<const char*>(&size), sizeof(size));
            write_vec(f, vec);
        }
    }

    // --- word_to_traces Map ---
    map_size = word_to_traces.size();
    f.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));
    for(const auto& pair : word_to_traces) {
        f.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        uint32_t vec_size = pair.second.size();
        f.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        write_vec(f, pair.second);
    }

    // --- Word Decay Priority Queue ---
    auto word_pq_copy = _word_decay_pq;
    uint32_t pq_size = word_pq_copy.size();
    f.write(reinterpret_cast<const char*>(&pq_size), sizeof(pq_size));
    while(!word_pq_copy.empty()) {
        auto item = word_pq_copy.top();
        f.write(reinterpret_cast<const char*>(&item), sizeof(DecayItem));
        word_pq_copy.pop();
    }

    // --- Trace Decay Priority Queue ---
    auto trace_pq_copy = _trace_decay_pq;
    pq_size = trace_pq_copy.size();
    f.write(reinterpret_cast<const char*>(&pq_size), sizeof(pq_size));
    while(!trace_pq_copy.empty()) {
        auto item = trace_pq_copy.top();
        f.write(reinterpret_cast<const char*>(&item), sizeof(DecayItem));
        trace_pq_copy.pop();
    }
}

std::shared_ptr<VocabularyModel> VocabularyModel::load_fast(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        // Return a new, empty model if the file doesn't exist
        return std::make_shared<VocabularyModel>();
    }

    // --- Header ---
    char magic_buf[sizeof(MAGIC_STRING) - 1];
    f.read(magic_buf, sizeof(magic_buf));
    if (std::string(magic_buf, sizeof(magic_buf)) != MAGIC_STRING) {
        throw std::runtime_error("Invalid model file format.");
    }
    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != MODEL_VERSION) {
        throw std::runtime_error("Incompatible model version. Expected " + std::to_string(MODEL_VERSION) + ", got " + std::to_string(version));
    }

    // --- Sizes ---
    uint32_t n_processed_words, n_traces, n_total_words_in_idx;
    f.read(reinterpret_cast<char*>(&n_processed_words), sizeof(n_processed_words));
    f.read(reinterpret_cast<char*>(&n_traces), sizeof(n_traces));
    f.read(reinterpret_cast<char*>(&n_total_words_in_idx), sizeof(n_total_words_in_idx));
    
    // --- Parameters ---
    double params[8];
    f.read(reinterpret_cast<char*>(params), sizeof(params));
    auto m = std::make_shared<VocabularyModel>(
        params[0], params[3], params[1], params[2], params[4], params[5], 1.0, params[6], params[7]
    );

    // --- Core Dense Vectors ---
    read_vec(f, m->prof, n_processed_words);
    read_vec(f, m->vol, n_processed_words);
    read_vec(f, m->eff_prof, n_processed_words);
    read_vec(f, m->encounters, n_processed_words);
    read_vec(f, m->_word_last_decay_h, n_processed_words);

    // --- Mapping Structures ---
    read_vec(f, m->_processed_to_global_id, n_processed_words);
    uint32_t map_size;
    f.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    m->_global_to_processed_id.clear();
    m->_global_to_processed_id.reserve(map_size);
    for(uint32_t i=0; i<map_size; ++i) {
        uint32_t key, val;
        f.read(reinterpret_cast<char*>(&key), sizeof(key));
        f.read(reinterpret_cast<char*>(&val), sizeof(val));
        m->_global_to_processed_id[key] = val;
    }

    // --- WordIndex (i2w) ---
    uint32_t i2w_size;
    f.read(reinterpret_cast<char*>(&i2w_size), sizeof(i2w_size));
    m->idx.clear();
    for(uint32_t i=0; i<i2w_size; ++i) {
        uint32_t len;
        f.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0');
        f.read(&word[0], len);
        // Use add_word_for_load which correctly populates both i2w and w2i
        m->idx.add_word_for_load(word, i);
    }

    // --- Trace Data ---
    if (n_traces > 0) {
        read_vec(f, m->_trace_timestamps_h, n_traces);
        read_vec(f, m->_trace_activations, n_traces);
        read_vec(f, m->_trace_decay_factors, n_traces);
        m->_trace_word_ids.resize(n_traces);
        for(uint32_t i=0; i<n_traces; ++i) {
            uint32_t size;
            f.read(reinterpret_cast<char*>(&size), sizeof(size));
            read_vec(f, m->_trace_word_ids[i], size);
        }
    }

    // --- word_to_traces Map ---
    f.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
    m->word_to_traces.clear();
    m->word_to_traces.reserve(map_size);
    for(uint32_t i=0; i<map_size; ++i) {
        uint32_t key;
        f.read(reinterpret_cast<char*>(&key), sizeof(key));
        uint32_t vec_size;
        f.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        std::vector<int> vec;
        read_vec(f, vec, vec_size);
        m->word_to_traces[key] = vec;
    }

    // --- Word Decay Priority Queue ---
    uint32_t pq_size;
    f.read(reinterpret_cast<char*>(&pq_size), sizeof(pq_size));
    for(uint32_t i=0; i<pq_size; ++i) {
        DecayItem item;
        f.read(reinterpret_cast<char*>(&item), sizeof(DecayItem));
        m->_word_decay_pq.push(item);
    }
    
    // --- Trace Decay Priority Queue ---
    f.read(reinterpret_cast<char*>(&pq_size), sizeof(pq_size));
    for(uint32_t i=0; i<pq_size; ++i) {
        DecayItem item;
        f.read(reinterpret_cast<char*>(&item), sizeof(DecayItem));
        m->_trace_decay_pq.push(item);
    }

    m->set_model_path(path);
    return m;
}

int promotion_times(float b, float v) {
    int t = 0;
    while (true) {
        if (b * (1.0f - 0.3f * v) > 0.3f) return t;
        float d = std::abs(b - 0.5f);
        float C = 0.15f * d * v;
        b = b * (1.0f - C) + C;
        v = v * (0.95f + 0.025f * d);
        v = std::max(0.1f, v);
        t++;
    }
}

} // namespace i_plus_one
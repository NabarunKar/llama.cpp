#pragma once

#include "llama.h"
#include <memory>
#include <unordered_map>
#include <vector>

// Forward declarations
//
// Radix Tree Node for KV Cache Reuse
//
class llama_kv_cache;

struct llama_radix_node {
    // The token at this node (root has special value -1)
    llama_token token = -1;
    
    // KV cache slot information
    // Which cells in the KV cache contain the KV pairs for this prefix
    std::vector<uint32_t> cache_slots;  // Per-layer cache slot indices
    
    // Reference count - how many sequences are using this prefix
    uint32_t ref_count = 0;
    
    // Last access time (for LRU eviction)
    uint64_t last_access_time = 0;
    
    // Parent node (nullptr for root)
    llama_radix_node * parent = nullptr;
    
    // Children nodes - map from token to child node
    std::unordered_map<llama_token, std::unique_ptr<llama_radix_node>> children;
    
    // Depth in the tree (number of tokens from root)
    uint32_t depth = 0;
    
    // Lock status - true if this node's cache slots are immutable
    bool locked = false;
    
    // Constructor
    llama_radix_node() : ref_count(1), parent(nullptr), token(0), last_access_time(0) {}
    
    // Get the full token sequence from root to this node
    std::vector<llama_token> get_token_sequence() const {
        std::vector<llama_token> sequence;
        const llama_radix_node * node = this;
        
        while (node && node->token != -1) {
            sequence.push_back(node->token);
            node = node->parent;
        }
        
        std::reverse(sequence.begin(), sequence.end());
        return sequence;
    }
    
    // Increment reference count
    void inc_ref() {
        ref_count++;
    }
    
    // Decrement reference count
    void dec_ref() {
        if (ref_count > 0) {
            ref_count--;
        }
    }
    
    // Check if node can be evicted
    bool can_evict() const {
        return ref_count == 0 && !locked;
    }
    
    // Update last access time
    void touch(uint64_t time) {
        last_access_time = time;
    }
};

//
// Radix Tree for managing shared prefixes
//
class llama_radix_tree {
public:
    explicit llama_radix_tree(uint32_t n_layers) : n_layers(n_layers) {
        root = std::make_unique<llama_radix_node>();
    }

    // Find the longest matching prefix for a given token sequence
    // Returns the node representing the longest matching prefix
    std::pair<llama_radix_node *, uint32_t> find_prefix(const std::vector<llama_token> & tokens);
    
    // Insert a new sequence into the tree
    // Returns the newly created or existing node for this sequence
    llama_radix_node * insert_sequence(
        const std::vector<llama_token> & tokens,
        const std::vector<uint32_t> & cache_slots);
    
    // Remove a sequence from the tree (decrement ref counts)
    void remove_sequence(const std::vector<llama_token> & tokens);
    
    // Evict unused nodes to free cache slots (LRU policy)
    // Returns the cache slots that were freed
    std::vector<std::vector<uint32_t>> evict_lru(uint32_t max_evict_count);
    
    // Get root node
    llama_radix_node * get_root() { return root.get(); }
    
    // Increment global time counter
    uint64_t tick() { return ++current_time; }
    
    // Get total number of nodes in the tree
    size_t get_node_count() const;
    
    // Get total number of tokens cached
    size_t get_cached_token_count() const;
    
private:
    std::shared_ptr<llama_radix_node> root;
    uint32_t n_layers; // Number of layers in the model (used for cache_slots sizing)
    uint64_t current_time;
    
    // Helper function to recursively count nodes
    size_t count_nodes(const llama_radix_node * node) const;
    
    // Helper function to collect evictable nodes
    void collect_evictable_nodes(
        llama_radix_node * node,
        std::vector<llama_radix_node *> & nodes);
};
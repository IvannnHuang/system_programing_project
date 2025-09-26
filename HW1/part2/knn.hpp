#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};


// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.empty()) return nullptr;
    const size_t d = Embedding_T<T>::Dim();      // dimensionality 
    const size_t axis = (d == 0) ? 0 : (depth % d);

    // Sort by embedding, then by idx (stable tie-breaker)
    std::sort(items.begin(), items.end(),
              [axis, d](const auto& A, const auto& B) {
              for (size_t i = 0; i < d; ++i) {
                  size_t ax = (axis + i) % d;
                  float va = getCoordinate(A.first, ax);
                  float vb = getCoordinate(B.first, ax);
                  if (va < vb) return true;
                  if (va > vb) return false;
              }
              return false;
            });

    const size_t n   = items.size();
    const size_t mid = n / 2; 

    auto* node = new Node<T>();
    node->embedding = items[mid].first;
    node->idx       = items[mid].second;

    // Partition into left/right vectors (exclude median)
    std::vector<std::pair<T,int>> left(items.begin(), items.begin() + mid);
    std::vector<std::pair<T,int>> right(items.begin() + mid + 1, items.end());

    node->left  = buildKD(left,  depth + 1);
    node->right = buildKD(right, depth + 1);

    return node;
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
    if (!node) return;

    const T& q = Node<T>::queryEmbedding;

    const size_t d = Embedding_T<T>::Dim();
    const size_t axis = (d == 0) ? 0 : (depth % d);

    // Choose near vs far using only the splitting axis
    const float q_ax  = getCoordinate(q, axis);
    const float x_ax  = getCoordinate(node->embedding, axis);
    // Decide near vs far
    Node<T>* nearC = (q_ax < x_ax) ? node->left : node->right;
    Node<T>* farC  = (nearC == node->left) ? node->right : node->left;

    // Visit current node: compute distance and update heap
    const float dist = Embedding_T<T>::distance(q, node->embedding);
    if ((int)heap.size() < K) {
        heap.push(PQItem{dist, node->idx});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push(PQItem{dist, node->idx});
    }

    // Explore near side
    knnSearch(nearC, depth + 1, K, heap);

    // Prune or explore far side
    const float d_split = std::abs(q_ax - x_ax);
    if ((int)heap.size() < K || d_split < heap.top().first) {
        knnSearch(farC, depth + 1, K, heap);
    }
}
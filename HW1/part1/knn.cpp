#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.empty()) return nullptr;

    // Sort by embedding, then by idx (stable tie-breaker)
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) {
                  if (a.first < b.first) return true;
                  if (a.first > b.first) return false;
                  return a.second < b.second;
              });

    const size_t n   = items.size();
    const size_t mid = (n - 1) / 2; // lower median for even n

    Node* node = new Node();
    node->embedding = items[mid].first;
    node->idx       = items[mid].second;
    node->left = node->right = nullptr;

    // Partition into left/right vectors (exclude median)
    std::vector<std::pair<Embedding_T,int>> left(items.begin(), items.begin() + mid);
    std::vector<std::pair<Embedding_T,int>> right(items.begin() + mid + 1, items.end());

    node->left  = buildKD(left,  depth + 1);
    node->right = buildKD(right, depth + 1);

    return node;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
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

    const float q = Node::queryEmbedding;

    // Decide near vs far
    Node* nearC = (q < node->embedding) ? node->left : node->right;
    Node* farC  = (nearC == node->left) ? node->right : node->left;

    // Explore near side
    knnSearch(nearC, depth + 1, K, heap);

    // Visit current node: compute distance and update heap
    float d = distance(q, node->embedding);
    if ((int)heap.size() < K) {
        heap.push(PQItem{d, node->idx});
    } else if (d < heap.top().first) {
        heap.pop();
        heap.push(PQItem{d, node->idx});
    }

    // Prune or explore far side
    float d_split = distance(q, node->embedding);
    if ((int)heap.size() < K || d_split < heap.top().first) {
        knnSearch(farC, depth + 1, K, heap);
    }
}


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>
#include <chrono>


using json = nlohmann::json;


int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <query.json> <passages.json> <K> <eps>\n";
    return 1;
    }

    auto processing_start = std::chrono::high_resolution_clock::now();
    // Load and parse query JSON
    std::ifstream query_ifs(argv[1]);
    if (!query_ifs) {
        std::cerr << "Error opening query file: " << argv[1] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1) {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[2]);
    if (!passages_ifs) {
        std::cerr << "Error opening passages file: " << argv[2] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1) {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }


    // Convert JSON array to a dict mapping id -> element
    std::unordered_map<int, json> dict;
    for (auto &elem : passages_json) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }


    // Parse K and eps
    int k = std::stoi(argv[3]);
    double eps = std::stof(argv[4]);

    try{
        // Extract the query embedding
        auto query_obj   = query_json[0];
        size_t D         = query_obj["embedding"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d) {
            query[d] = query_obj["embedding"][d].get<double>();
        }
        /*
        TODO:
        1. Extract the passage embedding and store it in alglib::real_2d_array, store the idx of each embedding in alglib::integer_1d_array
        2. Build the KD-tree (alglib::kdtree) from the passages embeddings using alglib::buildkdtree
        3. Perform the k-NN search using alglib::knnsearch
        4. Query the results
            - Get the index of each found neighbour  using alglib::kdtreequeryresultstags
            - Get the distance between each found neighbour and the query embedding using alglib::kdtreequeryresultsdists
        */
        
        // Shapes & validation
        const size_t N = passages_json.size();
        if (N == 0) {
            std::cerr << "Passages array is empty\n";
            return 1;
        }
        for (const auto& elem : passages_json) {
            if (!elem.contains("embedding") || !elem["embedding"].is_array()) {
                std::cerr << "Each passage must contain an 'embedding' array\n";
                return 1;
            }
            if (elem["embedding"].size() != D) {
                std::cerr << "Dimension mismatch: passage embedding dim != query dim\n";
                return 1;
            }
            if (!elem.contains("id")) {
                std::cerr << "Each passage must contain an integer 'id'\n";
                return 1;
            }
        }

        // Build ALGLIB inputs (row-major N x D and tags)
        std::vector<double> data;
        data.resize(N * D);
        alglib::integer_1d_array tags;
        tags.setlength((alglib::ae_int_t)N);

        for (size_t i = 0; i < N; ++i) {
            const auto& emb = passages_json[i]["embedding"];
            for (size_t d = 0; d < D; ++d) {
                data[i * D + d] = emb[d].get<double>();
            }
            tags[(alglib::ae_int_t)i] = passages_json[i]["id"].get<int>();
        }
        // Wrap into real_2d_array
        alglib::real_2d_array allPoints;
        allPoints.setcontent((alglib::ae_int_t)N, (alglib::ae_int_t)D, data.data());

        // Build KD-tree
        auto buildtree_start = std::chrono::high_resolution_clock::now();
        alglib::kdtree tree;
        alglib::kdtreebuildtagged(allPoints, tags, (alglib::ae_int_t)N, (alglib::ae_int_t)D,
                                /*NY=*/0, /*normtype=*/2, tree);
        auto buildtree_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> buildtree_duration = buildtree_end - buildtree_start;

        // Query
        if (k <= 0) {
            std::cerr << "K must be positive\n";
            return 1;
        }
        int effective_k = k;
        if ((size_t)effective_k > N) {
            effective_k = static_cast<int>(N);
        }
        auto query_start = std::chrono::high_resolution_clock::now();
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, (alglib::ae_int_t)effective_k, eps);

        // Retrieve ascending sorted results
        alglib::real_1d_array dist;
        dist.setlength(count);
        alglib::kdtreequeryresultsdistances(tree, dist);

        alglib::integer_1d_array nn_ids;
        nn_ids.setlength(count);
        alglib::kdtreequeryresultstags(tree, nn_ids);
        auto query_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> query_duration = query_end - query_start;

        // Print output
        auto program_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> program_duration = program_end - program_start;
        std::chrono::duration<double, std::milli> processing_duration = query_end - processing_start; // coarse: everything after args

        // Query header
        std::cout << "query:\n";
        if (query_obj.contains("text")) {
            std::cout << "  text:    " << query_obj["text"] << "\n\n";
        } else {
            std::cout << "  text:    " << "(no text)\n\n";
        }

        // Neighbors
        for (int i = 0; i < count; ++i) {
            int id = nn_ids[i];
            double dval = dist[i];

            std::cout << "Neighbor " << (i + 1) << ":\n";
            std::cout << "  id:      " << id << ", dist = " << dval << "\n";

            auto it = dict.find(id);
            if (it != dict.end() && it->second.contains("text")) {
                std::cout << "  text:    " << it->second["text"] << "\n\n";
            } else {
                std::cout << "  text:    " << "(no text)\n\n";
            }
        }

        // Performance section
        std::cout << "#### Performance Metrics ####\n";
        std::cout << "Elapsed time: " << program_duration.count() << " ms\n";
        std::cout << "Processing time: " << processing_duration.count() << " ms\n";
        std::cout << "KD-tree build time: " << buildtree_duration.count() << " ms\n";
        std::cout << "K-NN query time: " << query_duration.count() << " ms\n";

        
    }
    catch(alglib::ap_error &e) {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    return 0;
}
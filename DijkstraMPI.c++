#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <limits>
#include <map>
#include <algorithm>

const int INF = std::numeric_limits<int>::max();

struct Node {
    int vertex, weight;
    bool operator>(const Node& other) const {
        return this->weight > other.weight;
    }
};

void dijkstra(const std::vector<std::vector<Node>>& graph, int src, std::vector<int>& dist) {
    int n = graph.size();
    dist.assign(n, INF);
    dist[src] = 0;

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({src, 0});

    while (!pq.empty()) {
        Node node = pq.top();
        pq.pop();
        int u = node.vertex;

        if (dist[u] < node.weight) continue;

        for (const auto& adj : graph[u]) {
            int v = adj.vertex;
            int weight = adj.weight;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({v, dist[v]});
            }
        }
    }
}

void readGraph(const std::string& filename, std::vector<std::vector<Node>>& graph, int& num_vertices) {
    std::ifstream file(filename);
    std::string line;
    std::map<int, std::vector<std::pair<int, int>>> edges;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            int from, to;
            iss >> from >> to;
            edges[from].push_back({to, 1});  // Assuming weight = 1 for each edge
            edges[to].push_back({from, 1});
            num_vertices = std::max(num_vertices, std::max(from, to));
        }
        file.close();
    }
    num_vertices++;  // Adjust for 0-indexing
    graph.resize(num_vertices);

    for (const auto& pair : edges) {
        int u = pair.first;
        for (const auto& edge : pair.second) {
            int v = edge.first, weight = edge.second;
            graph[u].push_back({v, weight});
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<std::vector<Node>> graph;
    int num_vertices = 0;

    if (world_rank == 0) {
        readGraph("facebook_combined.txt.gz", graph, num_vertices);
    }

    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        graph.resize(num_vertices);
    }

    // Broadcast graph structure
    for (int i = 0; i < num_vertices; ++i) {
        int num_adj = graph[i].size();
        MPI_Bcast(&num_adj, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank != 0) {
            graph[i].resize(num_adj);
        }

        for (int j = 0; j < num_adj; ++j) {
            MPI_Bcast(&graph[i][j], sizeof(Node), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }

    // Calculate distances from each vertex
    std::vector<double> local_closeness(num_vertices, 0.0);

    for (int i = world_rank; i < num_vertices; i += world_size) {
        std::vector<int> dist;
        dijkstra(graph, i, dist);
        double sum = 0;
        for (int d : dist) {
            if (d != INF) {
                sum += d;
            }
        }
        if (sum > 0) {
            local_closeness[i] = (num_vertices - 1) / sum;
        }
    }

    // Reduce results to root
    if (world_rank == 0) {
        std::vector<double> global_closeness(num_vertices);
        MPI_Reduce(local_closeness.data(), global_closeness.data(), num_vertices, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Output top 5 centrality values
        std::vector<std::pair<double, int>> centrality_indices;
        for (int i = 0; i < num_vertices; ++i) {
            centrality_indices.push_back({global_closeness[i], i});
        }
        std::sort(centrality_indices.rbegin(), centrality_indices.rend());  // Sort descending

        std::cout << "Top 5 Nodes by Closeness Centrality:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "Node " << centrality_indices[i].second << ": " << centrality_indices[i].first << "\n";
        }
    } else {
        MPI_Reduce(local_closeness.data(), nullptr, num_vertices, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

#include <mpi.h>
#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>

#define INF std::numeric_limits<int>::max()

void readGraph(const char* filename, std::vector<std::vector<int>>& graph, int& num_vertices) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::pair<int, int>> edges;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            int a, b;
            iss >> a >> b;
            edges.push_back({a, b});
            num_vertices = std::max(num_vertices, std::max(a, b));
        }
        file.close();
    }
    num_vertices++;  // Adjust for zero-indexing

    graph.assign(num_vertices, std::vector<int>(num_vertices, INF));

    for (auto& edge : edges) {
        graph[edge.first][edge.second] = 1;  // Assuming undirected graph
        graph[edge.second][edge.first] = 1;
    }
}

void dijkstra(const std::vector<std::vector<int>>& graph, int src, std::vector<int>& dist) {
    int num_vertices = graph.size();
    dist.assign(num_vertices, INF);
    dist[src] = 0;
    std::vector<bool> sptSet(num_vertices, false);

    for (int count = 0; count < num_vertices - 1; count++) {
        int u = -1;
        int min = INF;

        // Pick the minimum distance vertex from the set of vertices not yet processed.
        for (int v = 0; v < num_vertices; v++) {
            if (!sptSet[v] && dist[v] <= min) {
                min = dist[v], u = v;
            }
        }

        // Mark the picked vertex as processed
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the picked vertex.
        for (int v = 0; v < num_vertices; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INF
                && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<std::vector<int>> graph;
    std::vector<int> dist;
    int num_vertices = 0;

    if (world_rank == 0) {
        readGraph("facebook_combined.txt.gz", graph, num_vertices);
    }

    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        graph.resize(num_vertices, std::vector<int>(num_vertices, INF));
    }

    for (int i = 0; i < num_vertices; i++) {
        MPI_Bcast(&graph[i][0], num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    }

    dijkstra(graph, world_rank, dist);

    if (world_rank == 0) {
        for (int i = 0; i < num_vertices; i++) {
            std::cout << "Distance from 0 to " << i << " is " << dist[i] << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;

/**
 * Tools to analyse map data
 * NOTE: Index nodes starting from 0
 */
public class MapTool {
    /**
     * Represents a way to travel between two locations
     */
    public static class MapConnection {
        int nodeA, nodeB, distance;

        public MapConnection(int u, int v, int w) {
            nodeA = u;
            nodeB = v;
            distance = w;
        }

        @Override
        public String toString() {
            return "{" + nodeA + " -> " + nodeB + ": " + distance + "}";
        }
    }


    /**
     * Collection of all the connections between different locations
     */
    public static class MapGraph {
        ArrayList<ArrayList<MapConnection>> graph;


        public MapGraph() {
            graph = new ArrayList<>();
        }


        public int size() {
            return graph.size();
        }


        public ArrayList<MapConnection> get(int idx) {
            return graph.get(idx);
        }


        public void addEdge(int nodeA, int nodeB, int dist) {
            // Connect nodeA and nodeB with an edge of some distance

            while (graph.size() <= nodeA || graph.size() <= nodeB) {
                graph.add(new ArrayList<>());
            }

            graph.get(nodeA).add(new MapConnection(nodeA, nodeB, dist));
        }


        public void formGraph(int[][] dist) {
            for (int i = 0; i < dist.length; i++) {
                for (int j = 0; j < dist[i].length; j++) {
                    if (dist[i][j] > 0) {
                        addEdge(i, j, dist[i][j]);
                    }
                }
            }
        }
    }


    /**
     * Return the shortest path to visit all locations
     *     Graph must be fully connected
     *     Time Complexity: O(n^2 * 2^n)
     *     Memory Usage: O(n * 2^n)
     */
    public static ArrayList<MapConnection> findPath(MapGraph graph, int start, int end) {
        int[][] dist = toAdjacencyMatrix(graph);
        int N = dist.length;
        int VISITED_ALL = (1 << N) - 1;

        // dp[mask][i] will be the minimum cost to visit all nodes in mask, ending at node i
        int[][] dp = new int[1 << N][N];
        int[][] parent = new int[1 << N][N];

        // Initialize dp array with a large value (representing infinity)
        for (int[] row : dp) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }

        // Starting point
        dp[1 << start][start] = 0;

        // Iterate over all possible subsets of nodes
        for (int mask = 0; mask < (1 << N); mask++) {
            for (int u = 0; u < N; u++) {
                if ((mask & (1 << u)) != 0) { // If u is in the subset represented by mask
                    for (int v = 0; v < N; v++) {
                        if ((mask & (1 << v)) == 0) { // If v is not in the subset
                            int newMask = mask | (1 << v);
                            int newDist = dp[mask][u] + dist[u][v];
                            if (newDist < dp[newMask][v]) {
                                dp[newMask][v] = newDist;
                                parent[newMask][v] = u;
                            }
                        }
                    }
                }
            }
        }

        // Reconstruct the path
        ArrayList<Integer> path = new ArrayList<>();
        int mask = VISITED_ALL;
        int currentNode = end;
        while (mask != 0) {
            path.add(currentNode);
            int temp = mask;
            mask ^= (1 << currentNode);
            currentNode = parent[temp][currentNode];
        }
        path.add(start);
        Collections.reverse(path);

        // Convert the path to ArrayList<MapConnection>
        ArrayList<MapConnection> result = new ArrayList<>();
        for (int i = 0; i < path.size() - 1; i++) {
            int nodeA = path.get(i);
            int nodeB = path.get(i + 1);
            result.add(new MapConnection(nodeA, nodeB, dist[nodeA][nodeB]));
        }

        return result;
    }


    /**
     * Solve TSP using Held-Karp algorithm and return the path as an ArrayList<MapConnection>
     *     Graph must be fully connected
     *     Time Complexity: O(n^2 * 2^n)
     *     Memory Usage: O(n * 2^n)
     */
    public static ArrayList<MapConnection> solveTSP(MapGraph graph, int start) {
        int[][] dist = toAdjacencyMatrix(graph);
        int N = dist.length;
        int VISITED_ALL = (1 << N) - 1;

        // dp[mask][i] will be the minimum cost to visit all nodes in mask, ending at node i
        int[][] dp = new int[1 << N][N];
        int[][] parent = new int[1 << N][N];

        // Initialize dp array with a large value (representing infinity)
        for (int[] row : dp) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }

        // Starting point
        dp[1 << start][start] = 0;

        // Iterate over all possible subsets of nodes
        for (int mask = 0; mask < (1 << N); mask++) {
            for (int u = 0; u < N; u++) {
                if ((mask & (1 << u)) != 0) { // If u is in the subset represented by mask
                    for (int v = 0; v < N; v++) {
                        if ((mask & (1 << v)) == 0) { // If v is not in the subset
                            int newMask = mask | (1 << v);
                            int newDist = dp[mask][u] + dist[u][v];
                            if (newDist < dp[newMask][v]) {
                                dp[newMask][v] = newDist;
                                parent[newMask][v] = u;
                            }
                        }
                    }
                }
            }
        }

        // Find the minimum cost to visit all nodes and return to the starting node
        int minCost = Integer.MAX_VALUE;
        int lastNode = -1;
        for (int u = 0; u < N; u++) {
            if (u != start) {
                int cost = dp[VISITED_ALL][u] + dist[u][start];
                if (cost < minCost) {
                    minCost = cost;
                    lastNode = u;
                }
            }
        }

        // Reconstruct the path
        ArrayList<Integer> path = new ArrayList<>();
        int mask = VISITED_ALL;
        int currentNode = lastNode;
        while (mask != 0) {
            path.add(currentNode);
            int temp = mask;
            mask ^= (1 << currentNode);
            currentNode = parent[temp][currentNode];
        }
        path.add(start);
        Collections.reverse(path);

        // Convert the path to ArrayList<MapConnection>
        ArrayList<MapConnection> result = new ArrayList<>();
        for (int i = 0; i < path.size() - 1; i++) {
            int nodeA = path.get(i);
            int nodeB = path.get(i + 1);
            result.add(new MapConnection(nodeA, nodeB, dist[nodeA][nodeB]));
        }
        // Add the return path to the start
        int nodeA = path.get(path.size() - 1);
        int nodeB = path.get(0);
        result.add(new MapConnection(nodeA, nodeB, dist[nodeA][nodeB]));

        return result;
    }


    //TODO: Use Christofide's algorithm to provide a fast approximation of the optimal TSP solution
//    public static MapConnection solveTSPFast(MapGraph graph, int start) {
//        MapGraph mst = getMST(graph);
//        ArrayList<Integer> oddEdgeNodes = new ArrayList<>();
//        MapGraph minWeight =
//    }


    /**
     * Returns a minimum spanning tree from a starting point
     */
    public static MapGraph getMST(MapGraph graph, int start) {
        int numNodes = graph.size();
        if (numNodes == 0) return null;

        // Initialize the MST and visited array
        MapGraph mst = new MapGraph();
        boolean[] visited = new boolean[numNodes];

        // Use a priority queue to select the next edge with the minimum weight
        PriorityQueue<MapConnection> pq = new PriorityQueue<>((a, b) -> a.distance - b.distance);

        // Start from node 0
        visited[start] = true;
        pq.addAll(graph.get(start));

        while (!pq.isEmpty()) {
            MapConnection edge = pq.poll();
            if (visited[edge.nodeB]) continue; // Skip if the node is already in the MST

            // Add edge to the MST
            mst.addEdge(edge.nodeA, edge.nodeB, edge.distance);

            // Mark the node as visited
            visited[edge.nodeB] = true;

            // Add all edges from the new node to the priority queue
            for (MapConnection nextEdge : graph.get(edge.nodeB)) {
                if (!visited[nextEdge.nodeB]) {
                    pq.add(nextEdge);
                }
            }
        }

        return mst;
    }

    /**
     * Convert MapGraph to an adjacency matrix
     */
    public static int[][] toAdjacencyMatrix(MapGraph mapGraph) {
        int numNodes = mapGraph.graph.size();
        int[][] adjacencyMatrix = new int[numNodes][numNodes];

        // Initialize the matrix with a large value (representing infinity for no direct path)
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < numNodes; j++) {
                if (i != j) {
                    adjacencyMatrix[i][j] = Integer.MAX_VALUE;
                }
            }
        }

        // Populate the adjacency matrix with the distances
        for (int i = 0; i < numNodes; i++) {
            for (MapConnection connection : mapGraph.graph.get(i)) {
                adjacencyMatrix[connection.nodeA][connection.nodeB] = connection.distance;
            }
        }

        return adjacencyMatrix;
    }
}





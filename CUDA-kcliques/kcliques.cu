/*************************************
**** Grzegorz B. Zaleski (418494) ****
*************************************/

#include <set>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <unistd.h>
#include <iostream>
using namespace std;
using LL = long long;
using ULL = unsigned long long int;
using u_int = unsigned int;

const ULL MOD = 1e9;

// Scaling graph to [0; (n-1)) indexed nodes - O(nlogn)
vector<pair<int,int>> get_edges(const string path, int &nodes)
{
    ifstream myfile;
    myfile.open(path);

    u_int a, b;
    unordered_map<u_int, int> scaler;
    unordered_map<u_int, int> deg;
    set<u_int> nodes_set;
    set<pair<u_int, u_int>> edges;
    if (myfile.is_open()) // Reading and counting degree.
    {
        while (myfile.good()) 
        { 
            myfile >> a;
            myfile >> b;
            edges.insert({a, b});

            nodes_set.insert(a);
            nodes_set.insert(b);

            if (deg.find(a) == deg.end())
                deg[a] = 0;
            else 
                deg[a]++;

            if (deg.find(b) == deg.end())
                deg[b] = 0;
            else 
                deg[b]++;
        }
    }
    myfile.close();

    vector<pair<int, u_int>> nodes_vec;
    for (const auto &node: nodes_set)
    {
       nodes_vec.push_back({-deg[node], node});
    }
    sort(nodes_vec.begin(), nodes_vec.end());
    nodes = nodes_set.size();

    for (int i = 0; i < (int)nodes_vec.size(); ++i)
    {
       scaler[nodes_vec[i].second] = i;
    }

    vector<pair<int,int>> _edges;

    // Orienting & scaling graph
    for (const auto &edge : edges)
    {
        auto new_edge = edge;
        if (deg[new_edge.first] > deg[new_edge.second] 
        	|| (deg[new_edge.first] == deg[new_edge.second] && new_edge.first > new_edge.second))
        {
            auto tmp = make_pair(new_edge.second, new_edge.first);
            new_edge.first = tmp.first;
            new_edge.second = tmp.second;
        }
        _edges.push_back(make_pair(scaler[new_edge.first], scaler[new_edge.second]));
    }

    sort(_edges.begin(), _edges.end());
    return _edges;
}

const int MAXK = 12;
const int NO_BLOCKS = 128; 
const int BLOCKSIZE = 256;
const int WARPSIZE = 32;
const int NO_WARPS = BLOCKSIZE / WARPSIZE;
const size_t MAXDEG = 1024;
// Each warp per traverse call (each warp has its own stack)

struct cell // struct for dfs stack
{
    unsigned int set[WARPSIZE];
    char depth; // depth < 12
};

struct gcell // struct for graph cell
{
    unsigned int set[WARPSIZE];
};

const int NODE_NOT_FOUND = -1;
// Encoding node as a position on root's adjacency list
__device__ int encode(int root, int target, int *ends, int *adj)
{
    int a = root == 0 ? 0 : ends[root - 1];
    int b = ends[root];

    int low = a;
    int high = b - 1;

    while (low <= high)
    {
        int mid = low + (high - low) / 2;
        
        if (adj[mid] == target)
            return mid - a; // offset by root positon 
        else if (adj[mid] < target)
            low = mid + 1;
        else // adj[mid] > target
            high = mid - 1;
    }

    return NODE_NOT_FOUND;
}

/*
    Parallel clique search
    n - number of nodes 
    m - number of edges
    k - clique threshold
    ends, adj - graph representantion (list of adjacency)
    dfs_stack - structure for running DFS.
    result[k] - final number of k-cliques
*/

const int MEM_THRESHOLD = 366; // Maximal value to fit in shared memory
__global__ void kernel(int n, int m, int k, int *ends, int *adj, cell *dfs_stack, size_t stack_size, ULL *result)
{ 
    int v_id = blockIdx.x; // which root node (0 to n-1)
    const int pos = threadIdx.x; // id in block (0 to BLOCKSIZE-1)
    const int warp_id = pos / WARPSIZE; // which warp  (0 to NO_WARPS-1)
    const int wid = pos % WARPSIZE; // id in warp (0 to WARPSIZE-1) 0-32

    __shared__ ULL local_result[MAXK]; // saving results locally
    __shared__ int top_stack_pos[NO_WARPS]; // next empty place on stack
    __shared__ int end_of_stack[NO_WARPS]; // end of stack
    __shared__ cell aux_cell[NO_WARPS]; // intersection cell
    __shared__ cell front_cell[NO_WARPS]; // current front
    __shared__ gcell graph[MEM_THRESHOLD]; // scaled subgraph, graph[v] -> encoded intersection of N(v)

    if (pos == 0)
    {
        for (int i = 0; i <= k; ++i)
        {
            local_result[i] = 0;
        }
    }

    if (wid == 0)
    {
        top_stack_pos[warp_id] = 0;
        end_of_stack[warp_id] = 0;
        aux_cell[warp_id].depth = 0;
        front_cell[warp_id].depth = 0;

        size_t node_stack_size = stack_size / NO_BLOCKS; // allocation per root call (per node, block)
        size_t warp_stack_size = node_stack_size / NO_WARPS; // allocation per warp

        top_stack_pos[warp_id] = v_id * node_stack_size + warp_id * warp_stack_size;
        end_of_stack[warp_id] = top_stack_pos[warp_id];
    }

    aux_cell[warp_id].set[wid] = 0;
    front_cell[warp_id].set[wid] = 0;

    for (; v_id < n; v_id += NO_BLOCKS) // iteration over more blocks
    {
        __syncthreads();
        const int offset_root = v_id == 0 ? 0 : ends[v_id - 1];
        const int root_range_size = 1 + ends[v_id] - (v_id == 0 ? 0 : ends[v_id - 1]);

        if (pos == 0) // head in the entire block
        {
            // First call -> subtreeTraverse(G_v, k, 2, adj(v)), allocate node's subtree to warps.
            int a = v_id == 0 ? 0 : ends[v_id - 1];
            int b = ends[v_id];

            // root node;
            cell dfs_put;
            for (int i = 0; i < 32; ++i)
                dfs_put.set[i] = 0;

            dfs_put.depth = 2; // number of nodes on edge

            // Encoding
            for (u_int i = 0; i < b - a; ++i)
            {
                dfs_put.set[i / 32] ^= (1 << (i % 32));
            }

            for (int w = a; w < b; ++w) // assinging subtrees to warps
            {
                int nei_warp = adj[w];

                cell dfs_node_warp = dfs_put;
                dfs_node_warp.depth++;
                
                int aw = nei_warp == 0 ? 0 : ends[nei_warp - 1];
                int bw = ends[nei_warp];

                int enc[WARPSIZE];
                for (int i = 0; i < WARPSIZE; i++)
                    enc[i] = 0;
                
                for (int itr = aw; itr < bw; ++itr)
                {
                    int inner_nei = adj[itr];
                    int encoded = encode(v_id, inner_nei, ends, adj);

                    if (encoded != NODE_NOT_FOUND)
                    {
                        enc[encoded / 32] |= (1 << (encoded % 32));
                    }
                }

                for (int i = 0; i < WARPSIZE; ++i)
                    dfs_node_warp.set[i] &= enc[i];

                int which_warp = (w - a) % NO_WARPS;

                dfs_stack[top_stack_pos[which_warp]] = dfs_node_warp;
                top_stack_pos[which_warp]++; // points at first empty

            }
        }
        else if (root_range_size < MEM_THRESHOLD)
        {
            // Construct subgraph:
            int a = v_id == 0 ? 0 : ends[v_id - 1], b = ends[v_id];
            for (int pos_nei = a + pos - 1, p = pos - 1; pos_nei < b; pos_nei += BLOCKSIZE - 1, p += BLOCKSIZE - 1)
            {
                int v_nei = adj[pos_nei];
                for (int tt = 0; tt < 32; ++tt)
                    graph[p].set[tt] = 0;

                int aw = v_nei == 0 ? 0 : ends[v_nei - 1];
                int bw = ends[v_nei];

                for (int itr = aw; itr < bw; ++itr)
                {
                    int inner_nei = adj[itr];
                    int encoded = encode(v_id, inner_nei, ends, adj);

                    if (encoded != NODE_NOT_FOUND)
                    {
                        graph[p].set[encoded / 32] |= (1 << (encoded % 32));
                    }
                }
            }
        }

        __syncthreads();
 
        // Main traverse
        ULL val;
        while (top_stack_pos[warp_id] != end_of_stack[warp_id])
        {
            __syncwarp();

            if (wid == 0)
            {
                front_cell[warp_id] = dfs_stack[top_stack_pos[warp_id] - 1];
                top_stack_pos[warp_id]--; // next place for cell  

                val = 0;
                for (int tt = 0; tt < 32; ++tt) 
                {
                    val += __popc(front_cell[warp_id].set[tt]);
                }

                if (pos == 0 && local_result[front_cell[warp_id].depth] >= MOD) // only one in block 
                {
                    atomicAdd(local_result + front_cell[warp_id].depth, val - MOD);
                }
                else if (val > 0)
                {
                    atomicAdd(local_result + front_cell[warp_id].depth, val);
                }
            }

            __syncwarp();

            for (u_int i = 0; i < MAXDEG; ++i)
            {
                __syncwarp();
                if (front_cell[warp_id].set[i / 32] & (1 << (i % 32)))
                {
                    __syncwarp();
                    if (wid == 0)
                        aux_cell[warp_id].depth = front_cell[warp_id].depth + 1;
              
                    aux_cell[warp_id].set[wid] = 0; // intersection
             
                    __syncwarp();

                    if (root_range_size < MEM_THRESHOLD && i < MEM_THRESHOLD) // optimised lookup
                    {
                        int nextkin = i; // p -> graph[p];
                        
                        for (int tt = 0; tt < 32; ++tt)
                        {
                            int encoded_bit = graph[nextkin].set[wid] & (1 << tt);
                            int father_bit = front_cell[warp_id].set[wid] & (1 << tt);

                            if (encoded_bit & father_bit)
                            {
                                aux_cell[warp_id].set[wid] |= 1 << tt;
                            }
                        }
                    }
                    else // regular lookup
                    {
                        int nextkin = adj[offset_root + i]; 
                        // construct intersection N(nextkin) & N(front_cell)
                        int a1 = nextkin == 0 ? 0 : ends[nextkin - 1], b1 = ends[nextkin];
                        for (int w = a1 + wid; w < b1; w += WARPSIZE)
                        {
                            int inner_nei = adj[w];
                            int encoded = encode(v_id, inner_nei, ends, adj);
                            if (encoded != NODE_NOT_FOUND)
                            {
                                int father_bit = front_cell[warp_id].set[encoded / 32] & (1 << (encoded % 32));

                                if (father_bit) // `and` in intersection
                                {
                                    atomicOr(&aux_cell[warp_id].set[encoded / 32], (1 << (encoded % 32)));
                                }
                            }
                        }
                    }

                    __syncwarp();

                    if (wid == 0)
                    {
                        val = 0;
                        for (int tt = 0; tt < 32 && val == 0; ++tt) 
                        {
                            val += __popc(aux_cell[warp_id].set[tt]);
                        }

                        if (val > 0 && aux_cell[warp_id].depth <= k) 
                        {
                            dfs_stack[top_stack_pos[warp_id]] = aux_cell[warp_id];
                            top_stack_pos[warp_id]++; // new next empty
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    if (pos == 0)
    {
        for (int i = 3; i <= k; ++i)
        {
            atomicAdd(result + i, local_result[i] % MOD); 
        }
    }
}

const bool DEBUG = false;
int main(int argc, char* argv[])
{
    // Max clique size
    const int k = stoi(argv[2]);
    int _n;

    // Sorted, scaled input with nodes from 0 to (n-1), each with out-deg <= 1024
    auto edges = get_edges(argv[1], _n); 
    const int m = edges.size();
    const int n = _n;

    // STL representantion of a graph.
    if (DEBUG) printf("Graph: %d %d %d\n", n, m, k);
    vector<vector<int>> stl_graph(n);
    for (const auto &edge : edges)
    {
        stl_graph[edge.first].push_back(edge.second);
    }

    // Host data to copy to CUDA device
    int *ends = (int*) malloc(n * sizeof(int));
    int *adj = (int*) malloc(m * sizeof(int));
    ULL *result = (ULL*) malloc((MAXK + 1) * sizeof(ULL));

    // Saving graph onto 1D array form.
    int p = 0;
    size_t max_deg = 0;
    for (int v = 0; v < n; ++v)
    {
        max_deg = max(max_deg, stl_graph[v].size());
        for (auto nei : stl_graph[v])
        {
            adj[p++] = nei;
        }
        ends[v] = p;
    }
    if (DEBUG) printf("MaxDeg(%d, %d) = %ld\n", n, m, max_deg);

    int *dev_ends;
    int *dev_adj;
    ULL *dev_result;
    cell *dev_dfs_stack;

    const size_t stack_size = NO_BLOCKS * MAXK * MAXDEG * NO_WARPS;
    if (DEBUG) printf("Allocated: %ld cells x %ld Bytes for stack \n", stack_size, sizeof(cell));

    cudaMalloc((void**) &dev_ends, sizeof(int) * n);
    cudaMalloc((void**) &dev_adj, sizeof(int) * m);
    cudaMemcpy(dev_ends, ends, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_adj, adj, m * sizeof(int), cudaMemcpyHostToDevice);
    free(adj);
    free(ends);

    cudaMalloc((void**) &dev_result, sizeof(ULL) * (k + 1));
    cudaMalloc((void**) &dev_dfs_stack, sizeof(cell) * stack_size);

    cudaMemset(dev_result, 0, sizeof(ULL) * (k + 1));

    const size_t blocks_to_run = min(n, NO_BLOCKS);
    kernel<<< blocks_to_run, BLOCKSIZE >>>(n, m, k, dev_ends, dev_adj, dev_dfs_stack, stack_size, dev_result); 
    cudaDeviceSynchronize(); // waiting for result of computation

    cudaMemcpy(result, dev_result, (k+1) * sizeof(ULL), cudaMemcpyDeviceToHost);
    cudaFree(dev_ends);
    cudaFree(dev_adj);
    cudaFree(dev_result);
    cudaFree(dev_dfs_stack);

    ofstream result_file;
    const string result_path = argc > 3 ? string(argv[3]) : "output";
    result_file.open(result_path);    

    result[1] = n;
    result[2] = m;
    for (int i = 1; i <= k; ++i)
    {
        if (DEBUG) printf(i != k ? "%llu " : "%llu\n", result[i] % MOD);
        result_file << result[i] % MOD << (i != k ? " " : "\n");
    }

    free(result);
    result_file.close();

    return 0;
}   

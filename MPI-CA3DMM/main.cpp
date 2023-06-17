
#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "densematgen.h"

/************************************
 *** Grzegorz B. Zaleski (418494) ***
 ***********************************/

using namespace std;
using LL = long long;

const double MIN_P_UTIL = 0.95f;
const int ADMIN_PROC = 0;
const LL INF = 1e15;

// Factorisation of number
vector<int> factorise(int p)
{
    vector<int> res;
    for (int i = 2; i * i <= p; ++i)
    {
        while (p % i == 0)
        {
            res.push_back(i);
            p /= i;
        }
    }

    if (p != 1)
    {
        res.push_back(p);
    }

    return res;
}

// Calculating grid
void get_optimal_grid(vector<int> alloc, const vector<int> &factors, LL &bestWork, 
    const int m, const int n, const int k, int &pm, int &pn, int &pk, const int p)
{
    if (alloc.size() == factors.size())
    {        
        int cur_pm = 1, cur_pn = 1, cur_pk = 1;
        for (int i = 0; i < factors.size(); ++i)
        {
            if (alloc[i] == 0)
                cur_pm *= factors[i];
            else if (alloc[i] == 1)
                cur_pn *= factors[i];
            else if (alloc[i] == 2)
                cur_pk *= factors[i];
        }

        LL currentWork = (LL)cur_pm * n * k + (LL)m * cur_pn * k + (LL)m * n * cur_pk;

        if (max(cur_pm, cur_pn) % min(cur_pm, cur_pn) == 0)
        {
            if (currentWork < bestWork)
            {
                bestWork = currentWork;
                pm = cur_pm;
                pn = cur_pn;
                pk = cur_pk;
            }
        } 
    }
    else 
    {
        alloc.push_back(0);
        get_optimal_grid(alloc, factors, bestWork, m, n, k, pm, pn, pk, p);

        alloc.back() = 1;
        get_optimal_grid(alloc, factors, bestWork, m, n, k, pm, pn, pk, p);

        alloc.back() = 2;
        get_optimal_grid(alloc, factors, bestWork, m, n, k, pm, pn, pk, p);
    }
}

// Reads seeds from string
vector<pair<int,int>> parse_seeds(string str)
{
    vector<pair<int,int>> seed_pairs{{0,0}};
    bool first = true;
    for (const auto c : str)
    {
        if (c == ',')
        {
            if (!first)
            {
                seed_pairs.push_back({0, 0});
            }

            first = !first;
        }   
        else 
        {
            if (first)
                seed_pairs.back().first = seed_pairs.back().first * 10 + c - '0';
            else 
                seed_pairs.back().second = seed_pairs.back().second * 10 + c - '0';
        }       
    }

    return seed_pairs;
}

// Cannon matrix fragment structure
struct frag
{
    int grid_pos_x;
    int grid_pos_y;
    int k_divpos;
    int cannon_gip;

    int low_per_p_sizex;
    int wide_per_p_sizey;
    int operation_k_size;

    vector<vector<double>> arrA;
    vector<vector<double>> arrB;
    vector<vector<double>> arrC;
};

// ./ca3dmm n m k -s seeds [-g ge_value] [-v]
int main(int argc, char *argv[])
{
    // Read parametres
    // Matrix size
    const int n_upperbound = stoi(argv[1]);
    const int m_upperbound = stoi(argv[2]);
    const int k_upperbound = stoi(argv[3]);
    // argv[4] == "-s"
    string seeds_str = string(argv[5]);
    const vector<pair<int,int>> seeds = parse_seeds(seeds_str);
    const bool print_matrix = argv[6][1] == 'v';
    const double ge_value = print_matrix ? -1 : stod(argv[7]);

    // Launch MPI
    int numProcesses, myRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int pm = -1, pn = -1, pk = -1; // Optimal process grid allocation
    if (numProcesses > 1) // Parallel calculation of grid size
    {
        LL bestWork = INF;
        for (int _p = ceil(MIN_P_UTIL * numProcesses) + myRank; _p <= numProcesses; _p += numProcesses)
            get_optimal_grid({}, factorise(_p), bestWork, m_upperbound, n_upperbound, k_upperbound, pm, pn, pk, _p);
    }

    if (pm == -1 && myRank == ADMIN_PROC)
    {
        pm = min(k_upperbound, min(m_upperbound, (int)sqrt(numProcesses))); 
        pn = min(k_upperbound, min(n_upperbound, (int)sqrt(numProcesses))); 
        pk = 1;
    }

    auto calc_pn_recv = new int[myRank == ADMIN_PROC ? numProcesses : 1];
    auto calc_pm_recv = new int[myRank == ADMIN_PROC ? numProcesses : 1];
    auto calc_pk_recv = new int[myRank == ADMIN_PROC ? numProcesses : 1];

    MPI_Gather(&pm, 1, MPI_INT, calc_pm_recv, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);
    MPI_Gather(&pn, 1, MPI_INT, calc_pn_recv, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);
    MPI_Gather(&pk, 1, MPI_INT, calc_pk_recv, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);

    LL bestWork = INF;
    for (int i = 0; i < numProcesses; ++i)
    {
        int cur_pm = calc_pm_recv[i];
        int cur_pn = calc_pn_recv[i];
        int cur_pk = calc_pk_recv[i];

        if (cur_pm != -1)
        {
            LL currentWork = (LL)cur_pm * n_upperbound * k_upperbound 
                + (LL)m_upperbound * cur_pn * k_upperbound 
                + (LL)m_upperbound * n_upperbound * cur_pk;

            if (currentWork < bestWork)
            {
                bestWork = currentWork;
                pm = cur_pm;
                pn = cur_pn;
                pk = cur_pk;
            }
        }
    }

    // Receives grid size
    MPI_Bcast(&pm, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);
    MPI_Bcast(&pn, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);
    MPI_Bcast(&pk, 1, MPI_INT, ADMIN_PROC, MPI_COMM_WORLD);
    
    // Remove useless processes
    MPI_Comm comm_allused;
    MPI_Comm_split(MPI_COMM_WORLD, myRank < (pm * pn * pk), myRank, &comm_allused);
    if (myRank >= pm * pn * pk)
    {
        MPI_Finalize();
        return 0;
    }

    // Provide equal divion of matrices between processes
    const int n = ((n_upperbound + pn - 1) / pn) * pn;
    const int m = ((m_upperbound + pm - 1) / pm) * pm;
    //k to be calculated needs value of s
    const int cannon_groups = max(pn, pm) / min(pn, pm);
    const int s = min(pn, pm); // Length and width of grid of processes in single Cannon group
    // How are cannon groups allocated
    const bool cannons_vertical = pm > pn; 
    const bool single_cannnon_group = pm == pn;
    const bool cannons_horizontal = pm < pn;
    
    const int k = ((k_upperbound + (s*pk) - 1) / (s*pk)) * (s*pk); 

    frag myState;

    const int local_myRank = myRank % (s * s);
    
    myState.cannon_gip = (myRank / (s * s)) % cannon_groups;
    myState.grid_pos_x = local_myRank / s;
    myState.grid_pos_y = local_myRank % s;

    myState.k_divpos = myRank / (pm * pn);
    MPI_Comm comm_my_cannonpass; // For passing matrix copies to Cannon groups
    MPI_Comm_split(comm_allused, myRank % (s * s) + myState.k_divpos * (s * s), myRank, &comm_my_cannonpass);

    MPI_Comm comm_my_kline_reduce; // For dividng and reducing k-tasks
    MPI_Comm_split(comm_allused, myRank % (pm * pn), myRank, &comm_my_kline_reduce);

    MPI_Comm comm_frontiers; // Front part of grid
    MPI_Comm_split(comm_allused, myState.k_divpos, myRank, &comm_frontiers);

    myState.low_per_p_sizex = m / pm;
    myState.wide_per_p_sizey = n / pn;
    const int full_koperation_size = k / s;
    myState.operation_k_size = full_koperation_size / pk; 
    
    myState.arrA = vector<vector<double>>{(size_t)myState.low_per_p_sizex, vector<double>(myState.operation_k_size)};
    myState.arrB = vector<vector<double>>{(size_t)myState.operation_k_size, vector<double>(myState.wide_per_p_sizey)};
    myState.arrC = vector<vector<double>>{(size_t)myState.low_per_p_sizex, vector<double>(myState.wide_per_p_sizey)};

    auto rotateA = new double[myState.low_per_p_sizex * myState.operation_k_size];
    auto rotateB = new double[myState.operation_k_size * myState.wide_per_p_sizey];
    auto rotateA_recv = new double[myState.low_per_p_sizex * myState.operation_k_size];
    auto rotateB_recv = new double[myState.operation_k_size * myState.wide_per_p_sizey];

    auto aux_printer = new double[myState.wide_per_p_sizey];

    auto reduceC = new double[myState.low_per_p_sizex * myState.wide_per_p_sizey];
    auto reduceC_recv = new double[myState.k_divpos == 0 ? myState.low_per_p_sizex * myState.wide_per_p_sizey : 1];

    const int NO_TAG = 0;
    MPI_Status statuses[4];

    for (const auto seed_pair : seeds)
    {
        MPI_Barrier(comm_allused);
        
        // 1. Read matrix fragment
        // Matrix A
        for (int i = 0; i < myState.low_per_p_sizex; ++i)
        {
            for (int j = 0; j < myState.operation_k_size; ++j)
            {
                int true_x = i + myState.grid_pos_x * myState.low_per_p_sizex + s * myState.low_per_p_sizex * myState.cannon_gip;
                int true_y = j + ((myState.grid_pos_y + myState.grid_pos_x) % s) * full_koperation_size + myState.k_divpos * myState.operation_k_size;

                if (true_x < m_upperbound && true_y < k_upperbound && (cannons_vertical || myState.cannon_gip == 0))
                {
                    // Shift of preskewment 
                    myState.arrA[i][j] = generate_double(
                        seed_pair.first, 
                        true_x,
                        true_y
                    );
                }
                else
                {
                    myState.arrA[i][j] = 0;
                }
            }
        }

        // Matrix B
        for (int i = 0; i < myState.operation_k_size; ++i)
        {
            for (int j = 0; j < myState.wide_per_p_sizey; ++j)
            {
                int true_x = i + ((myState.grid_pos_x + myState.grid_pos_y) % s) * full_koperation_size + myState.k_divpos * myState.operation_k_size;
                int true_y = j + myState.grid_pos_y * myState.wide_per_p_sizey + s * myState.wide_per_p_sizey * myState.cannon_gip;

                if (true_x < k_upperbound && true_y < n_upperbound && (cannons_horizontal || myState.cannon_gip == 0))
                {
                    // Shift of preskewment
                    myState.arrB[i][j] = generate_double(
                        seed_pair.second,
                        true_x,
                        true_y
                    );
                }
                else 
                {
                    myState.arrB[i][j] = 0;
                }
            }
        }
  
        // Matrix C (clear from previous iteration)
        for (int i = 0; i < myState.low_per_p_sizex; ++i)
        {
            for (int j = 0; j < myState.wide_per_p_sizey; ++j)
            {
                myState.arrC[i][j] = 0;
            }
        }

        // Duplicate smaller matrix for multi cannon groups
        if (cannons_horizontal) // Duplicate A
        {
            // Pass matrix A
            int itr = 0;
            for (int i = 0; i < myState.low_per_p_sizex && myState.cannon_gip == 0; ++i)
            {  
                for (int j = 0; j < myState.operation_k_size; ++j)
                {
                    rotateA[itr++] = myState.arrA[i][j];
                }
            }

            MPI_Bcast(rotateA, myState.low_per_p_sizex * myState.operation_k_size, MPI_DOUBLE, 0, comm_my_cannonpass);

            itr = 0;
            for (int i = 0; i < myState.low_per_p_sizex && myState.cannon_gip != 0; ++i)
            {  
                for (int j = 0; j < myState.operation_k_size; ++j)
                {
                    myState.arrA[i][j] = rotateA[itr++];
                }
            }
        }
        if (cannons_vertical) // Duplicate B
        {
            // Pass matrix B
            int itr = 0;
            for (int i = 0; i < myState.operation_k_size && myState.cannon_gip == 0; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    rotateB[itr++] = myState.arrB[i][j];
                }
            }

            MPI_Bcast(rotateB, myState.operation_k_size * myState.wide_per_p_sizey, MPI_DOUBLE,
                0, comm_my_cannonpass);

            itr = 0;
            for (int i = 0; i < myState.operation_k_size && myState.cannon_gip != 0; ++i)
            {  
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    myState.arrB[i][j] = rotateB[itr++];
                }
            }
        }

        // Run Cannon algorithm
        for (int rot = 0; rot < s; rot++)
        {
            // Multiply
            for (int i = 0; i < myState.low_per_p_sizex; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    for (int w = 0; w < myState.operation_k_size; ++w)
                    {
                        myState.arrC[i][j] += myState.arrA[i][w] * myState.arrB[w][j];
                    }
                }
            }

            // Pass matrix A
            int itr = 0;
            for (int i = 0; i < myState.low_per_p_sizex; ++i)
            {  
                for (int j = 0; j < myState.operation_k_size; ++j)
                {
                    rotateA[itr++] = myState.arrA[i][j];
                }
            }

            // Pass matrix B
            itr = 0;
            for (int i = 0; i < myState.operation_k_size; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    rotateB[itr++] = myState.arrB[i][j];
                }
            }

            if (s > 1)
            {
                // MPI Shift Matrix A
                const int shift_cannon = myRank / (s*s);
                const int shift_value = shift_cannon * (s*s);
                myRank = myRank % (s * s);

                int left_rank = ((myRank % s == 0) ? (myRank + s - 1) : (myRank - 1)) + shift_value;
                int right_rank = ((myRank % s == s - 1) ? (myRank - s + 1) : (myRank + 1)) + shift_value;
                MPI_Sendrecv(rotateA, myState.low_per_p_sizex * myState.operation_k_size, MPI_DOUBLE,
                        left_rank, NO_TAG,
                        rotateA_recv, myState.low_per_p_sizex * myState.operation_k_size, MPI_DOUBLE,
                        right_rank, NO_TAG, MPI_COMM_WORLD, &statuses[0]);

                // MPI Shift Matrix B
                int upper_rank = ((myRank / s == 0) ? (myRank + s * (s - 1)) : (myRank - s)) + shift_value;
                int lower_rank = ((myRank / s == s - 1) ? (myRank - s * (s - 1)) : (myRank + s)) + shift_value;

                MPI_Sendrecv(rotateB, myState.operation_k_size * myState.wide_per_p_sizey, MPI_DOUBLE,
                            upper_rank, NO_TAG,
                            rotateB_recv, myState.operation_k_size * myState.wide_per_p_sizey, MPI_DOUBLE,
                            lower_rank, NO_TAG, MPI_COMM_WORLD, &statuses[1]);
                
                
                myRank = myRank + shift_value;
            }

            // Load matrix A
            itr = 0;
            for (int i = 0; i < myState.low_per_p_sizex; ++i)
            {  
                for (int j = 0; j < myState.operation_k_size; ++j)
                {
                    myState.arrA[i][j] = rotateA_recv[itr++]; 
                }
            }

            // Load matrix B
            itr = 0;
            for (int i = 0; i < myState.operation_k_size; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    myState.arrB[i][j] = rotateB_recv[itr++];
                }
            }
        }

        if (pk > 1)
        {
            // Reduce (sum) matrix -> 3D Cannon on k-dimension
            int itr = 0;
            for (int i = 0; i < myState.low_per_p_sizex; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    reduceC[itr++] = myState.arrC[i][j];
                }
            }

            MPI_Reduce(reduceC, reduceC_recv, myState.low_per_p_sizex * myState.wide_per_p_sizey, 
                    MPI_DOUBLE, MPI_SUM, 0, comm_my_kline_reduce);

            itr = 0;
            for (int i = 0; myState.k_divpos == 0 && i < myState.low_per_p_sizex; ++i)
            {
                for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                {
                    myState.arrC[i][j] = reduceC_recv[itr++];
                }
            }
        }

        // Count ge (if applicable)
        int local_ge_ans = 0;
        for (int i = 0; i < myState.low_per_p_sizex && !print_matrix && myState.k_divpos == 0; ++i)
        {
            for (int j = 0; j < myState.wide_per_p_sizey; ++j)
            {
                int true_x = myState.grid_pos_x * myState.low_per_p_sizex + i; 
                int true_y = myState.grid_pos_y * myState.wide_per_p_sizey + j;

                if (myState.arrC[i][j] >= ge_value && true_x < m_upperbound && true_y < n_upperbound)
                {
                    local_ge_ans++;
                }
            }
        }

        // Print result
        if (print_matrix)
        {
            int value = 0;
            if (myRank != ADMIN_PROC && myState.k_divpos == 0) // Send line part
            {
                for (int i = 0; i < myState.low_per_p_sizex; ++i)
                {
                    int true_x = myState.grid_pos_x * myState.low_per_p_sizex 
                        + i + (cannons_vertical ? s * myState.low_per_p_sizex * myState.cannon_gip : 0);
                    if (true_x < m_upperbound)
                    {
                        // Token for passing data
                        MPI_Recv(&value, 1, MPI_INT, ADMIN_PROC, NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                        {
                            aux_printer[j] = myState.arrC[i][j];
                        }

                        MPI_Send(aux_printer, myState.wide_per_p_sizey, MPI_DOUBLE, ADMIN_PROC, NO_TAG, MPI_COMM_WORLD);
                    }
                }
            }
            else if (myRank == ADMIN_PROC) // Reads and prints line
            {
                // Prints matrix size
                cout << m_upperbound << " " << n_upperbound << endl;

                for (int h = 0; h < m_upperbound; ++h)
                {
                    int grid_height = h / myState.low_per_p_sizex; 

                    for (int grid_wid = 0; grid_wid < n / myState.wide_per_p_sizey; ++grid_wid)
                    {
                        // Own part
                        if (grid_height == 0 && grid_wid == 0)
                        {
                            for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                            {
                                aux_printer[j] = myState.arrC[h][j];
                            }
                        }
                        else 
                        {
                            int data_holder_rank;
                            if (cannons_vertical || pm == pn)
                            {
                                data_holder_rank = s * grid_height + grid_wid;
                            }
                            else
                            {
                                data_holder_rank = (grid_wid / s) * (s * s) + s * grid_height + (grid_wid % s);
                            }

                            MPI_Send(&value, 1, MPI_INT, data_holder_rank, NO_TAG, MPI_COMM_WORLD);
                            MPI_Recv(aux_printer, myState.wide_per_p_sizey, MPI_DOUBLE, data_holder_rank, NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }

                        for (int j = 0; j < myState.wide_per_p_sizey; ++j)
                        {
                            int true_y = grid_wid * myState.wide_per_p_sizey + j;
                            if (true_y < n_upperbound) 
                            {
                                cout << aux_printer[j] << " ";
                            }
                        }
                    }
                    cout << endl;
                }
            }
        }
        else // Print ge results
        {
            if (myState.k_divpos == 0)
            {
                int ge_answer = 0;
                MPI_Reduce(&local_ge_ans, &ge_answer, 1, MPI_INT, MPI_SUM, ADMIN_PROC, comm_frontiers);
                if (myRank == ADMIN_PROC)
                {
                    cout << ge_answer << endl;
                }
            }
        }
    }

    // Finish programme 
    delete[](calc_pn_recv);
    delete[](calc_pm_recv);
    delete[](calc_pk_recv);
    delete[](rotateA);
    delete[](rotateB);
    delete[](rotateA_recv);
    delete[](rotateB_recv);
    delete[](aux_printer);
    delete[](reduceC);
    delete[](reduceC_recv);
    
    MPI_Finalize();   
    return 0;
}
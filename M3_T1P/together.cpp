#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <omp.h>

using namespace std;

void sequentialMatrixMultiplication(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C)
{
    int n = A.size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void multiplyHelper(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int rowStart, int rowEnd) {
    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = 0; j < B[0].size(); j++) {
            C[i][j] = 0;
            for (int k = 0; k < A[0].size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multithreadedMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) {
        numThreads = 1;
    }
    vector<thread> threads(numThreads);

    int numRows = A.size();
    int rowsPerThread = numRows / numThreads;

    int rowStart = 0;
    int rowEnd = rowsPerThread;

    for (int i = 0; i < numThreads; i++) {
        if (i == numThreads - 1) {
            rowEnd = numRows;
        }
        threads[i] = thread(multiplyHelper, ref(A), ref(B), ref(C), rowStart, rowEnd);
        rowStart = rowEnd;
        rowEnd += rowsPerThread;
    }

    for (auto& t : threads) {
        t.join();
    }
}

void mpiMatrixMultiplication(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size)
{
    int n = A.size();
    int p = size;
    int rows = n / p;
    int start = rank * rows;
    int end = (rank == p - 1) ? n : start + rows;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void openmpMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int n = A.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < B[0].size(); j++) {
            C[i][j] = 0;
            for (int k = 0; k < A[0].size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 1000;
    vector<vector<int>> A(n, vector<int>(n));
    vector<vector<int>> B(n, vector<int>(n));
    vector<vector<int>> C(n, vector<int>(n));

    // Initialize input matrices A and B
    srand(time(nullptr));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    


    if (rank == 0)
{
    auto multistart = chrono::steady_clock::now();
    multithreadedMatrixMultiplication(A,B,C);
    auto multiend = chrono::steady_clock::now();
    
    // Sequential matrix multiplication
    auto seqstart = chrono::steady_clock::now();
    sequentialMatrixMultiplication(A, B, C);
    auto seqend = chrono::steady_clock::now();

    auto mpistart = chrono::steady_clock::now();
    mpiMatrixMultiplication(A, B, C, rank, size);
    auto mpiend = chrono::steady_clock::now();

    auto ompstart = chrono::steady_clock::now();
    openmpMatrixMultiplication(A, B, C);
    auto ompend = chrono::steady_clock::now();

    cout << "Sequential time: " << chrono::duration_cast<chrono::duration<double>>(multiend - multistart).count() << " seconds" << endl;
    cout << "Multithreaded time: " << chrono::duration_cast<chrono::duration<double>>( seqend - seqstart ).count() << " seconds" << endl;
    cout << "MPI time: " << chrono::duration_cast<chrono::duration<double>>(mpiend - mpistart).count() << " seconds" << endl;
    cout << "OMP time: " << chrono::duration_cast<chrono::duration<double>>(ompend - ompstart).count() << " seconds" << endl;
}

MPI_Finalize();
return 0;
}

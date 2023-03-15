#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

// Function to print a matrix
void printMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to multiply two matrices
void multiplyMatrices(double** A, double** B, double** C, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    int size = 1000;
    
    // Allocate memory for matrices
    double** A = new double*[size];
    double** B = new double*[size];
    double** C = new double*[size];
    for (int i = 0; i < size; i++) {
        A[i] = new double[size];
        B[i] = new double[size];
        C[i] = new double[size];
    }
    
    // Initialize matrices with random values
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = dis(gen);
            B[i][j] = dis(gen);
        }
    }
    
    // Multiply matrices and measure execution time
    auto start = chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, size);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;
    
    // Write output to file
    ofstream output("output.txt");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            output << C[i][j] << " ";
        }
        output << endl;
    }
    output.close();
    
    // Free memory
    for (int i = 0; i < size; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}

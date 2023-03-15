#include <chrono>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <random>

using namespace std;
using namespace std::chrono;

const int N = 1000;
const int BLOCK_SIZE = 50;
const int NUM_THREADS = (N * N) / (BLOCK_SIZE * BLOCK_SIZE);

int A[N][N], B[N][N], C[N][N];
int current_block = 0;
pthread_mutex_t mutex_current_block;

void init_matrix(int mat[][N]) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(1, 10);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      mat[i][j] = dist(gen);
    }
  }
}
void *matrix_multiply(void *arg) {
  int id = *((int *)arg);
  int block_start_row, block_start_col, block_end_row, block_end_col;
  while (true) {
    pthread_mutex_lock(&mutex_current_block);
    if (current_block >= NUM_THREADS) {
      pthread_mutex_unlock(&mutex_current_block);
      break;
    }
    block_start_row = (current_block / (N / BLOCK_SIZE)) * BLOCK_SIZE;
    block_start_col = (current_block % (N / BLOCK_SIZE)) * BLOCK_SIZE;
    block_end_row = block_start_row + BLOCK_SIZE;
    block_end_col = block_start_col + BLOCK_SIZE;
    current_block++;
    pthread_mutex_unlock(&mutex_current_block);

    for (int i = block_start_row; i < block_end_row; ++i) {
      for (int j = block_start_col; j < block_end_col; ++j) {
        C[i][j] = 0;
        for (int k = 0; k < N; ++k) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  return NULL;
}

int main() {
  init_matrix(A);
  init_matrix(B);
  pthread_t threads[NUM_THREADS];
  int thread_ids[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    thread_ids[i] = i;
    pthread_create(&threads[i], NULL, matrix_multiply, (void *)&thread_ids[i]);
  }

  auto start_time = high_resolution_clock::now();

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

  auto end_time = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end_time - start_time);

  ofstream fout("output.txt");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      fout << C[i][j] << " ";
    }
    fout << "\n";
  }
  fout.close();

  cout << "Execution time: " << duration.count() << " microseconds" << endl;

  return 0;
}

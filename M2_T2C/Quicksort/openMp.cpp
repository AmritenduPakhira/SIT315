#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int partition(vector<int>& data, int left, int right) {
    int pivotIndex = (left + right) / 2;
    int pivotValue = data[pivotIndex];

    swap(data[pivotIndex], data[right]);

    int storeIndex = left;
    for(int i = left; i < right; i++) {
        if(data[i] < pivotValue) {
            swap(data[i], data[storeIndex]);
            storeIndex++;
        }
    }
    swap(data[storeIndex], data[right]);

    return storeIndex;
}

void quickSort(vector<int>& data, int left, int right) {
    if(left < right) {
        int pivotIndex = partition(data, left, right);
        #pragma omp parallel sections
        {
            #pragma omp section
            quickSort(data, left, pivotIndex - 1);
            #pragma omp section
            quickSort(data, pivotIndex + 1, right);
        }
    }
}

int main() {
    vector<int> data;
    for(int i = 0; i < 1000000; i++) {
        data.push_back(rand());
    }
    int n = data.size();

    auto start = high_resolution_clock::now();

    quickSort(data, 0, n - 1);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Sorted " << n << " elements in " << duration.count() << " microseconds" << endl;

    return 0;
}

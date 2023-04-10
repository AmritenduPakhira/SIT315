#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <omp.h>
 
using namespace std;
 
// Function to generate random points
vector<vector<double>> generateRandomPoints(int n, int d, double minVal, double maxVal) {
    vector<vector<double>> points(n, vector<double>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            points[i][j] = minVal + static_cast<double>(rand()) / RAND_MAX * (maxVal - minVal);
        }
    }
    return points;
}
 
// Function to compute the Euclidean distance between two points
double distance(const vector<double>& p1, const vector<double>& p2) {
    double dist = 0.0;
    for (int i = 0; i < p1.size(); i++) {
        dist += pow(p1[i] - p2[i], 2);
    }
    return sqrt(dist);
}
 
// Function to assign each point to the closest cluster center
vector<int> assignPointsToClusters(const vector<vector<double>>& points, const vector<vector<double>>& centers) {
    int n = points.size();
    int k = centers.size();
    vector<int> assignments(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double minDist = INFINITY;
        int closestCenter = 0;
        for (int j = 0; j < k; j++) {
            double dist = distance(points[i], centers[j]);
            if (dist < minDist) {
                minDist = dist;
                closestCenter = j;
            }
        }
        assignments[i] = closestCenter;
    }
    return assignments;
}
 
// Function to compute the sum of all the points assigned to each cluster
vector<vector<double>> computeSums(const vector<vector<double>>& points, const vector<int>& assignments, int k) {
    int d = points[0].size();
    vector<vector<double>> sums(k, vector<double>(d));
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < points.size(); j++) {
            if (assignments[j] == i) {
                for (int l = 0; l < d; l++) {
                    #pragma omp atomic
                    sums[i][l] += points[j][l];
                }
            }
        }
    }
    return sums;
}
 
// Function to update the cluster centers
vector<vector<double>> updateCenters(const vector<vector<double>>& points, const vector<int>& assignments, int k) {
    int d = points[0].size();
    vector<vector<double>> centers(k, vector<double>(d));
    vector<vector<double>> sums = computeSums(points, assignments, k);
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
      int count = std::count(assignments.begin(), assignments.end(), i);
        for (int j = 0; j < d; j++) {
            centers[i][j] = sums[i][j] / count;
        }
    }
    return centers;
}
 
// Function to perform K-Means clustering
vector<int> kMeansClustering(const vector<vector<double>>& points, int k, int maxIter) {
int n = points.size();
int d = points[0].size();
vector<vector<double>> centers = generateRandomPoints(k, d, -10.0, 10.0);
vector<int> assignments(n);
for (int iter = 0; iter < maxIter; iter++) {
assignments = assignPointsToClusters(points, centers);
centers = updateCenters(points, assignments, k);
}
return assignments;
}

int main() {
    srand(time(0));
    int n = 1000;
    int d = 10;
    int k = 10;
    int maxIter = 100;
    vector<vector<double>> points = generateRandomPoints(n, d, -100.0, 100.0);
    auto start = chrono::high_resolution_clock::now();
    vector<int> assignments = kMeansClustering(points, k, maxIter);
    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000000.0;
    cout << "Final cluster assignments:" << endl;
    for (int i = 0; i < n; i++) {
        cout << assignments[i] << " ";
    }
    cout << endl;
    cout << "Execution time: " << duration << " seconds" << endl;
    return 0;
}


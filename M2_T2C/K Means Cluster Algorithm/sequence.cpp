#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

double distance(vector<double> a, vector<double> b) {
    double dist = 0.0;
    for(int i = 0; i < a.size(); i++) {
        dist += pow(a[i] - b[i], 2);
    }
    return sqrt(dist);
}

vector<vector<double>> kMeans(vector<vector<double>> data, int k) {
    int n = data.size();
    vector<int> cluster(n, 0);
    vector<vector<double>> centroids(k, vector<double>(data[0].size(), 0.0));

    // Initialize centroids
    for(int i = 0; i < k; i++) {
        centroids[i] = data[i];
    }

    bool converged = false;
    while(!converged) {
        // Assign each data point to nearest centroid
        converged = true;
        for(int i = 0; i < n; i++) {
            int nearestCluster = cluster[i];
            double nearestDistance = numeric_limits<double>::max();
            for(int j = 0; j < k; j++) {
                double dist = distance(data[i], centroids[j]);
                if(dist < nearestDistance) {
                    nearestDistance = dist;
                    nearestCluster = j;
                }
            }
            if(nearestCluster != cluster[i]) {
                converged = false;
                cluster[i] = nearestCluster;
            }
        }

        // Compute new centroids
        vector<int> clusterSize(k, 0);
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < data[i].size(); j++) {
                centroids[cluster[i]][j] += data[i][j];
            }
            clusterSize[cluster[i]]++;
        }
        for(int i = 0; i < k; i++) {
            if(clusterSize[i] > 0) {
                for(int j = 0; j < centroids[i].size(); j++) {
                    centroids[i][j] /= clusterSize[i];
                }
            }
        }
    }

    return centroids;
}

int main() {
    vector<vector<double>> data = {
        {1, 2},
        {2, 1},
        {5, 8},
        {6, 7},
        {9, 10},
        {10, 9}
    };
    int k = 2;

    vector<vector<double>> centroids = kMeans(data, k);

    for(int i = 0; i < centroids.size(); i++) {
        cout << "Centroid " << i << ": ";
        for(int j = 0; j < centroids[i].size(); j++) {
            cout << centroids[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}

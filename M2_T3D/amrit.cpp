#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <sstream>

struct TrafficData {
    std::time_t timestamp;
    int light_id;
    int num_cars;
};

bool operator<(const TrafficData& left, const TrafficData& right) {
    return left.timestamp < right.timestamp;
}

int main() {
    const std::string data_file = "traffics_data.txt";
    const int top_n = 5;

    std::ifstream infile(data_file);

    // Read data into vector
    std::vector<TrafficData> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string timestamp_str, light_id_str, num_cars_str;
        std::getline(ss, timestamp_str, ',');
        std::getline(ss, light_id_str, ',');
        std::getline(ss, num_cars_str, ',');
        TrafficData td {
            .timestamp = std::stol(timestamp_str),
            .light_id = std::stoi(light_id_str),
            .num_cars = std::stoi(num_cars_str)
        };
        data.push_back(td);
    }

    // Sort data by timestamp
    std::sort(data.begin(), data.end());
    std::time_t current_hour = data.front().timestamp / 3600 * 3600;
    // Process data hourly
    std::unordered_map<int, int> car_counts;
    for (const TrafficData& td : data) {
        if (td.timestamp - current_hour >= 3600) {
            // Report top N congested traffic lights
            std::vector<std::pair<int, int>> light_counts(car_counts.begin(), car_counts.end());
            std::sort(light_counts.begin(), light_counts.end(),
                [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
                    return left.second > right.second;
                });
            std::cout << "Hour " << std::asctime(std::localtime(&current_hour)) << std::endl;
            for (int i = 0; i < std::min(top_n, (int)light_counts.size()); i++) {
                std::cout << "Traffic Light " << light_counts[i].first << ": " << light_counts[i].second << " cars" << std::endl;
            }
            std::cout << std::endl;
            // Reset car counts for next hour
            car_counts.clear();
            current_hour = td.timestamp - (td.timestamp % 3600);
        }
        car_counts[td.light_id] += td.num_cars;
    }

    // Report top N congested traffic lights for last hour
    std::vector<std::pair<int, int>> light_counts(car_counts.begin(), car_counts.end());
    std::sort(light_counts.begin(), light_counts.end(),
        [](const std::pair<int, int>& left, const std::pair<int, int>& right) {
            return left.second > right.second;
        });
    std::cout << "Hour " << std::asctime(std::localtime(&current_hour)) << std::endl;
    for (int i = 0; i < std::min(top_n, (int)light_counts.size()); i++) {
        std::cout << "Traffic Light " << light_counts[i].first << ": " << light_counts[i].second << " cars" << std::endl;
    }

    infile.close();

    return 0;
}

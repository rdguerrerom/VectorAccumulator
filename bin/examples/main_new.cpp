#include <VectorAccumulator/VectorAccumulator.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <atomic>

// Add a counter to track completed tasks
std::atomic<int> completed_tasks{0};

// Helper function to format time durations
std::string format_duration(std::chrono::milliseconds ms) {
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(ms);
    ms -= std::chrono::duration_cast<std::chrono::milliseconds>(secs);
    return std::to_string(secs.count()) + "." + 
           std::to_string(ms.count()) + "s";
}

// Helper function to print statistics
void print_stats(const VectorAccumulator::BufferStats& stats) {
    std::cout << "\n=== Buffer Statistics ===\n"
              << "Total checkouts: " << stats.total_checkouts << "\n"
              << "Total waits: " << stats.total_waits << "\n"
              << "Average wait time: " << std::fixed << std::setprecision(3) 
              << stats.average_wait_time_ms << "ms\n"
              << "Currently checked out: " << stats.currently_checked_out << "\n\n"
              << "Current checkouts:\n";
    
    for (const auto& checkout : stats.current_checkouts) {
        std::cout << "Buffer " << checkout.first << ": "
                  << format_duration(checkout.second) << "\n";
    }
}

// Worker function that performs computations
void worker_task(VectorAccumulator& acc, int id, int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < iterations; ++i) {
        auto buffer_opt = acc.try_checkout(std::chrono::milliseconds(100));
        if (!buffer_opt) {
            std::cout << "Thread " << id << " timeout on iteration " << i << "\n";
            continue;
        }

        double* buffer = *buffer_opt;

        // Simulate computation with random data
        for (size_t j = 0; j < 1024; ++j) {
            buffer[j] = dist(gen);
        }

        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        acc.checkin(buffer);  // Ensure buffer is checked in
    }
    
    completed_tasks++;  // Increment counter when task completes
}

int main() {
    try {
        const size_t BUFFER_SIZE = 1024;
        const size_t NUM_BUFFERS = 8;
        const size_t NUM_THREADS = 6;
        const int ITERATIONS_PER_THREAD = 10;
        
        std::cout << "Initializing VectorAccumulator with:\n"
                  << "- " << NUM_BUFFERS << " buffers\n"
                  << "- " << BUFFER_SIZE << " elements per buffer\n"
                  << "- " << NUM_THREADS << " worker threads\n"
                  << "- " << ITERATIONS_PER_THREAD << " iterations per thread\n\n";

        VectorAccumulator accumulator(NUM_BUFFERS, BUFFER_SIZE,
            [](const char* msg) {
                std::stringstream ss;
                ss << "[" << std::this_thread::get_id() << "] " << msg;
                std::cout << ss.str() << std::endl;
            }
        );

        std::vector<std::thread> threads;
        auto start_time = std::chrono::steady_clock::now();
        completed_tasks = 0;  // Reset counter

        // Launch worker threads
        for (size_t i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back(worker_task, 
                               std::ref(accumulator), 
                               i,
                               ITERATIONS_PER_THREAD);
        }

        // Monitor progress
        while (completed_tasks < NUM_THREADS) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            auto stats = accumulator.get_statistics();
            print_stats(stats);
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        // Rest of the code remains the same...
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                       (end_time - start_time);

        std::cout << "\n=== Final Results ===\n"
                  << "Total execution time: " << format_duration(duration) << "\n";

        // Rest of the statistics printing...
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

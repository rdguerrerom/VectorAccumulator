#include <VectorAccumulator/VectorAccumulator.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <atomic>

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
void worker_task(VectorAccumulator& acc, int id, int iterations, std::atomic<int>& completed_threads) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < iterations; ++i) {
        // Try checkout with timeout
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
        
        acc.checkin(buffer);
        
        // Add small delay between iterations
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Increment completed threads counter
    completed_threads++;
}

int main() {
    try {
        // Create accumulator with debug logging
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

        // Launch worker threads
        std::vector<std::thread> threads;
        std::atomic<int> completed_threads{0};
        auto start_time = std::chrono::steady_clock::now();

        for (size_t i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back(worker_task, 
                               std::ref(accumulator), 
                               i,
                               ITERATIONS_PER_THREAD,
                               std::ref(completed_threads));
        }

        // Monitor progress
        bool monitoring = true;
        while (monitoring) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            auto stats = accumulator.get_statistics();
            print_stats(stats);

            // Check if all threads are done
            if (completed_threads == NUM_THREADS) {
                monitoring = false;
            }
        }

        // Wait for all threads to complete
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                       (end_time - start_time);

        // Perform final reduction
        std::vector<double> output(BUFFER_SIZE, 0.0);
        accumulator.reduce(output.data());

        // Print final statistics
        std::cout << "\n=== Final Results ===\n"
                  << "Total execution time: " << format_duration(duration) << "\n";

        auto perf_stats = accumulator.get_performance_stats();
        std::cout << "\n=== Performance Statistics ===\n"
                  << "Total CPU cycles: " << perf_stats.total_cycles << "\n"
                  << "Reduction cycles: " << perf_stats.reduction_cycles << "\n"
                  << "Avg cycles per reduction: " << std::fixed 
                  << perf_stats.avg_cycles_per_reduction << "\n";

        // Print buffer usage map
        auto usage = accumulator.get_buffer_usage();
        std::cout << "\nFinal buffer usage map:\n";
        for (size_t i = 0; i < usage.size(); ++i) {
            std::cout << "Buffer " << i << ": " 
                      << (usage[i] ? "in use" : "free") << "\n";
        }

        // Print sample of reduced data
        std::cout << "\nSample of reduced data (first 10 elements):\n";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) 
                      << output[i] << " ";
        }
        std::cout << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

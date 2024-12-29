// Path: PerformanceVectorAccumulator.cpp
#include <VectorAccumulator/VectorAccumulator.hpp>
#include <benchmark/benchmark.h>
#include <memory>
#include <thread>
#include <vector>

static void Benchmark_VectorAccumulator(benchmark::State& state) {
    const size_t size = state.range(0);
    VectorAccumulator accumulator(std::thread::hardware_concurrency(), size);
    std::unique_ptr<double[]> output_buffer(new double[size]);

    for (auto _ : state) {
        // Use a thread pool to perform work in parallel
        std::vector<std::thread> threads;
        const size_t num_threads = std::thread::hardware_concurrency();
        const size_t chunk_size = size / num_threads;

        // Each thread gets its own portion of the buffer to work on
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&accumulator, chunk_size, t]() {
                auto buffer = accumulator.try_checkout(std::chrono::milliseconds(100));
                if (!buffer) {
                    return;  // Skip this thread if checkout failed
                }

                // Fill buffer with unique values for each thread
                for (size_t i = t * chunk_size; i < (t + 1) * chunk_size; ++i) {
                    (*buffer)[i] = static_cast<double>(i);
                }

                accumulator.checkin(*buffer);
            });
        }

        // Join all threads to ensure work completes
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        // Perform reduction after all threads have finished
        accumulator.reduce(output_buffer.get());

        benchmark::DoNotOptimize(output_buffer.get());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(Benchmark_VectorAccumulator)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<20)
    ->Threads(std::thread::hardware_concurrency())  // Dynamically use all threads
    ->UseRealTime();

BENCHMARK_MAIN();

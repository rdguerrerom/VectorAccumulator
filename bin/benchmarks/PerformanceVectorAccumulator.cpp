#include <VectorAccumulator/VectorAccumulator.hpp>
#include <benchmark/benchmark.h>
#include <memory>

static void Benchmark_VectorAccumulator(benchmark::State& state) {
    const size_t size = state.range(0);
    VectorAccumulator accumulator(4, size);
    std::unique_ptr<double[]> output_buffer(new double[size]);

    for (auto _ : state) {
        // Get a fresh buffer for this iteration
        auto buffer = accumulator.try_checkout(std::chrono::milliseconds(100));
        if (!buffer) {
            state.SkipWithError("Failed to checkout buffer");
            break;
        }

        // Fill the buffer
        for (size_t i = 0; i < size; ++i) {
            (*buffer)[i] = static_cast<double>(i);
        }

        // Ensure the buffer is checked in immediately
        accumulator.checkin(*buffer);

        // Now perform the reduction
        accumulator.reduce(output_buffer.get());

        benchmark::DoNotOptimize(output_buffer.get());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(Benchmark_VectorAccumulator)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<20)
    ->Threads(1)
    ->UseRealTime();

BENCHMARK_MAIN();

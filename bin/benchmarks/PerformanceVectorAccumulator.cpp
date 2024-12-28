#include "VectorAccumulator.h"
#include <benchmark/benchmark.h>

static void Benchmark_VectorAccumulator(benchmark::State& state) {
    VectorAccumulator accumulator(4, state.range(0));
    for (auto _ : state) {
        double* buffer = accumulator.checkout();
        for (size_t i = 0; i < state.range(0); ++i) {
            buffer[i] = static_cast<double>(i);
        }
        accumulator.checkin(buffer);
        accumulator.reduce(buffer);
    }
}

BENCHMARK(Benchmark_VectorAccumulator)->RangeMultiplier(2)->Range(1<<10, 1<<20);
BENCHMARK_MAIN();


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "VectorAccumulator.hpp"
#include <vector>
#include <stdexcept>

TEST_CASE("VectorAccumulator::reduce") {
    const size_t n_buffers = 4;
    const size_t buffer_size = 1000;
    VectorAccumulator accumulator(n_buffers, buffer_size);

    std::vector<double> output(buffer_size, 0.0);
}

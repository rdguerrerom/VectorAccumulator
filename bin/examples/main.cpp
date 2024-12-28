#include <VectorAccumulator/VectorAccumulator.hpp>
#include <iostream>

int main() {
    try {
        VectorAccumulator accumulator(4, 1024);

        // Set debug callback
        accumulator.set_debug_callback([](const char* msg) {
            std::cout << "[DEBUG] " << msg << std::endl;
        });

        // Example usage
        double* buffer = accumulator.checkout();
        for (size_t i = 0; i < 1024; ++i) {
            buffer[i] = static_cast<double>(i);
        }
        accumulator.checkin(buffer);

        double* output = new double[1024];
        accumulator.reduce(output);

        // Print output
        for (size_t i = 0; i < 10; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;

        delete[] output;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}


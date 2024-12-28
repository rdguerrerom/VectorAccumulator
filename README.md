# VectorAccumulator

A high-performance, thread-safe component for parallel vector accumulation in numerical applications, leveraging SIMD optimization through Agner Fog's Vector Class Library 2 (VCL2).

## Overview

VectorAccumulator provides an efficient solution for scenarios where multiple threads need to accumulate numerical data concurrently. It manages a pool of vector buffers that can be checked out, used for accumulation, and checked back in, with thread-safe operations and optimized SIMD-based reduction capabilities.

## Key Features

- Thread-safe buffer management for parallel accumulation
- SIMD-optimized operations using VCL2
- Cache-aligned data structures to prevent false sharing
- NUMA-aware memory allocation and initialization
- Thread-local buffer caching for improved performance
- Comprehensive performance monitoring capabilities
- Parallel processing for reset operations
- Support for timeout-based buffer acquisition
- Partial reduction capabilities for subset analysis

## Requirements

- C++17 compliant compiler
- Agner Fog's Vector Class Library 2 (VCL2)
- CPU with SSE2, AVX, AVX2, or AVX-512 support (optimized for available instruction sets)
- POSIX-compliant system for aligned memory allocation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VectorAccumulator.git
```

2. Include the header in your project:
```cpp
#include "VectorAccumulator/VectorAccumulator.hpp"
```

## Usage

### Basic Example

```cpp
// Create accumulator with 4 buffers of 1000 elements each
VectorAccumulator acc(4, 1000);

// Checkout a buffer
double* buffer = acc.checkout();

// Perform computations
for (size_t i = 0; i < 1000; ++i) {
    buffer[i] += compute_something(i);
}

// Check buffer back in
acc.checkin(buffer);

// Get final results
std::vector<double> result(1000);
acc.reduce(result.data());
```

### Advanced Usage

```cpp
// Timeout-based checkout
auto buffer = acc.try_checkout(std::chrono::milliseconds(100));
if (buffer) {
    // Use buffer
    acc.checkin(*buffer);
}

// Get performance statistics
auto stats = acc.get_performance_stats();
std::cout << "Average cycles per reduction: " 
          << stats.avg_cycles_per_reduction << "\n";

// Partial reduction
std::vector<size_t> buffer_indices = {0, 2};
std::vector<double> partial_result(1000);
acc.reduce_subset(partial_result.data(), buffer_indices);
```

## Design Criteria

### Thread Safety
- Multiple threads can simultaneously check out and check in buffers
- Single-thread operations (reduction, reset) are protected against concurrent access
- Atomic operations used for performance tracking
- Thread-local optimization for buffer allocation

### Memory Management
- Cache-line aligned buffers to prevent false sharing
- NUMA-aware memory allocation and initialization
- Proper cleanup of resources in destructors
- Move semantics supported, copy operations disabled

### Performance Optimization
- SIMD operations through VCL2 integration
- Manual prefetching for improved memory access patterns
- Thread-local buffer caching
- Parallel processing for reset operations
- Minimized lock contention
- Efficient buffer pool management

### Error Handling
- Comprehensive input validation
- Runtime checks for invalid operations
- Buffer ownership verification
- Timeout support for checkout operations

## Project Structure

```
.
├── LICENSE
├── README.md
├── VectorAccumulator/
│   └── VectorAccumulator.hpp    # Main header file
├── bin/
│   ├── benchmarks/              # Performance benchmarking
│   │   └── PerformanceVectorAccumulator.cpp
│   ├── examples/                # Usage examples
│   │   └── main.cpp
│   └── tests/                   # Unit tests
│       └── TestVectorAccumulator.cpp
└── lib/
    └── VectorAccumulator.cpp    # Implementation file
```

## Performance Considerations

- Buffer checkout operation optimized for quick acquisition
- Reduction operation uses SIMD instructions via VCL2
- Memory allocation occurs only during initialization
- Cache-friendly memory layout and access patterns
- Thread-local caching reduces synchronization overhead
- NUMA-aware memory allocation for multi-socket systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Agner Fog for the Vector Class Library 2
- Contributors to the project
- The C++ community for feedback and suggestions

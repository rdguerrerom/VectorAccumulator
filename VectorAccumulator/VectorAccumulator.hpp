#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <chrono>
#include <optional>
#include <thread>
#include <memory>
#include <functional>
#include "vectorclass.h"

/**
 * @brief A high-performance thread-safe vector accumulator with SIMD optimization
 * 
 * VectorAccumulator provides a pool of fixed-size double precision buffers that can be
 * checked out, modified, and checked in by multiple threads. It is designed for
 * high-performance numerical computations where multiple threads need to accumulate
 * results independently before combining them.
 * 
 * Key Features:
 * - Thread-safe buffer management with checkout/checkin mechanism
 * - SIMD optimization using VCL library
 * - Cache-line alignment to prevent false sharing
 * - Thread-local buffer caching for faster checkout
 * - Manual prefetching for improved memory access patterns
 * - NUMA-aware buffer allocation
 * - Comprehensive performance monitoring and statistics
 * - Debug logging support
 * 
 * Performance Considerations:
 * - Buffer size should be a multiple of Vec4d::size() for optimal SIMD performance
 * - Number of buffers should be chosen based on expected concurrent thread count
 * - Prefetch distance can be tuned based on specific hardware characteristics
 * - Thread-local caching reduces contention in multi-threaded scenarios
 * - NUMA awareness improves performance on multi-socket systems
 * 
 * Thread Safety:
 * This class is thread-safe for all operations. Multiple threads can safely
 * checkout/checkin buffers concurrently.
 */
class VectorAccumulator {
public:
  /**
     * @brief Debugging callback function type
     * Used for logging and monitoring buffer operations
     */
  using DebugCallback = std::function<void(const char*)>;

  /**
     * @brief Construct a new Vector Accumulator
     * 
     * @param n_buffers Number of buffers in the pool
     * @param size Size of each buffer in elements
     * @param debug_cb Optional debug callback for logging operations
     * @throw std::invalid_argument if n_buffers or size is 0
     * @throw std::bad_alloc if memory allocation fails
     */
  VectorAccumulator(size_t n_buffers, 
                    size_t size,
                    DebugCallback debug_cb = nullptr);

  // Move semantics
  VectorAccumulator(VectorAccumulator&& other) noexcept;
  VectorAccumulator& operator=(VectorAccumulator&&) = delete;

  // Prevent copying
  VectorAccumulator(const VectorAccumulator&) = delete;
  VectorAccumulator& operator=(const VectorAccumulator&) = delete;

  ~VectorAccumulator();

  /**
     * @brief Check out a buffer for exclusive use
     * 
     * Uses thread-local caching to optimize repeated checkouts from the same thread.
     * Will block if no buffers are available.
     * 
     * @return double* Pointer to the buffer
     * @throw std::runtime_error if no buffer is available or accumulator is shutting down
     */
  double* checkout();

  /**
     * @brief Return a previously checked out buffer
     * 
     * @param buffer Pointer to the buffer being returned
     * @throw std::invalid_argument if buffer is invalid
     * @throw std::runtime_error if buffer was not checked out
     */
  void checkin(double* buffer);

  /**
     * @brief Try to check out a buffer with timeout
     * 
     * @param timeout Maximum time to wait
     * @return std::optional<double*> Buffer pointer or nullopt if timeout
     */
  std::optional<double*> try_checkout(std::chrono::milliseconds timeout);

  /**
     * @brief Statistics about buffer usage
     */
  struct BufferStats {
    size_t total_checkouts;                    ///< Total number of successful checkouts
    size_t total_waits;                        ///< Number of times threads had to wait
    double average_wait_time_ms;               ///< Average wait time in milliseconds
    size_t currently_checked_out;              ///< Number of buffers currently in use
    std::vector<std::pair<size_t, std::chrono::milliseconds>> current_checkouts;  ///< Details of current checkouts
  };

  /**
     * @brief Get current buffer statistics
     * 
     * @return BufferStats Current statistics
     */
  BufferStats get_statistics() const;

  /**
     * @brief Reduce all buffers into output using SIMD operations
     * 
     * Performs an optimized reduction using SIMD instructions and manual prefetching.
     * All buffers must be checked in before calling this function.
     * 
     * @param output Pointer to output buffer (must be aligned and sized appropriately)
     * @throw std::runtime_error if any buffers are checked out
     */
  void reduce(double* output);

  /**
     * @brief Reset all buffers to zero in parallel
     * 
     * Uses multiple threads to efficiently zero all buffers.
     * All buffers must be checked in before calling this function.
     * 
     * @throw std::runtime_error if any buffers are checked out
     */
  void reset();

  /**
     * @brief Reduce specific buffers into output
     * 
     * @param output Output buffer
     * @param buffer_indices Indices of buffers to reduce
     * @throw std::out_of_range if any index is invalid
     */
  void reduce_subset(double* output, const std::vector<size_t>& buffer_indices);

  /**
     * @brief Read values from a buffer without checking it out
     * 
     * @param buffer_index Buffer to read from
     * @param start Starting position
     * @param length Number of elements to read
     * @return std::vector<double> Buffer values
     * @throw std::out_of_range if parameters are invalid
     */
  std::vector<double> peek_buffer(size_t buffer_index, size_t start, size_t length) const;

  /**
     * @brief Force release a potentially stuck buffer
     * 
     * Useful for handling thread crashes or debugging.
     * 
     * @param buffer_index Buffer to release
     * @return bool True if buffer was released, false if already free
     */
  bool force_release(size_t buffer_index);

  /**
     * @brief Get current buffer usage state
     * 
     * @return std::vector<bool> True for each in-use buffer
     */
  std::vector<bool> get_buffer_usage() const;

  /**
     * @brief Performance statistics
     */
  struct PerformanceStats {
    uint64_t total_cycles;            ///< Total CPU cycles
    uint64_t reduction_cycles;        ///< Cycles spent in reduction operations
    double avg_cycles_per_reduction;  ///< Average cycles per reduction operation
  };

  /**
     * @brief Get performance statistics
     */
  PerformanceStats get_performance_stats() const;

  /**
     * @brief Set debug callback for logging operations
     * 
     * @param callback Function to call for debug logging
     */
  void set_debug_callback(DebugCallback callback);

private:
  // Forward declaration of internal buffer structure
  struct Buffer {
    double* data;
    bool in_use;
    size_t id;
  };

  struct CheckoutInfo {
    std::chrono::steady_clock::time_point checkout_time;
    std::thread::id owner_thread;
    bool is_checked_out;
    
    CheckoutInfo() : is_checked_out(false) {}
};


  std::vector<CheckoutInfo> checkout_times;
  std::atomic<size_t> active_checkouts{0};

  // Remove this forward declaration since we now have the full definition
  // struct Buffer;

  // Constants for optimization
  static constexpr size_t CACHE_LINE_SIZE = 64;
  static constexpr size_t VEC_SIZE = Vec4d::size();
  static constexpr size_t PREFETCH_DISTANCE = 8;

  const size_t num_buffers;
  const size_t buffer_size;
  const size_t aligned_size;
  std::unique_ptr<Buffer[]> buffers;
  std::vector<std::atomic<bool>> buffer_ownership;
  //std::vector<std::atomic<bool>> buffer_ready;

  mutable std::mutex mtx;
  std::condition_variable cv;
  std::atomic<bool> is_shutting_down{false};

  std::atomic<size_t> total_checkouts{0};
  std::atomic<size_t> total_waits{0};
  std::atomic<uint64_t> accumulated_wait_time{0};

  DebugCallback debug_callback;
  static thread_local size_t last_used_buffer;

  std::atomic<uint64_t> total_cycles{0};
  std::atomic<uint64_t> reduction_cycles{0};

  template<typename T>
  T* allocate_aligned(size_t size, size_t alignment);
  static inline uint64_t rdtsc();
  void cleanup();
  void before_cleanup();
  void log_debug(const std::string& message);

  std::vector<CheckoutInfo> checkout_info;  
};

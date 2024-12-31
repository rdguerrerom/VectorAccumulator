#include <algorithm>
#include <chrono>  // Added for timing functions
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>  // Added this header for stringstream
#include <stdexcept>
#include <thread>
#include <atomic>

#include <vector>
#include <mutex>
#include <cmath> 

#include <VectorAccumulator/VectorAccumulator.hpp>

// Static member initialization
thread_local size_t VectorAccumulator::last_used_buffer = 0;

// Constructor
VectorAccumulator::VectorAccumulator(size_t n_buffers,
                                     size_t size,
                                     std::function<void(const char*)> debug_cb)
    : num_buffers(n_buffers),
      buffer_size(size),
      aligned_size((size + Vec4d::size() - 1) & ~(Vec4d::size() - 1)),
      buffers(std::make_unique<Buffer[]>(n_buffers)),
      buffer_ownership(n_buffers),
      checkout_times(n_buffers),
      checkout_info(n_buffers),
      debug_callback(debug_cb){
  if (n_buffers == 0 || size == 0) {
    throw std::invalid_argument("Buffer count and size must be positive");
  }

  for (size_t i = 0; i < num_buffers; ++i) {
    buffers[i].data = allocate_aligned<double>(aligned_size, 64);
    buffers[i].in_use = false;
    buffers[i].id = i;
    buffer_ownership[i] = false;
    //buffer_ready[i].store(false, std::memory_order_release);

    // NUMA-aware allocation
    for (size_t j = 0; j < aligned_size; j += 64 / sizeof(double)) {
      buffers[i].data[j] = 0.0;
    }
  }
}

// Destructor
VectorAccumulator::~VectorAccumulator() {
  cleanup();
}

// Move constructor
VectorAccumulator::VectorAccumulator(VectorAccumulator&& other) noexcept
    : num_buffers(other.num_buffers),
      buffer_size(other.buffer_size),
      aligned_size(other.aligned_size),
      buffers(std::move(other.buffers)),
      buffer_ownership(std::move(other.buffer_ownership)),
      debug_callback(std::move(other.debug_callback)) {
  other.is_shutting_down.store(true);
}

// Debug callback setter
void VectorAccumulator::set_debug_callback(DebugCallback callback) {
  debug_callback = callback;
}

// Checkout a buffer
double* VectorAccumulator::checkout() {
    std::unique_lock<std::mutex> lock(mtx);
    
    // First check if all buffers are in use
    bool all_in_use = std::all_of(checkout_info.begin(), checkout_info.end(),
                                 [](const CheckoutInfo& info) { return info.is_checked_out; });
    
    // If all buffers are in use, throw immediately instead of waiting
    if (all_in_use) {
        throw std::runtime_error("All buffers are currently in use");
    }
    
    // First try the thread-local hint
    if (last_used_buffer < num_buffers && !checkout_info[last_used_buffer].is_checked_out) {
        checkout_info[last_used_buffer].is_checked_out = true;
        checkout_info[last_used_buffer].owner_thread = std::this_thread::get_id();
        checkout_info[last_used_buffer].checkout_time = std::chrono::steady_clock::now();
        buffers[last_used_buffer].in_use = true;
        active_checkouts++;
        total_checkouts++;

        if (debug_callback) {
            std::stringstream ss;
            ss << "Buffer " << last_used_buffer << " checked out by thread " << std::this_thread::get_id();
            debug_callback(ss.str().c_str());
        }

        return buffers[last_used_buffer].data;
    }
    
    // Wait for a buffer to become available
    auto pred = [this]() {
        if (is_shutting_down) return true;
        return std::any_of(checkout_info.begin(), checkout_info.end(), 
                          [](const CheckoutInfo& info) { return !info.is_checked_out; });
    };
    
    cv.wait(lock, pred);
    
    if (is_shutting_down) {
        throw std::runtime_error("Shutting down");
    }
    
    // After waiting, check again if all buffers are in use
    all_in_use = std::all_of(checkout_info.begin(), checkout_info.end(),
                            [](const CheckoutInfo& info) { return info.is_checked_out; });
                            
    if (all_in_use) {
        throw std::runtime_error("All buffers are currently in use");
    }
    
    // Find an available buffer
    for (size_t i = 0; i < num_buffers; ++i) {
        if (!checkout_info[i].is_checked_out) {
            checkout_info[i].is_checked_out = true;
            checkout_info[i].owner_thread = std::this_thread::get_id();
            checkout_info[i].checkout_time = std::chrono::steady_clock::now();
            buffers[i].in_use = true;
            last_used_buffer = i;
            active_checkouts++;
            total_checkouts++;

            if (debug_callback) {
                std::stringstream ss;
                ss << "Buffer " << i << " checked out by thread " << std::this_thread::get_id();
                debug_callback(ss.str().c_str());
            }

            return buffers[i].data;
        }
    }
    
    // This should never be reached due to the earlier checks
    throw std::runtime_error("No buffer available");
}

// Checkin a buffer
void VectorAccumulator::checkin(double* buffer) {
    std::unique_lock<std::mutex> lock(mtx);
    
    auto it = std::find_if(buffers.get(), buffers.get() + num_buffers,
                          [buffer](const Buffer& b) { return b.data == buffer; });
    
    if (it == buffers.get() + num_buffers) {
        throw std::invalid_argument("Invalid buffer returned");
    }
    
    size_t buffer_id = std::distance(buffers.get(), it);
    
    if (!checkout_info[buffer_id].is_checked_out) {
        throw std::runtime_error("Buffer was not checked out");
    }
    
    if (checkout_info[buffer_id].owner_thread != std::this_thread::get_id()) {
        throw std::runtime_error("Buffer must be checked in by the thread that checked it out");
    }
    
    checkout_info[buffer_id].is_checked_out = false;
    buffers[buffer_id].in_use = false;
    active_checkouts--;

    if (debug_callback) {
        std::stringstream ss;
        ss << "Buffer " << buffer_id << " checked in by thread " << std::this_thread::get_id();
        debug_callback(ss.str().c_str());
    }
    
    // Notify waiting threads
    cv.notify_all();
    
    lock.unlock();
}
// Try checkout with timeout
std::optional<double*> VectorAccumulator::try_checkout(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mtx);
    auto start_wait = std::chrono::steady_clock::now();

    bool got_buffer = cv.wait_for(lock, timeout, [this]() {
        if (is_shutting_down)
            return true;
        return std::any_of(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return !b.in_use; });
    });

    if (!got_buffer || is_shutting_down) {
        total_waits++;
        auto wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_wait);
        accumulated_wait_time.fetch_add(wait_duration.count());
        return std::nullopt;
    }

    auto it =
        std::find_if(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return !b.in_use; });
    size_t buffer_id = it->id;

    it->in_use = true;
    buffer_ownership[buffer_id] = true;
    checkout_info[buffer_id].is_checked_out = true;
    checkout_info[buffer_id].checkout_time = std::chrono::steady_clock::now();
    checkout_info[buffer_id].owner_thread = std::this_thread::get_id();
    active_checkouts++;
    total_checkouts++;

    if (debug_callback) {
        std::stringstream ss;
        ss << "Buffer " << buffer_id << " try-checked out by thread " << std::this_thread::get_id();
        debug_callback(ss.str().c_str());
    }

    return it->data;
}

// Reset all buffers
void VectorAccumulator::reset() {
  if (std::any_of(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return b.in_use; })) {
    throw std::runtime_error("Cannot reset while buffers are checked out");
  }

  const size_t num_threads = std::thread::hardware_concurrency();
  const size_t buffers_per_thread = (num_buffers + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  for (size_t t = 0; t < num_threads && t * buffers_per_thread < num_buffers; ++t) {
    threads.emplace_back([this, t, buffers_per_thread]() {
      const size_t start = t * buffers_per_thread;
      const size_t end = std::min(start + buffers_per_thread, num_buffers);

      for (size_t i = start; i < end; ++i) {
        Vec4d zero(0.0);
        for (size_t j = 0; j < aligned_size; j += Vec4d::size()) {
          zero.store(buffers[i].data + j);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

// Reduce all buffers
// TODO: Review and fix this method.
void VectorAccumulator::reduce(double* output) {
    uint64_t start_cycles = rdtsc();
    
    std::unique_lock<std::mutex> lock(mtx);
    
    if (active_checkouts > 0) {
        throw std::runtime_error("Cannot reduce while buffers are checked out");
    }

    // Handle edge case: zero buffers
    if (num_buffers == 0) {
        std::fill(output, output + buffer_size, 0.0);
        return;
    }
    
    // Initialize the output array to zero
    std::fill(output, output + buffer_size, 0.0);
    
    // Release the lock before parallel processing
    lock.unlock();
    
    // Determine number of threads for parallel reduction
    const size_t num_threads = std::min(
        static_cast<size_t>(std::thread::hardware_concurrency()), 
        num_buffers
    );
    
    std::vector<std::thread> reduction_threads;
    reduction_threads.reserve(num_threads);
    
    // Thread-local partial sums
    std::vector<std::vector<double>> thread_local_sums(num_threads, std::vector<double>(buffer_size, 0.0));
    
    // Launch reduction threads
    for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        reduction_threads.emplace_back([&, thread_idx]() {
            // Calculate range of buffers for this thread
            const size_t start_buffer = thread_idx * (num_buffers / num_threads);
            const size_t end_buffer = (thread_idx == num_threads - 1) 
                                        ? num_buffers  // Last thread processes the remaining buffers
                                        : (thread_idx + 1) * (num_buffers / num_threads);
            
            auto& local_sums = thread_local_sums[thread_idx];

            // Log assigned range for the thread
            //std::cout << "Thread " << thread_idx << ": processing buffers " 
                      //<< start_buffer << " to " << end_buffer - 1 << "\n";
            
            // Process assigned buffers
            for (size_t buf_idx = start_buffer; buf_idx < end_buffer; ++buf_idx) {
                Buffer& buffer = buffers[buf_idx];
                
                // Prefetch next buffer's data
                if (buf_idx + 1 < num_buffers) {
                    __builtin_prefetch(buffers[buf_idx + 1].data, 0, 3);
                }
                
                // Process current buffer
                for (size_t i = 0; i < buffer_size; ++i) {
                    double current_value = buffer.data[i];
                    if (!std::isnan(current_value)) { // Ignore NaN
                        local_sums[i] += current_value;
                    }
                }
            }

            // Log thread-local sums for this thread
            //std::cout << "Thread " << thread_idx << ": local sums = ";
            //for (size_t i = 0; i < std::min(size_t(10), buffer_size); ++i) { // Log first 10 elements
            //    std::cout << local_sums[i] << " ";
            //}
            //std::cout << "\n";
        });
    }
    
    // Wait for all reduction threads to complete
    for (auto& thread : reduction_threads) {
        thread.join();
    }
    
    // Combine thread-local results into the output buffer
    for (size_t i = 0; i < buffer_size; ++i) {
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            output[i] += thread_local_sums[thread_idx][i];
        }
    }

    // Log final output values
    //std::cout << "Final output = ";
    //for (size_t i = 0; i < std::min(size_t(10), buffer_size); ++i) { // Log first 10 elements
    //    std::cout << output[i] << " ";
    //}
    //std::cout << "\n";
    
    reduction_cycles.fetch_add(rdtsc() - start_cycles, std::memory_order_relaxed);
}
// Cleanup buffers
void VectorAccumulator::cleanup() {
  is_shutting_down = true;
  cv.notify_all();

  // Wait for all buffers to be checked in
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [this]() { return active_checkouts == 0; });

  for (size_t i = 0; i < num_buffers; ++i) {
    if (buffers[i].data) {
      free(buffers[i].data);
      buffers[i].data = nullptr;
    }
  }
}

// Get statistics
VectorAccumulator::BufferStats VectorAccumulator::get_statistics() const {
  std::lock_guard<std::mutex> lock(mtx);
  BufferStats stats;

  stats.total_checkouts = total_checkouts;
  stats.total_waits = total_waits;
  stats.currently_checked_out = active_checkouts.load();

  if (total_waits > 0) {
    stats.average_wait_time_ms = static_cast<double>(accumulated_wait_time.load()) / total_waits;
  } else {
    stats.average_wait_time_ms = 0.0;
  }

  auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < num_buffers; ++i) {
    if (buffer_ownership[i]) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - checkout_times[i].checkout_time);
      stats.current_checkouts.emplace_back(i, duration);
    }
  }

  return stats;
}

// Force release a buffer
bool VectorAccumulator::force_release(size_t buffer_index) {
  if (buffer_index >= num_buffers) {
    throw std::out_of_range("Invalid buffer index");
  }

  std::lock_guard<std::mutex> lock(mtx);
  if (!buffer_ownership[buffer_index]) {
    return false;  // Buffer was already free
  }

  buffer_ownership[buffer_index] = false;
  buffers[buffer_index].in_use = false;
  cv.notify_one();

  log_debug("Buffer forcefully released: " + std::to_string(buffer_index));
  return true;
}

// Peek at buffer contents
std::vector<double> VectorAccumulator::peek_buffer(size_t buffer_index,
                                                   size_t start,
                                                   size_t length) const {
  if (buffer_index >= num_buffers) {
    throw std::out_of_range("Invalid buffer index");
  }
  if (start + length > buffer_size) {
    throw std::out_of_range("Invalid range");
  }

  std::lock_guard<std::mutex> lock(mtx);
  std::vector<double> result(length);
  std::copy(buffers[buffer_index].data + start, buffers[buffer_index].data + start + length,
            result.begin());
  return result;
}

// Aligned allocation
template <typename T>
T* VectorAccumulator::allocate_aligned(size_t size, size_t alignment) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
  return static_cast<T*>(ptr);
}

void VectorAccumulator::log_debug(const std::string& message) {
  if (debug_callback) {
    debug_callback(message.c_str());
  }
}

// Add these implementations to VectorAccumulator.cpp

std::vector<bool> VectorAccumulator::get_buffer_usage() const {
  std::lock_guard<std::mutex> lock(mtx);
  std::vector<bool> usage(num_buffers);
  for (size_t i = 0; i < num_buffers; ++i) {
    usage[i] = buffer_ownership[i];
  }
  return usage;
}

VectorAccumulator::PerformanceStats VectorAccumulator::get_performance_stats() const {
  PerformanceStats stats;
  stats.total_cycles = total_cycles.load();
  stats.reduction_cycles = reduction_cycles.load();

  // Calculate average cycles per reduction
  uint64_t total_reductions = total_checkouts.load();
  if (total_reductions > 0) {
    stats.avg_cycles_per_reduction = static_cast<double>(stats.reduction_cycles) / total_reductions;
  } else {
    stats.avg_cycles_per_reduction = 0.0;
  }

  return stats;
}

// RDTSC (Read Time-Stamp Counter)
uint64_t VectorAccumulator::rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

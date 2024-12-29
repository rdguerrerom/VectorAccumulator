#include <VectorAccumulator/VectorAccumulator.hpp>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <functional>
#include <chrono> // Added for timing functions

// Static member initialization
thread_local size_t VectorAccumulator::last_used_buffer = 0;

// Constructor
VectorAccumulator::VectorAccumulator(size_t n_buffers, size_t size, std::function<void(const char*)> debug_cb)
    : num_buffers(n_buffers),
      buffer_size(size),
      aligned_size((size + Vec4d::size() - 1) & ~(Vec4d::size() - 1)),
      buffers(std::make_unique<Buffer[]>(n_buffers)),
      buffer_ownership(n_buffers),
      debug_callback(debug_cb){
    if (n_buffers == 0 || size == 0) {
        throw std::invalid_argument("Buffer count and size must be positive");
    }

    for (size_t i = 0; i < num_buffers; ++i) {
        buffers[i].data = allocate_aligned<double>(aligned_size, 64);
        buffers[i].in_use = false;
        buffers[i].id = i;
        buffer_ownership[i] = false;

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
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (last_used_buffer < num_buffers && !buffers[last_used_buffer].in_use) {
            buffers[last_used_buffer].in_use = true;
            buffer_ownership[last_used_buffer] = true;
            return buffers[last_used_buffer].data;
        }
    }

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this]() {
        if (is_shutting_down) throw std::runtime_error("Shutting down");
        return std::any_of(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return !b.in_use; });
    });

    for (size_t i = 0; i < num_buffers; ++i) {
        if (!buffers[i].in_use) {
            buffers[i].in_use = true;
            buffer_ownership[i] = true;
            last_used_buffer = i;
            return buffers[i].data;
        }
    }

    throw std::runtime_error("No buffer available");
}

// Checkin a buffer
void VectorAccumulator::checkin(double* buffer) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = std::find_if(&buffers[0], &buffers[num_buffers], [buffer](const Buffer& b) { return b.data == buffer; });

    if (it == &buffers[num_buffers]) {
        throw std::invalid_argument("Invalid buffer returned");
    }

    if (!buffer_ownership[it->id]) {
        throw std::runtime_error("Buffer was not checked out");
    }

    it->in_use = false;
    buffer_ownership[it->id] = false;
    cv.notify_one();

    if (debug_callback) {
        debug_callback(("Buffer " + std::to_string(it->id) + " checked in").c_str());
    }
}

// Try checkout with timeout
std::optional<double*> VectorAccumulator::try_checkout(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mtx);
    auto start_wait = std::chrono::steady_clock::now();

    bool got_buffer = cv.wait_for(lock, timeout, [this]() {
        if (is_shutting_down) return true;
        return std::any_of(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return !b.in_use; });
    });

    if (!got_buffer || is_shutting_down) {
        total_waits++;
        return std::nullopt;
    }

    auto it = std::find_if(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return !b.in_use; });

    it->in_use = true;
    buffer_ownership[it->id] = true;
    total_checkouts++;

    if (debug_callback) {
        debug_callback(("Buffer " + std::to_string(it->id) + " checked out").c_str());
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
void VectorAccumulator::reduce(double* output) {
    uint64_t start_cycles = rdtsc();

    if (std::any_of(&buffers[0], &buffers[num_buffers], [](const Buffer& b) { return b.in_use; })) {
        throw std::runtime_error("Cannot reduce while buffers are checked out");
    }

    std::memcpy(output, buffers[0].data, aligned_size * sizeof(double));

    for (size_t buf = 1; buf < num_buffers; ++buf) {
        for (size_t i = 0; i < aligned_size; i += Vec4d::size()) {
            if (i + PREFETCH_DISTANCE * Vec4d::size() < aligned_size) {
                _mm_prefetch(reinterpret_cast<const char*>(&buffers[buf].data[i + PREFETCH_DISTANCE * Vec4d::size()]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&output[i + PREFETCH_DISTANCE * Vec4d::size()]), _MM_HINT_T0);
            }

            Vec4d sum;
            sum.load(output + i);
            Vec4d vec;
            vec.load(buffers[buf].data + i);
            sum += vec;
            sum.store(output + i);
        }
    }

    reduction_cycles.fetch_add(rdtsc() - start_cycles);
}

// Cleanup buffers
void VectorAccumulator::cleanup() {
    for (size_t i = 0; i < num_buffers; ++i) {
        if (buffers[i].data) {
            free(buffers[i].data);
            buffers[i].data = nullptr;
        }
    }
    is_shutting_down = true;
    cv.notify_all();
}

// Get statistics
VectorAccumulator::BufferStats VectorAccumulator::get_statistics() const {
    std::lock_guard<std::mutex> lock(mtx);
    BufferStats stats;
    
    stats.total_checkouts = total_checkouts;
    stats.total_waits = total_waits;
    stats.currently_checked_out = 0;
    
    // Calculate average wait time
    if (total_waits > 0) {
        stats.average_wait_time_ms = 
            static_cast<double>(accumulated_wait_time.load()) / total_waits;
    } else {
        stats.average_wait_time_ms = 0.0;
    }
    
    // Count currently checked out buffers and gather their details
    for (size_t i = 0; i < num_buffers; ++i) {
        if (buffer_ownership[i]) {
            stats.currently_checked_out++;
            // Note: In a real implementation, you might want to track checkout times
            stats.current_checkouts.emplace_back(i, std::chrono::milliseconds(0));
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
    std::copy(buffers[buffer_index].data + start,
              buffers[buffer_index].data + start + length,
              result.begin());
    return result;
}

// Aligned allocation
template<typename T>
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

// RDTSC (Read Time-Stamp Counter)
uint64_t VectorAccumulator::rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}


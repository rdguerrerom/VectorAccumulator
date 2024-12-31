#include <VectorAccumulator/VectorAccumulator.hpp>
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <optional>
#include <mutex>
#include <condition_variable>
#include <vector>

class VectorAccumulatorTest : public ::testing::Test {
protected:
    static void AssertInThread(const std::function<void()>& test_func) {
        std::exception_ptr exception_ptr = nullptr;
        std::thread worker([&]() {
            try {
                test_func();
            } catch (...) {
                exception_ptr = std::current_exception();
            }
        });
        worker.join();
        if (exception_ptr) {
            std::rethrow_exception(exception_ptr);
        }
    }
};

// Test constructor and initial state
TEST_F(VectorAccumulatorTest, ConstructorInitialization) {
    VectorAccumulator accumulator(4, 1024);
    EXPECT_NO_THROW(accumulator.get_statistics());
    EXPECT_EQ(accumulator.get_statistics().total_checkouts, 0);
}

// Test buffer checkout and checkin from the same thread
TEST_F(VectorAccumulatorTest, BufferCheckoutAndCheckin) {
    VectorAccumulator accumulator(4, 1024);
    AssertInThread([&]() {
        double* buffer = accumulator.checkout();
        ASSERT_NE(buffer, nullptr);

        for (size_t i = 0; i < 1024; ++i) {
            buffer[i] = static_cast<double>(i);
        }
        ASSERT_NO_THROW(accumulator.checkin(buffer));
    });
}

// Test exception on invalid checkin
TEST_F(VectorAccumulatorTest, InvalidBufferCheckin) {
    VectorAccumulator accumulator(4, 1024);
    double invalid_buffer[1024];
    EXPECT_THROW(accumulator.checkin(invalid_buffer), std::invalid_argument);
}

// Test simultaneous checkouts
TEST_F(VectorAccumulatorTest, SimultaneousCheckouts) {
    VectorAccumulator accumulator(4, 1024);
    std::atomic<int> successful_checkouts{0};
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::vector<double*> buffers(4, nullptr);

    // First, checkout all buffers
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            buffers[i] = accumulator.checkout();
            {
                std::lock_guard<std::mutex> lock(mtx);
                successful_checkouts++;
                if (successful_checkouts == 4) {
                    ready = true;
                    cv.notify_one();
                }
            }
            // Wait a bit before checking in
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            accumulator.checkin(buffers[i]);
        });
    }

    // Wait for all checkouts to complete
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return ready; });
    }

    // Try to checkout when all buffers are in use
    ASSERT_THROW(accumulator.checkout(), std::runtime_error);

    // Join all threads
    for (auto& t : threads) {
        t.join();
    }
}

// Test reset functionality
TEST_F(VectorAccumulatorTest, ResetBuffers) {
    VectorAccumulator accumulator(4, 1024);
    
    AssertInThread([&]() {
        double* buffer = accumulator.checkout();
        ASSERT_NE(buffer, nullptr);
        buffer[0] = 42.0;
        ASSERT_NO_THROW(accumulator.checkin(buffer));
    });

    EXPECT_NO_THROW(accumulator.reset());
    EXPECT_EQ(accumulator.get_statistics().currently_checked_out, 0);
}

// Test reduce functionality
// Test reduce functionality
/*TEST_F(VectorAccumulatorTest, ReduceBuffers) {
    const size_t buffer_size = 1024;
    VectorAccumulator accumulator(4, buffer_size);
    std::vector<std::thread> threads;
    std::atomic<int> completed_threads{0};
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    
    // Fill buffers in separate threads
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            double* buffer = accumulator.checkout();
            ASSERT_NE(buffer, nullptr);
            
            // Fill buffer with consistent values
            for (size_t j = 0; j < buffer_size; ++j) {
                buffer[j] = static_cast<double>(i);  // Each thread writes its index
            }
            
            // Ensure memory visibility
            std::atomic_thread_fence(std::memory_order_release);
            accumulator.checkin(buffer);
            
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                completed_threads++;
                completion_cv.notify_one();
            }
        });
    }

    // Wait for all threads to complete
    {
        std::unique_lock<std::mutex> lock(completion_mutex);
        ASSERT_TRUE(completion_cv.wait_for(lock, std::chrono::seconds(5),
            [&]() { return completed_threads == 4; }));
    }

    // Now reduce
    alignas(64) std::vector<double> output(buffer_size);
    ASSERT_NO_THROW(accumulator.reduce(output.data()));

    // Verify results
    const double expected_sum = 0.0 + 1.0 + 2.0 + 3.0;  // Sum of thread indices
    for (size_t j = 0; j < buffer_size; ++j) {
        EXPECT_DOUBLE_EQ(output[j], expected_sum)
            << "Mismatch at position " << j << ": expected " << expected_sum
            << ", got " << output[j];
    }

    for (auto& t : threads) {
        t.join();
    }
}*/
// Additional Test: ReduceBuffersAdvanced
TEST_F(VectorAccumulatorTest, ReduceBuffersAdvanced) {
    const size_t buffer_size = 1024;
    const size_t num_buffers = 4;
    VectorAccumulator accumulator(num_buffers, buffer_size);

    // Simulate buffers being filled by different threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_buffers; ++i) {
        threads.emplace_back([&, i]() {
            double* buffer = accumulator.checkout();
            ASSERT_NE(buffer, nullptr);

            // Fill buffer with values based on thread index and buffer index
            for (size_t j = 0; j < buffer_size; ++j) {
                buffer[j] = static_cast<double>(i + j * 0.1);
            }

            // Release the buffer
            accumulator.checkin(buffer);
        });
    }

    // Wait for threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // Verify all buffers are checked in before reduction
    auto buffer_usage = accumulator.get_buffer_usage();
    for (bool is_in_use : buffer_usage) {
        ASSERT_FALSE(is_in_use) << "Buffer is still in use before reduction.";
    }

    // Reduce the buffers into a single output buffer
    alignas(64) std::vector<double> output(buffer_size, 0.0);
    ASSERT_NO_THROW(accumulator.reduce(output.data()));

    // Verify the reduced results
    for (size_t j = 0; j < buffer_size; ++j) {
        double expected_sum = 0.0;
        for (size_t i = 0; i < num_buffers; ++i) {
            expected_sum += static_cast<double>(i + j * 0.1);
        }
        EXPECT_NEAR(output[j], expected_sum, 1e-9)
            << "Mismatch at position " << j << ": expected " << expected_sum 
            << ", got " << output[j];
    }
}
// Test try_checkout with timeout
TEST_F(VectorAccumulatorTest, TryCheckoutTimeout) {
    VectorAccumulator accumulator(1, 1024);
    
    AssertInThread([&]() {
        double* buffer = accumulator.checkout();
        ASSERT_NE(buffer, nullptr);

        auto result = accumulator.try_checkout(std::chrono::milliseconds(100));
        EXPECT_EQ(result, std::nullopt);

        ASSERT_NO_THROW(accumulator.checkin(buffer));
        
        result = accumulator.try_checkout(std::chrono::milliseconds(100));
        EXPECT_NE(result, std::nullopt);
        if (result) {
            ASSERT_NO_THROW(accumulator.checkin(*result));
        }
    });
}

// Test debug callback
TEST_F(VectorAccumulatorTest, DebugCallback) {
    VectorAccumulator accumulator(4, 1024);
    std::atomic<bool> callback_called{false};
    
    accumulator.set_debug_callback([&callback_called](const char* msg) {
        callback_called = true;
    });

    AssertInThread([&]() {
        double* buffer = accumulator.checkout();
        ASSERT_NE(buffer, nullptr);
        ASSERT_NO_THROW(accumulator.checkin(buffer));
    });

    EXPECT_TRUE(callback_called);
}

// Test peek_buffer functionality
TEST_F(VectorAccumulatorTest, PeekBuffer) {
    VectorAccumulator accumulator(4, 1024);

    AssertInThread([&]() {
        double* buffer = accumulator.checkout();
        ASSERT_NE(buffer, nullptr);
        
        for (size_t i = 0; i < 1024; ++i) {
            buffer[i] = static_cast<double>(i);
        }
        ASSERT_NO_THROW(accumulator.checkin(buffer));
    });

    auto peeked = accumulator.peek_buffer(0, 0, 10);
    ASSERT_EQ(peeked.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(peeked[i], static_cast<double>(i));
    }
}

TEST_F(VectorAccumulatorTest, ReduceShouldCorrectlyReduceAllBuffersIntoOutput) {
const size_t n_buffers = 4;
const size_t buffer_size = 1000;
VectorAccumulator accumulator(n_buffers, buffer_size);

// Fill buffers with known values
std::vector<double*> checked_out_buffers;
for (size_t i = 0; i < n_buffers; ++i) {
    double* buffer = accumulator.checkout();
    checked_out_buffers.push_back(buffer);
    for (size_t j = 0; j < buffer_size; ++j) {
        buffer[j] = static_cast<double>(i + 1);  // Each buffer filled with its index + 1
    }
}

// Check in all buffers
for (auto buffer : checked_out_buffers) {
    accumulator.checkin(buffer);
}

std::vector<double> output(buffer_size, 0.0);
accumulator.reduce(output.data());

// Check if the reduction is correct
for (size_t i = 0; i < buffer_size; ++i) {
    EXPECT_DOUBLE_EQ(output[i], 10.0);  // Sum of 1 + 2 + 3 + 4
}

// Test that reduce throws an exception if any buffer is checked out
double* buffer = accumulator.checkout();
EXPECT_THROW(accumulator.reduce(output.data()), std::runtime_error);
accumulator.checkin(buffer);
}

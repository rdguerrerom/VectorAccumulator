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
TEST_F(VectorAccumulatorTest, ReduceBuffers) {
    VectorAccumulator accumulator(4, 1024);
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> completed_threads{0};

    // Fill buffers in separate threads
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&accumulator, i, &completed_threads]() {
            double* buffer = accumulator.checkout();
            ASSERT_NE(buffer, nullptr);
            
            for (size_t j = 0; j < 1024; ++j) {
                buffer[j] = static_cast<double>(j + i);
            }
            ASSERT_NO_THROW(accumulator.checkin(buffer));
            completed_threads++;
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // Ensure all threads completed successfully
    ASSERT_EQ(completed_threads, 4);

    double output[1024];
    EXPECT_NO_THROW(accumulator.reduce(output));

    for (size_t j = 0; j < 1024; ++j) {
        EXPECT_EQ(output[j], 6.0 * j + 6);  // Sum of 4 buffers
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

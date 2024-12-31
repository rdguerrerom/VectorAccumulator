#include <VectorAccumulator/VectorAccumulator.hpp>
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <optional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cstring>
#include <random>
#include <future>

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

    static std::vector<double> generateRandomData(size_t size) {
        std::vector<double> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1000.0, 1000.0);
        for (auto& val : data) val = dis(gen);
        return data;
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

TEST_F(VectorAccumulatorTest, HighContention) {
    const size_t n_buffers = 4;
    const size_t buffer_size = 1024;
    VectorAccumulator accumulator(n_buffers, buffer_size);
    
    std::atomic<bool> stop{false};
    std::atomic<size_t> successful_ops{0};
    std::atomic<size_t> failed_ops{0};
    
    auto worker = [&](int id) {
        while (!stop) {
            try {
                if (auto buffer = accumulator.try_checkout(std::chrono::milliseconds(50))) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    accumulator.checkin(*buffer);
                    successful_ops++;
                } else {
                    failed_ops++;
                }
            } catch (...) {
                failed_ops++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back(worker, i);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    stop = true;

    for (auto& t : threads) t.join();

    EXPECT_GT(successful_ops, 0);
    auto stats = accumulator.get_statistics();
    EXPECT_EQ(stats.total_checkouts, successful_ops);
}

TEST_F(VectorAccumulatorTest, LargeDataReduction) {
    const size_t n_buffers = 4;
    const size_t buffer_size = 1024 * 1024;  // 1M elements
    VectorAccumulator accumulator(n_buffers, buffer_size);

    std::vector<std::future<void>> futures;
    std::vector<std::vector<double>> expected_values(n_buffers);
    
    for (size_t i = 0; i < n_buffers; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            double* buffer = accumulator.checkout();
            expected_values[i] = generateRandomData(buffer_size);
            std::memcpy(buffer, expected_values[i].data(), buffer_size * sizeof(double));
            accumulator.checkin(buffer);
        }));
    }

    for (auto& f : futures) f.wait();

    std::vector<double> output(buffer_size);
    accumulator.reduce(output.data());

    for (size_t i = 0; i < buffer_size; ++i) {
        double expected = 0.0;
        for (const auto& vec : expected_values) expected += vec[i];
        EXPECT_NEAR(output[i], expected, std::abs(expected) * 1e-10);
    }
}

TEST_F(VectorAccumulatorTest, EdgeCaseBufferHandling) {
    VectorAccumulator accumulator(1, 1);
    
    {
        double* buffer = accumulator.checkout();
        buffer[0] = std::numeric_limits<double>::max();
        accumulator.checkin(buffer);
    }
    
    {
        double* buffer = accumulator.checkout();
        buffer[0] = std::numeric_limits<double>::min();
        accumulator.checkin(buffer);
    }

    std::vector<double> output(1);
    EXPECT_NO_THROW(accumulator.reduce(output.data()));
}
TEST_F(VectorAccumulatorTest, RaceConditionResilience) {
    VectorAccumulator accumulator(4, 1024);
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&]() {
            while (!start) std::this_thread::yield();
            while (!stop) {
                try {
                    if (auto buffer = accumulator.try_checkout(std::chrono::milliseconds(0))) {
                        accumulator.checkin(*buffer);
                    }
                } catch (...) {}
            }
        });
    }

    start = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    for (auto& t : threads) t.join();
    
    // Give time for any pending operations to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto stats = accumulator.get_statistics();
    EXPECT_TRUE(stats.current_checkouts.empty());
}

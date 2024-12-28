#include <VectorAccumulator/VectorAccumulator.hpp>  // Ensure this matches the updated header file's name
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <optional>

// Test constructor and initial state
TEST(VectorAccumulatorTest, ConstructorInitialization) {
    VectorAccumulator accumulator(4, 1024);
    EXPECT_NO_THROW(accumulator.get_statistics());
    EXPECT_EQ(accumulator.get_statistics().total_checkouts, 0);
}

// Test buffer checkout and checkin
TEST(VectorAccumulatorTest, BufferCheckoutAndCheckin) {
    VectorAccumulator accumulator(4, 1024);

    double* buffer = accumulator.checkout();
    ASSERT_NE(buffer, nullptr);

    // Fill the buffer and checkin
    for (size_t i = 0; i < 1024; ++i) {
        buffer[i] = static_cast<double>(i);
    }
    EXPECT_NO_THROW(accumulator.checkin(buffer));
}

// Test exception on invalid checkin
TEST(VectorAccumulatorTest, InvalidBufferCheckin) {
    VectorAccumulator accumulator(4, 1024);
    double invalid_buffer[1024];
    EXPECT_THROW(accumulator.checkin(invalid_buffer), std::invalid_argument);
}

// Test simultaneous checkouts
TEST(VectorAccumulatorTest, SimultaneousCheckouts) {
    VectorAccumulator accumulator(4, 1024);

    double* buffers[4];
    for (int i = 0; i < 4; ++i) {
        buffers[i] = accumulator.checkout();
        ASSERT_NE(buffers[i], nullptr);
    }

    // Ensure all buffers are checked out
    EXPECT_THROW(accumulator.checkout(), std::runtime_error);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NO_THROW(accumulator.checkin(buffers[i]));
    }
}

// Test reset functionality
TEST(VectorAccumulatorTest, ResetBuffers) {
    VectorAccumulator accumulator(4, 1024);

    double* buffer = accumulator.checkout();
    ASSERT_NE(buffer, nullptr);
    buffer[0] = 42.0;
    accumulator.checkin(buffer);

    // Reset and ensure all buffers are cleared
    EXPECT_NO_THROW(accumulator.reset());
    EXPECT_EQ(accumulator.get_statistics().currently_checked_out, 0);
}

// Test reduce functionality
TEST(VectorAccumulatorTest, ReduceBuffers) {
    VectorAccumulator accumulator(4, 1024);

    for (int i = 0; i < 4; ++i) {
        double* buffer = accumulator.checkout();
        for (size_t j = 0; j < 1024; ++j) {
            buffer[j] = static_cast<double>(j + i);
        }
        accumulator.checkin(buffer);
    }

    double output[1024];
    EXPECT_NO_THROW(accumulator.reduce(output));

    for (size_t j = 0; j < 1024; ++j) {
        EXPECT_EQ(output[j], 6.0 * j + 6);  // Sum of 4 buffers
    }
}

// Test try_checkout with timeout
TEST(VectorAccumulatorTest, TryCheckoutTimeout) {
    VectorAccumulator accumulator(1, 1024);

    double* buffer = accumulator.checkout();
    ASSERT_NE(buffer, nullptr);

    auto result = accumulator.try_checkout(std::chrono::milliseconds(100));
    EXPECT_EQ(result, std::nullopt);

    accumulator.checkin(buffer);
    result = accumulator.try_checkout(std::chrono::milliseconds(100));
    EXPECT_NE(result, std::nullopt);
}

// Test force_release functionality
TEST(VectorAccumulatorTest, ForceRelease) {
    VectorAccumulator accumulator(1, 1024);

    double* buffer = accumulator.checkout();
    ASSERT_NE(buffer, nullptr);

    EXPECT_TRUE(accumulator.force_release(0));
    EXPECT_NO_THROW(accumulator.checkout());
}

// Test debug callback
TEST(VectorAccumulatorTest, DebugCallback) {
    VectorAccumulator accumulator(4, 1024);

    bool callback_called = false;
    accumulator.set_debug_callback([&callback_called](const char* msg) {
        callback_called = true;
    });

    double* buffer = accumulator.checkout();
    accumulator.checkin(buffer);

    EXPECT_TRUE(callback_called);
}

// Test peek_buffer functionality
TEST(VectorAccumulatorTest, PeekBuffer) {
    VectorAccumulator accumulator(4, 1024);

    double* buffer = accumulator.checkout();
    for (size_t i = 0; i < 1024; ++i) {
        buffer[i] = static_cast<double>(i);
    }
    accumulator.checkin(buffer);

    auto peeked = accumulator.peek_buffer(0, 0, 10);
    ASSERT_EQ(peeked.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(peeked[i], static_cast<double>(i));
    }
}


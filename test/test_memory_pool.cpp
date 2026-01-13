#include "memory_pool.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using mm::MemoryPool;

void test_basic_alloc_free() {
    std::cout << "[TEST] basic alloc / free\n";

    MemoryPool pool;

    size_t bytes = 128;
    void* p = pool.alloc(bytes);

    assert(p != nullptr);
    assert(pool.allocated() == bytes);

    pool.free(p, bytes);
    assert(pool.allocated() == 0);
}

void test_calloc_zeroed() {
    std::cout << "[TEST] calloc zeroed\n";

    MemoryPool pool;

    const size_t count = 64;
    const size_t size = sizeof(int);
    int* p = static_cast<int*>(pool.calloc(count, size));

    assert(p != nullptr);
    assert(pool.allocated() == count * size);

    for (size_t i = 0; i < count; ++i) {
        assert(p[i] == 0);
    }

    pool.free(p, count * size);
    assert(pool.allocated() == 0);
}

void test_multiple_allocs() {
    std::cout << "[TEST] multiple allocations\n";

    MemoryPool pool;

    std::vector<void*> ptrs;
    std::vector<size_t> sizes;

    for (size_t i = 1; i <= 1000; ++i) {
        size_t sz = i * 16;
        void* p = pool.alloc(sz);
        assert(p != nullptr);

        ptrs.push_back(p);
        sizes.push_back(sz);
    }

    size_t expected = 0;
    for (size_t sz : sizes) expected += sz;
    assert(pool.allocated() == expected);

    for (size_t i = 0; i < ptrs.size(); ++i) {
        pool.free(ptrs[i], sizes[i]);
    }

    assert(pool.allocated() == 0);
}

void test_stress_alloc_free() {
    std::cout << "[TEST] stress alloc/free\n";

    MemoryPool pool;

    constexpr size_t rounds = 10000;
    constexpr size_t block_size = 64;

    for (size_t i = 0; i < rounds; ++i) {
        void* p = pool.alloc(block_size);
        assert(p != nullptr);
        pool.free(p, block_size);
    }

    assert(pool.allocated() == 0);
}

void test_raii_destroy() {
    std::cout << "[TEST] RAII destroy\n";

    size_t before = 0;
    {
        MemoryPool pool;
        void* p1 = pool.alloc(1024);
        void* p2 = pool.alloc(2048);

        before = pool.allocated();
        assert(before == 3072);

        // 故意不 free
    }
    // 出作用域，如果没有崩溃，说明 mi_heap_destroy 正常工作
}

int main() {
    std::cout << "==== MemoryPool Tests ====\n";

    test_basic_alloc_free();
    test_calloc_zeroed();
    test_multiple_allocs();
    test_stress_alloc_free();
    test_raii_destroy();

    std::cout << "All tests passed ✅\n";
    return 0;
}

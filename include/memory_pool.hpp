#include <mimalloc.h>

namespace mm {
class MemoryPool {
public:
    MemoryPool()
        : heap_(mi_heap_new()), allocated_(0) {}

    ~MemoryPool() {
        mi_heap_destroy(heap_);
    }

    // 禁止拷贝（非常重要）
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void* alloc(size_t bytes) {
        void* p = mi_heap_malloc(heap_, bytes);
        if (p) allocated_.fetch_add(bytes, std::memory_order_relaxed);
        return p;
    }

    void* calloc(size_t count, size_t size) {
        size_t bytes = count * size;
        void* p = mi_heap_calloc(heap_, count, size);
        if (p) allocated_.fetch_add(bytes, std::memory_order_relaxed);
        return p;
    }

    void free(void* ptr, size_t bytes) {
        if (!ptr) return;
        mi_heap_free(heap_, ptr);
        allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    }

    size_t allocated() const {
        return allocated_.load(std::memory_order_relaxed);
    }

private:
    mi_heap_t* heap_;
    std::atomic<size_t> allocated_;	



}

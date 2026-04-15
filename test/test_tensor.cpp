#include "../include/tensor.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>

namespace test {

static int g_total = 0;
static int g_passed = 0;

// =========================
// 打印工具
// =========================
void print_line(char ch = '=', int n = 60) {
    for (int i = 0; i < n; ++i) std::cout << ch;
    std::cout << "\n";
}

void print_title(const std::string& title) {
    print_line('=');
    std::cout << "[TEST] " << title << "\n";
    print_line('=');
}

void print_step(const std::string& msg) {
    std::cout << "  -> " << msg << "\n";
}

void print_pass(const std::string& msg) {
    std::cout << "  [PASS] " << msg << "\n";
}

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "  [FAIL] " << msg << "\n";
    std::abort();
}

template <typename T>
void expect_eq(const T& a, const T& b, const std::string& msg) {
    if (!(a == b)) {
        std::cerr << "  [FAIL] " << msg
                  << " | lhs = " << a
                  << ", rhs = " << b << "\n";
        std::abort();
    }
    print_pass(msg);
}

void expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        fail(msg);
    }
    print_pass(msg);
}

template <typename T>
void expect_vec_eq(const std::vector<T>& a,
                   const std::vector<T>& b,
                   const std::string& msg) {
    if (a.size() != b.size()) {
        std::cerr << "  [FAIL] " << msg
                  << " | size mismatch: "
                  << a.size() << " vs " << b.size() << "\n";
        std::abort();
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "  [FAIL] " << msg
                      << " | mismatch at index " << i
                      << ": " << a[i] << " vs " << b[i] << "\n";
            std::abort();
        }
    }
    print_pass(msg);
}

template <typename ExceptionType, typename Func>
void expect_throw(Func fn, const std::string& msg) {
    bool thrown = false;
    try {
        fn();
    } catch (const ExceptionType& e) {
        thrown = true;
        std::cout << "  [INFO] caught exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "  [FAIL] " << msg << " | wrong exception type\n";
        std::abort();
    }
    if (!thrown) {
        std::cerr << "  [FAIL] " << msg << " | expected exception not thrown\n";
        std::abort();
    }
    print_pass(msg);
}

template <typename T>
void print_flat_data(const Tensor<T>& x, const std::string& name) {
    std::cout << "  " << name << ".data ptr = "
              << static_cast<const void*>(x.data()) << "\n";

    if (!x.isContiguous()) {
        std::cout << "  " << name << " is non-contiguous, skip flat operator[] print\n";
        return;
    }

    std::cout << "  " << name << ".flat = [ ";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "]\n";
}

template <typename T>
void print_tensor_state(const Tensor<T>& x, const std::string& name) {
    std::cout << "  Tensor <" << name << ">\n";
    std::cout << "    shape   = [ ";
    for (auto v : x.shape()) std::cout << v << " ";
    std::cout << "]\n";

    std::cout << "    strides = [ ";
    for (auto v : x.strides()) std::cout << v << " ";
    std::cout << "]\n";

    std::cout << "    numel   = " << x.size() << "\n";
    std::cout << "    ndim    = " << x.ndim() << "\n";
    std::cout << "    contig  = " << (x.isContiguous() ? "true" : "false") << "\n";
    std::cout << "    data    = " << static_cast<const void*>(x.data()) << "\n";
}

// =========================
// 测试执行器
// =========================
template <typename Func>
void run_case(const std::string& name, Func fn) {
    ++g_total;
    print_title(name);
    fn();
    ++g_passed;
    std::cout << "[DONE] " << name << "\n\n";
}

// =========================
// 测试用例
// =========================
void test_constructor_basic() {
    print_step("construct Tensor<int> a({2,3,4})");
    Tensor<int> a(std::vector<int64_t>{2, 3, 4});

    print_tensor_state(a, "a");
    print_flat_data(a, "a");

    expect_eq(a.size(), static_cast<size_t>(24), "size should be 24");
    expect_eq(a.ndim(), static_cast<size_t>(3), "ndim should be 3");
    expect_vec_eq(a.shape(), std::vector<int64_t>{2, 3, 4}, "shape correct");
    expect_vec_eq(a.strides(), std::vector<int64_t>{12, 4, 1}, "strides correct");
    expect_true(a.isContiguous(), "constructor result should be contiguous");
}

void test_fill_and_linear_rw() {
    print_step("construct Tensor<int> a({2,3}) and fill with 7");
    Tensor<int> a(std::vector<int64_t>{2, 3});
    a.fill(7);

    print_tensor_state(a, "a");
    print_flat_data(a, "a");

    for (size_t i = 0; i < a.size(); ++i) {
        expect_eq(a[i], 7, "filled value should be 7");
    }

    print_step("overwrite with linear values");
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int>(i);
    }

    print_flat_data(a, "a");

    for (size_t i = 0; i < a.size(); ++i) {
        expect_eq(a[i], static_cast<int>(i), "linear write/read correct");
    }
}

void test_view_shared_storage() {
    print_step("construct contiguous tensor a({2,3})");
    Tensor<int> a(std::vector<int64_t>{2, 3});
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int>(i);
    }

    print_tensor_state(a, "a");
    print_flat_data(a, "a");

    print_step("view a as b({3,2})");
    Tensor<int> b = a.view(std::vector<int64_t>{3, 2});

    print_tensor_state(b, "b");
    print_flat_data(b, "b");

    expect_vec_eq(b.shape(), std::vector<int64_t>{3, 2}, "view shape correct");
    expect_vec_eq(b.strides(), std::vector<int64_t>{2, 1}, "view strides correct");
    expect_true(a.data() == b.data(), "view should share the same data pointer");

    print_step("modify b[0] = 100 and check a[0]");
    b[0] = 100;
    print_flat_data(a, "a");
    print_flat_data(b, "b");
    expect_eq(a[0], 100, "view writes should reflect back to original tensor");
}

void test_view_invalid_shape() {
    print_step("construct a({2,3}) and try invalid view({4,2})");
    Tensor<int> a(std::vector<int64_t>{2, 3});
    expect_throw<std::invalid_argument>(
        [&]() { a.view(std::vector<int64_t>{4, 2}); },
        "view with wrong numel should throw invalid_argument"
    );
}

void test_permute() {
    print_step("construct a({2,3,4})");
    Tensor<int> a(std::vector<int64_t>{2, 3, 4});
    a.fill(1);

    print_tensor_state(a, "a");

    print_step("permute a with order {1,0,2}");
    Tensor<int> b = a.permute(std::vector<size_t>{1, 0, 2});

    print_tensor_state(b, "b");

    expect_vec_eq(b.shape(), std::vector<int64_t>{3, 2, 4}, "permute shape correct");
    expect_vec_eq(b.strides(), std::vector<int64_t>{4, 12, 1}, "permute strides correct");
    expect_true(a.data() == b.data(), "permute should share data");
    expect_true(!b.isContiguous(), "permute result should be non-contiguous");
}

void test_permute_invalid() {
    print_step("construct a({2,3,4})");
    Tensor<int> a(std::vector<int64_t>{2, 3, 4});

    expect_throw<std::invalid_argument>(
        [&]() { a.permute(std::vector<size_t>{0, 1}); },
        "permute with wrong order size should throw"
    );

    expect_throw<std::invalid_argument>(
        [&]() { a.permute(std::vector<size_t>{0, 0, 2}); },
        "permute with duplicate dims should throw"
    );

    expect_throw<std::invalid_argument>(
        [&]() { a.permute(std::vector<size_t>{0, 1, 5}); },
        "permute with out-of-range dim should throw"
    );
}

void test_transpose() {
    print_step("construct a({2,3,4})");
    Tensor<int> a(std::vector<int64_t>{2, 3, 4});

    print_tensor_state(a, "a");

    print_step("transpose dim0 and dim1");
    Tensor<int> b = a.transpose(0, 1);

    print_tensor_state(b, "b");

    expect_vec_eq(b.shape(), std::vector<int64_t>{3, 2, 4}, "transpose shape correct");
    expect_vec_eq(b.strides(), std::vector<int64_t>{4, 12, 1}, "transpose strides correct");
    expect_true(a.data() == b.data(), "transpose should share data");
    expect_true(!b.isContiguous(), "transpose result should be non-contiguous");
}

void test_transpose_invalid() {
    print_step("construct a({2,3,4})");
    Tensor<int> a(std::vector<int64_t>{2, 3, 4});

    expect_throw<std::out_of_range>(
        [&]() { a.transpose(0, 5); },
        "transpose with invalid dim should throw out_of_range"
    );
}

void test_reshape_contiguous() {
    print_step("construct contiguous tensor a({2,3})");
    Tensor<int> a(std::vector<int64_t>{2, 3});
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int>(i);
    }

    print_tensor_state(a, "a");
    print_flat_data(a, "a");

    print_step("reshape to b({3,2})");
    Tensor<int> b = a.reshape(std::vector<int64_t>{3, 2});

    print_tensor_state(b, "b");
    print_flat_data(b, "b");

    expect_true(a.data() == b.data(), "reshape on contiguous tensor should share data");
    expect_vec_eq(b.shape(), std::vector<int64_t>{3, 2}, "reshape contiguous shape correct");
    expect_vec_eq(b.strides(), std::vector<int64_t>{2, 1}, "reshape contiguous strides correct");
}

void test_shape_with_one_dim() {
    print_step("construct a({2,1,3})");
    Tensor<int> a(std::vector<int64_t>{2, 1, 3});

    print_tensor_state(a, "a");

    expect_vec_eq(a.strides(), std::vector<int64_t>{3, 3, 1}, "stride with size-1 dim correct");
    expect_true(a.isContiguous(), "tensor with dim=1 should still be contiguous");
}

void test_randomize_runs() {
    print_step("construct float tensor and randomize");
    Tensor<float> a(std::vector<int64_t>{4, 4});
    a.randomize();

    print_tensor_state(a, "a");
    print_flat_data(a, "a");

    volatile float x = a[0];
    (void)x;
    print_pass("randomize runs successfully");
}

} // namespace test

int main() {
    using namespace test;

    run_case("constructor basic", test_constructor_basic);
    run_case("fill and linear read/write", test_fill_and_linear_rw);
    run_case("view shared storage", test_view_shared_storage);
    run_case("view invalid shape", test_view_invalid_shape);
    run_case("permute", test_permute);
    run_case("permute invalid cases", test_permute_invalid);
    run_case("transpose", test_transpose);
    run_case("transpose invalid", test_transpose_invalid);
    run_case("reshape contiguous", test_reshape_contiguous);
    run_case("shape with size-1 dim", test_shape_with_one_dim);
    run_case("randomize", test_randomize_runs);

    test::print_line('=');
    std::cout << "[SUMMARY] passed " << test::g_passed
              << " / " << test::g_total << " tests\n";
    test::print_line('=');

    return 0;
}
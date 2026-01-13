# 内存池测试
add_executable(test_memory_pool test_memory_pool.cpp)
target_link_libraries(test_memory_pool mimalloc)
target_include_directories(test_memory_pool 
						   PRIVATE 
						   ${CMAKE_SOURCE_DIR}/include 
						   ${CMAKE_SOURCE_DIR}/third_party/mimalloc/include)
# 设置输出目录
set_target_properties(test_memory_pool PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/bin/tests
)

# Tensor测试
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor mimalloc)
target_include_directories(test_tensor
						   PRIVATE 
						   ${CMAKE_SOURCE_DIR}/include 
						   ${CMAKE_SOURCE_DIR}/third_party/mimalloc/include)
# 设置输出目录
set_target_properties(test_tensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/bin/tests
)

# 添加测试
add_test(NAME memory_pool_test COMMAND test_memory_pool)
add_test(NAME tensor_test COMMAND test_tensor)
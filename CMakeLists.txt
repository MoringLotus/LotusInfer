cmake_minimum_required(VERSION 3.15)
project(MyTensorProject VERSION 1.0.0 LANGUAGES CXX)
enable_testing()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 全局包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/mimalloc/include  # 添加这里！
)

# 包含第三方库
add_subdirectory(third_party/mimalloc)

# 查找源文件
file(GLOB_RECURSE SOURCES "csrc/*.cpp" "csrc/*.cc" "csrc/*.cxx")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

# 可执行文件配置
if(SOURCES)
    add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
    
    # 设置目标属性
    target_include_directories(${PROJECT_NAME} PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/mimalloc/include
    )
    
    # 链接 mimalloc
    target_link_libraries(${PROJECT_NAME} mimalloc)
    
    # 编译选项
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
    endif()
endif()

# 测试
add_subdirectory(test)

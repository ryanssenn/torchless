cmake_minimum_required(VERSION 3.20)
project(torchless CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(APPLE)
    set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${LIBOMP_ROOT}/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LIBOMP_ROOT}/lib -lomp")
else()
    find_package(OpenMP REQUIRED)
endif()

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE TEST_MISTRAL CONFIGURE_DEPENDS "test/mistral/*.cpp")

add_executable(torchless main.cpp ${SRC})
add_executable(test_exec ${SRC} ${TEST_MISTRAL})

if(NOT APPLE)
    target_link_libraries(torchless PRIVATE OpenMP::OpenMP_CXX)
endif()
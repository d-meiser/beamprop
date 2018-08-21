cmake_minimum_required(VERSION 3.0.0)
project(beamprop)

set(CMAKE_C_STANDARD 11)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(FFTW)


add_library(common_settings INTERFACE)
target_compile_options(common_settings INTERFACE -Wall -Wextra -Werror)


add_subdirectory(fft)

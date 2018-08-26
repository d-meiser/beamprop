cmake_minimum_required(VERSION 3.0.0)
project(beamprop)
enable_testing()

set(CMAKE_C_STANDARD 11)


set(BEAMPROP_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(FFTW_ROOT ${BEAMPROP_THIRD_PARTY_DIR})
find_package(FFTW)

cmake_policy(SET CMP0063 NEW)
include(GenerateExportHeader)


add_library(common_settings INTERFACE)
target_compile_options(common_settings INTERFACE -Wall -Wextra -Werror)


add_subdirectory(fft)

add_library(fft fft.c)
target_link_libraries(fft common_settings)
target_include_directories(fft
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${BEAMPROP_THIRD_PARTY_DIR}/include)
target_link_libraries(fft ${FFTW_DOUBLE_LIB} m)
generate_export_header(fft BASE_NAME FFT)

add_executable(fft_test fft_test.c)
target_link_libraries(fft_test fft)
add_test(fft fft_test)

add_executable(four_f_system four_f_system.c)
target_link_libraries(four_f_system fft)

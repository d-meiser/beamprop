add_executable(fft fft.c)
target_link_libraries(fft common_settings)
target_include_directories(fft
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BEAMPROP_THIRD_PARTY_DIR}/include)
target_link_libraries(fft ${FFTW_DOUBLE_LIB} m)
add_test(fft fft --testing)


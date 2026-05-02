add_executable(test_turboquant_cpu
    test_turboquant_cpu.c
    turboquant_cpu.c
    turboquant_cpu_x86.c)

target_compile_features(test_turboquant_cpu PRIVATE c_std_11)
target_compile_options(test_turboquant_cpu PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_link_libraries(test_turboquant_cpu PRIVATE m)

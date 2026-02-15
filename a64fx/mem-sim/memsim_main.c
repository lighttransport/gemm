// memsim_main.c - Main program for A64FX Memory Access Simulator
//
// Usage: ./memsim <input_file>
//
// Input file format (pseudo-assembly):
//   ld1 <addr> <size>         - Load operation
//   st1 <addr> <size>         - Store operation
//   prfm <addr>               - Prefetch operation
//   sdot                      - Arithmetic op (treated as compute)
//   fmla                      - Arithmetic op (treated as compute)
//   <any other instruction>   - Treated as compute

#include "memsim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char* prog_name) {
    printf("A64FX Memory Access Simulator\n\n");
    printf("Usage: %s <input_file>\n\n", prog_name);
    printf("Input file format (pseudo-assembly):\n");
    printf("  ld1 <addr> <size>         - Load operation\n");
    printf("  st1 <addr> <size>         - Store operation\n");
    printf("  prfm <addr>               - Prefetch operation\n");
    printf("  sdot                      - Arithmetic op (1 cycle)\n");
    printf("  fmla                      - Arithmetic op (1 cycle)\n");
    printf("  <any other>               - Treated as 1-cycle compute\n\n");
    printf("Example:\n");
    printf("  ld1 0x10000 64            # Load 64 bytes from address 0x10000\n");
    printf("  sdot                      # SDOT instruction (compute)\n");
    printf("  prfm 0x20000              # Prefetch address 0x20000\n\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* input_file = argv[1];

    printf("A64FX Memory Access Simulator\n");
    printf("==============================\n\n");
    printf("Input file: %s\n\n", input_file);

    // Create simulator
    memsim_t* sim = memsim_create();
    if (!sim) {
        fprintf(stderr, "Error: Failed to create simulator\n");
        return 1;
    }

    // Run simulation from file
    printf("Running simulation...\n\n");
    memsim_run_file(sim, input_file);

    // Print results
    printf("\n");
    printf("Simulation Complete\n");
    printf("===================\n");

    memsim_print_detailed_stats(sim);

    // Cleanup
    memsim_destroy(sim);

    return 0;
}

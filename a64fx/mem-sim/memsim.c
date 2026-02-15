// memsim.c - A64FX Memory Access Simulator Implementation

#include "memsim.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// ============================================================================
// A64FX-Specific Cache Index Functions
// ============================================================================

// L1 index calculation: (A mod 16384) / 256
int cache_l1_index(uint64_t addr) {
    return ((addr % 16384) / L1_LINE_SIZE);
}

// L2 index calculation: XOR-based hashing
// index<10:0> = ((PA<36:34> xor PA<32:30> xor PA<31:29> xor PA<27:25> xor PA<23:21>) << 8) xor PA<18:8>
int cache_l2_index(uint64_t addr) {
    // Extract bit fields
    uint64_t pa_36_34 = (addr >> 34) & 0x7;   // bits 36:34
    uint64_t pa_32_30 = (addr >> 30) & 0x7;   // bits 32:30
    uint64_t pa_31_29 = (addr >> 29) & 0x7;   // bits 31:29
    uint64_t pa_27_25 = (addr >> 25) & 0x7;   // bits 27:25
    uint64_t pa_23_21 = (addr >> 21) & 0x7;   // bits 23:21
    uint64_t pa_18_8  = (addr >> 8)  & 0x7FF; // bits 18:8

    // XOR upper bits
    uint64_t xor_upper = pa_36_34 ^ pa_32_30 ^ pa_31_29 ^ pa_27_25 ^ pa_23_21;

    // Compute index
    uint64_t index = (xor_upper << 8) ^ pa_18_8;

    return (int)(index & 0x7FF);  // 11-bit index (0-2047)
}

// ============================================================================
// Cache Implementation
// ============================================================================

cache_t* cache_create(int size, int ways, int line_size, int latency) {
    cache_t* cache = (cache_t*)calloc(1, sizeof(cache_t));

    cache->num_ways = ways;
    cache->line_size = line_size;
    cache->latency = latency;
    cache->num_sets = size / (line_size * ways);

    // Allocate sets
    cache->sets = (cache_line_t**)calloc(cache->num_sets, sizeof(cache_line_t*));
    for (int i = 0; i < cache->num_sets; i++) {
        cache->sets[i] = (cache_line_t*)calloc(ways, sizeof(cache_line_t));
    }

    return cache;
}

void cache_destroy(cache_t* cache) {
    if (!cache) return;

    for (int i = 0; i < cache->num_sets; i++) {
        free(cache->sets[i]);
    }
    free(cache->sets);
    free(cache);
}

// Cache access with LRU replacement
bool cache_access(cache_t* cache, uint64_t addr, access_type_t type, uint32_t* lru_counter) {
    // Calculate set index using A64FX-specific functions
    int set_index;
    uint64_t line_addr = addr / cache->line_size;

    // Determine cache type by latency (L1=11, L2=40)
    if (cache->latency == L1_LATENCY) {
        // L1 cache: use simple modulo index
        set_index = cache_l1_index(addr);
    } else {
        // L2 cache: use XOR-based hashing
        set_index = cache_l2_index(addr);
    }

    // Ensure set_index is within bounds
    set_index = set_index % cache->num_sets;

    // Calculate tag
    uint64_t tag = line_addr;  // Use full line address as tag for uniqueness

    cache_line_t* set = cache->sets[set_index];

    // Check for hit
    for (int way = 0; way < cache->num_ways; way++) {
        if (set[way].valid && set[way].tag == tag) {
            // Hit!
            cache->hits++;
            set[way].lru_counter = (*lru_counter)++;

            if (type == ACCESS_LOAD) {
                cache->bytes_read += cache->line_size;
            } else if (type == ACCESS_STORE) {
                cache->bytes_written += cache->line_size;
            }

            return true;
        }
    }

    // Miss - need to find victim
    cache->misses++;

    // Find LRU victim
    int lru_way = 0;
    uint32_t min_lru = set[0].lru_counter;

    for (int way = 1; way < cache->num_ways; way++) {
        if (!set[way].valid) {
            // Empty slot, use it
            lru_way = way;
            break;
        }
        if (set[way].lru_counter < min_lru) {
            min_lru = set[way].lru_counter;
            lru_way = way;
        }
    }

    // Replace victim
    if (set[lru_way].valid) {
        cache->evictions++;
    }

    set[lru_way].valid = true;
    set[lru_way].tag = tag;
    set[lru_way].lru_counter = (*lru_counter)++;

    if (type == ACCESS_LOAD) {
        cache->bytes_read += cache->line_size;
    } else if (type == ACCESS_STORE) {
        cache->bytes_written += cache->line_size;
    }

    return false;
}

void cache_invalidate(cache_t* cache, uint64_t addr) {
    uint64_t line_addr = addr / cache->line_size;
    int set_index = line_addr % cache->num_sets;
    uint64_t tag = line_addr / cache->num_sets;

    cache_line_t* set = cache->sets[set_index];

    for (int way = 0; way < cache->num_ways; way++) {
        if (set[way].valid && set[way].tag == tag) {
            set[way].valid = false;
            return;
        }
    }
}

// ============================================================================
// Memory Simulator Implementation
// ============================================================================

memsim_t* memsim_create(void) {
    memsim_t* sim = (memsim_t*)calloc(1, sizeof(memsim_t));

    sim->l1 = cache_create(L1_SIZE, L1_WAYS, L1_LINE_SIZE, L1_LATENCY);
    sim->l2 = cache_create(L2_SIZE, L2_WAYS, L2_LINE_SIZE, L2_LATENCY);

    sim->global_lru_counter = 0;

    return sim;
}

void memsim_destroy(memsim_t* sim) {
    if (!sim) return;

    cache_destroy(sim->l1);
    cache_destroy(sim->l2);
    free(sim);
}

void memsim_reset(memsim_t* sim) {
    // Reset caches
    cache_destroy(sim->l1);
    cache_destroy(sim->l2);

    sim->l1 = cache_create(L1_SIZE, L1_WAYS, L1_LINE_SIZE, L1_LATENCY);
    sim->l2 = cache_create(L2_SIZE, L2_WAYS, L2_LINE_SIZE, L2_LATENCY);

    // Reset statistics
    sim->total_cycles = 0;
    sim->compute_cycles = 0;
    sim->memory_stall_cycles = 0;
    sim->instructions = 0;
    sim->l1_bandwidth_bytes = 0;
    sim->l2_bandwidth_bytes = 0;
    sim->dram_bandwidth_bytes = 0;
    sim->prefetches = 0;
    sim->useful_prefetches = 0;
    sim->useless_prefetches = 0;
    sim->current_cycle = 0;
    sim->global_lru_counter = 0;
}

// Perform memory access
uint64_t memsim_access(memsim_t* sim, access_type_t type, uint64_t addr, int size) {
    uint64_t cycles = 0;

    // Align to cache line
    uint64_t line_addr = (addr / L1_LINE_SIZE) * L1_LINE_SIZE;

    // Handle zero-fill separately (allocates L1 line without fetching from memory)
    if (type == ACCESS_ZFILL) {
        // Zfill: Allocate cache line in L1, set to zero, no memory fetch
        cache_access(sim->l1, line_addr, type, &sim->global_lru_counter);
        cycles = L1_LATENCY;  // Just L1 allocation time
        sim->l1_bandwidth_bytes += L1_LINE_SIZE;
        return cycles;
    }

    // Check L1
    bool l1_hit = cache_access(sim->l1, line_addr, type, &sim->global_lru_counter);

    if (l1_hit) {
        // L1 hit
        cycles = L1_LATENCY;
        sim->l1_bandwidth_bytes += size;
    } else {
        // L1 miss, check L2
        bool l2_hit = cache_access(sim->l2, line_addr, type, &sim->global_lru_counter);

        if (l2_hit) {
            // L2 hit
            cycles = L1_LATENCY + L2_LATENCY;
            sim->l2_bandwidth_bytes += size;
        } else {
            // L2 miss, go to DRAM
            cycles = L1_LATENCY + L2_LATENCY + DRAM_LATENCY;
            sim->dram_bandwidth_bytes += size;
        }

        // Fill L1 on L2 hit or DRAM access
        // (already done by cache_access)
    }

    if (type == ACCESS_PREFETCH) {
        sim->prefetches++;
        // Prefetch doesn't stall (non-blocking)
        return 0;
    }

    // Track memory stalls
    if (cycles > L1_LATENCY) {
        sim->memory_stall_cycles += (cycles - L1_LATENCY);
    }

    return cycles;
}

// Parse address from instruction
static uint64_t parse_address(const char* str) {
    // Handle formats: [x0], [x0, #offset], 0x1234, etc.
    uint64_t base = 0, offset = 0;

    const char* p = strchr(str, '[');
    if (p) {
        // Register-based addressing
        p++; // Skip '['

        // Skip whitespace
        while (*p && isspace(*p)) p++;

        // Parse base (assume x0-x31 map to some value)
        if (*p == 'x') {
            p++;
            int reg = atoi(p);
            base = reg * 0x10000; // Simple mapping
        }

        // Look for offset
        const char* comma = strchr(p, ',');
        if (comma) {
            comma++;
            while (*comma && isspace(*comma)) comma++;
            if (*comma == '#') {
                offset = strtoll(comma + 1, NULL, 0);
            }
        }

        return base + offset;
    }

    // Direct address
    return strtoull(str, NULL, 0);
}

// Execute single instruction
uint64_t memsim_execute_instruction(memsim_t* sim, const char* instruction) {
    char inst[256];
    strncpy(inst, instruction, sizeof(inst) - 1);
    inst[sizeof(inst) - 1] = '\0';

    // Skip whitespace
    char* p = inst;
    while (*p && isspace(*p)) p++;

    // Skip empty lines and comments
    if (*p == '\0' || *p == '/' || *p == '#') {
        return 0;
    }

    sim->instructions++;
    uint64_t cycles = 0;

    // Parse instruction type
    if (strncmp(p, "ld1", 3) == 0 || strncmp(p, "ld", 2) == 0) {
        // Load instruction
        // Format 1 (assembly): ld1 z0.b, p0/z, [x0, #offset]
        // Format 2 (simple):   ld1 0x10000 64
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;
        int size = SVE_VECTOR_SIZE;

        if (bracket) {
            // Assembly format with brackets
            addr = parse_address(bracket);
            if (strstr(p, ".b")) size = SVE_VECTOR_SIZE;
            else if (strstr(p, ".h")) size = SVE_VECTOR_SIZE;
            else if (strstr(p, ".s")) size = SVE_VECTOR_SIZE;
            else if (strstr(p, ".d")) size = SVE_VECTOR_SIZE;

            cycles = memsim_access(sim, ACCESS_LOAD, addr, size);
        } else {
            // Simple format: ld1 <addr> <size>
            // Skip instruction name
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            // Parse address
            if (*p) {
                char* endptr;
                addr = strtoull(p, &endptr, 0);
                p = endptr;

                // Parse size if present
                while (*p && isspace(*p)) p++;
                if (*p && isdigit(*p)) {
                    size = atoi(p);
                }

                cycles = memsim_access(sim, ACCESS_LOAD, addr, size);
            }
        }
    } else if (strncmp(p, "st1", 3) == 0 || strncmp(p, "st", 2) == 0) {
        // Store instruction
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;
        int size = SVE_VECTOR_SIZE;

        if (bracket) {
            // Assembly format
            addr = parse_address(bracket);
            cycles = memsim_access(sim, ACCESS_STORE, addr, size);
        } else {
            // Simple format: st1 <addr> <size>
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            if (*p) {
                char* endptr;
                addr = strtoull(p, &endptr, 0);
                p = endptr;

                while (*p && isspace(*p)) p++;
                if (*p && isdigit(*p)) {
                    size = atoi(p);
                }

                cycles = memsim_access(sim, ACCESS_STORE, addr, size);
            }
        }
    } else if (strncmp(p, "prfm", 4) == 0) {
        // Prefetch instruction
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;

        if (bracket) {
            // Assembly format
            addr = parse_address(bracket);
            cycles = memsim_access(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE);
        } else {
            // Simple format: prfm <addr>
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            if (*p) {
                addr = strtoull(p, NULL, 0);
                cycles = memsim_access(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE);
            }
        }
    } else if (strncmp(p, "zfill", 5) == 0) {
        // Zero-fill instruction (A64FX optimization)
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;

        if (bracket) {
            // Assembly format
            addr = parse_address(bracket);
            cycles = memsim_access(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE);
        } else {
            // Simple format: zfill <addr>
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            if (*p) {
                addr = strtoull(p, NULL, 0);
                cycles = memsim_access(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE);
            }
        }
    } else if (strncmp(p, "sdot", 4) == 0 ||
               strncmp(p, "fmla", 4) == 0 ||
               strncmp(p, "fmul", 4) == 0 ||
               strncmp(p, "fadd", 4) == 0 ||
               strncmp(p, "add", 3) == 0 ||
               strncmp(p, "mul", 3) == 0) {
        // Arithmetic instruction (treated as compute)
        cycles = 1; // Assume 1 cycle for arithmetic
        sim->compute_cycles += cycles;
    } else {
        // Other instructions (control flow, etc.)
        cycles = 1;
        sim->compute_cycles += cycles;
    }

    sim->total_cycles += cycles;
    sim->current_cycle += cycles;

    return cycles;
}

// Run from file
void memsim_run_file(memsim_t* sim, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return;
    }

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), fp)) {
        line_num++;

        // Remove newline
        char* newline = strchr(line, '\n');
        if (newline) *newline = '\0';

        memsim_execute_instruction(sim, line);
    }

    fclose(fp);

    printf("Executed %d lines from '%s'\n", line_num, filename);
}

// ============================================================================
// Statistics and Reporting
// ============================================================================

void memsim_print_stats(memsim_t* sim) {
    printf("\n");
    printf("================================================================\n");
    printf("A64FX Memory Access Simulator - Statistics\n");
    printf("================================================================\n");

    printf("\n--- Execution Summary ---\n");
    printf("Total instructions:      %llu\n", (unsigned long long)sim->instructions);
    printf("Total cycles:            %llu\n", (unsigned long long)sim->total_cycles);
    printf("Compute cycles:          %llu (%.1f%%)\n",
           (unsigned long long)sim->compute_cycles,
           sim->total_cycles > 0 ? 100.0 * sim->compute_cycles / sim->total_cycles : 0);
    printf("Memory stall cycles:     %llu (%.1f%%)\n",
           (unsigned long long)sim->memory_stall_cycles,
           sim->total_cycles > 0 ? 100.0 * sim->memory_stall_cycles / sim->total_cycles : 0);
    printf("CPI (Cycles Per Instr):  %.2f\n",
           sim->instructions > 0 ? (double)sim->total_cycles / sim->instructions : 0);

    printf("\n--- L1 Cache (64 KB, 4-way) ---\n");
    uint64_t l1_total = sim->l1->hits + sim->l1->misses;
    printf("Hits:        %llu\n", (unsigned long long)sim->l1->hits);
    printf("Misses:      %llu\n", (unsigned long long)sim->l1->misses);
    printf("Hit rate:    %.2f%%\n",
           l1_total > 0 ? 100.0 * sim->l1->hits / l1_total : 0);
    printf("Miss rate:   %.2f%%\n",
           l1_total > 0 ? 100.0 * sim->l1->misses / l1_total : 0);
    printf("Evictions:   %llu\n", (unsigned long long)sim->l1->evictions);

    printf("\n--- L2 Cache (8 MB, 16-way) ---\n");
    uint64_t l2_total = sim->l2->hits + sim->l2->misses;
    printf("Hits:        %llu\n", (unsigned long long)sim->l2->hits);
    printf("Misses:      %llu\n", (unsigned long long)sim->l2->misses);
    printf("Hit rate:    %.2f%%\n",
           l2_total > 0 ? 100.0 * sim->l2->hits / l2_total : 0);
    printf("Miss rate:   %.2f%%\n",
           l2_total > 0 ? 100.0 * sim->l2->misses / l2_total : 0);
    printf("Evictions:   %llu\n", (unsigned long long)sim->l2->evictions);

    printf("\n--- Memory Bandwidth ---\n");
    uint64_t total_bandwidth = sim->l1_bandwidth_bytes + sim->l2_bandwidth_bytes + sim->dram_bandwidth_bytes;
    printf("L1 accesses:     %.2f MB (%.1f%%)\n",
           sim->l1_bandwidth_bytes / (1024.0 * 1024.0),
           total_bandwidth > 0 ? 100.0 * sim->l1_bandwidth_bytes / total_bandwidth : 0);
    printf("L2 accesses:     %.2f MB (%.1f%%)\n",
           sim->l2_bandwidth_bytes / (1024.0 * 1024.0),
           total_bandwidth > 0 ? 100.0 * sim->l2_bandwidth_bytes / total_bandwidth : 0);
    printf("DRAM accesses:   %.2f MB (%.1f%%)\n",
           sim->dram_bandwidth_bytes / (1024.0 * 1024.0),
           total_bandwidth > 0 ? 100.0 * sim->dram_bandwidth_bytes / total_bandwidth : 0);

    if (sim->total_cycles > 0) {
        printf("\nEffective bandwidth:\n");
        printf("  L1:    %.2f GB/s\n",
               (sim->l1_bandwidth_bytes / (double)sim->total_cycles) * 2.0e9 / 1e9);
        printf("  L2:    %.2f GB/s\n",
               (sim->l2_bandwidth_bytes / (double)sim->total_cycles) * 2.0e9 / 1e9);
        printf("  DRAM:  %.2f GB/s\n",
               (sim->dram_bandwidth_bytes / (double)sim->total_cycles) * 2.0e9 / 1e9);
    }

    if (sim->prefetches > 0) {
        printf("\n--- Prefetch Statistics ---\n");
        printf("Prefetches issued:   %llu\n", (unsigned long long)sim->prefetches);
    }

    printf("\n================================================================\n");
}

void memsim_analyze_bottlenecks(memsim_t* sim) {
    printf("\n");
    printf("================================================================\n");
    printf("Bottleneck Analysis\n");
    printf("================================================================\n");

    double mem_stall_pct = sim->total_cycles > 0 ?
        100.0 * sim->memory_stall_cycles / sim->total_cycles : 0;

    uint64_t l1_total = sim->l1->hits + sim->l1->misses;
    double l1_miss_rate = l1_total > 0 ?
        100.0 * sim->l1->misses / l1_total : 0;

    uint64_t l2_total = sim->l2->hits + sim->l2->misses;
    double l2_miss_rate = l2_total > 0 ?
        100.0 * sim->l2->misses / l2_total : 0;

    printf("\n--- Performance Characterization ---\n");

    if (mem_stall_pct > 50) {
        printf("⚠ MEMORY-BOUND (%.1f%% stalls)\n", mem_stall_pct);
        printf("  Primary bottleneck: Memory access latency\n");

        if (l1_miss_rate > 20) {
            printf("  → High L1 miss rate (%.1f%%)\n", l1_miss_rate);
            printf("    Recommendations:\n");
            printf("    - Improve temporal locality (reuse data in L1)\n");
            printf("    - Reduce working set size\n");
            printf("    - Consider tiling/blocking\n");
        }

        if (l2_miss_rate > 10) {
            printf("  → High L2 miss rate (%.1f%%)\n", l2_miss_rate);
            printf("    Recommendations:\n");
            printf("    - Add software prefetching\n");
            printf("    - Improve spatial locality\n");
            printf("    - Optimize memory layout\n");
        }
    } else if (mem_stall_pct > 20) {
        printf("⚠ MEMORY-LIMITED (%.1f%% stalls)\n", mem_stall_pct);
        printf("  Bottleneck: Mix of compute and memory\n");
        printf("  Recommendations:\n");
        printf("  - Balance compute/memory with prefetching\n");
        printf("  - Hide memory latency with computation\n");
    } else {
        printf("✓ COMPUTE-BOUND (%.1f%% stalls)\n", mem_stall_pct);
        printf("  Bottleneck: Computation (memory well-optimized)\n");
        printf("  L1 hit rate: %.1f%%\n", 100.0 - l1_miss_rate);
    }

    printf("\n--- Cache Efficiency ---\n");

    if (l1_miss_rate < 5) {
        printf("✓ Excellent L1 utilization (%.1f%% hit rate)\n", 100.0 - l1_miss_rate);
    } else if (l1_miss_rate < 15) {
        printf("⚠ Good L1 utilization (%.1f%% hit rate)\n", 100.0 - l1_miss_rate);
    } else {
        printf("✗ Poor L1 utilization (%.1f%% hit rate)\n", 100.0 - l1_miss_rate);
    }

    if (l2_miss_rate < 2) {
        printf("✓ Excellent L2 utilization (%.1f%% hit rate)\n", 100.0 - l2_miss_rate);
    } else if (l2_miss_rate < 10) {
        printf("⚠ Good L2 utilization (%.1f%% hit rate)\n", 100.0 - l2_miss_rate);
    } else {
        printf("✗ Poor L2 utilization (%.1f%% hit rate)\n", 100.0 - l2_miss_rate);
    }

    // Estimate efficiency loss
    printf("\n--- Estimated Efficiency Loss ---\n");
    double efficiency_loss = mem_stall_pct;
    printf("Memory stalls: %.1f%%\n", efficiency_loss);
    printf("Achievable efficiency: %.1f%%\n", 100.0 - efficiency_loss);

    printf("\n================================================================\n");
}

void memsim_print_detailed_stats(memsim_t* sim) {
    memsim_print_stats(sim);
    memsim_analyze_bottlenecks(sim);
}

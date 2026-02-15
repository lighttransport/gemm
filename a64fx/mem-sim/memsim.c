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

    // Initialize sector cache (disabled by default)
    cache->sector_cache_enabled = false;
    // Default: equal way allocation across sectors
    int ways_per_sector = ways / NUM_SECTORS;
    for (int s = 0; s < NUM_SECTORS; s++) {
        cache->sector_ways[s] = ways_per_sector;
        cache->sector_hits[s] = 0;
        cache->sector_misses[s] = 0;
        cache->sector_evictions[s] = 0;
    }
    // Give remaining ways to sector 0 (normal data)
    cache->sector_ways[0] += ways % NUM_SECTORS;

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

    // Enable out-of-order model by default for accurate A64FX simulation
    sim->use_ooo_model = true;
    sim->load_ready_cycle = 0;
    sim->compute_ready_cycle = 0;
    sim->hidden_latency_cycles = 0;
    sim->prefetch_queue_head = 0;

    // Initialize prefetch queue
    for (int i = 0; i < MAX_INFLIGHT_PREFETCH; i++) {
        sim->prefetch_queue[i].valid = false;
    }

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

    // Reset OOO model state
    sim->load_ready_cycle = 0;
    sim->compute_ready_cycle = 0;
    sim->hidden_latency_cycles = 0;
    sim->prefetch_queue_head = 0;
    sim->total_memory_cycles = 0;
    sim->total_compute_only = 0;
    for (int i = 0; i < MAX_INFLIGHT_PREFETCH; i++) {
        sim->prefetch_queue[i].valid = false;
    }
}

// Check if prefetch has completed for this address
static bool check_prefetch_ready(memsim_t* sim, uint64_t line_addr) {
    for (int i = 0; i < MAX_INFLIGHT_PREFETCH; i++) {
        if (sim->prefetch_queue[i].valid &&
            sim->prefetch_queue[i].addr == line_addr &&
            sim->prefetch_queue[i].ready_cycle <= sim->current_cycle) {
            // Prefetch completed, mark as useful
            sim->prefetch_queue[i].valid = false;
            sim->useful_prefetches++;
            return true;
        }
    }
    return false;
}

// Add prefetch to queue
static void queue_prefetch(memsim_t* sim, uint64_t line_addr, uint64_t latency) {
    int slot = sim->prefetch_queue_head;
    sim->prefetch_queue[slot].addr = line_addr;
    sim->prefetch_queue[slot].ready_cycle = sim->current_cycle + latency;
    sim->prefetch_queue[slot].valid = true;
    sim->prefetch_queue_head = (sim->prefetch_queue_head + 1) % MAX_INFLIGHT_PREFETCH;
}

// Perform memory access with out-of-order execution model
uint64_t memsim_access(memsim_t* sim, access_type_t type, uint64_t addr, int size) {
    uint64_t cycles = 0;
    uint64_t memory_latency = 0;

    // Align to cache line
    uint64_t line_addr = (addr / L1_LINE_SIZE) * L1_LINE_SIZE;

    // Handle zero-fill separately (allocates L1 line without fetching from memory)
    if (type == ACCESS_ZFILL) {
        cache_access(sim->l1, line_addr, type, &sim->global_lru_counter);
        cycles = L1_LATENCY;
        sim->l1_bandwidth_bytes += L1_LINE_SIZE;
        return cycles;
    }

    // Check L1 cache
    bool l1_hit = cache_access(sim->l1, line_addr, type, &sim->global_lru_counter);

    if (l1_hit) {
        memory_latency = L1_LATENCY;
        sim->l1_bandwidth_bytes += size;
    } else {
        // Check L2 cache
        bool l2_hit = cache_access(sim->l2, line_addr, type, &sim->global_lru_counter);

        if (l2_hit) {
            memory_latency = L1_LATENCY + L2_LATENCY;
            sim->l2_bandwidth_bytes += size;
        } else {
            memory_latency = L1_LATENCY + L2_LATENCY + DRAM_LATENCY;
            sim->dram_bandwidth_bytes += size;
        }
    }

    // Handle prefetch: queue it and return immediately (non-blocking)
    if (type == ACCESS_PREFETCH) {
        sim->prefetches++;
        queue_prefetch(sim, line_addr, memory_latency);
        return 0;  // Prefetch doesn't block
    }

    // For loads/stores with OOO model: check if prefetch already brought data
    if (sim->use_ooo_model && !l1_hit) {
        // Check if a prefetch has already completed for this line
        if (check_prefetch_ready(sim, line_addr)) {
            // Prefetch completed! Data is ready, only pay L1 latency
            memory_latency = L1_LATENCY;
            sim->hidden_latency_cycles += (L1_LATENCY + L2_LATENCY + DRAM_LATENCY - L1_LATENCY);
        }
    }

    // Track memory cycles for overlap calculation
    sim->total_memory_cycles += memory_latency;

    // Out-of-order execution model: overlap memory with compute
    if (sim->use_ooo_model) {
        // In OOO model, memory and compute run in parallel
        // Only count the non-overlapped portion
        // A64FX has 2 load ports with pipelined L1 access

        // Track if this would cause a stall (L2/DRAM miss)
        if (memory_latency > L1_LATENCY) {
            sim->memory_stall_cycles += (memory_latency - L1_LATENCY);
        }

        // For OOO, we'll calculate total cycles at the end
        // Here we just track the memory latency
        cycles = memory_latency;
    } else {
        // Simple in-order model
        cycles = memory_latency;
        if (memory_latency > L1_LATENCY) {
            sim->memory_stall_cycles += (memory_latency - L1_LATENCY);
        }
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

// Parse sector annotation from instruction (e.g., ld1.s2 -> SECTOR_REUSE)
static sector_id_t parse_sector(const char* inst, char** after_sector) {
    sector_id_t sector = SECTOR_NORMAL;
    const char* p = inst;

    // Find the instruction name end
    while (*p && !isspace(*p)) {
        // Check for .s0, .s1, .s2, .s3 suffix
        if (*p == '.' && *(p+1) == 's' && *(p+2) >= '0' && *(p+2) <= '3') {
            sector = (sector_id_t)(*(p+2) - '0');
            p += 3;
            break;
        }
        p++;
    }

    if (after_sector) *after_sector = (char*)p;
    return sector;
}

// Execute single instruction
uint64_t memsim_execute_instruction(memsim_t* sim, const char* instruction) {
    char inst[256];
    strncpy(inst, instruction, sizeof(inst) - 1);
    inst[sizeof(inst) - 1] = '\0';

    // Skip whitespace
    char* p = inst;
    while (*p && isspace(*p)) p++;

    // Skip empty lines and comments (but handle directives starting with #)
    if (*p == '\0' || *p == '/') {
        return 0;
    }

    // Handle directives
    if (*p == '#') {
        // #sector_enable - enable sector cache
        if (strncmp(p, "#sector_enable", 14) == 0) {
            memsim_enable_sector_cache(sim, true);
            return 0;
        }
        // #sector_config <s0_ways> <s1_ways> <s2_ways> <s3_ways>
        if (strncmp(p, "#sector_config", 14) == 0) {
            p += 14;
            while (*p && isspace(*p)) p++;
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (sscanf(p, "%d %d %d %d", &s0, &s1, &s2, &s3) == 4) {
                memsim_configure_l2_sectors(sim, s0, s1, s2, s3);
            }
            return 0;
        }
        // Regular comment
        return 0;
    }

    sim->instructions++;
    uint64_t cycles = 0;

    // Parse sector annotation if present (e.g., ld1.s0, ld1.s1, ld1.s2, ld1.s3)
    sector_id_t sector = parse_sector(p, NULL);
    bool use_sector = sim->l1->sector_cache_enabled;

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

            if (use_sector) {
                cycles = memsim_access_sector(sim, ACCESS_LOAD, addr, size, sector);
            } else {
                cycles = memsim_access(sim, ACCESS_LOAD, addr, size);
            }
        } else {
            // Simple format: ld1 <addr> <size> or ld1.s2 <addr> <size>
            // Skip instruction name (including sector suffix)
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

                if (use_sector) {
                    cycles = memsim_access_sector(sim, ACCESS_LOAD, addr, size, sector);
                } else {
                    cycles = memsim_access(sim, ACCESS_LOAD, addr, size);
                }
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
            if (use_sector) {
                cycles = memsim_access_sector(sim, ACCESS_STORE, addr, size, sector);
            } else {
                cycles = memsim_access(sim, ACCESS_STORE, addr, size);
            }
        } else {
            // Simple format: st1 <addr> <size> or st1.s3 <addr> <size>
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

                if (use_sector) {
                    cycles = memsim_access_sector(sim, ACCESS_STORE, addr, size, sector);
                } else {
                    cycles = memsim_access(sim, ACCESS_STORE, addr, size);
                }
            }
        }
    } else if (strncmp(p, "prfm", 4) == 0) {
        // Prefetch instruction
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;

        if (bracket) {
            // Assembly format
            addr = parse_address(bracket);
            if (use_sector) {
                cycles = memsim_access_sector(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE, sector);
            } else {
                cycles = memsim_access(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE);
            }
        } else {
            // Simple format: prfm <addr> or prfm.s1 <addr>
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            if (*p) {
                addr = strtoull(p, NULL, 0);
                if (use_sector) {
                    cycles = memsim_access_sector(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE, sector);
                } else {
                    cycles = memsim_access(sim, ACCESS_PREFETCH, addr, L1_LINE_SIZE);
                }
            }
        }
    } else if (strncmp(p, "zfill", 5) == 0) {
        // Zero-fill instruction (A64FX optimization)
        const char* bracket = strchr(p, '[');
        uint64_t addr = 0;

        if (bracket) {
            // Assembly format
            addr = parse_address(bracket);
            if (use_sector) {
                cycles = memsim_access_sector(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE, sector);
            } else {
                cycles = memsim_access(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE);
            }
        } else {
            // Simple format: zfill <addr> or zfill.s3 <addr>
            while (*p && !isspace(*p)) p++;
            while (*p && isspace(*p)) p++;

            if (*p) {
                addr = strtoull(p, NULL, 0);
                if (use_sector) {
                    cycles = memsim_access_sector(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE, sector);
                } else {
                    cycles = memsim_access(sim, ACCESS_ZFILL, addr, L1_LINE_SIZE);
                }
            }
        }
    } else if (strncmp(p, "sdot", 4) == 0 ||
               strncmp(p, "fmla", 4) == 0 ||
               strncmp(p, "fmul", 4) == 0 ||
               strncmp(p, "fadd", 4) == 0 ||
               strncmp(p, "add", 3) == 0 ||
               strncmp(p, "mul", 3) == 0) {
        // Arithmetic instruction (treated as compute)
        // A64FX: 2 FPUs, each can do 2 SDOT/cycle = 4 SDOT/cycle throughput
        cycles = COMPUTE_LATENCY;
        sim->compute_cycles += cycles;
        sim->total_compute_only += cycles;  // Track pure compute for overlap
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

    // Print sector stats if sector cache was used
    if (sim->l1->sector_cache_enabled) {
        memsim_print_sector_stats(sim);
    }

    // Calculate overlapped execution for OOO model
    if (sim->use_ooo_model) {
        // In OOO execution, compute and memory run in parallel
        // Total time = max(compute_time, memory_time) + non-overlapped stalls

        // Memory time with 2 load ports: each L1 hit is pipelined
        // For L1 hits: 2 loads/cycle throughput, 11-cycle latency hidden by pipelining
        // For L2/DRAM: full latency applies (tracked in memory_stall_cycles)

        // Calculate effective memory time:
        // - L1 accesses are pipelined (1 cycle throughput with 2 ports)
        // - L2/DRAM stalls are serial
        uint64_t l1_accesses = sim->l1->hits;
        uint64_t l1_effective_cycles = (l1_accesses + 1) / 2;  // 2 loads/cycle

        // Total effective cycles with overlap
        uint64_t compute_time = sim->total_compute_only;
        uint64_t memory_time = l1_effective_cycles + sim->memory_stall_cycles;

        // Overlapped execution: max of compute and memory
        uint64_t overlapped_cycles = (compute_time > memory_time) ? compute_time : memory_time;

        // Add overhead for prefetch, zfill, stores (minimal)
        overlapped_cycles += sim->prefetches;  // 1 cycle each
        overlapped_cycles += 12;  // zfill/store overhead

        // Calculate hidden latency
        sim->hidden_latency_cycles = (sim->total_cycles > overlapped_cycles) ?
            (sim->total_cycles - overlapped_cycles) : 0;

        // Update total cycles to reflect overlap
        sim->total_cycles = overlapped_cycles;

        // Recalculate memory stalls based on overlap
        if (compute_time >= memory_time) {
            // Memory completely hidden by compute
            sim->memory_stall_cycles = 0;
        } else {
            // Some memory not hidden
            sim->memory_stall_cycles = memory_time - compute_time;
        }
    }
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
        printf("Useful prefetches:   %llu (%.1f%%)\n",
               (unsigned long long)sim->useful_prefetches,
               sim->prefetches > 0 ? 100.0 * sim->useful_prefetches / sim->prefetches : 0);
    }

    if (sim->use_ooo_model && sim->hidden_latency_cycles > 0) {
        printf("\n--- Out-of-Order Execution ---\n");
        printf("Hidden latency:      %llu cycles\n", (unsigned long long)sim->hidden_latency_cycles);
        uint64_t total_latency = sim->hidden_latency_cycles + sim->memory_stall_cycles;
        if (total_latency > 0) {
            printf("Latency hiding:      %.1f%%\n",
                   100.0 * sim->hidden_latency_cycles / total_latency);
        }
        double efficiency = sim->total_cycles > 0 ?
            100.0 * sim->compute_cycles / sim->total_cycles : 0;
        printf("Compute efficiency:  %.1f%%\n", efficiency);
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

// ============================================================================
// Sector Cache Implementation
// ============================================================================

void memsim_enable_sector_cache(memsim_t* sim, bool enable) {
    sim->l1->sector_cache_enabled = enable;
    sim->l2->sector_cache_enabled = enable;
}

void memsim_configure_l2_sectors(memsim_t* sim, int s0_ways, int s1_ways, int s2_ways, int s3_ways) {
    // Validate total ways doesn't exceed L2_WAYS
    int total = s0_ways + s1_ways + s2_ways + s3_ways;
    if (total > L2_WAYS) {
        fprintf(stderr, "Warning: Total sector ways (%d) exceeds L2 ways (%d)\n", total, L2_WAYS);
        return;
    }

    sim->l2->sector_ways[0] = s0_ways;
    sim->l2->sector_ways[1] = s1_ways;
    sim->l2->sector_ways[2] = s2_ways;
    sim->l2->sector_ways[3] = s3_ways;
    sim->l2->sector_cache_enabled = true;

    // Also configure L1 proportionally (4 ways total)
    sim->l1->sector_ways[0] = (s0_ways > 0) ? 1 : 0;
    sim->l1->sector_ways[1] = (s1_ways > 0) ? 1 : 0;
    sim->l1->sector_ways[2] = (s2_ways > 0) ? 1 : 0;
    sim->l1->sector_ways[3] = (s3_ways > 0) ? 1 : 0;
    sim->l1->sector_cache_enabled = true;
}

// Sector-aware cache access
static bool cache_access_sector(cache_t* cache, uint64_t addr, access_type_t type,
                                uint32_t* lru_counter, sector_id_t sector) {
    // Calculate set index
    int set_index;
    uint64_t line_addr = addr / cache->line_size;

    if (cache->latency == L1_LATENCY) {
        set_index = cache_l1_index(addr);
    } else {
        set_index = cache_l2_index(addr);
    }
    set_index = set_index % cache->num_sets;

    uint64_t tag = line_addr;
    cache_line_t* set = cache->sets[set_index];

    // Determine way range for this sector
    int way_start = 0, way_end = cache->num_ways;

    if (cache->sector_cache_enabled) {
        // Calculate way range for this sector
        way_start = 0;
        for (int s = 0; s < (int)sector; s++) {
            way_start += cache->sector_ways[s];
        }
        way_end = way_start + cache->sector_ways[sector];

        // Clamp to valid range
        if (way_end > cache->num_ways) way_end = cache->num_ways;
        if (way_start >= way_end) {
            // Sector has no ways allocated - use sector 0
            way_start = 0;
            way_end = cache->sector_ways[0];
        }
    }

    // Check for hit within sector's ways
    for (int way = way_start; way < way_end; way++) {
        if (set[way].valid && set[way].tag == tag) {
            // Hit!
            cache->hits++;
            cache->sector_hits[sector]++;
            set[way].lru_counter = (*lru_counter)++;

            if (type == ACCESS_LOAD) {
                cache->bytes_read += cache->line_size;
            } else if (type == ACCESS_STORE) {
                cache->bytes_written += cache->line_size;
            }
            return true;
        }
    }

    // Also check other sectors for hit (data may have been placed there before)
    for (int way = 0; way < cache->num_ways; way++) {
        if (way >= way_start && way < way_end) continue;  // Already checked
        if (set[way].valid && set[way].tag == tag) {
            // Hit in different sector - move to correct sector if space available
            cache->hits++;
            cache->sector_hits[sector]++;
            set[way].lru_counter = (*lru_counter)++;
            set[way].sector = sector;
            return true;
        }
    }

    // Miss - find victim within sector's ways
    cache->misses++;
    cache->sector_misses[sector]++;

    int lru_way = way_start;
    uint32_t min_lru = set[way_start].lru_counter;

    for (int way = way_start + 1; way < way_end; way++) {
        if (!set[way].valid) {
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
        cache->sector_evictions[set[lru_way].sector]++;
    }

    set[lru_way].valid = true;
    set[lru_way].tag = tag;
    set[lru_way].lru_counter = (*lru_counter)++;
    set[lru_way].sector = sector;

    if (type == ACCESS_LOAD) {
        cache->bytes_read += cache->line_size;
    } else if (type == ACCESS_STORE) {
        cache->bytes_written += cache->line_size;
    }

    return false;
}

// Memory access with sector specification
uint64_t memsim_access_sector(memsim_t* sim, access_type_t type, uint64_t addr,
                              int size, sector_id_t sector) {
    uint64_t cycles = 0;
    uint64_t memory_latency = 0;
    uint64_t line_addr = (addr / L1_LINE_SIZE) * L1_LINE_SIZE;

    // Handle zfill
    if (type == ACCESS_ZFILL) {
        cache_access_sector(sim->l1, line_addr, type, &sim->global_lru_counter, sector);
        cycles = L1_LATENCY;
        sim->l1_bandwidth_bytes += L1_LINE_SIZE;
        return cycles;
    }

    // Check L1 with sector
    bool l1_hit = cache_access_sector(sim->l1, line_addr, type,
                                      &sim->global_lru_counter, sector);

    if (l1_hit) {
        memory_latency = L1_LATENCY;
        sim->l1_bandwidth_bytes += size;
    } else {
        // Check L2 with sector
        bool l2_hit = cache_access_sector(sim->l2, line_addr, type,
                                          &sim->global_lru_counter, sector);

        if (l2_hit) {
            memory_latency = L1_LATENCY + L2_LATENCY;
            sim->l2_bandwidth_bytes += size;
        } else {
            memory_latency = L1_LATENCY + L2_LATENCY + DRAM_LATENCY;
            sim->dram_bandwidth_bytes += size;
        }
    }

    // Handle prefetch
    if (type == ACCESS_PREFETCH) {
        sim->prefetches++;
        queue_prefetch(sim, line_addr, memory_latency);
        return 0;
    }

    sim->total_memory_cycles += memory_latency;

    // Track stalls
    if (memory_latency > L1_LATENCY) {
        sim->memory_stall_cycles += (memory_latency - L1_LATENCY);
    }

    return memory_latency;
}

void memsim_print_sector_stats(memsim_t* sim) {
    printf("\n");
    printf("================================================================\n");
    printf("Sector Cache Statistics\n");
    printf("================================================================\n");

    const char* sector_names[] = {"Normal", "Streaming", "Reuse", "Temp"};

    printf("\n--- L1 Sector Cache (4-way) ---\n");
    if (sim->l1->sector_cache_enabled) {
        printf("Sector      Ways  Hits       Misses     Evictions  Hit Rate\n");
        printf("------      ----  ----       ------     ---------  --------\n");
        for (int s = 0; s < NUM_SECTORS; s++) {
            uint64_t total = sim->l1->sector_hits[s] + sim->l1->sector_misses[s];
            double hit_rate = total > 0 ? 100.0 * sim->l1->sector_hits[s] / total : 0;
            printf("%-10s  %4d  %-10llu %-10llu %-10llu %.1f%%\n",
                   sector_names[s],
                   sim->l1->sector_ways[s],
                   (unsigned long long)sim->l1->sector_hits[s],
                   (unsigned long long)sim->l1->sector_misses[s],
                   (unsigned long long)sim->l1->sector_evictions[s],
                   hit_rate);
        }
    } else {
        printf("Sector cache disabled\n");
    }

    printf("\n--- L2 Sector Cache (14-way) ---\n");
    if (sim->l2->sector_cache_enabled) {
        printf("Sector      Ways  Hits       Misses     Evictions  Hit Rate\n");
        printf("------      ----  ----       ------     ---------  --------\n");
        for (int s = 0; s < NUM_SECTORS; s++) {
            uint64_t total = sim->l2->sector_hits[s] + sim->l2->sector_misses[s];
            double hit_rate = total > 0 ? 100.0 * sim->l2->sector_hits[s] / total : 0;
            printf("%-10s  %4d  %-10llu %-10llu %-10llu %.1f%%\n",
                   sector_names[s],
                   sim->l2->sector_ways[s],
                   (unsigned long long)sim->l2->sector_hits[s],
                   (unsigned long long)sim->l2->sector_misses[s],
                   (unsigned long long)sim->l2->sector_evictions[s],
                   hit_rate);
        }
    } else {
        printf("Sector cache disabled\n");
    }

    printf("\n================================================================\n");
}

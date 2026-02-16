// memsim.h - A64FX Memory Access Simulator
// Simulates cache hierarchy and memory access patterns

#ifndef MEMSIM_H
#define MEMSIM_H

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// A64FX Cache Configuration (Accurate Specifications)
// ============================================================================

// L1 Data Cache (per CMG)
#define L1_SIZE (64 * 1024)        // 64 KiB
#define L1_WAYS 4                  // 4-way set associative
#define L1_LINE_SIZE 256           // 256 bytes per cache line
#define L1_LATENCY 11              // 11 cycles
// L1 index: index_A = (A mod 16384) / 256
// MESI protocol, writeback policy

// L2 Cache (per CMG, 1 MB reserved for assistant core)
#define L2_SIZE (7 * 1024 * 1024)  // 7 MiB (8 MiB - 1 MiB assistant core)
#define L2_WAYS 14                 // 14-way set associative
#define L2_LINE_SIZE 256           // 256 bytes per cache line
#define L2_LATENCY 40              // 35-45 cycles (average 40)
// L2 index: Complex XOR-based hashing (see cache_l2_index)
// MESI protocol, writeback policy

// HBM2 Main Memory
#define DRAM_LATENCY 260           // 260 cycles for HBM2
#define DRAM_BANDWIDTH_LD 256      // 256 GB/s load bandwidth
#define DRAM_BANDWIDTH_ST 128      // 128 GB/s store bandwidth

#define SVE_VECTOR_SIZE 64         // 512-bit = 64 bytes

// A64FX Execution Units
#define NUM_LOAD_PORTS 2           // 2 load/store ports per core
#define NUM_FPU_PIPES 2            // 2 FPUs, each can do 2 SDOT/cycle
#define SDOT_THROUGHPUT 4          // 4 SDOT per cycle (2 FPU × 2)
#define COMPUTE_LATENCY 1          // SDOT latency (pipelined)

// A64FX Sector Cache Configuration
// L1: 4 sectors (4-way, typically 1 way per sector)
// L2: 4 sectors in 2 groups, 14 usable ways
#define NUM_SECTORS 4
#define L1_SECTOR_WAYS 1           // 1 way per sector in L1
#define L2_DEFAULT_SECTOR_WAYS 4   // Default ways per sector in L2

// Sector types for cache partitioning
typedef enum {
    SECTOR_NORMAL = 0,      // Sector 0: Normal data (most ways)
    SECTOR_STREAMING = 1,   // Sector 1: Streaming/non-temporal data
    SECTOR_REUSE = 2,       // Sector 2: High-reuse data (keep in cache)
    SECTOR_TEMP = 3         // Sector 3: Temporary data
} sector_id_t;

// Derived constants
#define L1_NUM_SETS (L1_SIZE / (L1_LINE_SIZE * L1_WAYS))
#define L2_NUM_SETS (L2_SIZE / (L2_LINE_SIZE * L2_WAYS))

// ============================================================================
// Cache Line Structure
// ============================================================================

typedef struct {
    uint64_t tag;           // Tag bits
    bool valid;             // Valid bit
    uint32_t lru_counter;   // LRU counter (higher = more recently used)
    uint8_t sector;         // Sector assignment (0-3)
} cache_line_t;

// ============================================================================
// Cache Structure
// ============================================================================

typedef struct {
    cache_line_t** sets;    // Array of sets, each containing ways
    int num_sets;
    int num_ways;
    int line_size;
    int latency;

    // Statistics
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_read;
    uint64_t bytes_written;

    // Sector cache configuration
    bool sector_cache_enabled;
    int sector_ways[NUM_SECTORS];    // Ways allocated per sector
    uint64_t sector_hits[NUM_SECTORS];
    uint64_t sector_misses[NUM_SECTORS];
    uint64_t sector_evictions[NUM_SECTORS];
} cache_t;

// ============================================================================
// Prefetch Tracking (for latency hiding model)
// ============================================================================

#define MAX_INFLIGHT_PREFETCH 64

typedef struct {
    uint64_t addr;              // Cache line address
    uint64_t ready_cycle;       // Cycle when data will be ready
    bool valid;
} prefetch_entry_t;

// ============================================================================
// Memory Simulator
// ============================================================================

typedef struct {
    cache_t* l1;
    cache_t* l2;

    // Global statistics
    uint64_t total_cycles;
    uint64_t compute_cycles;
    uint64_t memory_stall_cycles;
    uint64_t instructions;

    // Bandwidth tracking
    uint64_t l1_bandwidth_bytes;
    uint64_t l2_bandwidth_bytes;
    uint64_t dram_bandwidth_bytes;

    // Prefetch statistics
    uint64_t prefetches;
    uint64_t useful_prefetches;
    uint64_t useless_prefetches;

    // Current cycle (for timing)
    uint64_t current_cycle;

    // LRU global counter
    uint32_t global_lru_counter;

    // Prefetch queue for latency hiding
    prefetch_entry_t prefetch_queue[MAX_INFLIGHT_PREFETCH];
    int prefetch_queue_head;

    // Pipeline model
    bool use_ooo_model;         // Use out-of-order execution model
    uint64_t load_ready_cycle;  // When load ports are free
    uint64_t compute_ready_cycle; // When FPUs are free
    uint64_t hidden_latency_cycles; // Latency hidden by overlapping

    // Parallel execution tracking (for final calculation)
    uint64_t total_memory_cycles;  // Sum of all memory latencies
    uint64_t total_compute_only;   // Compute cycles only (for overlap calc)
} memsim_t;

// ============================================================================
// API Functions
// ============================================================================

// Initialize simulator
memsim_t* memsim_create(void);
void memsim_destroy(memsim_t* sim);
void memsim_reset(memsim_t* sim);

// Memory access operations
typedef enum {
    ACCESS_LOAD,
    ACCESS_STORE,
    ACCESS_PREFETCH,
    ACCESS_ZFILL        // Zero-fill cache line
} access_type_t;

// Perform memory access (returns cycles taken)
uint64_t memsim_access(memsim_t* sim, access_type_t type, uint64_t addr, int size);

// Execute instruction (returns cycles taken)
uint64_t memsim_execute_instruction(memsim_t* sim, const char* instruction);

// Parse and execute from file
void memsim_run_file(memsim_t* sim, const char* filename);

// Statistics and reporting
void memsim_print_stats(memsim_t* sim);
void memsim_print_detailed_stats(memsim_t* sim);
void memsim_analyze_bottlenecks(memsim_t* sim);

// Cache operations (internal)
cache_t* cache_create(int size, int ways, int line_size, int latency);
void cache_destroy(cache_t* cache);
bool cache_access(cache_t* cache, uint64_t addr, access_type_t type, uint32_t* lru_counter);
void cache_invalidate(cache_t* cache, uint64_t addr);

// A64FX-specific cache index functions
int cache_l1_index(uint64_t addr);  // L1 index: (A mod 16384) / 256
int cache_l2_index(uint64_t addr);  // L2 index: XOR-based hashing

// Sector cache functions
void memsim_enable_sector_cache(memsim_t* sim, bool enable);
void memsim_configure_l2_sectors(memsim_t* sim, int s0_ways, int s1_ways, int s2_ways, int s3_ways);
uint64_t memsim_access_sector(memsim_t* sim, access_type_t type, uint64_t addr, int size, sector_id_t sector);
void memsim_print_sector_stats(memsim_t* sim);

#endif // MEMSIM_H

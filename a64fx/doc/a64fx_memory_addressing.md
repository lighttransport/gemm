# A64FX memory addressing, cache indexing, and stream placement

This note extracts the address-dependent behavior documented in Fujitsu's
[A64FX Microarchitecture Manual, Japanese, version 1.8.1](https://github.com/fujitsu/A64FX/blob/master/doc/A64FX_Microarchitecture_Manual_jp_1.8.1.pdf).
Section and printed-page references below refer to that edition. Fujitsu's
[English companion manual](https://github.com/fujitsu/A64FX/blob/master/doc/A64FX_Microarchitecture_Manual_en_1.8.1.pdf)
was used to cross-check terminology.

## CMG-local physical memory path

Section 9.1 (pp. 73–74) describes four cache-coherent NUMA Core Memory Groups
(CMGs). Physical memory space is divided by CMG. Memory belonging to a CMG is
connected only to that CMG's shared L2, and requests travel through:

```text
core L1D -> CMG L2 -> Move-In Buffer (MIB) -> Memory Access Controller (MAC)
            -> the CMG's directly attached HBM2 stack
```

Each CMG has one MAC and one point-to-point-connected 8 GiB HBM2 stack
(§10.1, p. 79). Remote CMG access remains coherent through the inter-CMG L2
ring but is NUMA access, not the local path.

The documented per-CMG data-path ceilings at 2 GHz are:

| Path | Documented rate |
|:-----|----------------:|
| L2 -> L1D | 512 B/cycle |
| L1D -> L2 | 256 B/cycle |
| memory -> L2 | 128 B/cycle = 256 GB/s |
| L2 -> memory | 64 B/cycle = 128 GB/s |

The MAC has a 244-entry scheduling queue and may reorder accesses to maximize
HBM2 throughput (§10.1). Fujitsu gives local-memory load-to-use latency as
135.5–144.5 ns and peak read/write bandwidth as 256/128 GB/s per MAC (§10.2).

## Cache-line size and address formulas

Both L1 and L2 use 256-byte cache lines (§9.2, pp. 75–76).

### L1 index

L1I and L1D are 64 KiB, 4-way VIPT caches. Their documented index is:

```text
L1_index(A) = (A mod 16,384) / 256
```

The index therefore spans a 16 KiB virtual-address interval. With 4 KiB pages,
address bits `[13:12]` can create synonyms; A64FX resolves those synonyms in
hardware. This 16 KiB period is an L1 indexing property, not an HBM-channel
interleave specification.

### L2 bank and set index

L2 is an 8 MiB, 16-way PIPT cache shared by the CMG. It has two banks selected
by physical-address bit `PA[8]`, so adjacent 256-byte lines alternate banks.
The manual explicitly says the set index is hashed to reduce inter-process
conflicts:

```text
H = PA[36:34] xor PA[32:30] xor PA[31:29]
    xor PA[27:25] xor PA[23:21]

L2_index[10:0] = (H << 8) xor PA[18:8]
```

The ranges in `H` are three-bit fields. Consequently, changing a high physical
address bit can change the low L2 set index, and equal virtual strides need not
produce a simple repeating set sequence once physical-page placement and carry
propagation are included. “Address coloring” is therefore concrete on A64FX
even before requests reach the MAC: buffer placement controls both the L2 bank
bit and this physical-index hash.

## Hardware-prefetch addressing

Section 11.5 (pp. 84–88) documents a 16-entry Prefetch Queue (PFQ) in each
core. Stream-detect mode learns ascending or descending continuous streams.
Its initial address offset and prefetch distance are 256 bytes; it increases
prefetch distance in 256-byte steps and can issue both L1 and L2 prefetches.
Keeping a kernel's reads as one sequential 256-byte-line stream therefore uses
fewer PFQ entries and is easier to predict than several streams separated by
kilobytes.

The manual also says stream-detect addresses are rounded to cache-line units,
but its following sentence says the lower seven address bits are ignored. That
sentence is inconsistent with the repeatedly documented 256-byte line size,
which would normally correspond to eight bits. Code should rely on the
256-byte line size and measured behavior rather than interpreting the “seven
bits” sentence as a separate 128-byte mapping rule.

## What the manual does not disclose

The manual does **not** publish the physical-address mapping for HBM2 channels,
pseudo-channels, banks, bank groups, rows, or columns. It also does not map
physical-address bits to MAC scheduler partitions. Therefore:

- the published formula can predict L2 bank and set coloring;
- it cannot prove that a particular stride collides in an HBM bank or channel;
- an observed placement-sensitive bandwidth change can arise in L2 indexing,
  MIB occupancy, MAC scheduling, undocumented HBM mapping, prefetch behavior,
  or a combination of them.

Use “memory-hierarchy address color” unless counters or a dedicated experiment
isolate the collision below L2.

## HBM channel/bank discovery benchmark

[`bench_hbm_color`](../dequant-pipe/bench_hbm_color.c) is a read-only probe for
the undocumented mapping. It pins 1--12 readers in one CMG, first-touches one
arena there, and changes exactly one placement variable while every reader
executes four 64-byte SVE loads per 256-byte line. It reports median and best
bandwidth as CSV. Useful experiments are:

```sh
# Relative spacing, one address bit, and common-base sweeps.
a64fx/dequant-pipe/bench_hbm_color --cores 12 --mib 240 \
  --min-skew-kib 0 --max-skew-kib 64 --step-bytes 256
a64fx/dequant-pipe/bench_hbm_color --bit-sweep --max-skew-kib 1024
a64fx/dequant-pipe/bench_hbm_color --sweep-base --fixed-skew-kib 0 \
  --max-base-kib 2048 --step-bytes 16384

# Compare transparent-huge-page and base-page advice.
a64fx/dequant-pipe/bench_hbm_color --page-mode thp --bit-sweep
a64fx/dequant-pipe/bench_hbm_color --page-mode base --bit-sweep

# Use Fujitsu XOS/libmpg 2 MiB hugetlbfs pages.
LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
XOS_MMM_L_HUGETLB_FALLBACK=0 \
  a64fx/dequant-pipe/bench_hbm_color --page-mode xos

# Sample independent allocation epochs and retain machine-readable output.
cd a64fx/dequant-pipe
EPOCHS=20 OUTPUT=hbm-color-epochs.csv ./run_hbm_color_epochs.sh
```

The probe prints its virtual arena address and the mapping's `KernelPageSize`,
`MMUPageSize`, and `AnonHugePages` values from `/proc/self/smaps`. Linux
normally withholds PFNs from unprivileged `/proc/self/pagemap`, so virtual bits
above the page offset cannot be substituted into the physical L2 formula.
Controlled huge pages or privileged PFN collection are needed to infer a
physical HBM bit function rather than merely find a good allocation.
The probe explicitly prints `core_start_pfns` and `physical_address_bits`.
`masked` means the pagemap entry is present but Linux cleared its PFN field;
the bandwidth data remains useful for finding good allocations, but not for
regressing a physical-address bit function.

`run_hbm_color_epochs.sh` starts a fresh process and allocation for every
epoch. Classify each epoch by bandwidth first, then compare PFNs when they are
available. At least two independently varying physical address bits are
required to test a four-resource hypothesis. For each candidate bit pair,
group the twelve stream starts by the two-bit value and test whether balanced
groups predict the approximately 229 GB/s state. Reject candidates that only
fit one allocation or one CMG.

An early cross-process comparison found about 120 GB/s at 0/32 KiB gaps and
228.84 GB/s at 16 KiB. Controlled sweeps inside one allocation subsequently
found allocations where every tested gap was about 229 GB/s and others where
every gap was about 120 GB/s. The 16 KiB conclusion was therefore confounded
by physical allocation/page placement. The supported result is that placement
can produce a roughly twofold bandwidth state; the responsible address bits
and whether the conflict is in L2, MIB/MAC, or HBM remain unknown.

With XOS 2 MiB hugetlbfs backing, a 240 MiB twelve-core sweep measured
229.712 GB/s at zero skew and 227.759--228.641 GB/s for 16--64 KiB skews.
`smaps` confirmed 2048 KiB kernel and MMU pages. This removes the earlier
all-slow allocation state in this run and reinforces that 16 KiB skew is not
the selector; controlled large-page placement is the useful configuration.

Operational rules:

1. Bind and first-touch the arena in the target CMG.
2. Preserve 256-byte cache-line and kernel-layout alignment.
3. Sweep common base placement separately from inter-stream spacing.
4. Repeat across allocations, page policies, CMGs, and node images.
5. Use physical addresses and PMU evidence before naming an HBM selector.

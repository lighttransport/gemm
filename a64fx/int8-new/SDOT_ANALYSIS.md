# SDOT Behavior Analysis

## SDOT Instruction Semantics

`SDOT Zda.S, Zn.B, Zm.B` computes:
```
for i = 0 to 15:  # 16 int32 lanes
    Zda.S[i] += Zn.B[4*i] * Zm.B[4*i]
              + Zn.B[4*i+1] * Zm.B[4*i+1]
              + Zn.B[4*i+2] * Zm.B[4*i+2]
              + Zn.B[4*i+3] * Zm.B[4*i+3]
```

## What We Want to Compute

For C[m, n] = sum_k A[m,k] * B[n,k]:
- Lane i computes C[m, n_i]
- Over multiple iterations, accumulate: A[m,0:3]⋅B[n_i,0:3] + A[m,4:7]⋅B[n_i,4:7] + ...

## Current A Packing (Row-Major)

Apack: [row0: 256 bytes][row1: 256 bytes]...[row4: 256 bytes]

With `ld1rqb z24.b, p0/z, [x13]` at offset k:
- Loads bytes [k, k+1, ..., k+15]
- Replicates to: [k:k+15, k:k+15, k:k+15, k:k+15]
- Lane 0 (bytes 0-3): A[m, k], A[m, k+1], A[m, k+2], A[m, k+3] ✓
- Lane 1 (bytes 4-7): A[m, k+4], A[m, k+5], A[m, k+6], A[m, k+7] ✓
- Lane 4 (bytes 16-19): A[m, k], A[m, k+1], A[m, k+2], A[m, k+3] ✓ (same as lane 0)

**This is correct!** All lanes get the appropriate A values.

## Current B Packing (WRONG!)

Bpack[k][lane] = B[n0+lane][k]

With `ld1b z25.b, p0/z, [x5]` at k:
- Loads [B[n0,k], B[n0+1,k], B[n0+2,k], ..., B[n0+63,k]]
- Lane 0 (bytes 0-3): [B[n0,k], B[n0+1,k], B[n0+2,k], B[n0+3,k]] ✗

**This is WRONG!** Lane 0 should get [B[n0,k], B[n0,k+1], B[n0,k+2], B[n0,k+3]] but instead gets B values from 4 different columns at the same k.

## Correct B Packing

For lane i to compute C[m, n_i]:
- Bytes [4*i, 4*i+1, 4*i+2, 4*i+3] should be [B[n_i, k], B[n_i, k+1], B[n_i, k+2], B[n_i, k+3]]

So B should be packed as:
```
Bpack layout per k-chunk (k=0, 4, 8, ...):
  [B[n0, k:k+3], B[n1, k:k+3], ..., B[n15, k:k+3]] for vector 0 (64 bytes)
  [B[n16, k:k+3], B[n17, k:k+3], ..., B[n31, k:k+3]] for vector 1 (64 bytes)
  [B[n32, k:k+3], B[n33, k:k+3], ..., B[n47, k:k+3]] for vector 2 (64 bytes)
  [B[n48, k:k+3], B[n49, k:k+3], ..., B[n63, k:k+3]] for vector 3 (64 bytes)
```

Each vector load gets 16 columns worth of data (4 bytes per column), for one k-chunk (4 k values).

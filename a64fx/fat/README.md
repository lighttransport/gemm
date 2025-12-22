# 2-Pass Flash Attention for A64FX

OoO テストの洞察を活かした Flash Attention タイル実装。

## 設計概要

```
Pass 1: S = Q @ K^T + row_max
  ↓ S を L1 に保存
Pass 2: O = softmax(S) @ V
  ↓ O を出力
Normalize: O /= rowsum
```

## OoO テストからの洞察

1. **FMA密度が重要**: 41%効率 @ 49% FMA比率 → FMA比率を上げる
2. **2-4個の一時レジスタ再利用は問題なし**: OoO がリネーミング
3. **ループ展開が重要**: 分岐予測があっても展開が有利
4. **命令発行幅がボトルネック**: FMA以外の命令を減らす

## パラメータ

```
BR = 4    (Query行数)
BC = 64   (Key/Value列数)  
D  = 64   (Head次元)
```

## メモリ配置

```
L1D (64 KB):
  S[4, 64] = 1 KB (Pass1出力 → Pass2入力)

Registers:
  Pass 1: Q[4, 64] = 16 regs (z0-z15)
  Pass 2: O[4, 64] = 16 regs (z0-z15)
          P[4, 16] = 4 regs  (z16-z19, exp結果)
          V[j, 64] = 4 regs  (z20-z23, ストリーム)

L2/Memory:
  K[64, 64] = 16 KB (Pass1でストリーム)
  V[64, 64] = 16 KB (Pass2でストリーム)
```

## Pass 1: S = Q @ K^T

```
Q[4, 64] をレジスタに常駐 (z0-z15)

for j in 0..64:
    Load K[j, 0:64]
    S[0, j] = dot(Q[0], K[j])  // fmul + 3 fmla + faddv
    S[1, j] = dot(Q[1], K[j])
    S[2, j] = dot(Q[2], K[j])
    S[3, j] = dot(Q[3], K[j])
    Store S[0:4, j]
    Update row_max

命令/j: 4 LD + 16 FMA + 4 FADDV + 4 STR + 4 FMAX ≈ 32
FMA比率: 16/32 = 50%
```

## Pass 2: O = softmax(S) @ V

### exp 償却戦略

```
S を 16 要素チャンク (BC/4 = 4 chunks) で処理:

for chunk in 0..4:
    Load S[0:4, chunk*16:(chunk+1)*16]  // 4 vectors
    
    // exp 計算 (4 行分)
    P[0] = exp(S[0] - m[0])
    P[1] = exp(S[1] - m[1])
    P[2] = exp(S[2] - m[2])
    P[3] = exp(S[3] - m[3])
    
    // P の各要素を再利用 (16 j × 1回)
    // V の各行を 4 回使用 (4 O行)
    for j in 0..16:
        Load V[chunk*16+j, 0:64]
        O[0] += P[0,j] * V[j]  // 4 FMA
        O[1] += P[1,j] * V[j]  // 4 FMA
        O[2] += P[2,j] * V[j]  // 4 FMA
        O[3] += P[3,j] * V[j]  // 4 FMA

exp/chunk: 4 行 × 16 要素 = 64 exp
FMA/chunk: 16 j × 4 行 × 4 vectors = 256 FMA
exp 償却: 1 exp → 4 FMA使用 (D/16 = 4)
```

### レジスタ時間多重化

```asm
// z24-z27 を broadcast 用に再利用
// FMA 発行後、即座に上書き可能 (OoO がリネーミング)

dup     z24.s, z16.s[0]     // P[0, j]
dup     z25.s, z17.s[0]     // P[1, j]
dup     z26.s, z18.s[0]     // P[2, j]
dup     z27.s, z19.s[0]     // P[3, j]

ld1w    z20-z23             // V[j, 0:64]

fmla    z0-z3, z24, z20-z23 // O[0] += P[0,j] * V
fmla    z4-z7, z25, z20-z23 // O[1] += P[1,j] * V
// z24, z25 は次の j で再利用可能
...
```

## FLOPs 分析

```
Pass 1 (Q @ K^T): 4 × 64 × 64 × 2 = 32,768
exp:             4 × 64 × 8      = 2,048
Pass 2 (P @ V):  4 × 64 × 64 × 2 = 32,768
------------------------------------------
Total:                            67,584 FLOPs
```

## サイクル予測

OoO テストで 0.83 FMA/cycle を達成:

```
Pass 1: 1024 FMA → 1024 / 0.83 = 1233 cycles (理想)
Pass 2: 256 exp + 1024 FMA → ~1500 cycles

Total: ~2700 cycles
予測 GFLOPS: 67,584 / 2700 × 2.0 = 50 GFLOPS
予測効率: 50 / 128 = 39%
```

## ビルドと実行

```bash
# GCC
make
./flash_2pass 10000

# Verbose output
./flash_2pass 1000 1

# Fugaku
make fugaku
```

## ファイル構成

```
flash_attn_2pass.h   - ヘッダ
pass1_qkt_rowmax.S   - Pass 1 ASM
pass2_softmax_pv.S   - Pass 2 ASM
flash_attn.c         - C ラッパー + リファレンス
main.c               - ベンチマーク
Makefile
```

## 今後の最適化

1. **Pass 1 の FADDV 削減**: 部分和をレジスタで保持
2. **V の double buffering**: LD レイテンシ隠蔽
3. **exp の INT/FP 並列**: FSCALE を整数演算で置換
4. **より大きな BR**: O の行を L1 に spill して BR=8 に拡張

了解。ここでは Mr×Nr = 5×64（本命）をベースに、A64FX/SVE512 の int8 sdot 前提で、
	1.	メモリレイアウト（pack 形式）
	2.	演算の疑似コード（microkernel）
	3.	外側ブロッキング（macro-kernel）

を「そのまま実装に落とせる」粒度で書きます。
（sdot の 並行和/横方向reduce不要 という条件に合わせて、N方向は lane に固定します）

⸻

0. 記号と前提
	•	行列積：C[ M × N ] = A[ M × K ] × B[ N × K ]^T
	•	ML/attention 的には：S = Q × K^T と同型
	•	K = d = 256（縮約軸）
	•	Nr = 64（SVE512の int8 64 lanes に合わせて N=64 列を一気に処理）
	•	Mr = 5（Q側 5行を同時）
	•	出力 C は int32（accum を int32 で保持）

⸻

1) パック済みメモリレイアウト

1.1 B パック（N=64 ブロック、K=256）

狙い：sdot zAcc.s, zA.b, zB.b を回すとき、
zB.b がそのまま **「N方向 64列分」**として lane に載る。

なので B は **「K方向に沿った 64B ベクトルの列」**として置くのが最も単純です。

Bpack の形（推奨）
Bpack[n0 : n0+63, k0 : k0+255] を
	•	k を 0..255
	•	1ステップは 64B (= int8×64)

として、連続メモリに：

Bpack layout (for one N-block of 64):
  for kchunk in 0..255 step 64:
      store 64 bytes vector: B[n0 + lane, kchunk + t]  (lane=0..63, t fixed per load group)

より具体的に、実装で扱いやすい表現にすると：
	•	Bpack は [K][Nr] の K-major（ただし K は byte で持つ）
	•	アドレスは Bpack + k*Nr + lane

つまり：

Bpack[k][lane] = B[n0 + lane][k]

で、K方向に連続。

これでカーネル側は ld1b zB, [pB, #k] のように k を進めるだけになります。

⸻

1.2 A パック（M=Mr 行、K=256）

A は N方向のような lane 並列は無いので、素直に 行連続が楽です。

Apack layout (for one M-block of Mr=5):
  Apack[m][k] = A[m0 + m][k]

K=256 は 64B×4 なので、ロードは 4 回。

⸻

1.3 C（出力）の置き方

C は行列 [Mr × Nr] のブロック出力。
	•	microkernel 内では Zレジスタ int32 accum に保持
	•	最後にストア（または後段で fp16/fp32 に変換）

レイアウトは通常の row-major でも col-major でもいいですが、ここでは row-majorを例にします：

C[(m0+m) * ldc + (n0+lane)]


⸻

2) microkernel の疑似コード（Mr=5, Nr=64, K=256）

2.1 使うレジスタのイメージ
	•	zB0..zB3 : B の 64B ベクトル（int8×64）
	•	zA : A の 64B ベクトル（int8×64） ※各行ごとにロード
	•	zAcc[m][v] : int32 accum
	•	Nr=64 を int32 lane(16) に分けるので v=0..3（4本）
	•	m=0..4（5行）
	•	合計 20本

⸻

2.2 sdot ループの構造

sdot は 4要素ぶん積和/命令（概念的）なので、K=256 は 64 ステップに相当。
ただし B と A は 64B=64要素を一気にロードするので、実装では
	•	kchunk = 0,64,128,192 の 4 回ロード
	•	各 chunk 内で sdot を回して K=64 を消化

…という二重ループの形が扱いやすいです。

⸻

2.3 疑似コード（SVE風、かなり実装寄り）

// Microkernel: C[Mr x Nr] += Apack[Mr x K] * Bpack[Nr x K]^T
// Mr=5, Nr=64, K=256
// Apack: [Mr][K] int8
// Bpack: [K][Nr] int8  (K-major, 64 bytes per k)
// C: int32, row-major with ldc

void ukernel_5x64_k256_i8s32(
    const int8_t* Apack,     // base of 5x256
    const int8_t* Bpack,     // base of 256x64 (K-major)
    int32_t* C, int ldc)
{
    // ---- accumulators: 5 rows x 4 vectors (each vector = 16 int32 lanes) ----
    // zAcc[m][v] represent 16 columns each; total 64 columns.
    int32x16_t acc[5][4];
    for (int m = 0; m < 5; ++m)
      for (int v = 0; v < 4; ++v)
        acc[m][v] = 0;

    // ---- loop over K in 64-byte chunks (kchunk = 0,64,128,192) ----
    for (int kchunk = 0; kchunk < 256; kchunk += 64) {

        // Load B vectors for this kchunk.
        // Layout: Bpack[k][lane] contiguous in lane for each k.
        // We need 64 bytes worth of B at each k, but sdot consumes 4 k-steps per op conceptually.
        // A practical way is to treat Bpack as bytes and let sdot pick groups; here we show it abstractly.
        int8x64_t b = load_i8x64(&Bpack[kchunk * 64]); 
        // ^ This assumes Bpack is stored as [kchunk][lane], contiguous 64B per kchunk,
        //   i.e. Bpack base points at kchunk start (kchunk/64 entries). Adjust indexing as per actual packing.

        // Load A vectors for each row (64 bytes each row at this kchunk)
        int8x64_t a0 = load_i8x64(&Apack[0 * 256 + kchunk]);
        int8x64_t a1 = load_i8x64(&Apack[1 * 256 + kchunk]);
        int8x64_t a2 = load_i8x64(&Apack[2 * 256 + kchunk]);
        int8x64_t a3 = load_i8x64(&Apack[3 * 256 + kchunk]);
        int8x64_t a4 = load_i8x64(&Apack[4 * 256 + kchunk]);

        // Now perform dot over this 64-byte chunk (which represents 64 elements of K).
        // Conceptually, we need 64/4 = 16 "sdot steps" per chunk.
        // In actual SVE asm, you'd use sdot with the right arrangement so each instruction advances 4 k-elements.
        for (int t = 0; t < 16; ++t) {
            // b_t is the appropriate view/slice of b for this t (depends on your pack + sdot form).
            // Here we keep it abstract as "sdot_step".
            acc[0][0..3] = sdot_step(acc[0][0..3], a0, b, t);
            acc[1][0..3] = sdot_step(acc[1][0..3], a1, b, t);
            acc[2][0..3] = sdot_step(acc[2][0..3], a2, b, t);
            acc[3][0..3] = sdot_step(acc[3][0..3], a3, b, t);
            acc[4][0..3] = sdot_step(acc[4][0..3], a4, b, t);
        }
    }

    // ---- store accumulators to C ----
    // Each acc[m][v] corresponds to 16 columns; v=0..3 => 64 columns
    for (int m = 0; m < 5; ++m) {
        store_i32x16(&C[(m * ldc) +  0], acc[m][0]);
        store_i32x16(&C[(m * ldc) + 16], acc[m][1]);
        store_i32x16(&C[(m * ldc) + 32], acc[m][2]);
        store_i32x16(&C[(m * ldc) + 48], acc[m][3]);
    }
}

超重要ポイント
	•	Bpack の添字は 実装の pack 方式で変わるので、上の Bpack[kchunk * 64] は「概念」です。
	•	実機の sdot は「b のどの byte グループを使うか」が命令形式（laneの取り方）に依存するので、sdot_step() を抽象化しました。
	•	ただし、あなたの前提「並行和なし」「事前pack済み」なら、Bpack を sdot の期待する並びに合わせるのが正攻法です。

⸻

3) pack 具体例（そのまま作れる形）

3.1 Bpack 作成（K-major, 64B/step）

// pack one N-block (64 columns): Bblock is [64][256] int8, row = n, col = k
// produce Bpack as [256][64] int8 (K-major), contiguous 64 bytes per k
void pack_B_64x256_kmajor(int8_t* Bpack, const int8_t* B, int ldb, int n0)
{
    // B[n][k] with n in [n0..n0+63]
    for (int k = 0; k < 256; ++k) {
        for (int lane = 0; lane < 64; ++lane) {
            Bpack[k*64 + lane] = B[(n0 + lane)*ldb + k];
        }
    }
}

これを使うと microkernel は k を回すだけで B を ld1b できます。

⸻

3.2 Apack 作成（Mr=5）

void pack_A_5x256_rowmajor(int8_t* Apack, const int8_t* A, int lda, int m0)
{
    for (int m = 0; m < 5; ++m) {
        for (int k = 0; k < 256; ++k) {
            Apack[m*256 + k] = A[(m0 + m)*lda + k];
        }
    }
}


⸻

4) 外側（macro）ブロッキング：L2→L1 再利用の作り方

あなたの条件だと 1core あたり L2 帯域 ≒ 66.7GB/s なので、
B を L1 に置いて Mr 回使い回すのが効きます。

典型のループ：

for (m0 = 0; m0 < M; m0 += 5) {
  pack_A_5x256(Apack, A, lda, m0);

  for (n0 = 0; n0 < N; n0 += 64) {
    pack_B_64x256(Bpack, B, ldb, n0);

    // Cblock points to C[m0:m0+5][n0:n0+64]
    ukernel_5x64_k256_i8s32(Apack, Bpack, &C[m0*ldc + n0], ldc);
  }
}

これで Bpack(256×64=16KB) は L1 に乗り、Mr=5 回の再利用が効きます。

⸻

5) C を L1 退避するか？（結論だけ）

この K=256 固定 で microkernel が acc を最後までレジスタ保持できるなら、
	•	C を L1 に退避して部分和を読む/書くより
	•	acc をレジスタで保持して最後に 1 回 store

の方が基本的に速いです（L2帯域が支配なので、余計な load/store を増やしたくない）。

例外は：
	•	K を分割して複数回 ukernel を呼び、C に加算しなきゃいけない設計（Kc < 256）
	•	その場合だけ C の read-modify-write が発生する

⸻

必要なら次は、あなたの “pack済みで並行和なし” を前提に、sdot が素直に回る Bpack の「4要素グルーピング（t=0..15）」を具体化して、
sdot zAcc.s, zA.b, zB.b[?] 相当の「どの byte をどこに置くか」まで落としたパック規約（実データ配置）を書けます。

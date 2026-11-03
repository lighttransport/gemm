# シームレスな顔 UV テクスチャ生成の設計（逆レンダリング版、2026-11-01）

前提ノート: [ear-seam-delight-mv-texture-2026-11.md](ear-seam-delight-mv-texture-2026-11.md)（以下「耳ノート」）、`resume-vhuman.md`（B3 まで）、承認済みプラン `~/.claude/plans/see-pixal3d-quality-ab-cycles-prompt-md-prancy-dusk.md`（inverse_albedo 案）。
対象画像: `tmp/vhuman-ab-cycles/bench/b3_final_s1.png`（左 B2、右 B3）、`tmp/vhuman-ab-cycles/bench/ears03_s1_atlas.png`。

---

## 0. 結論

1. **現在の焼き込み（`materials.bake_portrait`）は「texel → 画像」方向の逆投影**で、各 texel が各ビューの 1 画素を点サンプルし、重み付き平均する。B3 で足した guard / feather / gain / bg-reject はその上のパッチで、残っている症状（耳甲介のブロック状の引き伸ばし、耳の付け根の段差、耳の上のピンクの染み、ビュー境界の色差）は方式の限界。
2. **置き換えは「画像 → texel」方向の最適化（逆レンダリング）にする。**固定 UV テクスチャ A を未知数にして、各ビューをレンダした結果が画像に一致するように解く。レンダは「画素 → UV 座標 → `grid_sample`」で、ジオメトリとカメラが固定なら**微分可能ラスタライザは不要**（PyTorch の `grid_sample` だけで勾配が出る）。これで次が自動的に起きる。
   - 正面写真で引き伸ばされた耳甲介は、1 画素が多数の texel を覆うので**平均しか拘束しない**（フットプリント対応のミップサンプリング）。ヤコビアンによる棄却を損失側で行う形になり、ブロック状の模様が出ない。
   - テクスチャが全ビューで共有なので、**ビュー境界のシームは構造的に出ない**。露出・色温度・照明差はビューごとの SH9 照明（RGB）に吸収される（gain_match の一般化）。
   - 写真と生成の境界は Poisson 的な合成ではなく、同一の未知数の上の重み差になる。feather も seam leveling も要らない。
3. **PyTorch3D は ROCm でビルドでき、実ジオメトリで動作を確認した**（6 章: 1024² ラスタ 90 ms、被覆 IoU 0.992、ソフトシルエットの頂点勾配あり）。使う場面は (a) ビューごとのラスタ（UV・法線・深度バッファ）、(b) 頂点を動かす段（耳の再フィット）の勾配とソフトシルエット損失。テクスチャ・照明・ワープの勾配そのものは `grid_sample` だけで出るので、段 1〜2 は PyTorch3D がなくても成立する（依存を最小に保つ）。Mitsuba 3 はこのホストでは CPU（LLVM）版のみで、影付きの検証レンダ（低解像度）に限る。
4. 写真の texel の「バイト単位の保護」は**保持できる**: 正面ビューのフットプリントが 1.5 texel 以下で信頼度が高い texel は固定し、その他だけを解く。写真の色に生成ビューの照明が合わせに行く形になる（逆ではない）。
5. 順序: 段 0（診断マップ、半日） → 段 1（純 torch の texopt v1、1〜2 日） → 段 2（ビューごとの粗いワープ + AO + LPIPS、1〜2 日） → 段 3（耳の頂点オフセット。PyTorch3D か輪郭 chamfer） → 段 4（未観測部: 耳テンプレート・ミラー・生成ビューを同じソルバの事前分布として入れる）。

---

## 1. 現状の整理（コードから確認したこと）

| 項目 | 事実 |
|---|---|
| 入力 | `view_0.png`（正面写真）、`view_1/2.png`（Edit-2511 の ±40 生成。実測 +38 / −32°）、各 1024²、RGBA。`exclusion_<i>.png`、`silhouette_<i>.png` |
| ジオメトリ | `geometry.npz`: `captured` (3, 11460, 3) ビューごとの頂点（表情・姿勢込み）、`triangles` 22636、`triangle_uvs`。耳は `ear_dense_*`（5094 頂点、head の `ear_removed_triangles` 2620 面を置換、**同じアトラス UV を共有**） |
| UV | 単一の連結した島、UV の切れ目は **76 エッジ**だけ。耳ノート 1 章 (e) の「島の切れ目」は主因ではない |
| アトラス | 2048²。`skin_basecolor.png` + confidence / coverage / front_protected / generated_support |
| 現在の融合 | texel ごとに `conf = valid × depth 一致(2 mm, 512 raster) × facing × mask`、`estimate()` で SH 照明を除いた albedo を `fuse_samples` で重み付き平均 → `harmonic` 補完 → `feather` → 正面 texel をバイト単位で復元（`protect_front`）。B3 では guard 3 px / feather 6 px / gain / bg-reject / front-blend |
| 耳 | `ears02`: `ear_texture.py` + clay 条件付き Edit-2511 で合成。`ear_rebake_mask`（22 万 texel）内は ears02、外は B3 の rebake を 4 px Gaussian で合成（`make_ears03.py`） |
| 既存の部品 | `mv_delight.sh9`、`torch_reference.ggx`（torch の GGX）、`occlusion.py`、`mv_texture.seam_energy`、`rebake.py` の統計、`stage5_report.score`（LPIPS/DISTS）、`skin_templates`（領域マップ）、rocm venv に lpips |
| 環境 | ROCm torch 2.11、gfx1201 16 GB（Blender B レンダが 13 GB、GPU ジョブは 1 つずつ）。PyTorch3D / Mitsuba / nvdiffrast 未導入 |

### 1.1 画像からの所見（b3_final_s1）
- 右列（B3）で消えたもの: 頬〜顎の階段状境界、耳たぶ下の白斑、首の白い楔。
- 残るもの: ① 耳甲介・対耳輪の内側の引き伸ばされたブロック（正面写真の texel が斜めに投影）② 耳の付け根の周囲の色の輪（ears02 と rebake の合成境界、または dense ear と head の**メッシュ接合部の法線不連続**）③ 耳の上・後ろのピンクの丸い染み（ears02 の `anatomical_tint` か耳合成の赤みが `ear_rebake_mask` の Gaussian で外に出た可能性）④ 側頭部〜後頭部の平坦な塗り（未観測）。

---

## 2. 設計: `inverse_texture.py`（texopt）

### 2.1 前処理（numpy、ビューごとに 1 回、キャッシュ）
既存の `reference.rasterize` でビューごとに 1024² のラスタを取り（22k 三角形の Python ループは数十秒。PyTorch3D が通れば置換）、画素ごとに次を保存する。
- 三角形 id、重心座標 → **UV 座標 (u, v)**、世界法線 n、視線 v、深度
- **UV のヤコビアン** J = ∂(u,v)/∂(x,y)（隣接画素の UV 差分、三角形境界をまたがないように id で分ける）→ フットプリント = 特異値 × texres。ミップレベル λ = log2(最大特異値 × texres)
- 重み w_v(px) = 既存の conf（depth 一致、facing、exclusion、bg-reject）× シルエット距離ランプ（Callieri の項。guard と同じ距離変換）× 画像縁からの距離
- 正面ビューの**保護マスク**: w_0 > .5 かつ最大特異値 × texres ≤ 1.5 の画素が覆う texel → `pinned`

### 2.2 未知数
| 未知数 | 形 | 初期値 |
|---|---|---|
| アルベド A | ピラミッド和 A = Σ_l up(A_l)、l = 256/512/1024/2048（linear RGB、log 空間で最適化） | 現在の B3 の basecolor を分解 |
| 照明 L_v | ビューごとの SH9 × RGB（露出・WB を含む） | `mv_delight.sh9` の IRLS 解。正面の DC は 1 に固定（ゲージ = 校正済み露出を保つ） |
| 鏡面 | 領域ごとの roughness / F0（`skin_templates` の領域）、ビューごと共通 | manifest の .55 / .028 |
| AO | texel ごとの半球可視率（`occlusion.py` かレイキャストで事前計算、固定） | 計算値 |
| ワープ W_v（段 2） | ビューごとの 16×16 の 2D 変位格子（bicubic 補間） | 0 |
| 耳の頂点オフセット δ（段 3） | 耳バンドの頂点 × 3 | 0 |

### 2.3 順モデル（1 ビュー、画素 px）
```
uv, n, v = 前処理(px)                          # 固定
a = sample_pyramid(A, uv, λ(px))               # grid_sample をレベルごとに、λ でトリリニア
E = max(SH9(n) · L_v, 0)                       # 拡散放射照度(RGB)
diffuse, spec = torch_reference.ggx(a, n, v, l_v, rough(region), f0(region))
R_v(px) = a * E * AO(uv) + spec * E_dir        # 線形
I_v'(px) = I_v(px + W_v(px))                   # 段 2: 画像側をワープ
```
損失:
```
L = Σ_v Σ_px w_v · | tone(R_v) − tone(I_v') |₁          # tone = linear→sRGB、写真は高重み、生成は 0.3
  + λ_lp Σ_v LPIPS(crop(R_v), crop(I_v'))                 # 256² クロップ、ビューごと 4〜8 枚。生成ビューの 1〜3 px ずれでぼけない
  + λ_lap Σ_l | Lap_mesh(A_l) |²                          # メッシュ隣接(3D kd-tree, 法線一致 > .7)のラプラシアン。UV の切れ目をまたぐ
  + λ_pin | A − A_B3 |² on pinned                         # 実運用では固定(勾配を 0 にする)。バイト単位の保護
  + λ_leash | log A − log A_B3 |₁ on observed             # 観測済みだが保護外。暴走防止
  + λ_W | Lap(W_v) |²  + λ_δ | Lap(δ) |²  （段 2, 3）
```
- 未観測 texel は、データ項がなくラプラシアン項だけで決まる（harmonic 補完と同じ解になる）。段 4 で事前分布を足す。
- 鏡面の `E_dir` は SH の主方向（`log_direction`）1 本で近似。写真のハイライトを albedo に焼かないための項で、精密さは要らない。

### 2.4 なぜ既存の症状が消えるか
| 症状 | 機構 |
|---|---|
| 耳甲介のブロック | 正面画素の λ が 3〜5（8〜32 texel を覆う）なので、その画素は A の粗いレベルしか拘束しない。細部は ±40 ビューか事前分布から来る |
| ビュー境界の色差 | A が共有、L_v がビューごと。境界では両側の R_v が同じ A から出る |
| 耳の付け根の輪 | ears02/rebake の合成を廃止し、耳 texel も同じソルバで解く（ears02 は初期値と leash にだけ使う） |
| ピンクの染み | 出どころがソルバ外（合成耳の tint）なら消える。ソルバ内なら段 0 の source-ID マップで特定 |
| 影の焼き込み | AO × E で説明されるので、耳甲介の暗さは A に入らない |

### 2.5 計算量
3 ビュー × 1024² ≈ 3 M 画素、2048² ピラミッド。`grid_sample` の順逆で 1 反復 100 ms 程度、Adam 300〜800 反復で**数分 / 被写体**。LPIPS は 1024² を直接入れず 256² クロップで。VRAM は 4 GB 未満の見込み。Blender レンダと同時に走らせない。

---

## 3. 段階と完了判定

| 段 | 内容 | 学習 | 完了判定 |
|---|---|---|---|
| 0 | 診断マップ（`scripts/texture_debug.py`）: source-view ID、λ（フットプリント）マップ、保護マスク、UV 切れ目、**平坦な灰色 albedo での耳レンダ**（輪が残れば法線・接合部の問題で、テクスチャでは直らない） | なし | 残る 4 症状それぞれの原因が 1 つに決まる |
| 1 | `inverse_texture.py` v1: 固定ジオメトリ、A ピラミッド + L_v + 領域鏡面 + AO、front pin、メッシュラプラシアン。`rebake.py --solver inverse` から呼ぶ。出力は `skin_basecolor.png` 互換 + `inverse_texture.json`（損失推移、L_v、pinned 数、leash 逸脱量） | なし | `seam_ratio` < 1.2（B3 1.47）、`seam_energy` 半減、pinned texel がバイト一致、耳クロップでブロックが消える、held-out ±90 の LPIPS が B3 以下 |
| 2 | v2: ビューごとの 16×16 ワープ W_v（Color Map Optimization の torch 版）、LPIPS クロップ、生成ビューの重み .3 | なし | 耳輪の二重線が消える。±40 ビューの再投影残差が下がる |
| 3 | v3: 耳バンドの頂点オフセット δ。シルエット損失は (a) PyTorch3D が通れば SoftSilhouetteShader、(b) 通らなければ**耳輪郭の頂点をパーシングの耳マスク輪郭へ chamfer**（純 torch） | なし | 正面オーバーレイで耳の輪郭が一致。段 2 の W_v が小さくなる |
| 4 | 未観測部の事前分布: 耳テンプレート（GNM UV で一度作る）への leash、左右ミラー、生成 ±90 / 後方ビューを低重みの観測として追加（VDA-lite が来たらそのまま入る） | なし | 後頭部・耳裏の比較シートで B3 より良い |
| 5 | delight の統一: 段 1〜2 で L_v + AO が全ビュー共通の照明モデルになるので、`mv_delight` は初期値にだけ使う。OpenDelight（GPL、重みは要確認）は評価用に比較 | なし | ビュー境界 ΔE が小さい。写真との skin L* 一致を維持（校正済み露出） |

段 1 の時点で B3 の後処理（gain_match、feather、front_blend、`make_ears03` の合成）は不要になる見込み。残す opt-in は guard（重みマスクとして流用）と bg-reject。

---

## 4. 微分可能レンダラの使い分け

| 用途 | 道具 | 理由 |
|---|---|---|
| テクスチャ・照明・ワープの勾配 | **素の torch**（`grid_sample` + `torch_reference.ggx`） | ジオメトリ固定なら十分。ROCm で今日動く。依存なし |
| ビューごとのラスタ（id / 重心 / 深度） | 既存 `reference.rasterize`（numpy） → PyTorch3D `MeshRasterizer` が通れば置換 | 速度だけの問題。`torch_reference.backend_probe` が既に PyTorch3D を検出する作り |
| 頂点の勾配、ソフトシルエット（段 3） | **PyTorch3D**（BSD） | `setup.py` に ROCm 分岐あり（auto-hipify）。ビルド結果は 6 章 |
| 影を含む検証レンダ、AO の精度確認 | **Mitsuba 3**（BSD、`llvm_ad_rgb`、CPU） | このホストに CUDA/OptiX はないので CPU のみ。512² で A・L_v を固定して影付きの再投影残差を見る用途に限る |
| 使わない | nvdiffrast / nvdiffrec（NVIDIA 非商用）、redner（ライセンス未確認） | ライセンス方針 |

---

## 5. 実装メモ

- 新規: `server/vhuman/reconstruction/inverse_texture.py`（前処理、ピラミッド、損失、ソルバ、書き出し）、`tmp/vhuman-ab-cycles/scripts/texture_debug.py`（段 0）。`rebake.py` に `--solver inverse` と `--iters`。
- 再利用: `materials.bake_portrait` の conf 計算（関数に切り出す）、`texture_completion.harmonic` の kd-tree 隣接（ラプラシアン行列に流用）、`mv_delight.sh9`、`torch_reference.ggx`、`skin_templates` の領域、`occlusion.py`、`rebake.stats`、`mv_texture.seam_energy`、`stage5_report.score`。
- 提供性（provenance）: `skin_material.json` に `fusion.method = 'inverse texture optimisation'`、pinned texel の sha、L_v、leash 逸脱の統計を記録。`validate_candidate` を通す。正面保護の意味は「pinned texel はバイト一致、保護外の観測 texel は低周波の補正量を記録」と明記する。
- 耳: 段 1 では `ear_rebake_mask` 内の初期値を ears02 にし、leash を弱めに（生成由来なので）。段 4 でテンプレートに置き換える。
- 実行環境: `env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib PYTHONPATH=$PWD tmp/vhuman-rocm-venv/bin/python`。Blender レンダ中は走らせない（GPU ロック `tmp/pixal3d/device-locks/rocm-0.lock` を取る）。
- 検証レンダ: `render_matrix.py` の neutral セット（front / ±40 / ±90 / ear_left / ear_right、studio 校正、1024²/128）、比較シートは B3 vs 新候補 `B4`。

---

## 6. PyTorch3D の ROCm ビルド結果（2026-11-01、成功）

- ソース: `facebookresearch/pytorch3d` main（0.7.9）。`uv build --wheel --no-build-isolation --python tmp/vhuman-rocm-venv/bin/python`、`PYTORCH_ROCM_ARCH=gfx1201 FORCE_CUDA=1`、ROCM_HOME `/opt/rocm/core-10.1`。setup.py の ROCm 分岐（auto-hipify、hipcc）だけで通り、パッチ不要。警告のみ（hipError_t の nodiscard）。
- 成果物: `tmp/wheels/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl`（出典とインストール手順は同じ場所の `.source.txt`）。venv には**まだ入れていない**（検証は wheel を展開した PYTHONPATH で実施）。
- 検証（s1 ears03 の実ジオメトリ、RX 9070 XT）:

| 項目 | 結果 |
|---|---|
| ハードラスタ 1024²（22.6k 面） | 90 ms（`reference.rasterize` の numpy ループより桁違いに速い） |
| 被覆の一致（3 ビュー、256² で `reference.rasterize` と比較） | IoU 0.992 / 0.992 / 0.992、深度範囲一致 |
| ソフトシルエット 512² + 頂点勾配（`SoftSilhouetteShader`、faces_per_pixel 8） | 43 ms（初回 0.74 s）、勾配あり |
| UV 補間 → `grid_sample`（2048² テクスチャ）の順逆 | 94 ms、1.09 M texel に勾配 |
| ピーク VRAM | 0.46 GB |

- **カメラ変換**（`reference.Camera` → `PerspectiveCameras`）: `S = diag(-1, 1, -1)`、`R_p3d = (S R)ᵀ`、`T = -(S R) o`、`focal_length=(f, f)`、`principal_point=(cx, cy)`、`in_ndc=False`、`image_size=(H, W)`。これで 3 ビューとも被覆・深度が一致した（他の符号は IoU が大きく落ちる）。`inverse_texture.py` ではこの変換を関数にして単体テスト（IoU > .99）を付ける。
- 結論: 段 1 の前処理（ラスタ、UV・法線・深度バッファ）と段 3（ソフトシルエット）の両方を PyTorch3D で行える。`torch_reference.backend_probe` は PyTorch3D 検出時に `reference='pytorch3d'` を返す作りなので、そこに合流させる。Mitsuba 3 は影付き検証用のまま（未導入）。

---

## 7. 段 0・段 1 の実装結果（2026-11-01、s1 / ears03）

実装: `server/vhuman/reconstruction/inverse_texture.py`（PyTorch3D でビューごとの UV・法線・フットプリントを取り、ミップ付き `grid_sample` で A をサンプル。A = B3 + 6 段ピラミッドの補正、ビュー別 SH9 RGB、拡散のみ）、`tmp/vhuman-ab-cycles/scripts/inverse_preview.py`（アルベドの任意ヨー・プレビュー）、`inverse_metrics.py`。出力は `tmp/vhuman-ab-cycles/inverse/s1_*`（`debug/` に source-view ID、フットプリント、pinned、stretched、変化量×4）。300 反復で 1 分台。

**段 0 で分かったこと**
- 正面写真の頭部正対画素のフットプリントは **1.49 texel/画素**。アトラス 2048² は写真の約 1.5 倍の解像度なので、フットプリントの閾値は絶対値ではなく正面の代表値に対する相対値にする必要がある（最初の絶対閾値 1.5 では保護領域の 1/3 しか pin されなかった）。
- 側頭部・首の側面・耳介は、正面からは 3〜30 texel/画素に引き伸ばされる（`best_footprint_log2.png`）。`stretched` マスクは首〜側頭部の帯と耳甲介に出る。耳ノートの「耳内側は正面から取らない」はフットプリントで機械的に判定できる。
- 既存 bake の source-view マップには、**サンプル間隔が粗い領域の縞模様（モアレ）**が出る。点サンプルのスキャッターの副作用で、ヤコビアン無視の限界の可視化になっている。

**段 1 の数値**（再投影残差 = 重み付き sRGB 空間の Huber 残差、照明は評価時に再フィット。start = B3）

| 設定 | 正面 | +40 | −40 | pinned texel |
|---|---|---|---|---|
| B3（start） | .0099 | .0340 | .0380 | — |
| 解く（pin 1.3×、300 反復） | .0110〜.0132 | .0273〜.0270 | .0308〜.0277 | 29 万 |
| 解く（pin 2.0×、採用） | .0108 | .0302 | .0346 | 72 万 |
| −40 を held-out | .0110 | .0273 | **.0379（start .0380）** | 29 万 |
| +40 を held-out | .0116 | **.0339（start .0340）** | .0308 | 29 万 |

- pinned texel はバイト一致を確認（`pinned_bytes_equal: true`）。高周波の保持率 0.99（観測域）/ 1.01（耳）、平均変化 2.3/255 と保守的。`seam_ratio` は 1.19 → 1.25 で改善していない。
- **解釈（正直なところ）**: 学習に使ったビューの残差は 10〜20% 下がるが、**held-out ビューでは改善がない**。±40 の 2 枚はほとんど重ならないので当然の面もあるが、現状は「共有アルベドで照明差を吸収する」効果は出ていても、**側面の残差 0.03 の主因は照明ではなく、生成ビューと GNM のずれ（段 2 のワープ）と生成画像自体の質**と見るのが妥当。プレビュー（`scratchpad/prev_t3.png`、上 B3・下 解）では側頭部の色の粒状感が少し落ち着くが、耳の輪郭周りの見た目はほぼ変わらない。**見た目の改善はまだ示せていない。**
- 未実施: Cycles レンダでの比較（B4 候補の作成、`provenance` の更新、`write_candidates.py` の登録が必要）、s2、AO、ワープ、LPIPS、耳の頂点オフセット。

**次にやること（優先順）**
1. 段 2: ビューごとの 16×16 ワープ格子と生成ビューの LPIPS クロップを足し、held-out ビューの残差が下がるかで判定する（改善しなければこの方式の価値は限定的と見る）。
2. 耳領域だけ（`ear_rebake_mask`）の残差と、耳クロップのプレビューで比較する指標を追加。
3. 効果が出たら B4 候補を作って Cycles で B3 と比較する。

---

## 8. 段 2 の結果: ビューごとのワープと LPIPS（2026-11-01、s1 / ears03、結論: 効果なし）

実装（`inverse_texture.py`）: ビューごとの 16×16 ワープ格子（bicubic、tanh で ±25 px に制限、メッシュ非依存の滑らかさ正則化、正面は固定）で**ターゲット画像側を動かす**。LPIPS（alex）は 256² クロップ（学習時はビューごとにランダム 2 枚、評価時は被覆 50% 以上の格子クロップ）で、未観測画素は両画像で同値にして寄与させない。評価は「ワープなし、照明は再フィット、held-out は学習に未使用」。

| 実験（300 反復、lr .003） | 学習ビューの残差 | held-out の残差 | held-out の LPIPS（start .171） | ワープ中央値 |
|---|---|---|---|---|
| ワープのみ（−40 を held-out） | +40: .0340→.0301 | .0380→.0382 | .1716（変化なし） | 0.19 px |
| LPIPS のみ（ワープなし、−40 を held-out） | 変化小 | .0380→.0395 | **.234（悪化）** | — |
| ワープ + LPIPS（−40 を held-out） | 変化小 | .0380→.0394 | **.228（悪化）** | 0.7 px |
| ワープ + LPIPS（+40 を held-out） | 変化小 | .0340→.0357 | **.244（start .178、悪化）** | 0.7 px |
| 学習率 .01（最初の設定） | — | — | 全ビューで発散（データ項 .002→.005） | 1.7 px |

- **ワープ仮説は否定的**: 最適化が選ぶ変位は中央値 0.2〜0.7 px、最大でも 10 px 程度で、残差もほぼ動かない。16×16 の滑らかなずれで説明できる位置ずれは、残差 0.03 の主因ではない。主因は生成画像そのものの質（解像感・色・細部が GNM のテクスチャと一致しない）と見るのが妥当。
- **LPIPS 項は汎化を悪くする**: 学習ビューのクロップを鋭くしようとして、他の向きから見ると悪いテクスチャ（高周波ノイズ）を作る。held-out の LPIPS が 33〜38% 悪化し、ワープの有無に関係しない。→ 既定値を `lam_lpips = 0`、`lr = .003` に変更（ワープは既定でオン、効果は小さいが無害）。
- 学習率は .01 だと LPIPS の勾配ノイズで発散する（6 段ピラミッドの和で実効ステップが大きい）。

**結論**: この形（共有アルベド + 照明 + 滑らかなワープ + 知覚損失）では、B3 に対する held-out での改善が出ない。耳・側頭部の見た目の問題に効くのは、おそらく解法ではなく**入力の質と未観測領域の事前分布**。次の候補は、ソルバの調整ではなく次の 2 点。
1. 段 4（未観測・引き伸ばし領域の事前分布）: `stretched` マスク（耳甲介、首の側面）で、観測ではなく耳テンプレート・左右ミラー・低周波の転写を使う。これは `inverse_texture.py` の `ref`（leash の目標）を差し替える形で入れられる。
2. 耳の頂点オフセット（段 3、PyTorch3D のソフトシルエット）。耳の輪郭が合わないことは、画像側のワープでは直らない。

B4 の Cycles レンダは作っていない（held-out で改善が出ないため）。

---

## 9. 段 4 の結果: 耳の事前分布と、耳の形状の取り違えの発見（2026-11-01、s1）

**発見（本命）: ソルバと既存 bake は、耳をレンダ用の精密な耳ではなく粗い基本メッシュで観測していた。**`geometry.npz` の `captured` / `triangles`（11460 頂点）は耳が粗い基本 GNM のもの。レンダでは `ear_removed_triangles` で基本の耳面を外し、`ear_dense_*`（5094 頂点、`ear_bind_*` で頭に結合、アトラス UV は共通）を足している。`inverse_texture.render_surface` でこの面を組み立てた（基本面のうち外すもの 2620 面は全部一致、`ear_mesh.evaluate` の再現誤差 4e-9）。ビューごとの結合は `full_captured[vi]` から評価する。

| 耳 texel の再投影残差（照明は再フィット、start = B3） | 基本メッシュ | 精密な耳 |
|---|---|---|
| 正面（耳 8.9k 画素） | .0538 | **.0335**（−38%） |
| +40（耳 18k 画素） | .0772 | .0726 |
| −40（耳 17k 画素、held-out） | .0712 | .0732 |

→ 正面の耳は精密な耳の形状でないと写真と合わない。`ear_surface=True` を既定にし、`--no-ear-surface` で従来に戻せる。**既存の `bake_portrait` も同じ取り違えをしている**ので、B 候補の耳の外側の色が、粗い耳の面に写真の耳の色を貼っていた可能性がある（耳ノート 1 章の (c)）。ear_rebake_mask の外に耳由来の色が漏れる原因として確認する価値がある。

**観測の割合**: 耳マスク（24 万 texel）のうち、3 ビューのどれかが見ているのは約 17%。残り 83% は ears02 の合成テクスチャで、どのビューの拘束も受けない（`stretched` に分類されるのは 3% のみ）。つまり「引き伸ばされた観測」より「そもそも観測がない」が支配的。

**事前分布の検証**（held-out は証拠にも数えない形で再実装済み。最初の実装では held-out ビューの命中も「解像済み」に数えていて、事前分布がほぼ適用されなかった）:

| 事前分布（耳マスク内の未解像 texel に leash 目標を置く） | held-out の耳残差（start .0732 / .0726） |
|---|---|
| なし | .0740 / .0728（変化なし） |
| 左右ミラー + 滑らかなゲイン補正（4.3〜4.5 万 texel に適用） | **.0831 / .0820（悪化）** |
| 左右ミラーの低周波のみ（31 texel のぼかし） | **.0810 / .0784（悪化）** |

- **ミラーの事前分布は不採用**。アトラスは (u,v)→(1−u,v) で厳密に鏡映し（頂点誤差 0 texel、メッシュ対称誤差 0.24 mm）だが、精密な耳は左右別々にフィットされているので、同じ UV 座標に同じ構造・色が来るとは限らない。低周波だけ移しても悪化するので、左右の耳の色・陰影そのものが held-out ビューの観測と合わない。
- 学習ビューの耳残差は .0726 → .0603（−17%）まで下がるが、held-out は動かない。前の段と同じ結論。

**残る選択肢**:
1. 耳の合成（`ear_texture.py`、ears02）の改善。83% の未観測部の色と陰影は、結局そこで決まる。解法側では動かせない。
2. 精密な耳を使った `bake_portrait` の再実行（正しい面での写真の耳の色の取り込み）。ears02 は既にこの面で作っているので、差が出るのは ear_rebake_mask の外側への漏れ。
3. 耳の頂点オフセット（段 3）は、精密な耳をシルエットに合わせる用途になる。正面残差が .0335 まで下がっても、耳の輪郭の一致はまだ見ていない。

事前分布のコード（`mirror_reference`、`--ear-prior mirror|mirror-lf`）は実験用に残し、既定は `none`。

---

## 10. 精密な耳での再 bake（2026-11-01、s1 / s2）

実装: `ear_mesh.rendered_surface(geometry, view)`（基本面 − 外す耳面 + 精密な耳。`inverse_texture.render_surface` はこれを呼ぶ）、`rebake.py --ear-surface`（ビューごとの頂点を積んで `bake_portrait` に渡す。`bake_portrait` 自体は無改造）、`make_ears03.py` の環境変数 `REBAKE_DIR` / `OUT_NAME`、`inverse_preview.py ears`（耳の拡大シート）。bake の設定は B3 と同じ（guard 3、feather 6、gain、inner-lip、front-blend、bg-reject）。

- **対照**: 粗い耳で再実行した `rebake/s1/bg_base3` は既存の `bg` と**全 texel 一致**（差 0）。精密な耳の差分だけが効いている。
- **候補**: `candidates/<s>/B/reconstruction/ears04`（ears02 の耳 + 精密な耳での bake を耳マスクの外に合成。`validate_candidate` 通過）。`write_candidates.py` には**未登録**、Cycles レンダは**未実施**。
- bake 統計（s1 / s2）: bleed 7 / 0、ragged 17.9 / 17.6、seam_ratio 1.53 / 2.17（B3: 1.47 / 2.15）、lip_bright 36 / 2146。ほぼ同じ。
- 変化した texel（ears03→ears04、差 >16）: s1 7.2k（耳マスク内 0.4k、耳の周囲 60 texel の帯 6.7k、その他 87）、s2 5.1k（1.3k / 3.7k / 58）。つまり**変化は耳の周囲の帯に限られる**。
- 再投影残差（耳 texel）は ears03 とほぼ同じ（s1 正面 .0335→.0333、s2 .0386→.0385）。耳の中身は ears02 のままなので当然。

**見た目（unlit アルベド、±40 / ±90 の耳拡大）**
- **s1: 改善**。ears03 にあった耳の後ろ・上のピンクの丸い染みと小さな暗い点 2 つ（+90 側）、−90 側の淡いピンクの染みが、ears04 では消える。粗い耳の面に写真の耳の色が貼られていた、という推定と合う。
- **s2: 混在**。+90 側は大差なし。−90 / −40 側では、耳の上に**オレンジの染みが新たに出る**（ears03 では小さな線だけ）。耳の後ろの細いオレンジ線と、耳の下のリング状の模様は両方に残る（B2 以前からの既知）。精密な耳が観測する面が広がり、耳と頭皮の境の暖色の影や髪際の画素を拾ったと推定。未確認。

**判断**: s1 では ears04 を採用してよい。s2 は染みの原因（どのビューの画素か）を source-view ID で確かめるまで採用しない。修正候補は、耳の周囲の帯でだけ B3 の bake を残す（耳マスクの拡張マスクで合成する）か、精密な耳の面に対する髪・暗部の除外を強める。

---

## 11. s2 のオレンジの染みの出どころ（2026-11-01）

`tmp/vhuman-ab-cycles/scripts/smudge_source.py s2 <out>`: ears03 → ears04 で「赤みが増え（R−B が +10 以上）、差 >16」の耳周囲の texel を連結成分に分け、各ビューへ前方投影（bake と同じ向き。面 id 一致で可視判定）して、元画素の色・facing・シルエット距離・パーシングラベルを出す。切り出し画像は `tmp/vhuman-ab-cycles/smudge_s2/blob*_view*.png`。

- 染みは主に 1 か所: アトラスの画像左半分（x 265〜323、y 1356〜1446）、**2693 texel**。色は [244,213,185] → [235,187,154]。もう 1 か所（78 texel、右半分）は、前方投影ではどのビューにも可視にならず、原因は未確認。
- **見ているのは −40°の生成ビュー（view_2）だけ**。正面と +40° は 0 texel。962 texel が **79 画素**に落ちる（約 12 texel/画素に拡大）。facing 平均 **0.14**（ほぼ接線）、シルエットまで 2.7 px、支持（アルファ × 除外）は 1.0。
- その画素は、耳の**上端（耳輪の頂点）の、灰色の髪が耳にかかる境目**。元画素の色は sRGB [191,144,112]（髪の影が混ざった暗いオレンジ褐色）で、パーシングは「耳」68% / 「左耳」32%。髪ではなく耳と判定されているので、除外マスクでは落ちていない。
- bake 後の色は [238,188,153]、周囲は [249,219,191]。`estimate()` が照明を割る（照度を .5〜2 に制限）ので、暗い画素は最大 2 倍に持ち上げられ、さらに橙に寄る。

**原因の整理**: 粗い耳では、この領域は耳として観測されず（頭皮扱い）、−40°の画素が当たらなかった。精密な耳では耳頂点の面がこのビューで見えるが、ほぼ接線（facing 0.14）の数画素だけが約 12 倍に引き伸ばされて、髪境界の混色（暗橙）の画素が 960 texel に広がる。他のビューが見ていないので、重み（facing に比例）が小さくても正規化後の色はその画素の色になる。つまり**バグではなく、bake に最小 facing とフットプリント（拡大率）の制限がないこと**の現れで、s1 では髪境界でこの条件に当たらなかった。

**修正案（未実施）**
1. `bake_portrait` に `min_facing`（0.15〜0.35 でランプ）と、texel → 画素の拡大率の上限（約 6 texel/画素。隣接 texel の投影間隔から計算）を追加。該当 texel は観測外として harmonic 補完に回る。段 1 のソルバが使っている `foot` と同じ量。
2. 耳の周囲の帯だけ B3 の bake を残す合成（変更は帯に限られるので影響は小さい）。
3. 検証: s2 を再 bake（約 11 分）して染みが消えること、s1 の改善（耳の後ろのピンクの染みと暗い点）が残ることを確認する。

---

## 12. 最小 facing と拡大率の上限で s2 の染みを直す（2026-11-01、fast プリセット）

実装: `materials.bake_portrait` に opt-in の `min_facing`（その facing で重み 0、+0.2 で満額）と `magnification_cap`（1 画素あたり texel 数が上限以下で重み 1、2 倍で 0。隣接 texel の投影間隔の最小特異値から計算する `magnification_weight`）。どちらも既定は 0（従来どおり）。`protect_front` が復元する正面だけの bake には適用しない（そこは写真そのもの）。`rebake.py` に `--min-facing`、`--mag-cap`（2048 基準の値。`--res` に合わせて自動で縮尺）、`--res`（fast プリセット = 1024）を追加。

**fast プリセット（`--res 1024`）は 1 回約 100 秒**（2048 は約 11 分）。精密な耳の bake 3 本を並列で回して約 2 分。

s2、染みの領域（atlas x 253〜335、y 1344〜1458）の平均色。R−B が大きいほどオレンジ。B3（粗い耳）の基準は 57。

| 設定（min_facing / mag_cap） | 平均 sRGB | R−B | 領域外の変化 texel（差 >16） |
|---|---|---|---|
| B3（粗い耳） | 242, 211, 185 | 57 | — |
| 精密な耳、修正なし | 233, 192, 163 | 70 | 0（基準） |
| .15 / 6 | 235, 198, 171 | 64 | 2.4k |
| **.25 / 6（採用）** | 234, 201, 177 | **57** | 6.4k |
| .15 / 3 | 235, 200, 173 | 62 | 5.3k |
| .30 / 4 | 228, 195, 169 | 59 | 11.8k |

- **採用: `--min-facing .25 --mag-cap 6`**。染みの領域が B3 と同じ色味に戻り、−90°/−40° の耳拡大でオレンジの染みが消えた。
- 副作用: 領域外で 9.2k texel が変わるが、**全て確信度 80/255 未満（平均 3）**の、ほぼ観測がない texel（側頭部・顎の側面・目の周囲）。接線の数画素に引き伸ばされた色の代わりに補完が入る形。目の周囲の変化は目のメッシュに隠れる領域と思われるが未確認。
- **s1 の改善は維持**: fast で粗い耳 / 精密な耳 / 精密な耳 + 修正を並べると、粗い耳でだけ出る耳の上のピンクの染みと耳の後ろの淡い滲みが、精密な耳 + 修正では消える。seam_ratio は 1.55 / 1.59 / **1.50**。
- 未解決: s2 の +90° 側の耳の後ろのオレンジ線（B2 以前からの既知）と耳の下のリング。今回の原因とは別。
- fast の出力は 1024 なので、`make_ears03.py` の ears02 との合成（2048 前提）は未実施。次は同じ設定で 2048 の bake（約 11 分）→ `ears05` の合成 → 検証。

---

## 13. s2 の 2048 での再 bake と ears05（2026-11-01）

`rebake.py ... --ear-surface --min-facing .25 --mag-cap 6`（B3 と同じ guard 3 / feather 6 / gain / inner-lip / front-blend / bg-reject）→ `rebake/s2/bg_ear_fix`（約 14 分）→ `REBAKE_DIR=bg_ear_fix OUT_NAME=ears05 make_ears03.py s2` → `candidates/s2/B/reconstruction/ears05`（`validate_candidate` 通過。provenance の `multiview_cleanup.rebake_options` に ear_surface / min_facing / mag_cap を記録。ears04 も同様に再生成）。**`write_candidates.py` には未登録、Cycles レンダ未実施。**

| 項目 | ears03（B3） | ears04（精密な耳、修正なし） | ears05（精密な耳 + 修正） |
|---|---|---|---|
| 染みの領域 平均 sRGB | 241, 210, 183 | 238, 199, 171 | 237, 204, 179 |
| 染みの領域 R−B | 58 | 67 | **58** |
| bake 統計 bleed / ragged / seam / lip_bright | 0 / 17.4 / 2.15 / 2138 | 0 / 17.6 / 2.17 / 2146 | 0 / 17.6 / **2.04** / 2146 |
| 再投影残差 正面 / +40 / −40（耳 texel、同順） | .0059 / .0263 / .0268（.0386 / .0834 / .0578） | .0059 / .0263 / .0268（.0385 / .0824 / .0577） | .0061 / .0263 / .0271（.0386 / .0828 / .0574） |

- ears05 は ears03 に対して、差 >16 の texel が 3.2 万（耳マスク内 1.0k、耳の周囲の帯 3.3k、その他 27.6k）。**その他の 27.6k は全て ears03 での確信度が 80/255 未満（平均 7、128 以上は 0）**で、ほぼ観測がない texel（側頭部・顎の側面・目の周囲）が補完に置き換わったもの。位置はアトラスの中央の帯に分布する。
- 正面保護の整合性: 正面確信度 0.5 以上の保護 texel（83.6 万）のうち差 >2 は **312**（0.04%、ears04 では 10）。耳マスクのぼかし帯の近くで、`new` の変化が混ざったもの。クロスフェード帯（.2〜.5）と低確信度側の変化はブレンドの仕様どおり。
- 再投影残差は ears03 とほぼ同じ（正面が .0059 → .0061 と僅かに悪化、+40 の耳 .0834 → .0828）。耳の中身は ears02 のままなので差は出ない。
- 見た目（unlit、±40 / ±90 の耳拡大 `scratchpad/s2_ears03_vs_05.png`）: −90°側の耳の上の短いオレンジの線は、ears03 より小さく灰色寄りになり、ears04 で出たオレンジの染みは出ない。+90°側の耳の後ろのオレンジ線と耳の下のリングは**残る**（B2 以前からの既知で、今回の修正の対象外。リングは少し小さく淡くなる）。

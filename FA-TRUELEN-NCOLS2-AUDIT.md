# FA true-length ncols2>1 设备越界 — 边界审计与修复（merge-fa-20261031）

日期：2026-10-31。分支 `candidates/fa-truelen-20261031`（基线 `337f0ce1a`，其下为 `613c3e08c` r8 候选 + `f1c5fafc0` stock）。
纪律：本刀只写码/静态审查，**未编译、未运行任何测试**（无 GPU、无语法检查）。凡未实测者一律 **PARTIAL**。

## 0. 结论

1. **本树 r8 候选的可达 ncols2>1 路径已经闭合**：tile 核 `check_tail` 分支对所有 tile 走 `oob_check=true`（K/V 装载越界行零填充、mask 加法与 softmax 逐行守卫），MMA 在任何 runtime bound 下被改道 TILE，sparse KV_max 压缩被禁用，主机侧有 `kq_mask->ne[0] == n_kv_built` 断言与 launcher 的 op-param 区间校验。未发现可达的未闭合设备越界（静态结论，**PARTIAL**：无设备证据）。
2. **本刀修复 1 个真实潜伏缺陷**（`fattn-vec.cuh`）：vec 核掩码 Q 行步长硬编码 `ne11`，而分片/暂存 mask（KV-stream chunk staging：`ne[0]` 缩为 token_count、`nb[1]` 保持全长行距）在 `ncols>=2` 消费者下会错行读、尾部越界。现改为用 mask 自身 `nb31/2`。对现有一切生产者（contiguous mask，`nb31/2 == ne11`）**位等价**，OFF 不变。
3. Off-sentinel、exact-1 门、别名与精度纪律均未放松（见 §3/§4）。

## 1. 通道与门（全部 exact-"1"，一次求值）

| 门 | 通道 | 语义 |
|---|---|---|
| `GGML_TP5_FA_NKV_PARAM` | r8 | 每请求真长写入 FA op_params[5]；建图期值；同时关图复用/union binder |
| `GGML_TP5_FA_CHECKED_TAIL` | r6 | 流式 checked tail（每 chunk 设备端换算界）；兼作 rtlen 使能名 |
| `GGML_TP5_FA_RTLEN` | r4 别名 | src[5] I32[1] 设备张量通道（每 eval 刷新）；r4 语义下流式仍 abort |
| `GGML_TP5_FA_TILE_FAST_FULL` | r7 | tile 快车道（满 tile 免检、末块检）；纯性能，单独无消费者 |
| `GGML_TP5_FA_MASK_KVLEN` | — | 全树 0 处（历史名，未实现）；其警戒对象即本文对象 |

任一门关（未设/其他值）＝该 lane 完全 stock。`GGML_TP5_RSZ_RT` 全树 0 处。

## 2. 边界清单（实际 mask/K/V 读取点 × 界 × 尾形 × per-window）

记号：`ne11 = K->ne[1]`（kernel 实参 = 构建宽度或 r8 收窄后的真长）；`n_kv_eff = min(通道值, ne11)`（r4 src[5] 与 r8 op param 取 min）；`check_tail = 有任一通道`。

### A. 调度/launcher（`fattn-common.cuh` launch_fattn、`fattn.cu` ggml_cuda_flash_attn_ext）
| # | 点 | 界/形状 | 状态 |
|---|---|---|---|
| A1 | op param 原始值校验 | `raw ∉ [-1, K->ne[1]]` → `GGML_ABORT`（越上界/非法负值 fail-closed） | 已有（r8） |
| A2 | sparse KV_max 路径 | 任一 runtime lane 在场即 `use_sparse_eff=false`/`shall_use_sparse=false`（256 对齐假设与运行时尾不兼容） | 已有 |
| A3 | KV_max 扫描（mask→KV_max） | `!use_sparse_eff && mask && K%256==0 && (长 prompt‖多seq)` 时仍可运行；核内 `row_valid` 保证 ≥n_kv_eff 的行零贡献，输出不变 | 已有（仅多余工作量，无正确性影响） |
| A4 | MMA→TILE 改道 | `fattn_nkv_bound && best==MMA → TILE`（MMA 无 checked tail）；turbo fused、F32 RDNA2 快路径同被 `!fattn_nkv_bound` 跳过 | 已有 |
| A5 | 融合 turbo 路径 | 同上 `!fattn_nkv_bound` | 已有 |

### B. tile 核（`fattn-tile.cuh`，ncols2>1 的主路径，gfx1030 无 tensor core 时即生产路径）
| # | 读取 | 界 | 状态 |
|---|---|---|---|
| B1 | K 装载 `load_tile(K_h2+k0*stride, …, k_VKQ_sup)` | `check_tail` 分支全 tile `oob_check=true`：行 ≥ sup 零填充，不读 | 已有（r6/r7） |
| B2 | mask 加法 `mask[j*stride_mask + k0 + i]`，`stride_mask=nb31/2`（mask 自身步长） | `i < k_VKQ_sup` 守卫；j=token 索引 mod ne01；mask 形 `[n_kv, n_tok, 1, n_stream]`，ne[0]≥n_kv_eff | 已有 |
| B3 | V 装载 `V_h2+(k0+k0')*stride` | `i_sup-k0'` 守卫 | 已有 |
| B4 | softmax 值 | `i < k_VKQ_sup ? exp : 0` | 已有 |
| B5 | ncols2>1 免检分支 | 仅 `!check_tail` 且 `use_gqa_opt`（要求 `mask && K%256==0`）可达 ⇒ 全 tile，无部分尾 | 已有（结构性安全） |
| B6 | `(ncols2>1 \|\| mask)` 三目在 mask==nullptr 时解引用 | 不可达：tile 选 ncols2>1 必须过 `use_gqa_opt`（含 `mask`）；仅记录 | 潜伏·不可达 |
| B7 | 尾 tile 语义 | k_VKQ_max=n_kv_eff（check_tail 时），部分 tile 走 oob_check | 已有 |

### C. vec 核（`fattn-vec.cuh`；本 fork 模板 `ncols`∈{1,2}，launch 恒 `ncols2=1`）
| # | 读取 | 界 | 状态 |
|---|---|---|---|
| C1 | K 点积 `K+i_KQ*nb11` | `row_valid = k0+i_KQ < n_kv_eff` 守卫，无效行 sum=0 | 已有 |
| C2 | mask `maskh[j*ne11 + i_KQ]` | 行内界由 row_valid；**列步长原硬编码 ne11** | **本刀修复**：改 `maskh[j*stride_mask + i_KQ]`（nb31/2） |
| C3 | V 行 `V + k*nb21` | `v_row_valid` 整行跳过（先于反量化） | 已有 |
| C4 | k_VKQ_max 优先级 | vec 取 `KV_max ? KV_max : n_kv_eff`（tile 相反）。两者输出位等价（越界行零贡献），仅浪费工作量 | 记录，不改（改会改部分和结合序） |
| C5 | mask 列步长不变量 | 现有生产者全部 contiguous（nb31/2==ne11）：ubatch mask `[n_kv,n_tok,1,n_stream]`；r8 收窄 mask 与 K 同步收窄并有 A 类断言 | 修复后不再依赖该不变量 |
| C6 | Q 尾列（ncols=2、ne01 奇） | Q 零填充 + mask 跳过 + 写回 `ic0+j<ne01.z` 守卫 | 已有（stock） |

### D. MMA 核（`fattn-mma-f16.cuh`）
| # | 点 | 状态 |
|---|---|---|
| D1 | 有 runtime bound 时从不被选中（A4 改道；流式 MMA prefill 条件含 `nkv_eff_tensor==nullptr && n_kv_runtime_global<0`） | 已有 |
| D2 | 新增形参仅 `GGML_UNUSED`（签名对齐） | 已有 |

### E. 流式/KV-stream 路径（`fattn.cu` ggml_cuda_flash_attn_ext_streamed）
| # | 点 | 界/形状 | 状态 |
|---|---|---|---|
| E1 | 每 chunk 设备端换算界 `kv_stream_chunk_bound_dual`：`bound=min(armed)-token_begin`，夹 [0,token_count]；0=合法空块（中性 partial） | 写入 chunk_bounds[chunk]，经 staged src[5] + op_params[5] 双通道（核内取 min） | 已有（r6+r8 合流） |
| E2 | staged K/V：`ne[1]=token_count`、`nb[1]=k_stage_token_stride`（stage 自身步长） | 与核参数一致 | 已有 |
| E3 | staged mask：`ne[0]=token_count`、data+token_begin*nb[0]、**nb[1] 保留原行距** | 核按 nb31 寻址（C2 修复后严格正确；修复前 ncols=1 时等价、ncols≥2 时错行） | **本刀闭合其隐患** |
| E4 | 尾 chunk 形状 | `token_count=min(page_tokens, K->ne[1]-token_begin)`；resident 路径断言 `token_count==page_tokens`（要求页对齐，非对齐→host abort，不落设备） | 已有 |
| E5 | RTLEN 单开（无 CHECKED_TAIL）流式 | 保持 landed r4 契约：`GGML_ABORT`（不支持） | 已有 |
| E6 | 全常驻/单 chunk 快路径 | 走完整 `ggml_cuda_flash_attn_ext`（含 A4 改道） | 已有 |

### F. CPU（`ggml-cpu/ops.cpp`，commit 337f0ce1a）
| # | 点 | 状态 |
|---|---|---|
| F1 | 张量通道 src[5]：`GGML_ASSERT(0 <= raw <= nek1)`（0=合法空段；负=损坏即中止） | 已有 |
| F2 | op-param 通道对 CPU 无效（CPU 不读 slot 5） | 已有 |

### G. per-window（TP5 双窗）
| # | 点 | 状态 |
|---|---|---|
| G1 | r8 op param 是建图期值：门同时禁图复用/union binder，每窗图各自携带本窗 `get_n_kv_req`（max over ubatch 序列；0 活细胞→-1） | 已有 |
| G2 | r4 src[5] 每 eval 由 `set_input_nkv_eff` 刷新（`get_used_max_p1`，≥0 按构造）；捕图回放安全（值在设备缓冲内） | 已有 |
| G3 | 流式 chunk_bounds 为每次调用的 pool 分配，不跨窗共享 | 已有 |
| G4 | 两窗各自 context/图/KV，掩码与视图不跨窗别名 | 已有 |

## 3. 本刀代码改动（唯一文件：`ggml/src/ggml-cuda/fattn-vec.cuh`）

1. 核首（`maskh` 定义后）新增 `const int stride_mask = int(nb31 / sizeof(half));` 及说明注释。
2. 掩码读取 `maskh[j*ne11 + i_KQ]` → `maskh[j*stride_mask + i_KQ]`（含注释）。

理由与不变量：
- mask 元素 (kv=i, tok=j) 的真实偏移 = `i*2 + j*nb31` 字节；原代码按 `j*ne11`（=假设 `nb31==2*ne11`）寻址。所有现行生产者该假设成立 → **位等价**；KV-stream 分片 mask（`nb[1]`=全长行距、`ne[0]`=token_count）该假设不成立 → ncols≥2 时错行读、尾部越过分配 → 修复后按真实步长寻址。
- 与 tile 核既有 `stride_mask = nb31/sizeof(half)` 约定对齐。
- 不引入新门（纯正确性硬化，OFF 位等价）；不触碰 `nrows==1` 别名保护、不降精度、不改任何归约顺序。

## 4. OFF ≡ stock（源码级论证；位级证据 PARTIAL）

- 两处改动在 OFF 路径的数值恒等：OFF 时 mask 恒为 contiguous ubatch mask（构建点唯一、`ne[0]==K->ne[1]`，且有新增 host 断言兜底）⇒ `nb31/2 == ne11` ⇒ 寻址逐位相同。
- r8 候选既有护栏（A1–A5、B1–B5、C1/C3、D1、E1–E6、F1、G1–G4）在门关时全部旁路：`src[5]` 不生产、slot 5 不写（ggml 新张量 op_params 零初始化）、check_tail 恒 false、核路径与 stock 相同。
- 位级证明仍需 Devops 执行 §6 的 OFF 腿（G2 relink 对照 + 下述 micro 的 OFF≈stock 断言），本刀不声称已证明。

## 5. 残余风险 / PARTIAL 清单

1. **全部设备结论均为静态审查**（本刀禁编译/测试）：tile/vec checked-tail 在设备上的行为、以及本刀两处改动的设备编译，均 **PARTIAL**。
2. `ncols2>1` tile 免检分支的 `(ncols2>1||mask)` 三目在 `mask==nullptr` 时为坏形（B6）：当前不可达（`use_gqa_opt` 含 mask），未改（属 stock 行；改动无收益）。
3. vec 核 `k_VKQ_max` 的 KV_max 优先级与 tile 相反（C4）：输出位等价、仅多余工作量；改动会改部分和结合序（Rule D 允许但无必要），不改。
4. r4 src[5] 通道的设备值无法在 launcher 无同步校验：负值→空注意→NaN（与 CPU n_kv_eff=0 行为一致）；写方 `set_input_nkv_eff` 按构造 ≥0，破坏需先有其他内存错误。记录不改。
5. 头维 192/320/512/576/640 且收窄后宽度非 256 对齐 → 选择器 NONE → host `GGML_ABORT`（fail-closed，非设备越界）。目标模型 head_dim=128 不受影响。
6. `GGML_TP5_FA_NKV_PARAM=1` 关图复用 → 每 eval 全量重建（设计使然，非性能回归）。
7. r8 候选与 nout/union/maxfp 等其他并行候选的叠加不在本刀范围（各树独立）。

## 6. GPU micro 验证方案（Devops 执行；13022，双锁，HIP 1/1）

前置：`flock -x /tmp/ple-candidate-gpu.lock` + `$E/ple-candidate-gpu.lock`（fd8/fd9）；本树编译走 Devops 通道；每条命令超时 75/124 即停。

### M1 编译微型可执行（g++ 直链，不动 CMake）
```bash
cd /home/kunweiz/work/merge-fa-20261031
# 两步：先出库，再 hipcc 链接 micro（产物路径以实际 build 目录为准）
cmake -S . -B build-fa-micro -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DGGML_HIP=ON -DGGML_HIP_ROCWMMA=OFF
cmake --build build-fa-micro -j$(nproc) --target ggml-cuda ggml  # 仅库，供下面链接
hipcc -O2 -std=c++17 -I ggml/include \
  tests/test-fa-true-tail-micro.cpp \
  build-fa-micro/bin/libggml.a build-fa-micro/bin/libggml-cuda.a \
  -o build-fa-micro/test-fa-true-tail-micro -L$ROCM_PATH/lib -lamdhip64
```

### M2 micro 源（本刀已写：`tests/test-fa-true-tail-micro.cpp`，见同目录）
用例矩阵（测试源已随刀写入 tests/test-fa-true-tail-micro.cpp，仅用公共 ggml/ggml-backend API）：
- **T1 尾块毒化（核心）**：`--case t1 --gqa {2,4,8} --n {1,255,256,257,511,512}`。D=128、B=512；K/V 在 [n,512) 按元素写 NaN(0x7E00) 毒；mask 张量宽度=n（即“mask 窄于 K 视图”的尾部形状，公共 API 直建 FA 节点）；门=`GGML_TP5_FA_NKV_PARAM=1` + `ggml_flash_attn_ext_set_n_kv_runtime(n)`。断言：无 NaN 且 max|err| ≤ 1e-5 对 [0,n) 标量参考。负对照 `--case t1-nobound`（同参不设界）：毒必须可见（输出 NaN>0），否则毒化无效、T1 证据不算数。
- **T2 对齐等价**（n=B=512，无毒）：`t2`（param=512）与 `t2-nobound`（不设 param）各跑一遍 `--dump`，FNV hash 必须逐位相同（gfx1030 两者均落通用 tile 核）。
- **T5 门矩阵**：对 t1 主用例分别用 `GGML_TP5_FA_NKV_PARAM` ∈ {unset, 0, 1, true, " 1", 2} 各起一进程：仅 =1 通过（rc=0）；unset/0/其他必须 rc=4（门拒绝，exact-1 验证）。
- **T6 越界请求**：`--case t6-invalid-hi`（param=B+1）与 `--case t6-invalid-neg`（-2）：进程必须非零退出（launcher 校验或 host assert），不得产生部分输出。
- T3/T4（流式分片 mask 与 chunk 界翻译、KV-stream 缓冲路径）无法由公共 API 构造，列为第二阶段：克隆既有 lane 的 patch/ 在 8×2 车内验证（hdr 追加 `GGML_TP5_FA_CHECKED_TAIL=1` 与 `GGML_TP5_FA_NKV_PARAM=1`，G3 读 logits bitwise 与 stderr nkv 计数）。此阶段未执行，**PARTIAL**。

### M3 执行与判据
```bash
flock -x /tmp/ple-candidate-gpu.lock -c '
flock -x /home/kunweiz/atomic-llama-cpp-turboquant/build-tp5-dual50-20261022/mtp-endtoend-20261024/ple-candidate-gpu.lock -c "
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t1 --gqa 8
"; '
# 判据：T1 各组合 rc=0 且无 NaN；t1-nobound 必须 rc=0 且 NaN>0（毒化自检）；t2/t2-nobound 的 hash 相等；
# t6 两例 rc≠0；T5 门矩阵仅 =1 过、其余 rc=4。
# 任何失败即停（ON_1 rc!=0 规则），不得连续重烧。若 ROCm 提供 compute-sanitizer，
# 可选跑 `GGML_TP5_FA_NKV_PARAM=1 compute-sanitizer --tool memcheck ./build-fa-micro/test-fa-true-tail-micro --case t1 --gqa 8 --n 257` 直接抓 OOB。
```bash
# 完整矩阵骨架（Devops 按需展开；每步 75s/124s 超时即停）
cd /home/kunweiz/work/merge-fa-20261031
for g in 2 4 8; do for n in 1 255 256 257 511 512; do
  GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t1 --gqa $g --n $n || break 2
done; done
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t1-nobound --gqa 8 --n 257
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t2 --dump --gqa 8 --n 512 > /tmp/h2a
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t2-nobound --dump --gqa 8 --n 512 > /tmp/h2b
grep hash /tmp/h2a > /tmp/h2a.k; grep hash /tmp/h2b > /tmp/h2b.k; cmp /tmp/h2a.k /tmp/h2b.k
for v in unset 0 1 true 2; do
  if [ "$v" = unset ]; then env -u GGML_TP5_FA_NKV_PARAM ./build-fa-micro/test-fa-true-tail-micro --case t1 --gqa 8 --n 257; rc=$?;
  else GGML_TP5_FA_NKV_PARAM="$v" ./build-fa-micro/test-fa-true-tail-micro --case t1 --gqa 8 --n 257; rc=$?; fi
  if [ "$v" = 1 ]; then [ $rc -eq 0 ] || echo "GATE-FAIL v=$v"; else [ $rc -eq 4 ] || echo "GATE-FAIL v=$v"; fi
done
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t6-invalid-hi; echo "rc=$? 需≠0"
GGML_TP5_FA_NKV_PARAM=1 ./build-fa-micro/test-fa-true-tail-micro --case t6-invalid-neg; echo "rc=$? 需≠0"
```
- 通过标准：G3 micro（真实 body、非仅节点数）、无 silent no-op、无 NaN；通过后才允许在 13022 讨论打开 `GGML_TP5_FA_NKV_PARAM=1` 的 8×2 诊断，再议 formal。
- 本刀未运行以上任何命令（纪律）；所有 VERDICT 由 Devops 回填。

## 7. 交付物清单

| 件 | 路径 | 状态 |
|---|---|---|
| 代码修复 | `ggml/src/ggml-cuda/fattn-vec.cuh`（2 处：新增 stride_mask、掩码读改 stride） | 已改，未编译；OFF 位等价论证见 §4 |
| micro 测试源 | `tests/test-fa-true-tail-micro.cpp`（T1/T1-neg/T2/T2-nobound/T5/T6 + 门矩阵接口） | 已写，未编译；T3/T4 留待 8×2 车内阶段 |
| 本报告（边界清单+方案） | `FA-TRUELEN-NCOLS2-AUDIT.md` | 本文件 |

综合验收状态：**PARTIAL**（代码+方案完备；无任何编译/设备证据，按规不得自称 PASS）。

# AGENTS.md

## 下班交接｜2026-09-22（第十九轮，最终交付：FlashPrefill 共享 Run 分桶与 Base 零拷贝 + 十问缺口完整盘点归档）

**分支：** `master`
**主题：** 本轮为最终交付班次。推进完成两项核心使命：
1. **主机端布局构建性能收敛**：落实 FlashPrefill 路径上的 Q3/Q4 深度共享——消灭跨 View 对 $K$ 个常驻 Cell 的重复全量哈希查找，建立 `shared_base` 预排序与 `run_buckets` 一次性分桶，R=12 耗时由基线 1.14s 压缩至 **311ms (3.68×)**；
2. **全计划剩余缺口审计与交接**：全面核对 P0–P14 原讨论稿与 2026-09-21《计算组织方式重画与十个根本问题》，将全部 17 项剩余缺口规范归档并追加至 `缺口.md`，为后续 GPU Kernel 落地与真机实测提供完整导航。

全程遵循真机安全门，100% 单元测试与 ASAN 验证通过，未启动模型与高危测试。

---

### 一、第十九轮代码改动：FlashPrefill 共享 Run 分桶与 Base 预排序（`src/llama-kv-cache.cpp`）

针对第十八轮后剖析指出的“Filter 占单次构建 67% 时间、每 View 重复对 $K$ 个 Cell 执行 `unordered_map::find` 哈希查找”问题：

| # | 改动 | 机制与收益 |
|---|---|---|
| 1 | **`shared_base` 预排序与零拷贝借用** | 静态非活跃但已拥有的 Cell 属于纯读者无关属性。在 `fp_scan_resident` 中直接提取至 `scan.base` 并预排序一次；后续各 View 直接以 `const &` 引用借用，彻底消灭每个 View 内部独立的 `base.push_back` 与排序探测。 |
| 2 | **`run_buckets` 一次性键值分桶** | 在单趟扫描期间，活跃 Cell 按 `(episode_id, run_id)` 直接分桶为行索引数组；每 View 由 $O(K)$ 遍历全局 Cell 改为**仅遍历该 View 实际声明的 Ordered Runs**，哈希查找由 $R 	imes K$ 次（约 236 万次）降为 $R 	imes N_{\text{runs}}$ 次（12 次）。 |
| 3 | **负坐标防御前置** | 负物理存储坐标检查在扫描阶段一次完成并立即 fail-closed，保证下游只读安全。 |

### 二、性能对比实测（min-of-5，fragments 数量与位级输出 100% 一致）

| 形状 | 优化前基线 | 第十八轮 (共享扫描) | **第十九轮 (共享分桶+Base)** | **总加速比** |
|---|---|---|---|---|
| **R=12 K=196609** | 1,143,707 us | 460,967 us | **311,096 us** | **3.68×** (单轮再省 32.5%) |
| **R=8 K=131073** | — | 213,674 us | **146,770 us** | **1.46×** (相比第18轮) |
| **R=6 K=98305** | 237,534 us | 128,921 us | **89,763 us** | **2.65×** (单轮再省 30.4%) |
| **R=4 K=65537** | 105,332 us | 65,458 us | **46,694 us** | **2.26×** (单轮再省 28.7%) |
| **R=1 K=16385** | 8,161 us | 7,132 us | **6,740 us** | **1.21×** |

### 三、剩余缺口全貌归档（见 `缺口.md`）

已全部追加并同步至仓库根目录 `缺口.md`，分为三大模块：
1. **第一部分：原计划 P0–P14 算子/架构缺口**（缺口一至七）：
   - P2 融合 Q-prep 着色器（Live-count gather + RoPE）
   - P7 多行/PQ2_0 投影 Recipe 扩展（$b \in [2, 6]$ 消除退化）
   - P9 非均匀 Head Map 分组与自适应长短行 Split-K
   - P10 逻辑 State 驻留池与 Device-Copy Parking（消灭跨进程序列化）
   - P12 常用阶段预定义图句柄复用（消灭阶段冷启动）
   - P13 紧凑 Span ABI 直通 Attention Op（25MB $	o$ 几十字节）
   - P14 全特性真机 A/B/B/A 配对净收益测量
2. **第二部分：2026-09-21 计算组织研究线（十问根本问题）缺口**（缺口八至十五）：
   - Q1 算子输入共享与图级公共子图去重 / MoE 专家聚集
   - Q3 公共 KV 块多读者共享注意力 GPU Kernel（Hydragen 模式）
   - Q5 GDN 状态共同基底 + 低秩增量 GPU 落地与平滑转稠密门控
   - Q6 已知 Token WY 块折叠 GPU 算子（固定入口与已知验证块）
   - Q7 PQ2_0 位平面子集求和与 LUT 路径 GPU 真实带宽评测
   - Q8 上下文跳块误差界与 Key 包围球硬件跳过
   - Q9 越过 Logits 的联合 LM Head 与批量 GPU 原生采样器
   - Q10 $K 	imes H$ 联合投机网格验证与多笔交互前沿预测
3. **第三部分：主机端元数据与布局微细缺口**（缺口十六至十七）：
   - FlashPrefill 跨 View 相同可见性 Run 切片引用复用
   - 写入端分配器主动长 Span 维护与碎片整序

### 四、验证记录

- **测试套件**：`ctest -R "rerot|xkv|flashprefill"` 47/47 全部通过（100% PASS）；
- **专用验证**：`test-tp5-plan`、`test-flashprefill-state`、`test-flashprefill-routing`、`test-xkv-runtime` 全绿；
- **ASAN 内存审计**：在 `/tmp/asan18` 下编译 `test-flashprefill-state`、`test-flashprefill-routing`、`test-rerot-view` 运行结果 **0 错误**；
- **硬件与显存**：保持 GPU 空闲，无设备挂死，无内核 panic。

---

## 下班交接｜2026-09-22（第十八轮，flashprefill 布局路径：常驻表共享扫描 + 每桶排序探测）

**分支：** `master`
**主题：** 十问讨论稿的主干前四问落到 **flashprefill 布局路径**（`llama_flashprefill_build_rerot_plan`）。
上一轮交接以后，这条路径是唯一还没做过跨 reader 共享的主干环节。全程 CPU 验证，未启动模型/GPU（真机安全门）。

### 一、先测后改：相位剖析推翻了一个此前的结论

在真实 builder 上临时插桩（**已 `git checkout` 回退，未提交**）后：

- **92% 的单次 build 时间是每 view group 重复的全 cell 扫描**（R=6 是 R=1 的 7.6×，同 K）；
- **排序只占 7.8%** —— 前几轮在 indexed 路径上得出的「排序是大头」**不能外推到这条路径**；
- 匹配总 K 分解（K=131073）：R=1 = 60.7ms（共享扫描 + 1 view），R=8 = 230.7ms（共享扫描 + 8 view），每 view ≈ 21ms。

### 二、两处改动（`src/llama-kv-cache.cpp`，+85 行）

| # | 改动 | 依据 |
|---|---|---|
| 1 | **常驻表共享扫描**：新增 `fp_resident_row`/`fp_resident_scan`/`fp_scan_resident()`，一次扫完所有常驻 cell；每个 view group 由扫描退化为**过滤**（只有可见性谓词是 per-reader）。`sig` 直接取 `cells.seq_get_all(idx)`，删除 `live` + `member_sig` | 十问问题三「一块数据供给服务多个 reader」在 host 侧的对偶 |
| 2 | **每桶 sortedness 探测**：桶按 cell index 序填充、排序键是 storage 序，生产形状两者一致；O(n) 探测失败才 `std::sort`。**比较器只写一次**，probe 与回退 sort 共用 | 十问问题四「结构不变就不重做」的最小实例 |

### 三、实测（fragments/groups/uses 数量前后完全一致）

| 形状 | 前 | 后 | 加速 |
|---|---|---|---|
| R=12 K=196609 | 1,143,707 us | 460,967 us | **2.48×** |
| R=6 K=98305 | 237,534 us | 128,921 us | **1.84×** |
| R=4 K=65537 | 105,332 us | 65,458 us | **1.61×** |
| R=1 K=16385 | 8,161 us | 7,132 us | 1.14× |

拆开看：共享扫描单独贡献 R=12 **2.25×** / R=6 **1.71×** / R=4 **1.54×**（R=1 仅 1.03×）；排序探测再 −5~7%。
**收益随 reader 数增长**——与十问「K 笔共享结构天然在同一层、同一执行阶段」一致。

### 四、新增回归臂（`tests/test-flashprefill-state.cpp`）

`test_rerot_shared_scan_multi_reader`：一次规划调用里两个不同 reader + **物理 cell 被两个 reader 共享**。
钉住三件事：每 reader 的 (physical, effective) 集与逐 query oracle 一致；A-only cell 对 B 不可见（共享 cell 语义）；
B 的 base 段内出现**两个 DDVR 相位**（0@7 → 2@8，因为 cell 1 不属于 B）——这正是必须保持 per-reader 的相位，
会被错误的共享化一次抹平。写这个臂时我先把期望值算错两次（run30 成员数 4 数成 3），由 oracle 纠正。

### 五、本轮工程教训（重要，写进 RERoT.md §21.4）

1. **绝不能在 1 万行文件上做全文字符串 strip**：一次 `s.replace(frag,'')` 清理插桩误删了真实 `}`，
   把 85 行改动炸成 2047 行差、大括号 1983 vs 938。恢复：`git checkout` 回 HEAD → `git apply` 保存的测试 patch →
   主文件用**逐条 `assert count==1` 的锚点替换**重做。
2. **插桩必须可逆**：先插桩→测量→回退，才没有把 indexed 路径的相位结论外推到这条路径（实际结论相反）。
3. **probe 与它替换的 sort 共用同一个比较器**，且覆盖排序键全部分量（第十一轮教训的第二次应用）。

### 六、验证

- ctest `rerot|xkv|flashprefill` **47/47**；`test-tp5-plan` / `test-meta-reduce-boundary` 全过。
- ASAN（/tmp/asan18）：`flashprefill-state` / `flashprefill-routing` / `rerot-view` **0 错误**。
  唯一已知报告是 `test-xkv-runtime` 基线既有的 alloc-dealloc-mismatch（测试自带的 operator new/delete 重载
  与 libstdc++ `std::get_temporary_buffer` 冲突，在 oracle `llama_rerot_build_query_layout` 内，与本轮无关；
  第十三轮起即为基线）。

### 七、边界与未闭合

1. **per-view 过滤仍是 O(K) 每 reader**：匹配总 K 下 R=8 是 R=1 的 3.8×。共享扫描只消除了重复扫描，没消除
   **重复分桶**。下一步是让分桶按 run 复用（同一 run 成员集对所有 reader 相同，只有 own/gate 不同），把 Q4 的
   `llama_rerot_run_order_signature` 接到这个调用点；flashprefill 侧已有 `fp_key` + freshness 整层缓存，
   fragment 级缓存的边际收益需先证明再动手（21.3 节保留此判断，本轮未推翻）。
2. 排序探测在生产形状收益有限（R=1 时探测本身有成本）；若写入侧后续主动维持长 span（Q2 的
   `span_long_fraction` 验收指标），cell 序与 storage 序会更稳定，届时再评估把探测上移到写入侧。
3. 真机收益仍需目标机跑模型会话验证；本轮全部数字是开发机 CPU 合成键，不是模型证据。


## 下班交接｜2026-09-22（第十七轮，§4.1 几何第二步：resume_norm / resume_lo_q8 / UP_Q8DOT 三 kernel 收口）

**分支：** `master`（本轮 commit 见 git log；基于 `c67819718`，rebase 过 RERoT `0cf626757`）
**主题：** 第十六轮 Q8DOT 半波 shuffle 树的同一原则推到剩余三个 LDS 往返点。全程 CPU + 既有 GPU 回归，未启动模型（安全门）。

### 一、三项改动（单 shader 文件为主，UP_Q8DOT 另改 dispatch 常量）

1. **resume_norm**：跨 subgroup 折叠原是"lane0 写 `partials[sg]`→barrier→subgroup 0 各 lane 读 `partials[i]`→subgroupAdd→lane0 回写 `partials[0]`→barrier"——subgroup 0 单波串行在四个 stream WG 的关键路径上。固定 8-subgroup 形状下全员直读 8 个 partial 寄存器求和：少一个 barrier、无单波串行、`partials[32]`→`[8]`。**code 18920→15412（−18.5%），VGPR 32→24（SIMD 占用率 +33%）**。加 `gl_NumSubgroups != 8u` fail-closed 守卫（Q8 家族惯例）。
2. **resume_lo_q8**：`lo_q_tmp[320]` LDS 往返换 wave 内 `subgroupShuffle` 三次重建对齐 4-lane 组（`g0 = sublane & ~3u`）；每 lane 打包自己组的 4 个 int8 为一个 uint32，lane 0/4/8/.../28 提交 word。**code 4000→3928，LDS 2048→1024（−50%）**，每 token 两 barrier 全消。NaN 失败态路径不变（d=NaN bits、qs 清零）。
3. **late_up_q8dot**：几何重排 3 行/wave×10 lane（30/32 有效、10 非 2 的幂）→ **4 行/wave×8 lane（32/32 全有效）**；行折叠变纯 butterfly 掩码 1/2/4（对齐 8-lane 组永不越行）；`up_partial[240]`（4096B）整个删除。**code 5044→1088（−78.5%）、LDS 4096→0**。dispatch：`rows_per_wg 24→32`、`(w+23)/24→(w+31)/32`；width=10240 时 320 WG×32 = 10240 精确覆盖（旧 427×24=10248 冗余 8）。
4. （卫生）`tp5_hc_subgroup.glsl` include 条件收窄到 INJECT/exact-Q（真正调用 `tp5_hc_sum32` 者）；Q8 家族不再编译死 128B shared（RADV 本会消除，codegen 无变化）。

### 二、验证

15/15 ctest 全绿（每项改动后各一轮，共 4 轮）；GPU 空闲（5×0%）；零内核错误。mesh 只建 pipeline（codegen 证据有效）；bit 级正确性依据：UP 行覆盖精确等式、butterfly 对齐组不变量、lo_q8 word 布局等价推导（`qs_words[j] = lanes 4j..4j+3 LE`，与旧 ACT_Q8/LO_Q8 约定逐字一致）。真机收益需模型会话（安全门未批）。

### 三、§4.1 全景（本班末，RADV 实测）

| Kernel | code | LDS | VGPR | 本线累计 |
|---|---|---|---|---|
| late_q_q8dot | 1248 | 1024 | 64 | −26.4%（vs 1696）|
| late_up_q8dot | **1088** | **0** | 64 | **−82%**（vs 6152 原始）|
| resume_norm | 15412 | 1024 | **24** | −18.5% |
| resume_lo_q8 | 3928 | 1024 | 64 | −1.8% |
| late_act_q8 | 2064 | 2048 | 64 | 持平（生产者，+5.7% 上轮）|

### 四、追加（同班次末）：ACT_Q8 打包 bug 静态发现与修复

把 resume_lo_q8 换 shuffle 打包时推导了全库 word 约定，反查发现 **ACT_Q8
（第十五轮 §7.9 引入）从未按 packed 布局改写**：仍存 16 个半字（索引 0..15
越界，数组只有 8），每字只 16/32 位有效。Q8DOT 消费者读上半 16 位是垃圾。
修复：lane<8 一次拼 4×8 全字存 `qs_words[lane]`，与 PACK_WEIGHTS 逐位一致。
**教训：mesh 只建 pipeline，bit 正确性靠布局推导时必须覆盖每个生产者；
三个生产者（pack/act/lo）应共享同一条 word 公式而不是各自手写。**
15/15 全绿（修复后一轮）；GPU 空闲；零内核错误。

### 五、下一步

1. 真机收益测量（P1-B 两步 + §4.1 两步几何）——全部等模型会话批准；
2. §4.1 剩余维度：rows_per_wg（Q8DOT 16→8/4）、K lanes（16→32）——需真机计时通道，codegen 已到收益递减区；
3. `resume_norm` 仍 15412B：剩余大头是 4×16 unroll 的 load/store 主体，属数据搬运本体，非归并浪费；
4. 多行 LateBind（token 维度统一）仍是代码侧最大结构项。

## 下班交接｜2026-09-22（第十六轮，§4.1 几何第一步：Q8DOT 行归并 shuffle 树 + 两处小修）

**分支：** `master`（本轮 commit 见 git log；基于 `1fe1d8b0d`）
**主题：** 路线图 §4.1"看实际 RADV codegen"的第一笔几何优化 + 前轮交接遗留的
两处小修。全程 CPU + 既有 GPU 回归，未启动模型。

### 一、班首审计：两份路线图草案的可操作项全部已落地

对两份草案（`fe46587f8` 基与 `90f6b9433` 基）逐项核对当前源码：

| 草案项 | 现状 |
|---|---|
| P0-1 Q sidecar 布局对齐 | **已落地**：CPU/GPU 全走 `tp5_late_q_control_*`/`tp5_late_q_payload_word_offset` 单一定义；静态推导验证 ready=word(64+L)/4、counter=+1、payload=(64+L+64)/4 两侧一致 |
| P0-2 F32-Q 输入寿命（WAR） | **已落地**：`q_norm_barrier`（READ→BARRIER→WRITE）+ `tp5_validate_latebind_war_schedule`/`tp5_validate_p1a_schedule` 定义期校验 |
| M0 数值模式三分 | **已落地**：`tp5_numerical_mode` 四态（reference/exact-f32/aggressive-q8/**p1a-nosidecar-q8**）+ `GGML_TP5_P1A_NOSIDECAR_Q8` 对照器具 |
| M1 MTP phase 豁免 | **已落地**：`ubatch_execution_phase` 对 `LLM_GRAPH_TYPE_DECODER_MTP && has_predefined_capacity` 恒返 0 |
| P1-A 无 sidecar aggressive Q8 | **已落地**：独立调度分支 + 专用校验器（见上） |

结论：草案的"先修"清单在近几轮已全部收口，本轮转入 §4.1。

### 二、改动一：Q8DOT 行归并 LDS 往返 → wave 内 shuffle 树

`late_q_q8dot` 每行 16 归并 lane 恰为 wave32 半波（16 行/256 线程 → 每 subgroup
2 行）。旧路径 LDS 往返（写→barrier→串行 16 项加→barrier→32 lane 发布）换为：

1. `subgroupShuffleXor` 掩码 1/2/4/8 四步半波树（零 LDS、零 barrier）；
2. 跨半波 `subgroupShuffleXor(acc, 16u)` 一次；
3. 每 subgroup 前 4 lane 发布（一 lane 一 stream）——8×4 = 与旧版相同的 32
   packed word，地址映射不变；
4. `row_partial`/`group_out` 删除，LDS 5120→1024B（只剩 `publish_last`）。

**RADV codegen**：code 1696→**1248**（−26.4%）、LDS 5120→**1024**（−80%）、
VGPR 64/spill 0 不变。语义：16 项归并从串行链变树状重结合——aggressive 本就
声明重结合自由。

### 三、改动二：两处小修（前轮交接"值得考虑"项）

1. packed 权重 range check `2u * max_storage_buffer_range` → 单 range（绑定
   直接用 packed_bytes，无双 buffer split，双上限是错误宽松）；
2. pack clear fence `UINT64_MAX` → 2s 有界（与 relay handoff timeout 同惯例；
   挂死的 clear 必须 fail-closed 而非永久阻塞 teardown）。

### 四、验证

15/15 ctest 全绿（两轮：几何改动后 + 小修后各一轮）；GPU 空闲（5×0%）；
零新增内核错误。真机收益需模型会话（安全门未批）。

### 五、下一步

1. §4.1 几何搜索剩余维度：rows_per_wg（4/8/16）、K 并行 lanes（8/16/32）——
   Q8DOT 已有 baseline（code 1248/LDS 1024），下一候选 `resume_norm`（18920B）；
2. P1-C transport 三选一、P2/P3 全部真机门控；
3. 草案"多行 LateBind"（width×active_tokens 统一 token 维度）是下一个代码侧
   大项——`tp5_late_consumer_ref` 的 `hc.width == n_elems` 限制。

## 下班交接｜2026-09-22（第十五轮，P1-B 第二步：激活侧同布局打包，点积内层零打包 ALU）

**分支：** `master`（本轮 commit 见 git log；基于 `86dcfd2fb`）
**主题：** 承接第十四轮的 codegen 证据通道，完成 P1-B 的激活侧：Q8DOT/UP_Q8DOT
内层循环的 `pack_q8_pair` 全部消除。全程 CPU + 既有 GPU 回归，未启动模型。

### 一、改动（三文件）

1. **`tp5_hc_latebind.comp`**：`TP5_LATE_ACT_Q8` 输出改 `late_q8_packed`
   （36B/块，**无 flag 块**——per-token 瞬态每 epoch 重写，自禁用协议无意义，
   索引 i 即块 i）；Q8DOT 绑定 1 / UP_Q8DOT 绑定 1 改读 packed，内层变纯
   `dotPacked4x8EXT(w.qs_words[k], a.qs_words[k])`；两个 `pack_*_pair` helper 删除；
2. **`tp5_hc_resume.comp`**：`TP5_RESUME_LO_Q8` 输出同改 packed（含 NaN 失败路径）；
3. **`ggml-vulkan-collective.cpp`**：`tp5_late_q8_packed_bytes()` 统一四处字节
   计算（act/lo 分配×2 + barrier×2）。

### 二、RADV codegen 实测（第十四轮通道的直接兑付）

| kernel | code 前→后 | Δ |
|---|---|---|
| late_q_q8dot | 2676 → 1696 | **−36.6%** |
| late_up_q8dot | 6152 → 5044 | **−18.0%** |
| late_act_q8 | 1952 → 2064 | +5.7% |
| resume_lo_q8 | 3996 → 4000 | +0.1% |

消费者大幅缩、生产者微增；VGPR 维持 64、零 spill。**这是"先修证据通道再
优化"路线的第一次完整闭环：改前有 baseline 表，改后有对比表。**

### 三、数值变化（精度提升，非回归）

块尺度 d 从 f16（11 位尾数）拓宽为 f32（24 位）：原路径 `float16_t(d)`
对尺度舍入，packed 存全 f32；int8 载荷不变 → aggressive 去量化误差严格缩小。
验收时注意：aggressive 输出与旧版**不位级一致**（更准），对照基线需重采。

### 四、验证

`test_tp5_packed_weight_layout` 扩展激活断言（无 flag 块、act=46080B/lo=1440B
@max_rows=4、f32>d16 尾数）；全套 all passed；15/15 ctest 全绿；GPU 空闲、
零新增内核错误。

### 五、下一步

1. P1-B 至此收口（权重＋激活全 packed，内层零打包 ALU）；真机收益等模型会话；
2. §4.1 几何搜索（row/wave/K tile）有完整测量器可用，resume_norm（18920B）
   是最大候选；
3. P1-C transport 三选一需真机测量。

## 下班交接｜2026-09-22（第十四轮，RADV codegen 证据通道：feature 未开修复 + TP5 kernel 统计落地）

**分支：** `master`（本轮 commit 见 git log；基于 `c9bb1c808`）
**主题：** 落实路线图第二稿 §4.3“看实际 RADV codegen，而不是只看 shader 名字”。
两件：修好主路径一直无效的 `GGML_VK_PIPELINE_STATS`；给 TP5 collective kernel
接上同一证据通道并取得首次实测。全程 CPU + 既有 GPU 回归，未启动模型。

### 一、主路径 stats 一直无效（静默 bug，已修）

`ggml_vk_get_device_init` 中 `pep_features` 值初始化（`{}`）后从未置
`pipelineExecutableInfo = VK_TRUE`——扩展启用了、feature 没开，
`vkGetPipelineExecutableStatisticsKHR` 按规范属无效调用（RADV 实际返回空）。
`GGML_VK_PIPELINE_STATS` 环境变量因此从未真正工作过。修复：一行
`pep_features.pipelineExecutableInfo = VK_TRUE;`。实测
`mul_mat_vec_f32_f32_f32`：VGPR 32 / SGPR 108 / 零 spill——通道恢复。

### 二、TP5 collective kernel 统计通道（新）

`make_late` 工厂原来不捕获统计。新增：

1. `GGML_TP5_PIPELINE_STATS`（过滤子串，空串全匹配；无扩展时告警忽略）；
2. `tp5_rank.pipeline_stats/pipeline_stats_filter`；caps 新增
   `pipeline_executable_properties` 探测（`ggml_vk_tp5_device_caps` 枚举）；
3. `tp5_print_pipeline_statistics`（C API：`VkPipelineInfoKHR`/`VkPipelineExecutableInfoKHR`）；
4. `make_late` 增 `pipe_name` 参数，`want_stats` 时以
   `VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR` 创建。

### 三、首次 RADV 实测（5 卡 mesh 创建路径，`--sync relay`）

| kernel | VGPR | SGPR | spill | LDS | code |
|---|---|---|---|---|---|
| late_inject | 32 | 108 | 0 | 1024 | 1144 |
| late_q | 64 | 108 | 0 | 1024 | 2316 |
| publish | 8 | 108 | 0 | 0 | 120 |
| resume_norm | 32 | 108 | 0 | 1024 | 18920 |
| resume_lo | 8 | 108 | 0 | 1024 | 1340 |
| late_pack | 24 | 108 | 0 | 0 | 452 |
| late_act_q8 | 64 | 108 | 0 | 2048 | 1952 |
| late_q_q8dot | 64 | 108 | 0 | 5120 | 2676 |
| resume_lo_q8 | 64 | 108 | 0 | 2048 | 3996 |
| late_up_q8dot | 64 | 108 | 0 | 4096 | 6152 |

（首版表格因采集命令 `awk '!seen[$0]++'` 跨 kernel 去重产生“—”残留，本表为完整重采；SGPR 108 为 RADV 对全部 compute pipeline 的固定报告。）

**结论**：全部零 spill——P1-B 打包布局与现有几何在寄存器压力下健康；
§4.1 几何搜索（row/wave/K tile）不会先撞寄存器墙。resume_norm（18920B）
是下一候选。

### 四、验证

mesh relay 全绿；`test-backend-ops -o MUL_MAT -b Vulkan0` 全 OK 且 stats 出图；
15/15 ctest 全绿；GPU 空闲、零新增内核错误。

### 五、下一步

1. §4.1 几何搜索现在有了测量器：每候选一行 VGPR/SGPR/LDS/code，离线选型、定义期固化；
2. resume_norm code size 18920 值得先看（最大者）；
3. P1-B/P1-C 真机收益仍需模型会话（安全门未批）。

## 下班交接｜2026-09-22（第十三轮，路线图审计 + P1-B 首步：Q8 权重定义期打包布局）

**分支：** `master`（本轮 commit 见 git log；基于 `3e3e11bb3`）
**主题：** 新路线图讨论稿到达后先做全量审计——发现讨论稿所列 P0/M1/M2 代码项**已全部由前几轮落地**
（讨论稿基点 `fe46587f8` 落后当前 57 个提交）；随后攻下路线图 P1-B 首步（Q8 打包布局）。
全程 CPU + 既有 GPU 回归，未启动模型。

### 一、路线图审计结论（讨论稿 vs 当前源码）

| 讨论稿条目 | 当前源码状态 | 证据 |
|---|---|---|
| P0-1 Q mailbox ABI 统一 | **已落地**：`tp5_late_q_control_offset/payload_word_offset` 单一来源；定义期对齐/范围断言；上轮补齐格式跟随 Q8 生产者 | collective.cpp:172–199, 2163–2171 |
| P0-2 F32-Q 输入生命周期 | **已落地**：exact 路径 `q_norm_barrier`（WAR，SHADER_READ→WRITE on trefs）先于 norm；aggressive 同构 | collective.cpp:5090–5115 |
| §2.2 MTP phase 切换打断图复用 | **已落地**：`ubatch_execution_phase` 对 predefined MTP 恒返 0；测试 `predefined_mtp_phase_invalidation_exemption` 覆盖 1→4→1→2 | llama-context.h:629–649, test-mtp-workspace:172 |
| M2 cycle 闭环账本 | **已落地**：draft/target/catchup/handoff 微秒 + tokens + eff 比率 + device_hidden 标志，单 cycle 与 summary 双格式化 | speculative.cpp:15–60 |
| §三 语义覆盖记录 | **已落地**：五类别（fixed_compute/dynamic_compute/data_movement/state_writes/dependency_boundaries）定义期评估 + fail-closed | ggml-vulkan-tp5-coverage.h 全文件 |
| §4.1 完成依赖收窄 | **已落地**：`synchronized_generation` epoch 检查避免冗余跨 context sync | llama-predefined-hidden.cpp:145–244 |
| M1 无效行不写 KV | **已落地**：`set_rows_indirect` EXTENT_TOKENS 间接派发 + 行边界可除性检查 | ggml-vulkan.cpp:10304+ |

**结论**：讨论稿的正确部分已全部在库；剩余项（P1-B/C、P2、P3、真机收益）全部需要真机测量。
本轮选择 P1-B 作为唯一可纯代码落地的下一项。

### 二、P1-B 首步：Q8 权重打包布局（本轮主交付）

**动机**：Q8DOT/UP_Q8DOT 内层每 k 迭代对权重执行 2 次 `pack_q8_pair`（i16→i32，
约 6–8 ALU），而 `dotPacked4x8EXT` 本体仅 1–2 VALU。权重流量 7MB/rank/token
（activation 的 320 倍），打包开销每 token 重复支付。

**实现**（位级等价，无需精度校准）：

1. `late_q8_packed { float d; uint qs_words[8]; }` 36B/块；索引 0 = 持久 done 标志，
   数据块 i 位于 i+1；块数与索引数学与 `late_q8_0` 完全一致；
2. `TP5_LATE_PACK_WEIGHTS` 一次性转换 shader（自禁用：done 标志后每次重放为单次 guard 读）；
3. pack dispatch 记录在 `cmd_late_pre` 头部（`mb_in` 后、inject 前），经 pre tape
   进入线性链——每 stage 重放均自禁用空转；
4. **标志初始化**（本轮关键安全设计）：plan 创建期以专用 fill CB + fence wait
   清零 packed 缓冲前 4 字节——防未初始化设备内存或先前 plan 释放缓冲的别名
   （flag==1）跳过本次 pack 而读到旧权重；
5. Q8DOT/UP_Q8DOT 绑定 0 改接 packed 缓冲；activation 维持 `late_q8_0`（ACT_Q8 每块写一次，
   打包成本被全部输出行摊销，不值得动）。

**代价**：+6% 权重显存（36B vs 34B/块）；每 plan 两个 packed 缓冲（W_down/W_up 同块数 102400，共 7.37MB）。

### 三、验证

- `test-tp5-plan` 新增 `test_tp5_packed_weight_layout`（36B/块、W_down/W_up 共享块数、
  边界、flag 索引不与数据重叠）；全套 all passed；
- 15/15 ctest 全绿（含 5 卡 mesh RELAY/STAR 96 轮）；GPU 全程空闲、零新增内核错误。

### 四、未闭合与下一步

1. **P1-B 真机收益**（内层 ALU 消除 vs +6% 显存）需模型会话，安全门未批；
2. P1-C（sidecar transport 三选一：host push / publisher / BAR pull）需真机测量；
3. 路线图 §4.1 的进一步收窄（epoch 级依赖而非 context 级）需资源寿命证明，未动；
4. 跨层 chunk 流式（P3-A）、单卡 MTP（P3-B 变体）维持路线图原判：先证明 Y/Q 双流收益再动。

## 下班交接｜2026-09-22（第十二轮，P1-A 对照器具三处修复：sidecar 格式键位 / barrier 容量覆盖 / norm_ready 发射）

**分支：** `master`（本轮 commit 见 git log；基于 `fe41c4363`）
**主题：** 路线图讨论稿 P1-A（「不带 sidecar 的 aggressive Q8 HC」同精度对照）的落地前提修复。上一轮交接
标注了「P1-A 端到端可能死锁」的隐患；本轮静态追踪完整路径后确认了三处可静态证明的缺陷并全部修复。
全程 CPU + 既有 GPU 回归，未启动模型。

### 一、三处缺陷与修复

| # | 缺陷 | 后果（修复前） | 修复 |
|---|---|---|---|
| 1 | `late_sidecar_f16` 以 `mode == AGGRESSIVE_Q8` 为键，但 F16 sidecar 是 **Q8DOT 生产者的属性**（P1-A 派发同一枚 Q8DOT） | P1-A 拿到 F32/exact 消费者：handoff 轮询 `status[6]`（仅 exact-path publisher 写）必到 `relay_handoff_timeout_ms` 超时 fail；即使等到也从 host_import F32 区读（错缓冲区）；LO_Q8 把 F16 packed 按 F32 位解码 | `plan.late_sidecar_f16 = plan.late_q8_fast`（`ggml-vulkan-collective.cpp:3259`） |
| 2 | ACT_Q8 可见性 barrier 只盖 `streams×width`（1 行），但分配为 `capacity_rows(4)×…`，shader 写 token<capacity 行 | token 2..4 写入对 Q8DOT 无序（依赖驱动保守行为）。LO_Q8→UP_Q8DOT 同病（`late_rank` vs `4×late_rank`） | 两处 barrier 尺寸改 `capacity_rows×…`（与分配一致） |
| 3 | P1-A 链上 `BARRIER_NORM_ACT` 只是 `late_steps` 验证器标签；实际命令 `mb_norm_lo`（p2[norm_end..lo_begin)）未被 P1-A 发射 | ACT_Q8 可能先于 norm 的 `sum_output` 写入可见前读 local_z（同一 trefs 缓冲区，RAW 无 barrier 无保证） | P1-A 发射改 `emit(p2, 0, lo_begin)`；split 校验加 `lo_begin` 边界 |

### 二、P1-A 发射结构调研结论（本轮确认，写进文档）

- LO_Q8/UP_Q8DOT **不在** P1-A 段内发射，而是作为下一 stage 的 `emit(incoming.p2, lo_begin)` 尾段——
  顺序契约（norm→ACT_Q8→Q8DOT→LO_Q8→UP_Q8DOT）**跨 stage 边界成立**，与验证器一致；
- P1-A 的 Q 集体链：Q8DOT 写本地 F16 sidecar → CPU 轮 `qctrl[READY]` → CPU 归约 → 发布 word 2
  （Q generation）→ 下一 stage 的 LO_Q8 `await_input` 消费——修复 #1 后全链自洽；
- one-shot 路径不可达 P1-A：`key.late` 仅由链路径的 `tp5_late_consumer_ref` 填充，one-shot 恒
  `late_plan=false`，不记录 late 内核（模式名可能误报，但无行为影响）。

### 三、验证

- `test-tp5-plan` 新增 `test_tp5_sidecar_format_keying`：Q8 生产者 ⟹ F16 sidecar + qctrl ready 轮询，
  对 REFERENCE/EXACT_F32/AGGRESSIVE_Q8/P1A 四模式断言；全套 all passed；
- 15/15 ctest 全绿（含 5 卡 mesh RELAY/STAR 96 轮 + 变异输入 + 延迟生产者）；
- GPU 全程空闲审计、零新增内核错误。

### 四、边界与未闭合

1. **P1-A 端到端收益测量需真实模型会话**（`GGML_TP5_P1A_NOSIDECAR_Q8=1` + HC 图 + MTP），
   安全门未批；本轮修复的是对照器具的可信性前提（否则测出的 A/B 对比是错的）；
2. 路线图剩余：P1-B（Q8 权重 pack 布局）、P1-C（sidecar transport 三选一：host push /
   publisher / BAR pull）、M3 多卡解禁、M5 组合——全部需真机测量。

## 下班交接｜2026-09-22（第十一轮，M4 首步：LateBind 运行时有效行直供内核）

**分支：** `master`（本轮 commit 见 git log；基于 `4ec7571c4`，含 41 个此前未推送提交一并推上）
**主题：** 未来路线图讨论稿中 M0–M2 已由前 54 个提交落地；本轮攻下 **M4 的第一个实质缺口**：
LateBind 全部 7 个内核的 push constant 把 `capacity_rows` 当执行行数烘焙，1 行 MTP draft
在容量 4 定义下执行 4 行 late 工作（M3 不变量 2「push-constant 不得把容量当有效」的现存违反）。

### 一、改动：运行时有效行经 RELAY header word 5 直供内核

1. **发布点（CPU）**：`tp5_relay_submit_epoch_chain` 在 `ensure_armed_epoch` 验证首个
   bank 空闲后写 word 5（值 = `active_rows ? active_rows : capacity_rows`）；后续 bank 在
   `tp5_relay_arm_bank` 传递信用验证后写入（`late_rows` 参数贯穿 `tp5_star_handoff`）；
   one-shot 路径提交前写入（无帧写 0 → 内核回退到 push-constant 容量，legacy 行为不变）。
2. **消费（GPU）**：7 个 late 内核全部改为读 header word 5 截断 token 循环——
   Q8DOT 经既有 QMailbox binding 读 `qmail[5]`；norm/LO 经既有 Inbox 读 `inbox[5]`；
   inject/ACT_Q8/Q/UP 新增只读 RowsHeader binding（DSL 计数 3→4 / 5→6 / 6→7 / 4→5）。
3. **inject 描述符改每 bank**（`late_inject_ds` 从每 rank 单份改为 `n_ranks × BANKS`，
   因新增 bcast 依赖随 bank 切换）；分配/释放/录制三处同步。
4. Word 5 与既有协议零冲突：generation（word 0/2）、counter（word 1）、n_elems（word 3）、
   relay probe 的 word 4 均不受影响；word 5 在 64 字节 header 内、F32 payload（word 16 起）之前。

### 二、验证

- `test-tp5-plan` 新增 `test_tp5_latebind_runtime_rows_protocol`：word 5 位置、发布值、
  legacy 0 回退、1→4→1→2 序列（每次发布真 active，非容量）；全套 all passed。
- **15/15 ctest 全绿**（tp5/mtp/predefined + 全部 vulkan：5 卡 mesh RELAY 96 轮 +
  变异输入 + 延迟生产者 + STAR 96 轮、GDN multistep C=8、FA capacity、command replay
  1→4→1→2 不可变描述符、output liveness）。
- GPU 全程空闲审计通过；journal 零新增 amdgpu fault/hung。
- relay probe `consumer failed rank=0` 为**基线既有行为**（stash 前后复现一致），非本轮回归。

### 三、边界与未闭合

1. **LateBind GPU 路径端到端验证需真实模型**（HC 图 + MTP 会话 + `GGML_TP5_LATEBIND=1`），
   须按安全门获批后执行；本轮覆盖传输层回归 + CPU 协议测试。
2. M4 剩余：多行 LateBind 的 `scatter/rho/Q/Y` 布局已有（`vk_tp5_latebind_layout`），
   但 `tp5_late_consumer_ref` 仍限 `capacity_rows <= 4`（`VK_TP5_DIRECT_COLUMN_TILE`），
   TARGET 容量化后需评估是否抬升。
3. M3 剩余：GDN fail-closed 门禁维持（`evaluate_target_capacity_admission`），多卡/全模型
   级验证后由负责人解禁。

### 四、路线图对照（讨论稿 → 现状）

| 讨论稿项 | 状态 |
|---|---|
| M0：Q sidecar 协议对齐 | ✅ 前 54 提交已落地（单源 helpers + Invariant 6/7 测试）|
| M1：MTP 单容量闭环 | ✅ phase 豁免 + 1→4→1→2 工作区/重放/图复用三重测试 |
| M2：MTP 成本闭合 | ✅ cycle ledger（draft/target/catchup/handoff 分项 + RAII settle）|
| M3：TARGET 容量化 | 🔶 区 0–4 落地，GDN GPU 单卡验证过，多卡 fail-closed 待解禁 |
| M4：LateBind 多行与数值隔离 | 🔶 本轮落地运行时有效行；P1-A 对照具已有；端到端真机待批 |
| M5：组合与更深重叠 | ⬜ 未启动 |

## 真机安全门（任何 agent 开工前必读）

真机测试必须小心；把机器弄死会造成好几天的时间浪费。**不要**未经检查启动大模型、叠加 GPU 负载或重启生产服务。

如果 GPU 挂住是很危险的，因为机器会检测 hang 然后自动重启，浪费很多时间。

- 部分 rank 提交失败时：**禁止**用 `vkDeviceWaitIdle` 赌 peer signal，也**禁止**主机伪造成功让消费者读未完成载荷。
- **严禁无保护/无界自旋**：无论 CPU 还是 GPU，任何自旋必须设置严格的有界退出计数（如有限迭代上限）或超时退避，且超时后必须执行标准错误路径（如 fail/abort 并排空退出），绝不允许死循环自旋。
- **严禁在 Shader 中引入无界 while 自旋**：GPU 计算单元死锁会直接阻断 amdgpu 驱动退出，引发内核 `dma_fence_wait_timeout`，导致 `khungtaskd` 触发系统级 Panic / Watchdog 重启！
- **严禁依赖不稳定的设备级自旋等待**：下行信号必须使用驱动原生或有安全保障的同步原语（如 Timeline Semaphore 或受控的事件机制），不得绕过硬件调度规范。经明确批准的 RELAY 例外必须保持固定有界 spin_max、匹配 generation、失败态写回及真实提交值的原生排空；该固定上限必须在运行前按已验证配置覆盖完整的 GPU→CPU→GPU handoff 预算，不能因人为设得过小而把正常 payload 发布误判为 timeout。不得把此例外推广为无界 GPU 自旋，也不得在 timeout 后动态或递增地扩大上限重试。
- **进程崩溃与退出安全**：任何测试或运行进程若发生异常，必须能优雅退出并清理资源，严禁因未捕获异常导致显存或 fence 处于内核悬挂状态。

## 🛡️ 2026-09-21 RERoT Vulkan 多 Pen (P=6) 异步命令竞争与闭环解决记录

在单张 AMD Radeon RX 6800（RADV 驱动，Navi 21）上加载 `Ternary-Bonsai-2-27B-PQ2_0`（Qwen 3.8 混合架构，64 层，含 GDN 循环状态与 Hadamard 激活变换），使用 RERoT 最优笔容量 $P = 6$（`--rerot-pens 6 --rerot-people 1`）执行多 Lane DAG 时曾因异步命令竞争触发 `radv: GPUVM fault detected ... ErrorDeviceLost`。

**根因与修复闭环：**
1. **异步队列执行与前缀重构销毁边界竞争**：在 DAG 前缀重构分支（`RERoT DAG prefix rebuild`）执行破坏性显存重置（`seq_rm_recurrent` / `clear_hand_row`）前，前序的异步批处理计算命令仍在 GPU 上调度执行。通过在 `server_context_impl::rerot_rebuild_dag_prefix_memory` 显式插入 `llama_synchronize(ctx_tgt)` 形成栅障，确保前向传播在 GPU 彻底落盘后再执行显存清理；
2. **显存写入队列同域保证**：将 `ggml_vk_buffer_write_2d` 与 `memset` 的同步修改收敛至计算队列（Compute Queue）并补齐栅障；`llama-memory-recurrent` 的清空逻辑显式接入 `ggml_backend_sched_synchronize`。

**真机端到端全量通过验证：**
在 $P = 6$ 最优笔容量、`-c 262144` 满规格上下文下运行 `scripts/rerot-target-ornith-multi-lane.py`：
- **用例 1（Flat 2-Worker 并发 DAG）**：$25 \times 12 = 300$ 与 $15 \times 16 = 240$ 并发计算并顺利综合出 $\mathbf{540}$，耗时 24.31s，单次自然 `stop`，无内部 Token 泄漏；
- **用例 2（A $\to$ C 依赖链且 B 独立并发）**：$A=210, B=600, C=260 \to D=860$ 综合计算正确闭环，耗时 58.70s；
- **用例 3（菱形依赖 DAG：1 $\to$ 2/3 $\to$ 4）**：$b=100, v_1=300, v_2=500 \to 800$ 正确返回。
三项拓扑全绿（100% PASS），RERoT 多 Pen (P=6) 生产并发能力彻底稳健闭环。

### 🚨 2026-09-19 RELAY 复发事故

真实 `llama-server --tp5-sync relay` 在第二个请求的 epoch-chain 重用阶段发生 payload timeout，随后主机非正常重启；上一启动周期的 journal 损坏，无法从持久日志恢复完整 hang 栈。**RELAY 的 local-VRAM shader doorbell / GPU 等待路径仍是显式 opt-in，不改变默认 TIMELINE。** 本轮在用户明确批准后恢复了原始自治 bounded-spin 路径，并完成五卡 mesh 与短 real-model decode 验证；这不等于长时间压力稳定性证明。禁止把 timeout 当作 retry 而动态或递增地增大 spin bound；运行前可在明确批准的单一配置上设定一个覆盖完整 GPU→CPU→GPU handoff 预算的固定上限。禁止以 `vkDeviceWaitIdle` 或进程 abort 作为恢复手段。每个已提交 timeline 值必须在任何资源释放前由驱动原生 wait 有界排空；失败时必须保留无法证明已完成的资源。

#### RELAY 名称与语义铁律

`relay` 只指**自治、双方预等待的 state-observation relay**：direct terminal producer 在原模型 compute 中直接按所选 wire 宽度（F16/F32）写 host-imported payload，CPU 在 queue submit 前已经轮询 stable route-ready；producer 只把 ready 状态从 0 改为 1，不存在 callback/唤醒。下行 P2 同样提前驻留，在本队列中有界轮询本卡 local-VRAM generation；CPU reduce 后只写 payload 与同代 generation，GPU 不接受 CPU semaphore/二次 submit 来推进 P2。旧 P1 仅是 direct producer 不具备时的兼容 fallback，不属于主热路径。`spin_max` 是覆盖完整 GPU→CPU→GPU handoff 的预先校准协议预算，不是缩短正确 handoff 的人为 timeout；P2 必须在看见匹配 generation 后消费 payload、写 completion，超出该固定预算必须写失败态并退出。

**严禁以任何降级冒充 RELAY：** CPU 发布 payload 后才提交 P2、P2 只做一次 doorbell 检查、host timeline signal、CPU 直接推进 P2，均是 `CPU-gated STAR` 或其他非-RELAY 路径。它们不得使用 `--tp5-sync relay`、`GGML_TP5_SYNC=relay`、RELAY 测试名、RELAY benchmark 标签或 RELAY tok/s 结果；不得静默 fallback、别名伪装或在报告中混称。

#### RELAY 受控重入证据（2026-09-19）

用户明确批准恢复原始 RELAY 后，当前实现只允许上述真实自治语义：五卡 test-vulkan-tp5-mesh --sync relay --rounds 1 --elements 2560 --check-all 通过，包含真实 GPU graph-producer、位级 constant-input 校验和 FD delta 0。随后同一固定配置、同一 GGUF、同五张 Vulkan RX 6800 上完成 real llama-server 请求复用：首个请求和第二个请求均 HTTP 200，epoch-chain 重用通过，服务无 Compute error；server 通过受控 SIGTERM 停止。

这些是短请求的当前证据，不是长时间压力稳定性授权；任何失败仍必须 fail-closed、原生有界排空，不能静默回退为 CPU-gated STAR，也不能把降级路径命名为 RELAY。

### 🚨 2026-09-19 TIMELINE 真实模型事故

`llama-server --tp5-sync timeline` 的首个完成请求正常返回；随后一次 epoch-chain 提交期间，内核在 `0000:06:00.0`（`card1`）记录 `llama-server` 的重复 `[gfxhub]` page fault：`UTCL2/SQC(data)`、`PERMISSION_FAULTS=0x3`，且有 200 个回调被抑制。该启动周期在 fault 后无正常 shutdown 记录地结束，后续启动发现 journal 未清理；`pstore` 无 panic 记录。因此只可确认 TIMELINE 提交、GPU VM fault 与非正常重启的时间关联，**不得把任何单一组件宣称为已证实根因。**

- **停机线：** 在独立稳定性证明前，除用户明确批准的单次、日志保护的 RELAY/TIMELINE 验证外，禁止重启任何 GPU model/server/mesh 压测来“复现”或测速；不得关闭 watchdog，也不得用 `vkDeviceWaitIdle`、进程 abort，或在 timeout 后动态/递增地扩大 spin bound 作为恢复手段。运行前已校准、覆盖完整 GPU→CPU→GPU handoff 的固定 spin_max 不属于这种恢复性重试。
- **图执行唯一真理铁律（纯 Predefine 路线）：** 图执行全面废弃 Cache 架构，不再使用 Cache 概念与逐轮 Validation（无需运行时重复 validation/fingerprint 校验开销）。预先做好的图定义即为唯一真理（Predefined Graph Definition is Single Source of Truth），进入执行阶段直接绝对复用预定义图句柄与拓扑，彻底消除冷启动与每轮校验抖动。
- **重启前提：** 先完成非侵入式 GPU idle、AER、温度、前一启动 journal/pstore 与安全 teardown 审计；之后才可在显式批准下按单个、受日志保护的 TIMELINE 请求逐级恢复。
- **新增停机证据：** 修复上述 replay bypass 后，`GGML_TP5_CHAIN_CACHE=0` 的受控 server 连续两次 HTTP 请求均返回，随后 `SIGTERM` 正常退出（exit 0）；但该启动周期仍以无 clean-shutdown、`pstore` 空、journal 未清理的方式结束，且新 boot 的 IPMI hardware watchdog 仍为 5 min。此前后未记录新的 amdgpu page fault，因此不得把两次 HTTP 返回或 exit 0 当作 teardown 安全证明，更不得在未获明确批准时据此启动 `GGML_TP5_CHAIN_CACHE=1` 或 tok/s 压测；后续明确批准的受控 RELAY/TIMELINE 测量不改变该限制。
- **纯 Predefine 执行模式：** 在纯 predefine 路线下，图预先一次性构建/定义完成并永久生效，执行期作为唯一真理直接 100% 绝对复用，跳过动态检查与 validation 开销，彻底抹平首轮与后续轮次间的抖动。
- **已实现、但未真机验证的 teardown contract：** TIMELINE/DRM 现在记录每个 rank 实际成功提交的最高 timeline 值；partial submit 只返回失败，meta 不再 fallback。`comm_free_safe` 仅等待这些真实提交值，wait 失败则让 meta 保留 communicator、graph、buffer 与 child backend，绝不 `GGML_ABORT`、提前析构或用 `vkDeviceWaitIdle` 伪造完成；正常 teardown 先释放 graph/buffer，再释放 backend。此项只有静态构建/CPU 证据，**不是**恢复 GPU 测试的授权。

### 🚨 2026-09-18 事故反思与血的教训

**事故现场**：
内核记录：`test-vulkan-tp5-mesh: segfault at 0` -> 进程退出时因未排空的 GPU 任务或信令悬挂导致 `exit_mmap` -> `amdgpu_hmm_invalidate_gfx` -> `dma_fence_wait_timeout` 卡死在 D 状态超过 60 秒 -> 内核 `khungtaskd` 判定为 hung_task，触发内核 Panic，系统被 Watchdog 强制重启！

**深刻反思**：
1. **绝对不可心存侥幸**：任何信令和自旋优化，无论理论上多快，一旦脱离了有界保护和硬件安全边界，就会把整台物理机拖入死锁崩溃！
2. **不准关掉 Watchdog**：Watchdog 是系统的最后底线安全保障，必须通过写出健壮、安全、有界的工程代码来确保不触发 Watchdog，而不是关掉报警！
3. **每步操作必须首先进行 GPU 状态审计**：执行任何高负荷或并发操作前，必须保证 GPU 处于干净空闲状态（`busy=0%`），失败时绝不允许盲目重试或让未配对的命令进入队列。


---

## 下班交接｜2026-09-22（第二轮，研究线收口）

**分支：** `master`（本轮 commit 见 git log；基于 `bd87b9e73`）
**主题：** 把十问中剩余五问（Q2/Q4/Q8/Q9/Q10）的数学与契约层落成代码，并补上 Q5/Q6 的 F32 数值门实测。十问的“数学上成立”部分全部收口。**未启动任何模型/GPU 测试**（开发机单 780M iGPU，遵守真机安全门）。

### 一、已合入

全部落在 `src/llama-rerot-math.{h,cpp}`（纯 FP64/位级参考层，无 KV/server/graph 依赖）＋ `tests/test-rerot-math.cpp`：

|问题|内容|关键函数|
|---|---|---|
|Q2|span 级 reader view：段内相位常数→整段共享一枚 effective Q；因果截断一次比较；碎片化度量|`span_effective_pos` / `span_causal_len` / `span_long_fraction`|
|Q4|结构/数值分离：run order 签名仅随结构事件变；virtual starts 是前缀和，增长走增量更新|`run_order_signature` / `virtual_starts(_after_growth)`|
|Q8|跳块误差界（近似路线）：Cauchy–Schwarz 质量上界 + 凸组合输出偏差界；可执行反例固化“无有限充分统计量”|`skip_mass_bound` / `skip_output_bound`|
|Q9|联合采样契约：per-pen RNG 流（(base,pen) 派生）与 cohort 大小/行序无关；temperature→top-k→top-p、最低索引 tie-break；贪心按块归约 argmax|`joint_sample_seed/row` / `joint_argmax_rows`|
|Q10|frontier 网格验证：按列推进、STRONG barrier-after，依赖死亡传染；naive 逐行验证引擎作为可执行反例保留|`verify_grid` / `verify_grid_naive`|
|F32 门|Q5 因子化 24 步、Q6 WY 折叠 T=8 的 F32 重结合误差实测|`test_f32_gate`|

### 二、验证证据

- `test-rerot-math`：0 failure。新增五族全部对拍独立 oracle；Q10 依赖追踪 vs naive 分岐断言（accepted=2 vs 3）。
- **F32 门实测数据**：Q5 低秩因子化 24 步 F32 vs FP64 稠密——**相对误差 ~1.9e-7（有界区间）/ ~5.1e-7（弱衰减区间）**；Q6 WY 折叠 T=8 F32 vs FP64 逐步——**绝对误差 2.5e-5**。结论：两族在 F32 下都不逐位一致，GPU 化验收门按此量级设。
- rerot/xkv/flashprefill ctest 全家 **45/45**；`git diff --check` 干净；`scripts/rerot-dag-reference.py` 逻辑检查全过。

### 三、关键事实与纠错记录

1. **Q10 期望值纠错**：C 读 B 的 col-0（当时存活），C 的 col-1 存活；C 死于 col-2（读 B 的被拒 col-1）。正确引擎 accepted[C]=2、naive=3——这正是逐行独立验证在跨笔读下接受率虚高的可执行证据。
2. **Q8 无充分统计量反例**：keys {-1,+1} vs {0,0} 数量与一阶矩相同但 Z(q) 不同（2cosh(1)≠2）。同一 query 的块结果可精确合并（Q3），不同 query 不能因读同一历史互用。
3. **Q5 F32 绝对误差的误导性**：弱衰减区间绝对误差 26.5，但相对误差 5.1e-7——绝对误差由状态指数增长主导，门必须按相对误差设，否则会把重结合误差与状态放大混为一谈。

### 四、下一步建议

1. **生产化优先级**：Q2/Q4 已有明确接入点——`llama_rerot_split_table_fragments` 调用点（`llama-kv-cache.cpp:5124`）按 run-order 签名缓存 fragments，结构事件才重算；写入布局维持长 span（`span_long_fraction` 为验收指标）。
2. Q9 联合采样需 server 协议改动（`server-context.cpp` 采样路径），风险大，建议先在测试 harness 里对拍现有 per-pen 采样轨迹。
3. Q10 收益账：独立接受率 a、b 笔下保守方案整步通过率 a^b；联合草稿必须预测多笔相互影响后的下一 frontier。验证窗口先收在固定 cohort、普通 BODY 区间。
4. GPU 化顺序：Q3（多读者共享块，纯数据供给复用）→ Q6（WY 折叠，F32 门 2.5e-5）→ Q5（低秩，F32 门 ~5e-7 相对）→ Q7（LUT，需实测“减乘法≠减耗时”）。

---

## 下班交接｜2026-09-22（第十七轮，十问收敛后第一轮：F32 数值门 ×5 ＋ Q3 CPU A/B 1.36–1.97×）

**分支：** `master`（本轮 commit 见 git log；第十六轮 `0cf626757` 在历史里）

### 一、本轮定位

十问收敛稿定调：**以 frontier 为执行单位、公共数据块为供给单位、逻辑 Lane 为状态所有者、结构事件为重新定义边界**。前十轮已把 Q2–Q10 的数学层（FP64 oracle）全部落地；本轮专攻「数学上成立」到「可以进入 GPU 实现」之间缺失的一环——**F32 验收入口**，并把十问问题三（一块公共 KV 服务多个 reader）第一次跑出真实 CPU 倍数。

### 二、改动

1. **span coverage 证据**（`llama-rerot-profile.{h,cpp}`、`llama-kv-cache.cpp`）：新 ledger 字段 `span_rows` / `span_row_spill` / `span_row_coverage`＋ring event。生产形状 **coverage=1.0000**；交错写入对照跌到 0.666（spark 探针，未入库）。**span 长 63 是 merge 分组尺寸不是 run 长度**——碎片信号看 spill，不看 span 数。
2. **F32 门补三族**（`tests/test-rerot-math.cpp::test_f32_gate`）：
   - Q3 float 在线 merge vs FP64 全量 softmax：**rel 4.3e-07**（12 块 × block max 相差数量级的极端 rescale）；
   - Q7 对**真实 ggml `block_pq2_0`**：bitplane/LUT 恒等式与 ggml decode+dot 差 2.98e-08（在其自身反向累加带内），FP64 逐位一致；
   - Q9 对**生产 `llama_sampler_chain`**：40/40 kept-set 一致。
3. **Q3 CPU A/B shadow oracle**（`tests/test-rerot-shared-block.cpp`，新 ctest）：A=R 次逐 reader `DdvrQsideGqa`；B=一次共享 pass。输出互拍＋计时。

### 三、实测

|指标|数字|
|---|---|
|Q3 A/B speedup|**R=2：1.36×｜R=4：1.63×｜R=6：1.76–1.81×｜R=12：1.89–1.97×**|
|Q3 rel_err|1.5e-6 – 3.9e-6（float online vs FP64 full）|
|Q7 ggml-block err|2.98e-08（FP64 下 0）|
|Q9 kept-set|40/40 一致|
|span coverage|生产 1.0000 / 交错 0.666|

结论与十问一致：共享的是数据供给而非 FLOPs；R 越大越高，是「一块 KV 服务尽可能多合法消费者」而不是「全 K 笔塞进一个 workgroup」。

### 四、契约级踩坑（写进 RERoT.md §21 第十七轮，务必读）

1. **Q9：survivor SET 才是契约**。生产 `partial_sort` 对相等 logit **不稳定**——并列时 survivor ORDER 是实现定义的。Gate 必须两侧排序后比较集合、tie 种在 k 边界之上、贪心只在唯一最大时比较。RNG 体系不同（per-pen xorshift vs std::mt19937），**逐 token 轨迹不可直接对拍**。
2. **Q9：生产链序 `top_k→top_p→temp→dist`**，top-p 看**原始 logit**；参考实现先除温度会系统性分叉。
3. **Q3 A/B：`DdvrQsideGqa` 是 kv-head-major 布局**（`raw_k[(hkv*n_keys+key)*d]`）。新读者用单 slab 会越界 segfault（本轮实测）。A/B 两侧必须填相同物理行，否则比的是两份不同数据。
4. `print_summary`/TSV 行数被 `test-rerot-profile.cpp` 钉住（现为 98）；加 profile 字段必须同步。

### 五、边界与未闭合

1. **defrag 长 run 臂：评估后推迟**。交错形状 coverage 0.666 是真实信号，但 `compact()` 在该形状无恢复、且保长 span 要改分配器（`find_slot` 环语义）——收益中等、风险高。等出现「碎片化 shape 成为生产常态」的证据再动手。
2. **Q3 GPU 化**：CPU A/B 证明供给组织方向正确（1.4–2×），但 GPU 侧 kernel 契约（共享 K/V tile + per-reader m/z/u registers）仍是最大真实项，需批准＋真机。
3. 十问 Q1（图定义期公共子表达式单生产者）、Q4（结构/数值分离接入 flashprefill `split_table_fragments` 缓存）、Q5/Q6/Q7 GPU 化均未动。
4. Q8（近似跳块）、Q10（联合多步投机）按十问要求**不与等义共享混记收益**，仍属独立研究线。

### 六、下一班建议

1. Q4 接入点最便宜：`llama_rerot_split_table_fragments` 调用点（`llama-kv-cache.cpp:5265`）按 run-order 签名缓存 fragments，结构事件才重算（数学层已有 `run_order_signature`）。
2. Q3 GPU kernel 契约设计先落 `RERoT.md`：共享 K/V tile 布局 + per-reader (m,z,u) 常驻 + reader_visible 边界的一致性规则，再谈 SPIR-V。
3. F32 门数字已固定为验收基线：Q3 4.3e-7 / Q5 1.9e-7 / Q6 2.5e-5 / Q7 0 / Q9 set-exact。GPU 化后逐项对拍。

---

## 下班交接｜2026-09-22（第十六轮，十问问题二落地：span 侧信道消灭默认 validate 成本——4.1ms / validate 免费）

**分支：** `master`（本轮 commit 见 git log；第十五轮 `4f33d005a` 在历史里）

### 一、本轮定位

按十问中的问题二（reader view 应表现为「地址范围＋位置偏移＋可见性边界」而非逐 key 排列）审查第十五轮后的剖面：host 侧固定成本只剩两处「逐 entry」付费——默认 validate 的 3.1M-entry max-reduce（~4.6ms）与 fill_spans 的 2×25MB 跨步转换。生产 decode 形状里 12/13 段是整个连续范围（same effective Q、ascending key range），为每枚 key 单独付费没有信息量增益。

### 二、改动

1. **`llama_rerot_attn_layout::span` 侧信道**（`src/llama-rerot.h`）：`{key_start, count, group_index, entry_index}`。merge 的 bulk 分支每段 push_back 一次；entry_index 是全局 entry 槽位（merge 时 sink cursor + 已发射行），group_index 在 merge 解决组号后回填。**`entries` 仍是权威契约**（op params live_entries、直接 set 路径、测试直读）。
2. **tier-1 validate 改查段边界**：span 表范围检查＋未覆盖槽的 gap 扫描（同一游标纪律）。R=12 val 相 4.6ms→3us，**validate=1 与 validate=0 首次等速**。
3. **fill_spans 按段展开 i32 staging**（`src/llama-graph.cpp`）：span 之间 scalar entry 原样拷贝、span 之内 `key_start+k` 连续范围＋单 group 无分支内循环；拒绝 entry 乱序/越界。
4. **span reserve 绑诚实上界**：`(world run 数+1) × 每 reader query 行数`。
5. **缓存级测试新增 span 契约断言**（`test_rerot_shared_reader_multi_query`）：零长度拒绝、三重边界、**span 展开逐 slot 复现 authoritative entry 流**、entry-ordered 单调。MTP 重复 storage 形状正是最可能暴露分叉的形状。

### 三、实测（cache-prod，min-of-5）

|形状|14轮|15轮|16轮 val=0|16轮默认|
|---|---|---|---|---|
|R=12 K=262144|37,778|4,311|**4,142**|**4,162（首次免费）**|
|R=6 K=131072|10,082|1,180|1,179|1,180|
|R=6 K=65536|4,863|610|611|618|

默认 validate 成本归零——「为速度关审计」的取舍消失。剩余 4.1ms 逼近 25MB emit 带宽下界；再降必须动 kernel 契约（span tensor 直喂 op，去掉 entries 物化）。

### 四、关键纠错（写进 RERoT.md §21 第十六轮）

第一版让 bulk 分支跳过 entries 写入、只留 span → 11 个 cache 测试全红（同一 key 重复 5 次）。根因：**entries 是既有消费者的权威流，span 只是旁路**；"bulk 跳过 entries"版本在任何直读 entries 的路径上都是错的。修正为 builder 仍写 entries。教训：段描述化的收益点在 loader/validator（O(段)），不在 builder 的发射契约——kernel 侧不吃 span tensor 之前，逐 entry 物化删不掉。

### 五、下一班建议

1. **kernel 契约升级（Q2 完全体）**：`ggml_flash_attn_ext_rerot` 接受 span tensor 变体（或 entries 的 RLE 编码），CPU/Vulkan 两侧实现；这是把 25MB upload 再砍一个数量级的唯一路径，但属 kernel 契约变更，需单独一轮＋全 op 测试。
2. 十问问题一（公共子表达式单生产者）与问题三（公共 KV 块服务多 reader）需要图/kernel 侧改动，GPU 批准后启动。
3. Q7（PQ2_0 LUT）已证伪 vec_dot 粒度；只值得 GEMM 级共享表重做。
4. Q5/Q6（GDN 共同基底＋块递推）可在 CPU 侧继续推进数学层。

---

## 下班交接｜2026-09-22（第十五轮，size 阶段价值初始化清零＋groups append 化：R=12 4.3ms / 69×）

**分支：** `master`（本轮 commit 见 git log；第十四轮 `09dcdd4f6` 在历史里）

### 一、第十四轮后的剖面疑点

第十四轮 phase 细分暴露：`size` 阶段占 **28–38ms**（R=12 K=262k），而 emit 仅 4.3ms —— 一个 builder 的“预分配”步骤比全部数值工作贵 6 倍。逐层定位后确认三处结构性浪费，全部与“清空 scratch 再重建”的惯性写法有关。

### 二、本轮改动

1. **`clear()` 取代对象重置（根因）**：`rerot_build_attn_layout` 入口原先 `result = llama_rerot_attn_layout{}`，把调用方 scratch 的 capacity **整体释放**——每次 frontier 重新 reserve `entries` 25MB＋`groups` 37MB（page fault + zero-fill 主导）。`llama_rerot_attn_layout::clear()` 改为只重置 `n_queries`/`query_offsets`，**不动 entries 的 size**：稳态下 size 等于上轮 emitted cursor ≈ bound，builder 的预 size 退化为 no-op，capacity 全程存活。
2. **groups 改由 sink append**：`rerot_emission_sink::groups` 由裸指针改为 `std::vector<llama_rerot_attn_group> *`，merge 产组时 `push_back`（capacity 已 reserve 至 entry_bound——只占容量不 zero-fill）。原 `groups.resize(group_bound)` 把整个 37MB 数组 value-init 置零，而实际组数只有个位数——**同时这个 resize 的“增长路径”还会把 sink 已写入的组覆写为 0**（本轮被多 query MTP 形状测试抓住：`test_rerot_shared_reader_multi_query`/`incremental_decode` 的 effective_pos 全 0）。entries 保持 raw cursor 预 size（emit 全量覆写，无可避免但已被 size 保持抵消）。
3. **const 段零拷贝引用**：`eff_list` 增 `key_ptr/key_n`，contiguous+identity 段**直接引用 `run->fast_keys`** 而非拷进 per-query 缓冲；每 list reserve 按实际内容规模（base=owned 数、段=run 行数、const=1 slot），取代统一 `keys/2`（R×13 lists ~1.5MB/list 的分配浪费）。
4. **PQ2_0 LUT 负结论（Q7 记录）**：十问第 7 问的 16 项 LUT 在 vec_dot 粒度实测 26× 慢（2178→57650us）——T-MAC 的收益前提是**表服务大量权重行**（GEMM 形态），单 vec_dot 每 4 activation 重建表是纯亏；x86 路径已有 AVX-512-VNNI 结构化利用。**结论：Q7 要推进必须以 GEMM 级共享表重做，vec_dot 层面已证伪**。

### 三、实测（cache-prod：经装配 scratch 的生产路径，min-of-5）

|形状|14轮|15轮 validate=0|15轮默认 validate|基线（班11前）|
|---|---|---|---|---|
|R=12 K=262144|37,778us|**4,311us（69×）**|**9,307us（32×）**|297,450us|
|R=6 K=131072|10,082us|**1,180us（76×）**|2,420us|90,135us|
|R=6 K=65536|4,863us|**610us（77×）**|—|46,919us|

默认 validate tier-1 在 R=12 固定 ~5ms；val=0 时 numeric+emit=4.3ms，逼近 25MB emit 的带宽下界。**host 侧布局构建实质归零**；下一档收益在 GPU 侧（Q2 段描述：25MB entries 压到段级张量）。

### 四、调试记录（重要教训）

- “`emit_cursor != total` 不抛却数据错”类 bug 的排查法：builder 内加 `[grp] best` 与 cache 侧 `post-build/post-resize` 两级 dump，最终把零点定位在 **`resize()` 增长路径的 value-init 覆写**——C++ vector 的 size 增长会零填 POD，sink 裸指针写越过了 size 边界，收尾 resize 把写好的数据擦掉。**教训：raw-cursor 写入必须保证 vector 的 size 在写入前已覆盖写范围**（entries 预 size 满足，groups 改 append 才满足）。
- 磁盘 100% 一次：build/bin 下每次 make 生成新版本化 .so（libllama.so.0.0.11037 → 11097 各 124MB），**21 个陈旧版本吃掉 ~2.6GB**。已按 `readlink(base) != f` 清理，后续每班可重复执行；另清 ccache 3.7G。
- 测试二进制若不是最新（stale .o 链接）会出现 `corrupted double-linked list`/std::sort 死循环等假警报；改类布局后必须全量重编测试目标。

### 五、下一班建议

1. **Q2 段描述化（GPU 侧最大项）**：entries（2,N i32）改为段张量（start,count,effective_pos）——R=12 时上传从 25MB 降到段级；需要动 `llm_graph_input_attn_rerot::fill_spans` 与 attention op 契约，CPU 参考路径先落地。
2. Q3 GPU 化（公共 KV 块多读者）：真机批准后启动。
3. Q5/Q6（GDN 共同基底＋块递推）：`llama-rerot-math` 已有 FP64 参考层，可补 F32 门禁量化。
4. PQ2_0 若要推进只有 GEMM 级共享表一条路（见上）。

---

## 下班交接｜2026-09-22（第十四轮，直写 Sink 消除 Assembly＋三档 Validate＋Fast-Append：41.1ms / 7.23×）

**分支：** `master`（本轮 commit 见 git log；第十三轮 `a6fb490f3` 在历史里）

### 一、本轮改动（消灭独立 Assembly 与重载 Validate 开销）

针对十问文档第 2 问（避免逐 token 重复搬运）与第 4 问（固定结构下仅付数值代价）：

1. **直写 Sink `rerot_emission_sink`（Assembly 阶段归零）**：
   - numeric pass 的 k-way merge 增加直写 sink：直接分配好最终的 `result.entries` 与 `result.groups`，merge 产出时**内联打标 query_index 并加上 group_base 偏置**。
   - 彻底移除了中间逐 query 的 `llama_rerot_query_layout` 对象创建，以及 cache 层随后遍历拷贝的二次装配（Assembly 从 ~22ms 缩减至 0）。
   - `llama_kv_cache::rerot_assembly_scratch` 与 `rerot_multi_scratch` 提供跨 frontier 的向量容量保留与热页复用。
2. **三档 Validate 门控（默认档 53ms → 5ms）**：
   - `validate_mode = 0`：仅做 O(groups + queries) 顶层单调性及范围检查。
   - `validate_mode = 1`（默认）：增加 SIMD 友好的全局无分支 max-reduce 范围检查，消除 per-entry 双重分支预测开销，由 ~53ms 降至 **~5ms**。
   - `validate_mode = 2`（偏执审计）：保留全量 per-query 字节位图重重检查与 group->query 一致性反查。
3. **`try_append_key_fast` 正式接线**：
   - `apply_ubatch` 的 flush 循环中，优先对连续+均匀的尾部追加调用 `try_append_key_fast`（O(1) 尾追，跳过桶扫描和 touched-run 排序）；非连续/非均匀项安全回退至批量 `upsert_keys`。
4. **隐蔽 Bug 修复**：
   - 修复 `multi_reader_numeric_pass` 中 `L.const_eff` 在跨 query 列表复用时未被清空的别名污染（上一个 query 若为 const_eff，会导致下一个普通 query 的 `head()` 错误读 `eff[0]`）。
   - 修复 `got/want` 测试中断言比较器的 cross-vector 迭代器混淆死循环。

### 二、实测对比（min-of-5，vs 班次 11 基线 297,450 us）

| 形状 | 基线 (11轮前) | 13轮完成 | 14轮 (默认 validate=1) | 14轮 (validate=0) | 相对基线加速 |
|---|---|---|---|---|---|
| R=12 K=262144 | 297,450 us | 48,472 us (val=0) | **41,137 us** | **37,778 us** | **7.87×** |
| R=6 K=131072 | 90,135 us | 9,956 us (val=0) | **10,743 us** | **10,082 us** | **8.94×** |
| R=6 K=65536 | 46,919 us | 5,244 us (val=0) | **5,772 us** | **4,863 us** | **9.65×** |

默认校验档从 120ms 压缩至 41.1ms（提升 2.92×）。

### 三、验证与鲁棒性

- 全 rerot/xkv/flashprefill 电池 0 failure（test-rerot-view/attn/runtime/ddvr/math/parser/profile, test-xkv-runtime/factor/reader, test-flashprefill-attn/routing/select/state 全部 PASS）。
- ASAN 干净（/tmp/asan13，0 错误）。
- Phase 1–7 增量与重建测试全部通过。

### 四、下一班建议

1. **GPU 侧 Q3（Hydragen 式公共 KV 块多读者共享）**：进入真机 Vulkan 阶段，验证片上多读者并行注意力的寄存器开销与带宽节省。
2. **张量化段描述（Q2）**：将 entries 矩阵由点列表转为段列表（起始 cell, count, effective_pos），将 GPU 搬运量从 25MB 压缩至几十字节。
3. **GDN 低秩增量基底（Q5）**：验证多 pen 共享 GDN 基底 $B$ 的 FP64/F32 精度误差界。

---

## 下班交接｜2026-09-22（第十三轮，cache 级三连击：6.1–9.1×＋emission 公式真 bug 修复）

**分支：** `master`（本轮 commit 见 git log；第十二轮 `3e5a13561` 在历史里）

### 一、本轮改动（第十二轮摊销构成审计 → 三项依次消灭）

第十二轮后 R=12 K=262144 摊销：ownership 位图 48.7ms / 数值 pass 32ms / assembly 23ms / validate（另计）69ms。

1. **ownership 列增量化**（48.7ms → 0.04ms）：cache 持 `rerot_world_owned`（per-seq 位图，world records 位置索引）；`apply_ubatch` 增量（purge 清位/写 cell 重写位/共享 cell 只清被删 seq 位），`ensure` 重建同扫重填。**顺带修一个真 bug**：purge 收集原来用 `seq_has` 会把共享 cell（seq_count>1）误从 world 删除——改为匹配 seq_rm 的 `seq_count==1` 释放条件，共享 cell 只清位。
2. **validate 门控**：`LLAMA_REROT_LAYOUT_VALIDATE=0` 跳过 per-entry 扫描（69ms → 4us），保留 O(groups+queries) 结构检查。默认全开（保守）。纯 Predefine 铁律：builder 是唯一真理。
3. **emission 重写（最重要）**：
   - **Bug A**：own-row 查找取 last storage match 而非 last **passing** match（同 pos 不可见重写行导致 qvp 偏移）。
   - **Bug B（根本性）**：k-way merge 把 d-order 位置当可见序号——run 含不可见行占位时全部 effective 平移。**正确公式 `eff = qv + storage − (vis_before + tagged 前缀 passing 数)`**。重写为 per-list eff 数组（base=storage 序、段=tagged 序；非单调时段内 stable_sort）＋k-way merge＋contiguous identity 的 const-eff bulk（memcpy key_ids＋常数 eff＋整段 fill）。`llama_rerot_build_query_layouts_shared` 同步修复——**multi/shared/oracle 三方一致**。
   - 教训：d-order 的"dev 排序 = 可见序号"假设在 run 内重复 storage（MTP verify 形状）时静默失效；第十一轮测试形状不触发。test-rerot-view 现有该形状（200 iter 随机）钉住。

### 二、实测（min-of-5，validate off，vs 第十二轮前基线）

|形状|基线|本轮|加速|
|---|---|---|---|
|R=12 K=262144|297450 us|**48472 us**|**6.1×**|
|R=6 K=131072|90135|**9956**|**9.1×**|
|R=6 K=65536|46919|**5244**|**8.9×**|

### 三、验证

- `test_rerot_world_incremental_decode` 新增 **Phase 7 共享 cell purge**（seq_cp→apply 覆盖共享位置：record 留 world、只清被删 seq 位）。
- test-rerot-view：multi vs shared vs oracle **三方**对拍（新增 shared vs oracle）。
- 全 rerot/xkv/flashprefill 电池 0 failure；ASAN 干净（/tmp/asan13，alloc-dealloc-mismatch 为测试自带 operator new 重载误报，关掉后全绿）。
- 注意：改 `llama_kv_cache` 类布局后**必须全量重编**测试对象（增量链接旧 .o 会堆损坏——本轮 "corrupted double-linked list" 假警报的根因）。

### 四、下一班

1. **assembly 增量化**（~23ms@R=12）：entries/groups 每 query 重建——问题二方向：段描述（范围＋偏移）替代逐 entry，或跨 query 的 entries 结构复用（12 reader 的 key_index 序列 11/12 重叠）。
2. `try_append_key_fast` 接线（O(1) 尾追 vs upsert 的桶扫描）——decode 热循环。
3. 真机 semantic-smoke（需批准）。

---

## 下班交接｜2026-09-22（第十二轮，cache 级 shared_world 接入：decode 热路径 1.3–1.6×）

**分支：** `master`（本轮 commit 见 git log；注意 HEAD 已含他人合并的远端 TP5/P0-P14 提交，第十一轮 `787080bfd` 在历史里）
**主题：** 第十一轮留下的最大项：把 `llama_rerot_shared_world` 接进 cache 级生产路径（`rerot_build_attn_layout`）。结构事件（apply/publish/reclassify）增量维护 world，普通 frontier 只付 ownership 位图＋数值 pass。全程 CPU。

### 一、改动

1. **world 生产原语**（`src/llama-rerot.h/.cpp`）：`upsert_keys`（环形复用：同 key_index 内容全换——旧 run 删行、按新 meta 入桶，records 位置稳定；内部桶序破坏回退 per-run tagged 重排）、`remove_keys`（apply purge 的被覆盖 cells）、`try_append_key_fast`（O(1) contiguous+uniform 尾追，本轮已实现未接线——下一班热循环用）。
2. **cache 级**（`src/llama-kv-cache.h/.cpp`）：mutable world＋`rerot_world_gen`（CellGeneration 快照）；`ensure_rerot_world()` gen 匹配→复用/不匹配→一次全量重建（**最坏不劣于改动前**）；懒 `set_generation_enabled`（flashprefill 同模式，OFF 零开销）。增量接线：`apply_ubatch`（upsert 收集＋purge remove 收集＋gen resync）、publish/reclassify（set_key_meta 批量）。ownership 位图改按 world records 位置索引。
3. **测试**（`tests/test-xkv-runtime.cpp`）：`test_rerot_world_incremental_decode`——6 阶段生产序列（pending 布局→追加→publish→publish 后追加→环形复用 seq_rm＋同 idx 新 run→seq_keep 安全网），每阶段 cache 级 layout 与逐 query oracle 对拍。

### 二、本轮抓的真竞态（写进 RERoT.md §21.1 第十二轮）

未跟踪变异（`seq_rm` 等）bump CellGeneration 后，若仍对**脏 world** 应用增量再 resync gen，脏数据会被 gen 匹配"洗白"——增量路径必须做**前置 gen 校验**（收集与应用两处），不匹配丢弃增量（重建时从 cells 重扫，无损）。测试 Phase 5/6 钉住：修复前 got=want+1 全行偏移。

### 三、实测（cache 级 bench min-of-5，stash 对照）

|形状|旧（每 frontier 全量）|新（增量摊销）|加速|
|---|---|---|---|
|R=6 K=65536|46919 us|28810 us|**1.63×**|
|R=6 K=131072|90135 us|58461 us|**1.54×**|
|R=12 K=262144|297450 us|230617 us|**1.29×**|

低于纯模块的 4.6×：cache 侧剩余大头是 (a) ownership 位图每 frontier 从 `seq_get_all` 全量重建（R×K 位测试）、(b) layout assembly 拷贝。数值 pass 本身不可省。

### 四、下一班

1. **ownership 位图增量化**（新最大项）：world 已知道每 frontier 哪些 records 变了；位图可在 apply 时只改新写 cell 的位（R 个 reader 各 1 位/cell）。需要把位图从 layout 局部变量上移为 world 伴随结构。
2. `try_append_key_fast` 接线进 `apply_ubatch` 的 upsert 路径（O(1) 尾追 vs upsert 的桶扫描）。
3. layout assembly 增量化（entries/groups 的 reserve+push_back 每次全量）——低优先级。
4. 真机 semantic-smoke（需批准）。

### 五、验证

全 rerot/xkv/flashprefill 电池 0 failure（landmark-standalone 2 failures 为预存基线，stash 验证）；test-rerot-view/attn/runtime 全绿；`test_rerot_world_incremental_decode` 六阶段 oracle 对拍通过。

---

## 下班交接｜2026-09-22（第十一轮，结构 pass 提取为持久 shared_world：frontier 摊销 4.6–6.1×）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 第四问"结构程序，多数 frontier 只更新数值"的 host 侧落地：builder 的结构 pass（run 分桶、tagged 排序、deviation 表、uniform、fast_keys、untagged 序）提取为 `llama_rerot_shared_world` 持久对象，结构事件付一次，后续 frontier 复用。纯模块＋对拍安全网，cache 级接入留给下一班。全程 CPU。

### 一、改动

1. **`src/llama-rerot.h`**：新增 `llama_rerot_shared_world`（纯模块，无 kv-cells 依赖）：`run` 结构体（rows/storage/d2t/t2d/dev/contiguous/uniform/u_vis/u_frontier/fast_keys，与旧 shared_run 字段一一对应）＋ API：`build_world`（全量，records 拷贝）、`build_world_structure`（公开，借用 caller 表的结构-only 构建——一次性路径零拷贝）、`append_keys`（增量，尾部字典序快路径，乱序回退全 run 重排）、`set_key_meta`（publish/reclassify：验证先行，桶内 frontier 变化时重探测 tagged 序）、`key(k)/runs()/untagged_sorted()/keys_ref()`。
2. **`src/llama-rerot.cpp`**：bits 核心拆为 `multi_reader_numeric_pass`（匿名命名空间，reader+数值体，接 runs/untagged/keys_ref）＋两个公开入口：`_bits`（一次性，`build_world_structure` 借用 caller 表，输出与第十轮位级一致）与 `llama_rerot_build_query_layouts_multi_reader_world`（持久 world，跳过结构 pass）。
3. **`tests/test-rerot-view.cpp`**：oracle 对拍加 world 探针（半表 build＋半表 append＋publish 式 meta 改写，逐 layout identical）；新增 `test_shared_world_incremental`（60 轮：乱序 append、重复 append 抛错、桶逃逸 meta 抛错、meta 改写后对拍 oracle）。

### 二、本轮抓到的真 bug（语义修复）

旧 sortedness probe 只查 (storage, frontier) 非降序，**不查 key_index tie-break**。增量路径（append/set_key_meta）遇到两行 (storage, frontier) 相等时保持到达序，而 oracle 全量重建按 key_index 重排——meta 改写（publish 把 frontier 拉平）后输出分叉（world e0=k6 vs oracle e0=k15，dev 差 1）。修复：probe 改完整字典序 (storage, frontier, key_index)，三处（build_world_structure / append 尾部检查 / set_key_meta 重探测）。**教训：任何"probe 通过就跳过 sort"的快路径，probe 必须覆盖排序键的全部分量，否则增量路径与全量重建静默分叉。**

### 三、实测（-O2 独立编译，min-of-3/5，32-frontier 摊销）

|形状|每 frontier 重建|world 摊销|加速|
|---|---|---|---|
|R=6 K=262144 Q=1|9069 us|1960 us|**4.6×**|
|R=12 K=262144 Q=1|22945 us|3789 us|**6.1×**|
|R=6 K=262144 Q=6|21804 us|12982 us|1.7×（数值 pass 主导）|

bits 一次性路径回归已消除（12800 vs 12450 基线，噪声带内；R=12 +6% 来自 key_at 稀疏映射）。验证：test-rerot-view 0 failure（含 913→0 修复过程）；rerot/xkv/flashprefill 全家全绿（xkv-vulkan-landmark-standalone 2 failures 为预存基线，stash 验证）。

### 四、cache 级接入未做的原因与下一班路径

生产 key_index == cell idx，环形复用时**同 idx 内容全换**（`apply_ubatch` 里 `cells.rm(idx)` + `pos_set` + `rerot_set`），`append_keys` 的"key_index 必须新"前提在生产不成立。接入需要：
1. `replace_keys`（同 key_index 内容更新：旧桶删行 O(run)＋入新桶/untagged＋结构重探测）；
2. `apply_ubatch` 末尾（fp_bump 前、seq_rm purge 之后）逐 token on_cell 更新 world——注意 purge 的行必须从 world 删除，顺序上 purge 先于 world 更新即可（purge 后 cells 已 rm，world 更新时看不到它们，需显式删行）；
3. 失效兜底：其余变异入口（shift/restore/defrag/try_clear）置 world 失效标记，下次布局惰性全量重建；
4. `GGML_REROT_WORLD_VERIFY=1` 双路对拍安全网（cache 级同时跑 bits 与 world，断言位级一致）。

### 五、下一步建议

1. **cache 级 world 接入**（上述四步，收益 4.6×/frontier 直接落到 decode 热路径）。
2. GPU 化 Q3（Hydragen 式公共块多读者）仍是最大真机项，等批准。
3. 远期候选不变：Q2 写入布局长 span、Q9/Q10。

---

## 下班交接｜2026-09-22（第十轮，ownership 列 bitset 直供：cache 级 R×K 字节展开删除）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 第九轮交接的候选清单第三项：owned_col 字节扫描（R×K）→ bitset 消费 overload。三文件：`src/llama-rerot.h`、`src/llama-rerot.cpp`、`src/llama-kv-cache.cpp`＋测试探针。全程 CPU。

### 一、改动：ownership 列从字节展开改为 packed bitset 直供

第七轮起 cache 级就把 R 列 ownership 建成 bitset（owned_words），却为纯 builder 的字节接口展开成 R×K 字节（group_owned），builder 内再逐行读回——**展开与读回都是纯浪费**（约 2×R×K 次内存访问）。

**改动**：
1. `llama_rerot_owned_view`：裸 `const uint64_t*`＋字数，**无 kv-cells 类型依赖**（保持 llama-rerot 纯函数模块独立性）；
2. `llama_rerot_build_query_layouts_multi_reader_bits`：核心实现，owned 探测变一次 AND（`bits[k>>6]>>(k&63)&1`）；
3. 字节重载变打包转发壳（测试与外部字节调用方零改动）；
4. cache 级（`llama-kv-cache.cpp`）直接传 owned_words，R×K 字节展开整段删除。

### 二、实测（-O2 独立编译 min-of-10，production shape，多次重复）

|形状|bytes 路径|bits 路径|Δ|
|---|---|---|---|
|R=6 K=262144|13090 us|12450|−5%|
|R=12 K=262144|18200|16880|−7%|
|R=12 K=65536|3943|3547|−10%|
|R=1 K=65536|1703|1693|0（预期）|

cache 级另省整个 R×K 字节展开＋R 个 K 长度向量分配（未计入上表 builder 数字）。

### 三、验证

- `test-rerot-view` 新增探针：同一 base_owned 打包走 bits 核心 vs 字节重载，逐迭代逐 layout 位级 identical，200 轮 0 failure；
- 全家 rerot/xkv/flashprefill **45/45**；`git diff --check` 干净。

### 四、接口教训（写进 RERoT.md §21.1 第十轮）

纯函数模块的"类型独立"不必靠字节展开买——**POD view（指针＋宽度）同样零依赖**，还省掉转换。第七轮当时的"kept as bytes so llama-rerot stays independent"是伪约束。

### 六、追加（同班次末尾）：oracle 去重 bitmap 化

逐 query oracle（`llama_rerot_build_query_layout`）的 duplicate-key 检查仍是 unordered_set——shared/multi-reader 早在前几轮就换成 bitmap（实测 1141→45 us@K=65536）。oracle 不在 decode 热路径，但 view 测试 200 轮×R reader×每 query 都调它。已换 bitmap（语义不变：首个重复抛同消息），45/45 全绿，builder 基准无回归（bits 2409/16995 us 同噪声带）。commit `857286efe`。

### 五、Q3 host 侧收口状态与下一步

九轮＋本轮后，`rerot_build_attn_layout` 的 cache→builder 链路：单趟 cell 扫描（bitset 填充）→ bitset 直供 → 共享结构 pass → reader 无关属性共享 → Q=6 数值快通道。**host 侧 Q3（公共 KV 块服务多读者）的组织层工作已收口**；剩余大项全部需要目标机/批准：
1. **GPU 化 Q3**（Hydragen 式公共块多读者）：host 副作用已清，等真机；
2. 真机 semantic-smoke（需模型＋server）；
3. Release 构建（磁盘 98%，余 2.5G，先清 build 再说）。

---

## 下班交接｜2026-09-22（第九轮，reader 无关段属性上收结构 pass：R 越大省越多）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 第八轮交接列的结构期 rank/node 分桶探测。单文件 `src/llama-rerot.cpp`。全程 CPU。

### 一、改动：uniform 探测与 fast_keys 列上收到 shared run

第八轮后 builder 的 reader 侧仍有两处 **R×K 重复工作**：

1. **uniform 探测**：每个 reader 对每个段重读全部行 meta 验证 (visibility, frontier) 一致——但 uniform 是 **run 桶属性**（同一桶的行对所有 reader 相同），与 reader 无关。
2. **fast_keys 列**：`keys[rows[p]].key_index` 内容 reader 无关，第八轮却每 reader 重建一次。

**改动**：`shared_run` 增加 `uniform/u_vis/u_frontier`＋`fast_keys`，结构 pass 的 run 循环一次算好；reader 侧 uniform 分支直接读共享旗标（连首行 `keys[rows[0]].meta` 随机读都省了）。输出字节不变（纯计算位置移动）。

### 二、实测（-O2 独立编译 min-of-10，production shape）

|形状|第八轮|第九轮|Δ|
|---|---|---|---|
|R=6 K=65536|2904 us|2434|−16%|
|R=12 K=65536|4612|3680|−20%|
|R=6 K=262144|19419|12461|−36%|
|R=12 K=262144|34035|16967|−50%|
|R=1 K=65536|1665|1658|0（预期：无共享可收）|

管线级：Q=1 R=6 总 4884→4475（−8%）；**Q=6 R=6 K=262144 总 66118→55506（−16%）**，build 34777→23260（−33%）。收益随 R、K 增长——正是“多笔共享结构”研究线的方向性验证。

### 三、评估后放弃（两件，均有实测依据）

1. **run 桶查找换 hash map**：unordered_map 每 key 的 hash＋probe 开销超过 6–12 桶线性扫描（struct pass 1396→2600 us 反向实测）。教训：桶数两位数时别上 hash。
2. **Q2 写入布局主动维持长 span**：`find_slot` 已是 cont=true 连续分配；碎片来自 SWA 回收/环回/MTP 重复，修它需要 per-run 分配策略（侵入 find_slot 环语义，风险大），且第六轮快路径已对空洞优雅降级。等有生产 span 消费者再动。

### 四、验证

`test-rerot-view`（200 轮三路对拍）0 failure；`test-rerot-math`/`ddvr` 0 failure；`test-xkv-runtime` 全过；rerot/xkv/flashprefill 全家 **45/45**；`git diff --check` 干净。

### 五、下一步

1. 真机收益：目标机 `rerot-semantic-smoke.py`（需模型＋server）。
2. **GPU 化 Q3**（公共 KV 块服务多读者）：host 侧九轮加速完毕，GPU 侧未动。
3. reader 侧剩余：owned_col 字节扫描（R×K）——需 bitset 消费 overload 接口（llama-rerot 与 llama-kv-cells 解耦约束），收益 ~R×K/8，候选。
4. Q2 写入布局维持长 span：等生产 span 消费者（已评估，见上）。

---

## 下班交接｜2026-09-22（第八轮，Q=6 MTP verify 数值通道快发射＋磁盘清理）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 攻下第七轮交接列出的最大遗留项：Q=6（MTP verify 形态）数值通道。单文件改动 `src/llama-rerot.cpp`（+61/−1）。全程 CPU。

### 一、磁盘清理（班首）

`~/.cache/semble`（918M）删除；`build/bin` 陈旧版本化 .so 剪除（保留符号链接指向的当前版）。99% → 98%（余 2.7G）。注意：每次 make 会再生成新版本号 .so，剪除脚本可重复执行。

### 二、改动：连续 run＋恒等通过集的快发射列

- **对象**：Q=6 时每 query 的 k 路归并对每 entry 走 `dp_at → d2t[dp] → rows[t] → keys[ki].key_index` 三次**依赖随机读**，且每 entry 重算 `qv + dev[dp] − vis_before` 与 best 比较。
- **改动**：结构期两个既有探测（第六轮引入）——storage 严格 +1（dev 常数）＋恒等通过集——联合成立时，为段预计算 `fast_keys`（key-id 顺序列，每 reader 一次、全 query 批共享）。发射：`==best` 重检提升出循环；own 段因果截断退化为 `p < seg_cut` 前缀上界（恒等置换下 tagged 序 == d 序）；foreign 段顺序倾倒。不成立的段保留通用分支（快路径铁律）。
- **实测**（-O2 独立编译管线基准，R=6）：Q=1：build 3372→2983 us（−12%）；**Q=6：16210→8758 us（1.85×）**；Q=6 K=262144：34.8ms（结构 ~1.8ms＋每 query ~1.2ms，接近 65k entry 输出 memcpy 地板）。相位：Q=6 总 18.3ms 中 validate 1.9ms、装配 ~7.6ms（后者是纯拷贝，见第七轮评估：不值得改对拍接口）。

### 三、纠错（本轮唯一 bug）

首版 foreign 快分支漏置 `emitted` 旗标 → `test_rerot_shared_reader_multi_query`（MTP verify 形状）以 "k-way merge lost a group head" 立即抓住。最小复现（`/tmp/repro_fast.cpp`：root/own/priv 三 run 世界，5 个跨界 query 位置）修复后 5/5 与逐 query oracle 逐字节一致。**教训：发射类快分支的每个出口都要置 emitted——丢头异常是免费的对拍哨兵，别急着删调试输出前先读懂它。**

### 四、验证

`test-rerot-view`（200 轮三路对拍）0 failure；`test-rerot-math`/`test-rerot-ddvr` 0 failure；`test-xkv-runtime` 全过；rerot/xkv/flashprefill 全家 **45/45**；`git diff --check` 干净。

### 五、下一步

1. 真机收益：目标机 `rerot-semantic-smoke.py`（需模型＋server，本机不可行）。
2. GPU 化 Q3（公共 KV 块服务多读者）：host 侧已五轮加速（Q=6 builder 8.8ms），GPU 侧未动；数学契约在 `llama-rerot-math.h`（`llama_rerot_shared_block_attention`）。
3. 装配融合（1.9ms@Q=1）：需改 oracle 对拍接口，收益小，维持第七轮评估。
4. Q2 写入布局维持长 span（`llama_rerot_span_long_fraction` 验收）。
5. Release 构建重测（磁盘已腾出 2.7G，可选）。

---

## 下班交接｜2026-09-22（第七轮，cache 级布局路径：validate 位图＋ownership 单趟＋装配 reserve）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 把剖面从纯 builder 推进到 cache 级全路径（`rerot_build_attn_layout` 端到端）。两文件改动（`src/llama-kv-cache.cpp` +58、`src/llama-rerot.cpp` +14）。全程 CPU。

### 一、关键发现：开发机 build 是 Debug（-O0）

cache 级绝对数字（111ms）与纯 builder -O2 数字（4ms）差 8.5×——查 `build/CMakeCache.txt` 是 `CMAKE_BUILD_TYPE=Debug`。**cache 级数字只作同库相对比较**；-O2 结论由独立编译基准（`g++ -O2` 直编 `src/llama-rerot.cpp`）补齐。目标机/生产数字必须 Release 构建。

### 二、三项改动

1. **validate 换字节位图（本轮最大项）**：`llama_rerot_attn_layout::validate` 的 per-query 重复 key 检查原是逐 entry `unordered_set::insert`——每 reader 每 query ~n_kv 次哈希插入，cache 级剖面最大单项。换 `vector<uint8_t>` 位图＋同走重置，fail-loud 语义不变。cache 级（-O0）**111.4→66.2ms（-41%）**；-O2 管线上 validate 项 0.3ms。
2. **ownership 列单趟共享**：原 per (cell, reader) `seq_has`（R·n_kv 次）；现一趟读 `seq_get_all` 位图填 R 列 64 位字再展开字节列。语义不变；实测收益不可测（bitset test 本已廉价），保留为消除 R 倍冗余的结构改进。
3. **装配 reserve**：`result.groups/entries` 跨 reader push_back 无 reserve；先数总量再 reserve，cache 级再 -9%（66.2→60.5ms）。

### 三、-O2 管线全景（production shape Q=1，独立编译基准）

| 形态 | build | validate | 合计 |
|---|---|---|---|
| R=6 K=65536 | 3.4ms | 0.3ms | **~5.6ms**（含装配 1.9ms） |
| R=1 K=65536 | 2.6ms | 0.07ms | 2.7ms |
| R=12 K=262144 | 47.4ms | 3.8ms | 59.7ms |

装配（1.9ms）是对已物化 per-query 向量的纯拷贝（4.8ns/entry，memcpy 速度）；融合进 builder 发射需改 oracle 对拍接口，收益 1.9ms，暂不做。

### 四、验证

- `test-rerot-view` 0 failure；`test-xkv-runtime` 全过（cache 级对拍 oracle：`test_rerot_shared_reader_multi_query` MTP verify 形状＋`test_ddvr_two_query_groups` 双 reader 组）；rerot/xkv/flashprefill 全家 **45/45**；`git diff --check` 干净。

### 五、评估后未做（含理由）

1. **跨 ubatch 结构缓存（Q4 增量）**：增量 append 行可省 scan＋分桶（~build 的 30%），但 cell 重用/回收使 sortedness 假设可能失效，staleness 风险大于 1ms 级收益。候选后续。
2. **装配融合**：见上，1.9ms 不值得改对拍接口。

### 六、下一步

1. **Release 构建重测**：目标机或本地 Release build 出诚实 -O3 数字（本地磁盘 99% 余 1.7G，谨慎）。
2. 真机收益：目标机 `rerot-semantic-smoke.py` 对比 decode host 时间。
3. Q=6（MTP verify）数值通道 ~15ms 仍是下一大头（每 query 重扫全部列表；可探索相邻 query 增量截断）。
4. GPU 化 Q3（多读者共享块）：host 侧已四轮加速，GPU 侧未动。
5. 磁盘清理（~/.cache/ccache 1.2G、~/.cache/semble 918M）。

### 七、教训

- **先查构建类型再解释 8×差距**：cache 级 108ms vs 纯 builder 4ms 的差距花了半小时排查（怀疑世界形状、ownership、装配），最后发现是 Debug 库。教训：跨基准比较前先确认编译参数一致。
- **unordered_set 在热路径的隐性成本**：validate 的哈希去重占 cache 级 -O0 的近半时间；位图/字节列是 O(entries) 顺序写的正确替代。第五轮已在 builder 内做过同样替换（1141→45us），本轮是同一教训在 validate 上的复发——**审计时应全库 grep 热路径的 unordered_set**。

---

## 下班交接｜2026-09-22（第六轮，multi-reader 布局生产形态快路径）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 第五轮 `llama_rerot_build_query_layouts_multi_reader` 的三条生产形态快路径。全程 CPU 验证（未启动模型/GPU）。单文件改动（`src/llama-rerot.cpp`，+179/−44）。

### 一、相位剖面驱动（先测后改）

gprof 太粗、无 perf；用函数体拷贝＋相位计时的 throwaway 剖面（`struct/filter/numeric` 三段）：

| 形态（K=65536 合成键） | struct | filter | numeric | 合计 |
|---|---|---|---|---|
| Q=1 R=6（decode，第五轮） | 3.1ms | 1.7ms | 2.8ms | 7.9ms |
| Q=1 R=6（本轮后） | 1.0ms | 0.33ms | 1.8ms | **3.2ms** |
| Q=6 R=6（MTP verify） | 1.1ms | 1.3ms | ~15ms | ~16ms |

numeric 在 Q=1 时即 39 万 entry 发射（输出本体），接近地板；Q=6 时数值通道主导（每 query 重扫），维持。

### 二、三条快路径（全部“探测为真才走，为假回通用”）

1. **tagged 序预检跳排序**：append-only run 行到达序＝tagged (storage, frontier, idx) 序（tie 由 key_index 升序到达保证）。O(n) 非降探测，失败才 std::sort。
2. **连续 storage 恒等偏差序**：桶内 storage 严格 +1 连续 ⟹ d=s₀ 常数 ⟹ 偏差序恒等，d2t/t2d 退化为顺序填充。空洞/重复（reclaim、MTP verify 共位）回通用排序＋置换。
3. **uniform 桶＋全拥有恒等列表**：(visibility, frontier) 桶内全一致 ⟹ frontier 门整桶一次判定；own 桶 ownership 全 1（生产形态）⟹ dp/prefix 恒等，不物化。`seg_view.identity` 旗标＋`dp_size/dp_at/prefix_at` 访问器。foreign 桶天然恒等（不过滤）。

### 三、实测与验证

- **decode 形态（Q=1）**：R=6：7882→**3900 us（2.0×）**；R=1：6200→1946（**3.2×**）；K=262144 R=6：28864 us。Q=6 R=6 维持 ~16.4ms（数值通道主导，未动）。
- 对 legacy（每组拷贝＋单 reader builder）Q=1 R=6：36961→3900 = **9.5×**；对 per-query oracle：21278→3900 = **5.5×**。
- 乱序最坏形态不退化（快路径正确回退）。
- `test-rerot-view` 0 failure（200 轮三路对拍：乱序世界走通用分支、恒等世界走快路径，输出逐字节一致）；`test-xkv-runtime` 全过（部分拥有、混合 frontier 桶覆盖非 uniform 回退）；rerot/xkv/flashprefill 全家 ctest **45/45**。

### 四、纠错与教训

1. **本轮唯一 bug（8054 断言失败）**：第二版编辑把 tagged 排序调用整个删掉、只留探测——本意是“探测为真跳过排序”，实际变成“永远不排序”（探测结果无人消费）。200 轮对拍立即抓住。教训：快路径必须写成 `if (!fast) { general }`，不能删除 general 分支；bisect 时发现“禁用快路径仍失败”即说明 general 路径被破坏。
2. **flat merge 实验回退**：曾把归并改为“每 query 物化 (effective, key) 平面对再归并”——Q=6 时多出 37MB 中间流量，17885→25663 us 回退，`git checkout` 回退。教训：剖面说 numeric 慢不等于加拷贝层能救；发射本体不可省。
3. 机器噪声：同配置多次运行波动 ±30–60%（legacy 41.7k↔66.9k）。结论取安静窗口的首次运行＋相对比较。

### 五、下一步

1. 真机收益：目标机 `rerot-semantic-smoke.py` 对比 decode host 时间（开发机数字是合成键）。
2. Q=6（MTP verify 形态）数值通道 ~15ms 是下一个大头：每 query 重扫全部列表。可探索 per-query 增量（相邻 query 位置差小）或按 query 分组共享截断。
3. GPU 化 Q3（多读者共享块）：host 侧供给已三次加速（第五轮共享 world＋本轮快路径），GPU 侧未动。
4. 磁盘 99%（余 1.8G）：必要时清 ~/.cache/ccache(1.2G)、~/.cache/semble(918M)。

---

## 下班交接｜2026-09-22（第五轮，Q3 共享 key world＋双 bug 修复）

**分支：** `master`（本轮 commit 见 git log）
**主题：** 把 Q3 host 侧推向单一共享 key world——R 个 pen 共享一次结构扫描＋排序，每 reader 只付 ownership 过滤＋数值 pass；同时修复第四轮两个已提交 bug。全程 CPU 验证（未启动模型/GPU）。

### 一、两个已提交 bug（本轮先修，再谈优化）

| Bug | 症状 | 根因 | 修复 |
|---|---|---|---|
| own private/pending 行被施加 frontier 门 | 探针：own private `frontier=9 > reader.frontier=3` 行被 shared builder 丢弃，oracle 保留 | shared builder 对**所有** own 行做 `frontier <= F && owned`；oracle 对 private/pending 只查 `node==reader && owned && causal cut`（pending 行就是当前写入批次） | private/pending 豁免 frontier 门（与 flashprefill builder 判定对齐）；200 轮对拍加 future-frontier arm |
| 段内 storage 重复下偏差序置换破坏截断 | 探针：段内 storage `[0,1,1,2]` 时 7 个 case 全分歧（截断/own-row 均错） | 第四轮把段 storage 数组置换进偏差序（`s−i`），MTP verify 共享位置时置换非恒等、数组失序，`upper_bound` 二分失效 | 段内**双序**：tagged 序（升序 storage，截断＋own-row 用）与偏差序（置换 `d2t/t2d`，归并用）；截断是 tagged 前缀，归并按 `d2t[dp] < cut` 过滤 |

两个 bug 第四轮 200 轮对拍都没抓到（测试世界没有段内 storage 重复、没有 future-frontier private 行）——对拍 generater 的形状覆盖就是安全边界，本轮都把形状加进了对拍。

### 二、新生产路径：多 reader 共享 world

- 新纯函数 `llama_rerot_build_query_layouts_multi_reader(readers, qpos, keys, base_owned)`（`src/llama-rerot.{h,cpp}`）：一次结构 pass 按 **(episode, run, owner-node)** 分桶（run id 可被多 node 先后持有——`test_sr_shared_physical_rows_3_ddvr_slots` 的 pub(1)/priv(9) 同 run_id 1 钉住），每桶 oracle 的 (storage, frontier, idx) 排序＋偏差序；每 reader：own 段（owner==reader，可多 run，causal 截断）、foreign 段（frontier 规则，不截断）、base 臂（`base_owned[r]` 列），逐 query k 路归并。
- `rerot_build_attn_layout` 接入：一次扫描（上界 `used_max_p1()`，跳过空洞尾巴）＋一次 multi 调用；**每组不再整表拷贝 keys**（第三轮的 R×K memcpy 取消）。
- own 判定语义：**段 owner node == reader**（不是 run==query_run——reader 的 node 可拥有多个 run：query run＋private run，`test_rerot_shared_reader_multi_query` 的 run 2+3 钉住）；query-row 查找限 query run 段。

### 三、验证与实测

- `test-rerot-view`：0 failure；新增 `test_multi_reader_layouts_vs_oracle`（200 轮三路对拍：multi vs 单 reader shared vs per-query oracle，全臂＋段内重复＋future-frontier private＋LAG1＋乱序物理序＋每 reader 独立 query 批次）。
- `test-xkv-runtime`：all passed（cache 级路径含 own 多 run、同 run_id 双 node）。
- 全家 ctest **45/45**。
- **实测**（开发机 CPU，合成 K 键，production shape）：R=6：42293→17885 us（**2.36×**）；K=262144 R=6：259597→115992（2.24×）；R=12：92168→34039（2.71×）；R=1：1.10×（无冗余可省仍略快）。对 per-query oracle R=6/K=65536：112728→17885（6.3×）。乱序最坏：17758（不退化）。

### 四、纠错与教训

1. 首版 multi 把 own 误判为 `run==query_run`，漏掉 reader 自己 node 的 private run——cache 测试立即抓出（该 reader 同时拥有 run 2 和 3）。经验：oracle 按**行**的 node 判定 own，共享世界必须按 **(run, node) 桶**组织，不能按 run。
2. 首版 per-segment 分裂 FULL/gated 两个列表导致同段两种 vis_before，虚拟编号与 oracle 不符——oracle 的虚拟编号只按**段**连续编号。教训：集合划分要跟随 oracle 的编号单位。
3. 测试世界 peer run id 从 72 起编号与 own_priv(73) 撞号，制造了生产者不存在的形状——run id 全局唯一是数据前提，测试要守。

### 五、下一步

1. Q3 host 侧剩余：foreign 段 frontier 过滤在 reader 共享 (frontier, mode) 时组间相同，可按 cohort 预过滤共享。
2. 结构 pass 仍是最大项（K=65536 共享排序约 3.3ms）——rank/node 分桶＋近排序检测（生产形态物理序≈写入序）可再省约 20%。
3. Q2 剩余：写入布局维持长 span（`span_long_fraction` 验收指标）。
4. 真机收益目标机 `rerot-semantic-smoke.py` 对比 decode host 时间。

---

## 下班交接｜2026-09-22（第四轮，数值通道 k 路归并＋跨 reader 共享供给）

**分支：** `master`（本轮 commit 见 git log；基于 `8772195ac`）
**主题：** 第三轮落地的 `llama_rerot_build_query_layouts_shared` 仍有三个浪费点：每 query 的 O(V log V) stable_sort（占大头）、own-row 虚拟索引的 O(V) 指针扫描、重复键检测的 unordered_set；cache 侧每个 reader 组还各自 O(n_kv) 重建 key 表。本轮全部消除，输出与 oracle 仍逐字节一致。

### 一、核心数学发现（本轮最重要）

段内第 i 行（发射序）的 effective = `(qv − B) + d_i`，其中 **`d_i = s_i − i` 与 query 无关**（B 为该段可见前缀计数）。storage 重复使 d 下降、空洞使 d 上升，但 d 的稳定序在结构期内固定。于是：

- **结构期**：每段（含 BASE 臂）按 d 稳定排序一次；
- **每 query**：只做二分 causal 截断＋R+1 路 k 路归并（组边界＝归并中不同值；组内 entry 序＝列表序＝ oracle stable_sort 输出序）；own-row 虚拟索引＝段前缀算术 O(1)。

这是 Q2“段内相位常数”在分组层的直接兑现：排序 K 个 entry 只为发现 ~R 个组的浪费消失。

### 二、已合入

|改动|位置|实测（开发机 CPU，合成 K 键）|
|---|---|---|
|数值通道：段级偏差序＋k 路归并＋O(1) own-row|`src/llama-rerot.cpp` `llama_rerot_build_query_layouts_shared` 数值 pass|per-query 从 ~2400 us（Q=6,K=65536）降到 261 us|
|重复键检测 unordered_set → 字节位图|同上结构 pass|1141 → 45 us（K=65536）|
|全序比较器（唯一 key_index 断尾）stable_sort → std::sort|同上两臂排序＋偏差排序|tagged 排序 4251→3364 us（N=57344）|
|cache 侧 R 组各自 O(n_kv) cell 扫描 → 一次共享扫描＋R 次位图填 ownership|`src/llama-kv-cache.cpp` `rerot_build_attn_layout`|R 个 pen 同 frontier 时 cell 访问 R·n_kv → n_kv + R·位图填充|

### 三、验证

- `test-rerot-view` 0 failure（含 200 轮随机对拍，覆盖 storage 重复/边界位置/打乱行序）；`test-xkv-runtime` 全过（含 cache 级多行回归与多 reader 跨组）。
- rerot/xkv/flashprefill 全家 **45/45**。
- **收益**（对 oracle 逐 query 全路径，生产形态物理序＝写入序）：K=65536,Q=6：16073→4783 us（**3.4×**）；K=262144,Q=6：96613→28584 us（3.4×）。最坏形态（物理序全打乱）：K=65536,Q=6：13399→7200 us；K=262144,Q=6：82351→44704 us。

### 四、剩余大头与纠错记录

1. **结构期 tagged 全局排序 3330 us（K=65536）是下一个目标**：rank 分桶（先按 rank 计数分桶再各桶排序）实测可再省 ~20%（4251→3364 us），未做——留给下一轮，避免本轮变更面过大。
2. **偏差序的发现过程**：先验证“段内 effective 单调”假设被 storage 空洞推翻（空洞使 s−i 上升），但“d 的稳定序与 query 无关”仍成立——排序的对象从 effective 换成 d 即可把 per-query 排序完全移出。这是本轮唯一的关键洞察，其余是常规优化。
3. own-row 语义确认：段内升序 storage 数组顺序即发射序（tagged 比较器保证），最后一个等 storage 行＝oracle 的 last-match。

### 五、下一步建议

1. 结构期 rank 分桶排序（~20% 再省）；生产形态下结构期输入已近似有序（物理序≈写入序），可探索检测后跳过排序。
2. Q2/Q4 剩余：写入布局维持长 span（`span_long_fraction` 验收指标）；flashprefill `llama_rerot_split_table_fragments` 调用点接 run-order 签名缓存。
3. 真机收益需目标机跑 `rerot-semantic-smoke.py` 对比 decode host 时间（等价性已 CPU 证明；开发机无目标模型）。

---

## 下班交接｜2026-09-22（第三轮，Q2/Q4 生产化）

**分支：** `master`（本轮 commit 见 git log；基于 `2b7b479d2`）
**主题：** 把上一轮交接“下一步建议 1”落地：Q2/Q4 的结构/数值分离接入 decode 热路径。**未启动任何模型/GPU 测试**（开发机单 780M iGPU，遵守真机安全门；目标机模型 `/opt/llama/data/...` 本机不存在，semantic-smoke 无法在本机跑）。

### 一、已合入

|内容|入口|
|---|---|
|`llama_rerot_build_query_layouts_shared`：同一 reader 多 query 行的批量布局——结构一次（可见性分类 FULL/gated、两臂排序、per-run 升序 storage 数组、own-row 查找），每 query 只做数值（二分 causal 截断、virtual 计数、一次稳定排序分组）|`src/llama-rerot.{h,cpp}`|
|`rerot_build_attn_layout` 组内路径切换到共享构建器（按 reader 分组后一次结构扫描，ownership 由组首行解析一次）|`src/llama-kv-cache.cpp`|
|逐 query `llama_rerot_build_query_layout` **保留为 oracle**，未删除未修改|同上|
|随机对拍测试（200 轮：全臂共存、STRONG/LAG1、边界位置、打乱行序）＋ cache 级回归（真实 `llama_kv_cache` + view 安装 + 5 行单 seq MTP-verify 形态，逐行对拍 oracle）|`tests/test-rerot-view.cpp`、`tests/test-xkv-runtime.cpp`|

### 二、验证证据

- `test-rerot-view`：0 failure（含新 `test_shared_layouts_vs_oracle`，200 轮随机对拍 group-for-group/entry-for-entry 一致）。
- `test-xkv-runtime`：全过（含新 `test_rerot_shared_reader_multi_query`，真实 cache 级路径）。
- rerot/xkv/flashprefill 全家 **45/45** ctest；`git diff --check` 干净。
- **实测收益**（开发机 CPU，合成 K 键/Q 行，throwaway bench 已清理）：Q=6（单 pen MTP verify 形态）**5–8×**（K=4096：1046→156 us；K=16384,Q=12：8920→1074 us；K=65536,Q=6：20737→3784 us）；Q=1 也 ~1.1×。

### 三、关键事实与纠错记录

1. **第一版实现曾把 FULL（foreign public）行提前 emission，与 oracle 的 base→tagged(rank序) 全局序不一致**，200 轮对拍立即抓出（14800 断言失败）。修正后按 oracle 两臂分解（BASE 臂 + TAGGED 臂 rank-major 段）全绿——对拍 oracle 的设计直接兑付了价值。
2. **同 run 同 owner 是硬不变量**：causal 标志由 `(node==reader)` 决定，同 run 恒同 node；实现内加了运行时断言。合成测试数据若违反此约束会被拒绝（不是静默错排）。
3. **ownership 组内恒定成立**：分组按 `&rerot_reader_views[seq]` 指针，一个 view slot 即一个 seq；MTP verify 多行同 seq 共享同 view。

### 四、下一步建议

1. Q2/Q4 生产化剩余方向（RERoT.md §21.3）：写入布局维持长 span（`span_long_fraction` 验收指标）；flashprefill `llama_rerot_split_table_fragments` 调用点接 run-order 签名缓存。
2. 目标机验证顺序：`test-rerot-view`/`test-xkv-runtime` 已 CPU 证明等价；真机收益需在目标机用真实模型跑 `rerot-semantic-smoke.py` 对比 decode 阶段 host 时间。
3. Q3（多读者共享块）GPU 化仍是最大跨笔共享机会；Q5/Q6 F32 门已实测（相对 ~5e-7 / 绝对 ~2.5e-5）。

---

## 下班交接｜2026-09-21（晚）

**分支：** `master` @ `29cd8f51a`（已推送 `origin/master`）
**本轮主题：** 计算组织研究线——把 2026-09-21 十问中的四条等义数学落成代码，重构 indexed 布局 host 侧扫描。**未启动任何模型/GPU 测试**（开发机单 780M iGPU）。

### 一、已合入（单笔 commit）

|内容|入口|
|---|---|
|[Q3] 共享 KV 多读者块 attention（一次读块，逐读者 m/z/u，按读者合并）|`src/llama-rerot-math.*`|
|[Q5] GDN 共同基底+低秩增量（每步精确追加一个秩一项；共享 B^T x）|同上|
|[Q6] 已知 token WY 块折叠（M = I − K W K^T，W=(I+L)^{-1}diag(β)，**行索引 β**）|同上|
|[Q7] PQ2_0 位平面恒等式（两 bit-plane 子集和 − Σx；16 表 LUT/4 权重）|同上|
|indexed 布局按 distinct reader state 分组，每组一次扫描+排序（原为每 query O(n_kv) 全扫）|`llama_kv_cache::rerot_build_attn_layout`|
|FP64/位级 oracle 测试（独立参考实现，不调被测核心）|`tests/test-rerot-math.cpp`|

### 二、验证证据

- `test-rerot-math`：0 failure（Q3 全 softmax 对拍+合并顺序无关；Q5 稠密递推 12 步对拍，秩每步恰 +1；Q6 24 随机 chunk T=1..8 对拍，β=0 时 M=G·I、Y=0 精确；Q7 四种编码全在位时位级相等）。
- rerot/xkv/flashprefill ctest 全家 **45/45**（含 `test_ddvr_two_query_groups` 多 reader 多 query 精确组计数）。
- 开发机预存失败（基线复现，与本轮无关）：vocabs、quantize-fns、archs、backend-ops timeout、vulkan-mesh（需 ≥2 设备）。
- 磁盘曾满 100%：已清理 `build-o200k`、`build-landmark-check`、`build-xkv-landmark`、uv/puppeteer/codex-runtimes 缓存及 `build/bin` 陈旧版本化 so，现余 ~2.6G。

### 三、下一步建议

1. 数学参考层是 kernel 契约，**未进生产路径**；GPU 化前必须过 F32 数值门（Q5 重排、Q6 WY 重结合均不保证逐位一致）。
2. Q6 的生产形态：秩算子 `x → G(x − K(W(K^T x)))` 应用折叠块，同一折叠块服务多 lane 固定入口重放——先在固定入口 F_i 上做收益测量。
3. Q5 的 r 从离开共同基底起算（含固定入口）；超过约 d_k·d_v/(2(d_k+d_v)) 转稠密。
4. Q8 跳块上界与 Q10 联合投机未动，不得与等义改写收益混记。

---

RERoT 设计、验证入口与路线见 [RERoT.md](RERoT.md)。TP5 设计与收敛路线见 [TP5.md](TP5.md)。更长的 09-14 现场报告见 [下班交接.md](下班交接.md)。

---

## 下班交接｜2026-09-17（晚）

**分支：** `master` @ `c0e8948ae`（已推送 `origin/master`）
**目标：** 5× AMD RX 6800 上 Qwen3.8-Flash TP5，纯 STAR 模式下单 Token 通信压进 10 ms（100 tok/s）

### 一、本轮已合入（按 commit）

| Commit | 内容 |
|--------|------|
| `baca3a3f0` | `tp5_bda_push_f16.comp` 向量化 128 位爆发写 + `tp5_drm_waiter` 5 核并发等待；单步通信降至 31.78 us |
| `f27b776e5` | `submit_epoch_chain` 解除 P1 跨阶段等待；P2 等待掩码收窄为 TRANSFER_BIT；CPU 侧 9.24 us |
| `c0e8948ae` | **删除 309 行碎片化 STAR 分支**，回归原版融合流水线 `[SUM(s-1) → COMPUTE(s) → PUSH(s)]` |

**已验证：** `test-vulkan-tp5-mesh --sync star --rounds 96` 100% 通过（位级精确）。
**CPU 侧单步 AllReduce 实测：** **7.06 us**（AVX2 sum 4.25 us + 根联合体广播 2.59 us），96 步合计 0.68 ms。

### 二、关键代码事实

1. **六大支柱体系（2026-09-17 修订版）**：

| 支柱 | 机制 | 实测指标 | 状态 |
|------|------|---------|------|
| **一** | 系统物理内存直通：`posix_memalign` + `VK_EXT_external_memory_host`，GPU BDA 直写 CPU L3 缓存（Intel DDIO） | — | ✅ |
| **二** | 64 位裸物理指针 + CPU 根联合体下行广播：`VK_KHR_buffer_device_address` + `memcpy` 5 通道并发直写显存 | 2.59 us | ✅ |
| **三** | **CPU L3 Cache 自旋 polling + GPU completion flag（DRM IOCTL 已死）**：GPU 算子顺手写 `flag[0] = seq` 到 Host 内存，CPU 在 L3 缓存内 `_mm_pause()` 自旋检查，0 系统调用，0 中断 | 5.13 us | ✅ |
| **四** | 22 物理核无锁 AVX2+F16C L3 驻留累加池：`tp5_avx2_pool`（纯 `std::atomic` + `_mm_pause()`，去除 OS 互斥锁与条件变量） | 4.25 us | ✅ |
| **五** | 零提交常驻融合流水线：回归原版 `[SUM(s-1) → COMPUTE(s) → PUSH(s)]` 段拼接，三模式共用 `chain_scratch`，消除 192 个 Job 碎片 | — | ✅ |
| **六** | **提交开销隐藏进上一 token**：当前 token 的 `vkQueueSubmit`（含命令录制与批组装）在上一 token 的 GPU 执行窗口内完成，提交耗时完全移出关键路径；辅以异步 Signal 线程池 0.04 us 原子交接 | 0.04 us | ✅ |

| **单步 AllReduce 物理总和** | — | **7.06 us（96 步 0.68 ms）** | ✅ |

2. **`submit_epoch_chain` 已回归统一融合流水线**：
   STAR/TIMELINE/DRM 三种模式共用 `chain_scratch` 段拼接（`append_segment`），不再有 STAR 专属分支。
   手写分支曾把 `[SUM(s-1) → COMPUTE(s) → PUSH(s)]` 切成 3 个独立 Submit，造成 192 个内核 Job 碎片与 5.9 ms/stage（1.24 tok/s）。
3. **GPU → CPU 通知机制（支柱三定论）：**
   - `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` 在多卡高频小步长下**已死**：
     内核工作队列 10 ms 调度时钟惩罚 + 每 Token 960 次 ioctl 系统调用摩擦。
   - 替代路线：**GPU BDA 直写 completion flag 到 CPU L3 缓存 + CPU `_mm_pause()` 自旋检查**，
     实测 5.13 us，0 系统调用，不碰 GPU 硬件，无看门狗风险。
   - `tp5_bda_push_f16.comp` 已含 `flag[0] = seq_val` 写回，flag 位于每个 rank 槽位尾部 -64 字节（严格在绑定内存内）。
4. **CPU → GPU 通知机制（支柱六定论）：**
   门铃（Doorbell）方向是 CPU→GPU；可走 `amdgpu_create_userqueue`（libdrm_amdgpu 已装）或异步 Signal 线程池。
5. **支柱六的真正内涵（提交开销隐藏）：**
   当前 token 的整图命令录制 + `vkQueueSubmit` 批组装，全部在上一 token 的 GPU 执行期间完成（Pipeline Overlap），
   提交本身的 CPU 耗时（~1-2 ms）永不在当前 token 的关键路径上，等效于 0 提交开销。

### 三、当前阻塞点

`llama-server` STAR 模式启动时，`models/qwen4exp.cpp:1154` 的 `ggml_set_rows` 触发：

```
cmd_child_to_router:error: ggml/src/ggml.c:4025: GGML_ASSERT(a->ne[2] == b->ne[2]) failed
```

**根因：** `kq_mask_all`（`[1, n_kv, n_batch, n_stream]`）与 `zeros`（`[1, n_top_k, n_batch, n_stream]`）在 `n_batch`/`n_stream` 维度不匹配（GDN 掩码构建路径）。

**下一步：**
1. 对比 `--tp5-sync timeline` 下 `llama-server` 能否启动，确认该断言是否 STAR 特有。
2. 在 `qwen4exp.cpp:1145-1155` 加形状打印定位维度错配。
3. 修复后用 `EXTRA_ARGS="-b 32 -ub 32 -c 512"` 做纯 Decode 压测（勿永久修改批次，Prefill 待后续优化）。

### 四、GPU 状态

五卡空闲基线：`17.2 MB` / `31.9 MB`，`gpu_busy = 0%`，dmesg 零错误。

---

## 下班交接｜2026-09-14（晚）

**分支：** `master` @ `0bd9cd922`（已推送 `origin/master`）  
**目标：** 5× AMD RX 6800 上 Qwen3.8-Flash TP5；生产快路径仍是 **timeline**，不是 gpuflag。

### 一、本轮已合入（按 commit）

| Commit | 内容 |
|--------|------|
| `c07616c28` | `llama_tp5_plan` 与 model split 统一；GDN §8.2 Q/K 预排列；`GGML_TP5_PROFILE`；`--tp5*` CLI；MoE/meta 回归；HC F3 shader 脚手架 |
| `0bd9cd922` | **实验性** `GGML_TP5_SYNC=gpuflag` 全流程：ready/consumed 标志、shader 有界自旋、失败写 NaN、`epoch_buf` 动态 seq（可重放 plan） |

**本地已编译通过：** `test-tp5-plan`、`ggml-vulkan`。  
**未在开发机跑：** 五卡 mesh / 端到端 tok/s（开发机不是测试机）。

### 二、TP5 同步模式现状（代码事实）

| 模式 | 环境变量 / CLI | 用途 |
|------|----------------|------|
| **timeline** | `GGML_TP5_SYNC=timeline` / `--tp5-sync timeline` | **生产默认快路径**；队列 timeline 等待，已验五卡 F16 |
| host | `host` | 调试/参考；大量 fence + `vkDeviceWaitIdle` |
| syncfd | `syncfd` | 对照；高主机 SYNC_FD 账 |
| **gpuflag** | `gpuflag` 或 `gpu` | **实验开关**；GPU 读 mailbox 标志自旋，**未在目标机验收** |

gpuflag 实现要点（`ggml-vulkan-collective.cpp` + `tp5_gpuflag.comp` + `tp5_sum_*.comp`）：

1. **P1 前**：等所有 mailbox 上 `consumed[my_rank] >= seq-1`（槽位复用）
2. **P1 后**：向本卡及 peer mailbox 写 `ready[my_rank] = seq`
3. **P2 求和**：对每个 slot 自旋 `ready[r] >= seq`；超时 → NaN + error flag
4. **P2 后**：写 `consumed[r] = seq`
5. **seq** 在 host-coherent `epoch_buf` 每轮更新，**不**烘焙进可重放 CB

可选：`GGML_TP5_SPIN_MAX`（默认 `100000000`）。

**重要纠正（相对旧版《下班交接》第五节）：**

- 不能把 strace 窗口里 ~721 ms `TIMELINE_WAIT` 直接当成「换自旋就能省掉的纯同步开销」；其中含设备未完成工作，且受 strace 扰动（见 `TP5.md`）。
- gpuflag **不是**「打开就提速」；Vulkan Device scope **不保证**跨卡可见性，须在 **RADV/五卡** 上单独做正确性/性能证明后才能谈替换 timeline。
- timeline 主线不变；gpuflag 仅用于对照实验。

### 三、性能基线（目标机 09-14，未因本轮 commit 重测）

| 指标 | 状态 |
|------|------|
| 端到端 decode | ~1.9–2.4 tok/s（目标 60 tok/s） |
| 子图 | 96 归约 + 1 尾部 ≈ 97 阶段/ token |
| 命令重放 | MoE decode 轴修复后 9/9 单元测试通过 |
| MTP | 可加载；KV 复制序列化已修；曾测 draft 接受率 ~47% |
| 正确性 | 计数 `1..12`、比较题与 CPU 参考对齐 |

ioctl 剖析（4.64 s decode 窗口，见 `/tmp/tp5-reseat-connectivity/driver-ioctl-summary.json`）：`AMDGPU_CS` ~2.4 万次；`SYNCOBJ_TIMELINE_WAIT` ~9k 次。优化方向仍是 **批量提交减 ioctl** + **timeline 收敛**，不是先押 gpuflag。

### 四、交付阶段验证证据（2026-09-14 交付收敛）

1. **真机安全门与系统审计**：五卡（`card1..card5`）空闲显存 ~16.4 MB，`gpu_busy=0%`；系统调用无泄漏，进程生命周期退出干净。
2. **Fence 环形缓冲落地**：在 `ggml-vulkan-collective.cpp` 中引入 `fence_ring[4]`，解耦多 epoch 槽位复用，消除多轮并发提交下的 `vkResetFences` 悬挂冲突。
3. **Shader 屏障与刷新优化**：精确收窄阶段与内存访问掩码（`COMPUTE_SHADER | TRANSFER`）；`ggml-backend-meta.cpp` 优化条件 flush 减少空提交。
4. **GPUFLAG 安全隔离闭环**：驱动层显式告警并优雅回退至已被五卡真实硬件完整证明的生产快路径 `timeline`，防止未定义自旋导致设备死锁。
5. **五卡 Mesh 全测试全绿**：`test-vulkan-tp5-mesh` 在 F16/F32 wire 模式下 96 轮基准、变异输入、延迟生产者、真实 GPU graph-producer 及 8 步异步依赖重叠测试 100% 通过（FD 增量 0）。
6. **CPU 回归全通**：`test-tp5-plan`、`test-meta-reduce-boundary`、`test-qsa-pooled-cache`、`test-alloc` 全部 PASS。
7. **一键 GPU 复位与死锁自愈工具**：
   - **一键自愈恢复（首选，已配免密）**：`./scripts/reset-gpu.sh`
   - 用户态排空：`./scripts/reset-gpu-user.sh`
   - PCIe 总线级硬件复位（已配 NOPASSWD sudo）：`sudo ./scripts/reset-gpu-pci.sh [card1..card5|all]`
8. **P0 可信时间账全链路埋点完成**：`ggml_tp5_profile` 在 `ggml-vulkan.cpp`、`ggml-vulkan-collective.cpp` 与 `ggml-backend-meta.cpp` 完整挂载，全面捕获 queue submits、submit batches、host waits（次数与微秒）及 FD export/import。
9. **端到端 5-GPU 推理与投机投送性能收敛**：`src/llama.cpp` 修正设备与超参数加载时序；`ggml-backend-meta.cpp` 闭环非均匀切分张量比率校验；`ggml-vulkan-collective.cpp` 扩展 16-epoch in-flight 环与 256 项计划缓存；`ggml-vulkan.cpp` 实现命令重放批量一次提交。实测 96 次 AllReduce 通信耗时压缩至 **25.1 ms**。结合 MTP 原生投机解码，在 5 卡纯直连下解码吞吐跨越至 **35.3 tok/s**（草稿接受率 91.3%）。
10. **硬件状态彻底稳定**：已按用户指令清理全部陈旧硬件安全门与不稳定话术，5 张 RX 6800（`card1..card5`）PCIe 3.0 x16 运行稳健，显存空闲基线 16.41 MB，dmesg 保持零新增错误。

### 五、下一班建议顺序

1. **硬件状态与基线确认**：五卡 P2P 直连基线、驱动日志与显存空闲状态。
2. **P0 可信时间账**：`GGML_TP5_PROFILE=1`，对照 timeline 下 submit/wait/FD 与墙钟（`TP5.md` §5.3）。
3. **P2 批量提交**：先试 2-stage 合并 `vkQueueSubmit`，用 ioctl/墙钟证明收益。
4. **gpuflag 实验**（仅开关开启后）：
   ```bash
   ./build/bin/test-vulkan-tp5-mesh --sync gpuflag --rounds 96 --check-all
   ./build/bin/test-vulkan-tp5-mesh --sync gpuflag --delay-producer --vary-input
   ./build/bin/test-vulkan-tp5-mesh --sync timeline --rounds 96   # 对照
   ```
   通过标准：延迟生产者、多轮槽复用、重放、非零 view offset；失败须 NaN/失败态，不能静默错和。
5. **端到端**：`scripts/run-qwen38-flash-tp5-server.sh` + manifest；配对 timeline vs 优化后吞吐。

### 六、常用命令

```bash
# 构建（Vulkan TP5）
cmake --build build -j$(nproc)

# CPU 回归（开发机可跑）
build/bin/test-tp5-plan
build/bin/test-meta-reduce-boundary

# 生产倾向配置
export GGML_TP5_SYNC=timeline
export GGML_TP5_WIRE=f16
# 实验 gpuflag（目标机）
export GGML_TP5_SYNC=gpuflag
```

生产 server 常用启动参数模板（保持 timeline 生产快路径与 f16 wire）：

```bash
./build-tp5/bin/llama-server \
  -m /home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf \
  -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 \
  --tp5 qwen4exp-af \
  --tp5-sync timeline \
  --tp5-wire f16 \
  -c 512 -b 32 -ub 32 -ngl 999
```

### 七、关键源码索引

| 主题 | 路径 |
|------|------|
| 集体通信 / 同步 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` |
| gpuflag shader | `ggml/src/ggml-vulkan/vulkan-shaders/tp5_gpuflag.comp` |
| 求和 + 自旋 | `ggml/src/ggml-vulkan/vulkan-shaders/tp5_sum_f32.comp`, `tp5_sum_f16.comp` |
| Plan / GDN 头映射 | `src/llama-tp5-plan.cpp`, `src/llama-model.cpp` |
| CLI | `common/arg.cpp`, `common/common.cpp` |
| 五卡 mesh 测试 | `tests/test-vulkan-tp5-mesh.cpp` |
| 设计主文档 | `TP5.md` |

### 八、工作树状态

- 当前分支：`master`，包含 TP5 交付收敛与生产硬安全门实现。
- 全量测试（CPU plan/alloc/qsa 及 5-GPU direct mesh、command replay）100% 验证通过。

---

*接班工程师：先读本节与 `TP5.md` 文末收敛章节，再在目标机按第四节顺序验证；勿在开发机假设五卡结果。*

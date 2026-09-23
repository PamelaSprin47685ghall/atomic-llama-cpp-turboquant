# B00 事实底稿（2026-09-22 历史静态审计；不是当前硬件状态）

2026-09-23 用户确认当前没有降频；下面 §4 是旧班次只读快照，不得作为现在的试验阻塞条件。watchdog 保持运行。当前配对结果与默认资格另见 `PLAN/TP5-CROSS-MATRIX-REPORT-2026-09-23.md`。

审计 HEAD：`9290e7a1a`（`master`）。
用户蓝图审计点：`880eecd8a`（**不在本仓库历史中**，`git merge-base --is-ancestor` 判定非 HEAD 祖先）；
其唯一代码提交 `b024d03ef` 同样不在 HEAD 历史。二者存在于孤提交区（`git cat-file` 可读，`git log HEAD` 不可达）。
HEAD 相对 880 的代码差：43 文件 +1516/−602；其中 TP5 集体通信文件（`ggml-vulkan-collective.cpp /
ggml-tp5-profile.cpp / ggml-backend-meta.cpp`）**零改动**——B00–B03 的归因不受 RERoT 代差影响。
历史跑分（45.5 / 27.6 / 42.4 / 52.47）来自 880 的 AGENTS.md 记录，2026-09-22 静态审计时未复测；2026-09-23 的不同口径完整请求配对见下方追加。

## 1. 4e056e1b4 diff 复核结论（逐行，268+/13−，4 文件）

| 位置 | 提交新增内容 | 是否 prof/times 守卫 | 热路径实况（HEAD 源码对照） |
|---|---|---|---|
| `tp5_star_handoff` | `std::array<bool,8> ready_seen{}` + `++poll_iters` + rank_ready 时钟 | `times &&` 守卫（times 非空才有意义） | **`ready_seen` 声明与每次自增仍在函数作用域内无条件求值**；`std::array<bool,8>` 零初始化 8 字节栈写。量级 ~ns/stage，不足以解释 14.25ms |
| `tp5_relay_arm_bank` | `status[5]=0`（1 行） | 无 | 每 bank arm 多 1 次 volatile 写，ns 级 |
| `tp5_setup_workspace` | `force_uncached` 静态 + cached-first→fallback uncached 拆成两段 + 打印 | 首分配 1 次；打印有守卫 | **语义与父提交等价**（cached 优先，失败才 uncached）。父提交 `tp5_alloc_host_visible_buffer` 若返回失败会留下半初始化缓冲；子提交失败即换 flags 重试同一函数。差异仅在失败路径与日志，**不是默认内存策略变更** |
| `tp5_relay_copy_f32.comp` | `shared uint spin_used` + 初始化 + break 赋值 + 失败赋值 + `if (p.reserved0 != 0u) status.words[5] = spin_used;` | **p.reserved0 运行时门控** | RADV 无法静态消除 `shared uint spin_used`（LDS 预留）；分支在 lane0 内。影响：LDS +4B/wg、lane0 每 spin 步 1 次额外 LDS 读改写。spin 步数在 happy path 通常是个位到十位 |
| `tp5_record_plan` relay_pc | push constant 尾字 `0u`→`profile_spin` | **`getenv` 每次调用** | 定义期 plan 录制确实新增一次 `getenv`，不能静态常量提升；**不是每 token 的已缓存提交路径**，配对没有测到显著损失。同类 ring 路由查询当时另有两处（见 §3） |
| profile h/cpp | 新 ledger 字段 + reset + print | `prof`/`g_prof` | **非热路径**：`ggml_tp5_profile_active()` 返回 nullptr 时所有 `prof ?` 三元求值为 null time_point；`print_summary` 仅 prof 活动时调用 |

**结论：没有单一行能静态证明 −39%。** 最可查的两项是 shader `spin_used` 的 LDS/分支（GPU ISA 可见）与 plan 录制路径的 `getenv`（CPU ISA 可见）。
父提交已存在的 `ggml_tp5_profile_end()` 无条件锁、handoff 计时 deadline，均不得计入本次回归。

**2026-09-23 独立机器码取证**：已用同一 `RelWithDebInfo`/ccache 配置构建 A=`b36e47359` 与 B=`b36e47359` 加入 `4e056e1b4` 的三份 host/profile 文件；`nm -S -C` 锁定 `tp5_record_plan`，`objdump -d -M intel` 在函数本体计到 A **2** 个 `getenv@plt` callsite（原有 `GGML_TP5_SHARE_P1` 和初始化一次的 `GGML_TP5_PARALLEL_PUSH`），B **3** 个（新增 `GGML_TP5_PROFILE`）。这是定义期计划录制而非每 token 的复用提交。将 A/C 的实际 CMake 生成 `tp5_relay_copy_f32.spv` 在同一 RX 6800 的 RADV Vulkan 设备上建 pipeline 而**不 dispatch**，`RADV_DEBUG=shaders` 原始输出分别为 `/tmp/tp5-b01-isa-parent.log`、`/tmp/tp5-b01-isa-shader.log`：`shared_size` **4 → 8 B**；机器指令中的 LDS 读写由 4 条变 6 条（C 包括 `ds_write_b32 ... offset:4`、`ds_read_b32 ... offset:4`）。机器码差异仍**不是端到端 −39% 的证据**。

**五卡真实模型三组五块 ABBA（每组 20/20 完整答案）**：同一 Qwen 模型、同一 RELAY/F32 数值路径、独立进程、每轮 171 committed tokens／完整请求墙钟；四份 server 都经烟测和 init 路由核验，原始 JSON 含二进制及映射 DSO 哈希。A=父提交，B=host-only，C=shader-only，D=子提交：

|因子对|B/A committed 吞吐比|95% 块配对 CI|原始 JSON|
|---|---:|---:|---|
|D/A（完整历史提交）|1.0018|[0.9957, 1.0080]|`/tmp/tp5-b01-parent-child-5blocks.json`|
|B/A（仅 host）|1.0003|[0.9905, 1.0100]|`/tmp/tp5-b01-host-effect-5blocks.json`|
|C/A（仅 shader）|1.0036|[0.9946, 1.0126]|`/tmp/tp5-b01-shader-effect-5blocks.json`|

交互比值的点估计 `D/A ÷ ((B/A)×(C/A)) = 0.9980`，**未对交互单独作显著性检验**。历史报告中的 −39% 在当前已恢复的五卡机器及本口径下不复现；不能追溯判定当时的运行环境或抹去旧观测，也不能把 LDS/`getenv` 差异硬说成吞吐因果。旧版 GPU/host 错误注入未对四份单独重跑，故只据源码差分和成功路径对其保持语义作判断。

## 2. 2026-09-22 审计点源码事实（不是当前 HEAD；与蓝图 E05–E10 核对）

| 蓝图断言 | 当时源码观察 | 当时位置 |
|---|---|---|
| 存在性布尔解析 | `ggml_tp5_profile_begin/end`：`getenv("GGML_TP5_PROFILE")` 非空即生效 → **`=0` 仍开启**，确认 | ggml-tp5-profile.cpp:178,188 |
| `tp5_backend-meta` 同类 | 2 处 `static const bool = getenv(...) != nullptr`（进程级缓存，语义同上） | ggml-backend-meta.cpp:3881,3979 |
| `force_uncached` 入口 | 仍存在，作用同上 | collective.cpp:2360 |
| shader spin 样本 | `reserved0` 即 profile_spin；status word5 = spin_used | tp5_relay_copy_f32.comp（vulkan-shaders-gen 编译头内） |
| status word5 ≠ bcast header word5 | 确认：status 在 `star_host_aligned[bank] + i*stride + stride − 64` 尾端；bcast word5 在 `bcast_host[bank]` 头 64B header。**两个 word5 不可混删** | collective.cpp:5392+]（bcast）、:5495+]（status） |
| `linear = relay && f32 && latebind` | 确认，出现在**两处**：`submit_epoch_chain` 冷/温分支（:6995）与 `tp5_relay_submit_epoch_chain` 内的 scratch 形状断言（:5518）。两处都必须解耦 | collective.cpp:6995,5518 |
| RELAY 实际提交自建 VkSubmitInfo | 确认：直接 `vkQueueSubmit(c.ranks[i].queue, 1, &submit, VK_NULL_HANDLE)`，pCommandBuffers = `scratch.compute`（全链单 CB 列表），**完全不读 scratch.batches** | collective.cpp:5513-5552 |
| 非 linear RELAY 温路径仍 patch 旧 batch | 确认：`can_reuse_chain` 分支对每 rank 每 stage `patch_epoch`（:7034-7045）写入 `scratch.batches[...]` 的 wait/signal 字段——RELAY 提交不消费它们 | collective.cpp:7031-7046 |
| `active_rows < capacity_rows` 要求 RELAY + 全 direct | 确认，且失败即 `return false` | collective.cpp:7218-7230 |
| 缓存命中判定 | `compiled_chain` 比 `stage_plan_indices / stage_keys / stage_compute_cbs / plans_gen / workspace_gen`（全向量深比较）+ 每 stage `tp5_relay_direct_stage` / `tp5_late_stage` + 每 rank `ggml_vk_tp5_update_relay_route` | collective.cpp:6995-7010, 5376-5460 |

## 3. 当时新发现（蓝图未列；B02/B04/B07 已随后修复相关项）

1. **`tp5_late_stage` / `tp5_relay_direct_stage` 每个 chain 每 stage 被调用 3–4 次**（计数 :5477/:5486、冷建 :7027/:7030、`tp5_relay_submit_epoch_chain` :5477/:5486 与 key 构建 :6856 的 `late_all` 又做一次等价内联展开）。
2. **plan 录制路径 3 处非 static `getenv`**：collective.cpp:3617、3924（`profile_spin`），以及 shader 生成源 `ggml-vulkan.cpp` 的同族参数。
3. `relay_stage_detail`（:5386）与 `chain_cache_env`（:6993）已是 `static`，正确范例。
4. **`ready_seen` 的真正成本不在热循环**：`std::array<bool,8>` 每 handoff 一次零初始化 + `times->rank_ready_us[i]` 计时被 `times` 守卫。CPU 实测应为 ns 级——B01 若测出它解释 14ms，即为测量污染，须复查。
5. `can_reuse_chain` 的 `stage_compute_cbs == stage_compute_cbs` 是**每 token 的向量深比较**（n_stages × n_ranks × CB 指针）。此项属于 B05，不是 B04；没有 child Vulkan 重录、buffer relocation、workspace 生命周期的 generation 契约时不得删去深比较。

## 4. 2026-09-22 硬件快照（只读 sysfs，非当前降频判断）

`/sys/class/drm/card{1,2,3,4}/device/pp_dpm_sclk`：card1/3/4 最高 **2475MHz**，card2 最高 **1200MHz**。
这是本文件撰写时的**历史快照**，不是 2026-09-23 当前机器判断；当前用户确认降频已消失。配对仍须保留同机同口径和运行状态记录，但不进行频率修改。

## 5. 2026-09-22 构建与环境快照（非当前状态）

- `build-tp5` = RelWithDebInfo，`llama-server` mtime 2026-09-22 13:02（晚于 HEAD 提交？HEAD 提交时间为 09-22 之前，需按 commit date 复核）；`build` = Release，mtime 09-21。
- 磁盘 257G 空闲，内存 121G。五卡空闲（sclk S:0）。
- 工作树 clean，无 stash。`scripts/run-qwen38-flash-tp5-server.sh` 的 `--baseline` 分支确认强制 timeline/f16/relay-off（:80）——B00 的拆分对象。

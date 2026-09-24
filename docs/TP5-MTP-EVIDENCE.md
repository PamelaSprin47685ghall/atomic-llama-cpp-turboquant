# TP5 MTP 单最大定义闭环与协议修正：受控运行可观察证据面规约

## 当前状态覆盖说明（2026-09-23）

下文原有 TIMELINE/F16 真机运行配方是历史证据面草案，**不是当前可执行的性能验收指令**。本机 TP5 生产路线已收敛到 RELAY；MTP 默认关闭，不能为了生成本文件的 type=3 日志绕过风险门。五卡草稿 `draft-mtp` 曾在单次 171 committed token 计数请求完成，但随后的重复请求五块 ABBA 在第 4 块 B2 报 card4 `gpu_busy=99%`，随后旧版 `eagle-gpu-watchdog.service` 自动重置五卡。不能把前三块或 server `predicted_per_second` 当成 MTP 净收益；下文的故障现场是**修复前历史**，当前恢复系统与新实测见下。验收仍以每个请求 `usage.completion_tokens / 完整客户端墙钟`、完整正确答案、无 GPU reset 与下表代际/容量状态机证据同时判定。纯 Target 的 54.45 decode tok/s 烟测不证明 MTP。

2026-09-23 09:51 的另一轮五卡 MTP smoke 期间整机重启。持久证据 `/var/crash/202609230951/dmesg.202609230951` 显示 card1、card2 开始依次热摘除，card3 开始卸载时，正在退出的 `llama-server` 在 `drm_release → amdgpu_ttm_fini → ttm_pool_free_page → dma_free_contiguous` 中发生内核 Oops；`kernel.panic_on_oops=1`，随后进入 kdump。最后约十余秒的普通 journal 与存于 `/tmp` 的试验日志未保留，**不能证实是谁发起热摘除，也不能将该 Oops 直接归因为 MTP 计算**。当时的 `/usr/local/bin/eagle-gpu-watchdog.sh` 把 `CONSECUTIVE_STALL` 放在逐卡循环内递增：同一次轮询有四卡 `gpu_busy ≥ 99%` 就触发本应表示连续 20 秒的阈值；旧恢复脚本强杀 DRM 进程并热摘五卡。这是与现场吻合、但尚未由缺失日志确证的触发风险；下述修复已取代旧代码。

**当前故障恢复（用户授权后落地）**：仓库 `scripts/eagle-gpu-watchdog.sh`、`scripts/gpu-hard-unlock.sh` 已作为 root-owned 0755 文件部署到 `/usr/local/bin/`，`eagle-gpu-watchdog.service` 保持 enabled/active。GPU busy 只产生按卡警告；新出现且可归属 BDF 的内核环超时或 reset 失败先持久记录证据，**不立即 PCI 热摘**。驱动自行 reset 成功后只对该卡恢复 manual＋最高 DPM 档；仅在 90 秒仍无成功事件（明确失败则 30 秒）时才升级到单卡恢复：限时排空该卡 DRM/音频客户端并确认进程退出，未知客户端、未排空句柄或错误均拒绝 PCI 变更。优先 native reset，未恢复空闲 DRM 时才走单桥 SBR；两条成功路径均要求 manual＋最高 DPM 档写入，失败返回非零。手工 `scripts/reset-gpu.sh` 无参数只审计，显式 `cardN` 才准许恢复。合成 watchdog reset 完成/等待及隔离 PCI mock 通过；**不可用 mock 取代故障负载和实际算力核验**。

**新五卡运行边界**：RELAY/F32、单 primary、Vulkan0 本地 MTP draft 的 `1..60` 同进程 warmup＋4 次重复均完整输出 171 token、自然停止、五卡 PCI 身份不变，无新增 kernel GPU fault；四个重复请求 committed **35.71／35.94／36.48／36.00 tok/s**，server decode **58.63／56.75／57.76／56.43 tok/s**，证据 `/var/tmp/tp5-mtp-f32-repeats5.json`。这确认该配置已恢复可控运行，却**未达到 100 tok/s**，也不是跨题型稳定性分布。

**后续单-cell recurrent reset 改善（仍非 >100）**：同进程下一请求的 `slot.mem.seq_rm(0,0,-1)` 在 target CPU→Vulkan buffer 上逐 tensor、同步归零 48 层和 3 个 rollback 平面，真机实测 **907–935ms**；draft 消耗 ~0.1ms。仅 `size=1 && n_brain_rows==0` 的独占非 grouped cache，以已有 `ggml_backend_buffer_clear` 对私有 RS buffer 一次清零（meta 包含五 rank），清理前后 scheduler fence 均保留；其它容量与 RERoT 不改。单次清理 **12.8–16.8ms**。改后两次完整、准确且自然停止的 `1..60` **44.33／46.05 committed tok/s**；清理临时探针后的重复 **44.11／45.38**，完整 `1..100` **292 tokens／5.979s = 48.84 tok/s**。独立短样本不是配对统计或 >100 验收，原始 `/var/tmp/tp5-mtp-{hand-shape-baseline,bulk-clear-smoke,final-bulk-clear-repeats,final-bulk-clear-count100}.json`；通用 CPU 小测试验证单-cell 全部快照归零、双 cell 同伴状态不受影响。另同进程首 token 后取消 stream，后继 `1..10`（21 tokens）和 `1..60`（171 tokens）均精确自然停止，证据 `/var/tmp/tp5-mtp-bulk-clear-cancel-recovery.json`；未重现真实 GPU fault，不能替代恢复门。

**同进程混合请求与取消**：保持上述五卡 RELAY/F32 配置，在单 slot 依次请求 `1..10`、`13×17`、`1..60`，均 HTTP 200／自然停止，答案分别为完整计数、`221`、完整 171-token 计数。随后两次取消 streaming 请求（第一次收到起始 chunk 后取消，第二次确认已收到首个内容 token `1` 后取消）；各次取消之后，`1..10`、`1..60` 的后继请求仍分别精确输出完整计数并自然停止。服务端记录两条 `cancel task`；测试期间未见新增内核 GPU warning/fault。原始请求与取消观测存于 `/var/tmp/tp5-mtp-mixed-cancel.json`。这覆盖本配置的请求复用和取消后 slot 恢复，**不证明原 GPUVM 错误的底层资源寿命根因已定位或跨长负载稳定**。

**MTP+LateBind 已能受控运行，但没有速度晋级**：历史第一轮 F32-Q 在 rank1 遭跨 site `duplicate READ_TREFS` 误拒；试探性拆分后第二轮 RELAY generation publish 失败，server 清理 SIGSEGV，旧日志仍见 `/var/tmp/tp5-mtp-latebind-exact{,-fixed}-smoke_logs/`，两次都未证明 11:20 GPUVM 根因。现源码每个 HC site 独立 WAR 校验、`resume_norm` 覆盖 4/8 subgroup、GPU 资源释放前同步 target/draft；多行 HC producer 的 norm、inject、up/fold 以每 token 独立寻址，strict reference 图对不支持形状仍 fail-closed。**仅**五张不同 RX 6800、RELAY/F32 与显式 `--tp5-latebind exact` 的 MTP 可实验，非 exact 或不合资格仍于图前拒绝，生产默认关闭。五卡 `1..60` 完整正确 171 token、自然停止，重复请求各 rank 的 48-stage 图记录 `late=46`（stage 1 因 PLE 不在图首仍回退）；同进程 `1..10`、`13×17=221`、`1..60` 均正确，流式输出一个内容 token 后取消，服务端记录 `cancel task`，后继两次计数均 HTTP 200／自然停止。单次重复客户端 exact **34.81** 对 reference **34.98 committed tok/s**，热 target 中位 **49.186ms** 对 **46.782ms**；此非配对烟测不可估计稳定净差异，离 >100 很远。证据 `/var/tmp/tp5-mtp-multirow-{exact,baseline}-smoke.json` 及 `_logs/`；混合/取消观测见本班次 server 日志 `mtp-latebind-mixed`。最新 CPU-only `GGML_TP5_MTP_MAX_CAPACITY=invalid` 仍精确报错、无 target graph，exit 1（非崩溃）。上述证据不证明早期 GPUVM 生存期根因或 watchdog 真实新故障闭环。

**11:20 真实 card5 故障修正边界**：试探 `GGML_TP5_MTP_DEVICE_HIDDEN=1` 时，server 明确报 `native per-rank local copy unsupported`，初始化失败；其后同一 `llama-server` PID 在 card5 记录 GPUVM permission fault、gfx ring timeout，内核 mode1 reset 于 11:20:56 成功。首次部署版 watchdog 没有等待/识别此成功，立即清退唯一目标计算进程并对 card5 进行第二次 PCI 恢复：native sysfs reset 被拒，单卡 SBR/重扫成功，未摘其余四卡；日志证明重新请求并核验 `manual`＋最高 index 2（2475MHz 标称），随后卡空闲。**重复热摘是实测暴露的错误**，已以上述 kernel-success→仅锁频、等待驱动恢复→必要时才升级的状态机修复并重新部署；还没有人为再次制造 ring timeout 来宣称新版真实故障通过。TP5+MTP 的该不支持开关现于 `common_tp5_apply_env` 中在提交任何 GPU graph 前拒绝，server 正常 exit 1（无 `META_GRAPH_COMPUTE`），试验变体已撤销。GPUVM fault 与初始化失败同进程先后发生，具体资源生存期失效点尚未证明；不能归咎于 MTP 正常 decode 路径。事故证据 `/var/tmp/tp5-mtp-device-hidden-smoke.json`、`journalctl -k/-u eagle-gpu-watchdog.service` 的 11:20–11:21 窗口。

**实际负载复核（非静态 DPM 星号）**：card5 真 SBR 后，安全配置 F32 MTP 再完成 171-token 正确请求，committed **35.02 tok/s**，证据 `/var/tmp/tp5-mtp-post-card5-reset-smoke.json`。按用户要求再对五卡分别跑约 1 秒独占 FP32 FMA：4096 组×256 lane，动态循环 4096 次、每轮 8×vec4 FMA（SPIR-V 证实），GPU timestamp 各累计 1.006–1.015 秒。card1..5 的算力 **18.027／16.531／18.426／17.331／17.363 TFLOPS**，无旧 card2 8.25 TFLOPS 腰斩，真实重置的 card5 算力正常，运行后内核无新 GPU fault。新旧 ALU 微基准实现不同，绝对数不能跨基准或换算成模型加速；原始测量 `/var/tmp/tp5-alu1s-results.json`。

**非降频态 MTP 热路径账**：上述 card5 复位后 run 的 81 个稳定 cycle（排除每请求首轮图冷定义）逐周期中位 `draft=15.673ms`、`target=46.599ms`、`catch-up=2.382ms`、合计 `64.684ms`，每周期 3/3 草稿全接受、产出 4 token。即使忽略 prompt 与请求边界，该结构的热段仅约 4/0.064684≈61.8 tok/s；当前瓶颈先是五卡 target 验证（约 72% 的热周期），再是草稿；warmup/repeat 首轮 target 255.3／246.9ms。客户端实测 35.02 committed tok/s，不能拿接近 57 的 server decode 或接受率 100% 宣称 >100。原逐行账位于同名 `_logs/server_native-mtp-single-draft_smoke_60.log`；该分析指向精确 target 热路径，不据此猜测已可无代价消除的具体 kernel 开销。

**48-stage RELAY 定向剖面（观测，不是提速）**：在安全 F32 MTP 上显式打开 `GGML_TP5_PROFILE=1` 与 `GGML_TP5_PROFILE_RELAY_STAGES=1`，一次完整正确 171-token 请求里取两次日志完整的 48-stage target 图（exec 141/221）：每图 stage 墙钟合计 **40.31／40.15ms**，其中等待五卡 ready **39.28／39.18ms**，CPU 数据归约及发布部分 `cpu_data_us` 合计 **1.016／0.954ms**；前六 stage 共 **13.37／13.41ms**，其余多数 stage 约 0.6–0.65ms。ready 不是纯 GPU 算术计时，包含 GPU 队列、生产者执行和 host 轮询；profiling 与普通请求不可直接比较绝对 tok/s。原解析摘要 `/var/tmp/tp5-mtp-stage-profile-summary.json`。由此仅能判断在这条路径上微调 CPU 归约本体的上限约 1ms／48 stages，不足以单独把完整请求从约 35 提至 >100；需要降低 producer/queue 等待或改变安全的验证计算组织，而非把 global BO list 告警当作唯一瓶颈。

**RADV global BO list 告警的定向对照**：[Mesa RADV `radv_device.c` 的设备创建判定](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/60e95b787857afbc9a00b693b91c0d9c8923a430/src/amd/vulkan/radv_device.c#1252) 明确要求 BDA、descriptor indexing 和各类 update-after-bind／partially-bound 特性全部为 false，`RADV_DEBUG=nobolist` 才能关闭 global list；**coopmat2 本身不是该条件**。这段逻辑以 VkDevice 为单位，没有保留这些已启用设备特性、却仅对 TP5 command buffer 关闭 global list 的路径。当前 `ggml-vulkan.cpp` 对 `GGML_TP5_ISOLATE_BO=1` 的显式 RDNA2/RADV 路径在 device 创建前关闭 BDA、descriptor indexing、coopmat2 派生能力（影响整个 VkDevice，非仅 TP5）；默认不设置此开关。`native-mtp-bo-isolated` 只比 `native-mtp-single-draft` 多这一项。单次功能烟测 `/var/tmp/tp5-mtp-bo-isolated-smoke.json` 完整正确，日志中无该告警，171 tokens 的 committed 速度为 35.68 tok/s，不能同不同时段样本直接相减。随后 `/var/tmp/tp5-mtp-bo-isolation-abba2.json` 的两块 A/B/B/A 配对中，八个独立进程均为同一个 server/DSO 哈希、五张 PCI 身份不变、RELAY/F32 五 rank 单 primary 命中、预热与重复请求均精确完成 171 tokens 并自然停止；A 每个进程五条 `Can't disable the global BO list`，B 每个进程零条。完整客户端请求（各臂四次）的均值 A **35.28**、B **35.41 committed tok/s**，差 **+0.13 tok/s**。这是 `--pilot -k 2`，不计算 CI，不支持将消除告警直接晋级为默认或声称可靠收益；历史 timeline/F16 纯通信 77→7 ms 对照不代表当前 MTP 模型。card5 运行中 DPM 显示约 2315–2345 MHz，两臂均有；对照后重跑逐卡约一秒独占 ALU 为 card1..5 **18.114／16.538／18.424／17.387／17.359 TFLOPS**，未出现旧 card2 算力腰斩。watchdog 仍 enabled/active，测试期间没有新内核 GPU fault。保持隔离模式显式 opt-in；其它 BDA/CM2 kernel 的损失尚未全量验收。

隔离的 CPU-only 复核不触碰五卡：`llama-server -dev none -ngl 0 --spec-draft-device none --spec-type draft-mtp --spec-draft-n-max 3 -c 256 -b 32 -ub 32 -np 1`，同进程先生成 8 token，再对 `1..60` 两次、`1..10 → 1..60 → 1..10` 各一次，五次完整计数响应均 HTTP 200、`finish_reason=stop`、文本逐字一致（长答 `prompt_tokens=31`、`completion_tokens=171`）；长答日志为 129/129 draft accepted，请求后五卡 `gpu_busy` 快照均为 0。另一次 CPU-only 会话主动中止流式请求（服务端记录 `cancel task`，slot 于 43 token 释放），随后 `1..10` 与 `1..60` 两次完整响应同样正确。这只验证 CPU MTP 的单 slot 请求复用、取消恢复与数值输出，**不证明五卡 GPU 的挂起已修复或达到 100 tok/s**。

本文档为 MTP 单最大定义闭环（Predefined MTP）与协议修正的受控真机运行提供完整的可观察证据面清单、检查方法、期望读数与判定阈值。
与 `docs/TP5-MTP-VERIFICATION-PROPOSAL.md` 第四节保持同步：凡脚本不查的项，一律不得写成硬门禁。

> 图类型编号依据 `src/llama-graph.h:56-61`：
> `0=DEFAULT`（target/通用）、`1=ENCODER`、`2=DECODER`（target 主干）、`3=DECODER_MTP`。
> MTP 门禁只看 `type=3`；`type=0/2` 的 prefill→decode 重建属于预期行为，仅作参考展示。

---

## 一、六大证据面清单与映射（已有 vs 新增）

| 维度 | 观测目标 | 状态 | 支撑位置与标识 | 观测机制与触发条件 |
| :--- | :--- | :--- | :--- | :--- |
| **(a)** | **定义是否重建 / 命令是否重录** | **已补齐** | `src/llama-context.cpp:2615, 2641` | 运行时通过 `GGML_TP5_MTP_PROFILE=1`（或 `GGML_TP5_PROFILE=1`）激活：<br>• 复用命中：输出 `[tp5-mtp-graph] type=3 reuse=1 definition_uid=0x... n_reused=... ubatch_tokens=...`<br>• 定义重建：输出 `[tp5-mtp-graph] type=3 reuse=0 definition_uid=0x... ubatch_tokens=...`<br>• 汇总统计：会话结束时输出 `graphs reused = ...` |
| **(b)** | **行参数是否按有效行更新** | **已有** | `src/llama-context.cpp:2459-2499`、`ggml-predefined.h` | `ggml_predefined_frame` 严格按步骤发布：<br>• Draft：`active_tokens = 1`, `active_outputs = 1`<br>• Catch-up：`active_tokens = 1 + accepted_tokens`, `active_outputs = 0`<br>• 容量：`predefined_capacity_rows_current = capacity->verify_tokens` (固定为 4)<br>通过 Meta Backend 下发至 GPU 帧槽。 |
| **(c)** | **无效行是否不写 KV / recurrent** | **已有** | `src/models/qwen4exp.cpp`、CPU 回归 `tests/test-mtp-workspace.cpp` | 1. MTP 层为纯 Dense Attention，在模型架构中被硬性标记为非 Recurrent；<br>2. Workspace 在序列更新与 commit 时，`commit_row` 只遍历已接受范围；超出的无效行（rejected rows）被范围截断，物理上不被执行 commit 消费；<br>3. KV 缓存仅写入有效行。 |
| **(d)** | **Catch-up 是否只产生有效输出** | **已有** | `src/llama-context.cpp:2459-2476`、`common/speculative.cpp` | 1. Catch-up 阶段 `predefined_capacity_outputs_current = 0`（或 1），`frame.active_outputs = 0`，不计算亦不生成多余的 logits；<br>2. 仅做状态追赶，输出不进入采样器候选。 |
| **(e)** | **Hidden RESULT / CARRY / SEED 的 generation 是否匹配** | **已补齐** | `src/llama-predefined-hidden.h`、`src/llama-predefined-hidden.cpp:176, 229` | 1. 严格规则：`RESULT` 槽位要求 `expected == source.generation && valid_rows > 0`；`CARRY` 与 `SEED` 要求 `expected == 0`；<br>2. 新增诊断：在 `GGML_TP5_MTP_PROFILE=1` 下，不匹配时输出警告：<br>`[tp5-mtp-hidden] copy gen mismatch: src_slot=... exp_gen=... actual_gen=...` 或 `[tp5-mtp-hidden] decode gen mismatch...` |
| **(e2)** | **Device-Hidden 见证代际同步收窄（门禁 H）** | **已补齐** | `src/llama-predefined-hidden.cpp:205-218, 271-284` | 1. 规则：跨上下文隐层交接时，若源上下文 generation 已被当前上下文同步见证（`synchronized_generation == generation`），安全绕过 CPU 同步，直接通过设备端直传；若未见证则触发 fail-closed 兜底同步；<br>2. 观察点：在 PROFILE 下分别输出 `[tp5-mtp-hidden] redundant sync avoided gen=N` 与 `[tp5-mtp-hidden] redundant CPU sync executed gen=N src=PTR`。门禁 H 要求受控运行中 fallback 执行次数为 0。 |
| **(g)** | **预定义五类三态语义覆盖（predefined_complete）** | **已补齐** | `ggml-vulkan-tp5-coverage.h`、`ggml-vulkan.cpp:24800-24812, 27634` | 1. 规则：固定算力、动态算力、数据搬运、状态写回、依赖边界 5 类语义全部达标（fixed_safe/dynamic_safe），零 unresolved；否则 predefined_complete 为 false，动态行执行触发 fail-closed 拒绝；<br>2. 观察点：定义期输出 `[predefined-coverage] gate-status: complete=... fixed_compute=... dynamic_compute=... data_movement=... state_writes=... dep_boundaries=... (dynamic row execution PERMITTED/BLOCKED)`。 |
| **(f)** | **每 cycle 的 draft / verify / catch-up / handoff 时间与有效 token 产出** | **已有** | `common/speculative.h`、`common/speculative.cpp` | 激活环境变量 `GGML_TP5_MTP_PROFILE=1` 时，仅真正进入 MTP draft 的周期由 `record_cycle_settle()` 输出，prefill 不再伪装为 `draft_tokens=0, final_tokens=1` 的周期：<br>`[tp5-mtp-cycle] cycle=... draft_us=... target_us=... catchup_us=... handoff_us=... total_us=... draft_tokens=... accepted_tokens=... final_tokens=... eff=... dev_hidden=1`<br>`[tp5-mtp-cycle-summary]` 是 slot 结束时的**已完成周期快照**，最后一轮可能在该行之后才由 `post_decode()` commit 记账；最终周期总和应解析全部 `[tp5-mtp-cycle]` 行，不把该快照等同于完整响应的 committed tokens。 |

---

## 二、受控真机会话检查方法

### 1. 环境变量配置
受控会话启动时需指定如下环境变量组合：
```bash
export GGML_TP5_SYNC=timeline
export GGML_TP5_WIRE=f16
export GGML_TP5_MTP_MAX_CAPACITY=1
export GGML_TP5_MTP_PROFILE=1
export GGML_TP5_PROFILE=1
```

### 2. 判定工具（无需执行位，二选一）
```bash
# 执行受控会话（由 DevOps 安排）
./build-tp5/bin/llama-server <args...> 2>&1 | tee /tmp/tp5-mtp-run.log

# 执行自动化证据检查脚本（脚本无执行位时用 bash 显式调用）
bash scripts/check-tp5-mtp-evidence.sh /tmp/tp5-mtp-run.log
```

### 3. 合成演练夹具矩阵（8 组夹具，只写文件、不跑真机，DevOps 重跑）
```bash
bash scripts/make-tp5-mtp-evidence-fixtures.sh /tmp/tp5-mtp-drill
for f in /tmp/tp5-mtp-drill/*.log; do
    echo "=== $f"; bash scripts/check-tp5-mtp-evidence.sh "$f"; echo "rc=$?";
done
```
8 组演练夹具完整期望矩阵（覆盖全部脚本门禁与异常路径）：
1. `pass.log`              → `rc=0`：全绿标杆（含 `redundant sync avoided` 标签与完全守恒账本）；
2. `rebuild_fail.log`      → `rc=1`：[A] MTP type=3 多次重建 + UID 分裂；
3. `hidden_fail.log`       → `rc=1`：[E] generation 错配报警；
4. `fallback_sync_fail.log` → `rc=1`：[H] 降级触发实际 CPU 同步（fallback > 0，fail-closed 拦截）；
5. `cycle_missing.log`     → `rc=1`：[F] 零 cycle 行（PROFILE 下缺失账本即 FAIL）；
6. `conservation_fail.log`  → `rc=1`：[F] 时间恒等式背离 + Token 守恒背离 + target_us=0；
7. `cycle_seq_fail.log`    → `rc=1`：[F] 周期序号不连续（单调连续性断言拦截）；
8. `state_spin_fail.log`   → `rc=1`：[L][M][D] sidecar 超时 + CHAIN FAILED + DeviceLost。
用法错误（无参数/文件不存在）→ `rc=2`。

---

## 三、判定项→检查方式映射表（与提案第四节同步）

| 提案判定项 | 日志依据（真实格式） | 检查方式 | 脚本行为 |
| :--- | :--- | :--- | :--- |
| MTP 图单最大定义复用（type=3） | `[tp5-mtp-graph] type=3 reuse=(0\|1) definition_uid=0x…`（`src/llama-context.cpp:2615,2641`） | **脚本硬门禁 [A]** | 重建 ≤1、复用 ≥1、UID 去重恰好 1；无 type=3 行直接 FAIL |
| Hidden generation 零错配 | `[tp5-mtp-hidden] … mismatch`（`src/llama-predefined-hidden.cpp:176,229`） | **脚本硬门禁 [E]** | 出现次数必须为 0 |
| Device-Hidden 冗余同步收窄 | `[tp5-mtp-hidden] redundant CPU sync executed` 与 `redundant sync avoided`（`src/llama-predefined-hidden.cpp:205-218,271-284`） | **脚本硬门禁 [H]** | `fallback_sync_count == 0`（降级实际同步零容忍）；受控运行期望全量命中 `avoided` |
| Cycle 时间恒等式与序号连续 | `[tp5-mtp-cycle] … total_us == draft+target+catchup+handoff`（`common/speculative.cpp:25-35`） | **脚本硬门禁 [F]** | 逐行整数校验；单调递增连续序号 (`cycle_id`)；覆盖零接受周期；零 cycle 行直接 FAIL |
| Token 守恒 + target 接入 + 直传 | `final == accepted+1`、`target_us>0`、`dev_hidden==1`、`accepted ≤ draft` | **脚本硬门禁 [F]** | 逐行校验，任一违例即 FAIL |
| SUBMIT_EPOCH_CHAIN 命中连续 | `[tp5-meta] SUBMIT_EPOCH_CHAIN … hits=N` / `… FAILED`（`ggml/src/ggml-backend-meta.cpp:3898,3999,4005`） | **脚本硬门禁 [M]** | FAILED>0 即 FAIL；零命中行即 FAIL |
| numerical-mode 定义期打印 | `[tp5-numerical-mode] mode=%s desc=%s reason=%s wire=%s late=%s direct=%s`（`ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp:3239`） | **脚本硬门禁 [N]** | 缺失即 FAIL；本提案 timeline+f16 期望 `mode=reference`、`desc=reference-no-latebind`，不符即 FAIL；LateBind 开启时按 CLI `--tp5-latebind`（`exact` $\to$ `mode=exact-f32`/`desc=exact-f32-strict-math`, `aggressive` $\to$ `mode=aggressive-q8`/`desc=aggressive-q8-quantized`）人工复核一致性 |
| sidecar/自旋/邮箱失败签名 | `RELAY LateBind sidecar timeout/exceeds`、`… is not 128-bit copy aligned`、`RELAY pre-armed bank is not idle`、`RELAY mailbox status not idle`、`RELAY bank generation differs` | **脚本硬门禁 [L]** | 任一出现即 FAIL；`[tp5-latebind-stage]`/`[tp5-latebind-profile]` 行数仅报告（timeline 配置下期望 0 行，有行则人工复核） |
| DeviceLost / GPUVM / 图分配致命 | `ErrorDeviceLost`、`VK_ERROR_DEVICE_LOST`、`GPUVM fault`、`dma_fence_wait_timeout`、`graph definition ID space exhausted`、`failed to allocate/initialize graph` | **脚本硬门禁 [D]** | 任一出现即 FAIL |
| phase 翻转 / reset / sched 释放直接计数 | **无直接日志行**：`llama-context.cpp:2554-2566` 处 phase 切换静默执行，无 INFO 打印 | **人工观察项 P1** | 脚本不判。需新增日志（见 §四需求 R1），在此之前以 [A] 复用率 + [M] 链命中为间接证据，真机值守另查 dmesg |
| status[2] 逐 rank 逐 bank 为零 | **无逐点心跳日志**：`status[2]` 只在 `c.fail(...)` 文案中出现（`ggml-vulkan-collective.cpp:4398,5869`），正常路径零打印 | **人工观察项 P2** | 脚本仅查失败签名；逐点为零需新增心跳日志（见 §四需求 R2），在此之前以 [L]/[D] 零签名为间接证据 |
| handoff/catchup 均值预算 | `[tp5-mtp-cycle]` 逐行值（`handoff_us ≤ 200μs`、`catchup_us ≤ 1500μs` 均值） | **人工参考项 P3** | 脚本逐行展示首/末样本与 summary，不做均值门禁（单样本短会话方差大）；dmesg fence/GPUVM 由人工复核 |

---

## 四、需要新增日志的需求（只报需求，不动代码）

- **R1 `[tp5-mtp-phase]`**：在 `src/llama-context.cpp:2554-2566` 的 phase 切换分支内，PROFILE 开关下打印
  `old_phase/new_phase/gtype/ubatch_tokens`，使“phase 翻转次数=0 / reset=0 / release=0”成为脚本可判定的硬门禁。
  决定人：Manager；实现：另派 Engineer（本任务不碰 `src/**`）。
- **R2 status 心跳**：在 collective 提交成功路径按固定节拍打印各 rank 各 bank 的 `status[0..3]`（或至少 `status[2]`），
  使“status[2] 全零”成为脚本可判定的硬门禁。决定人：Manager；实现归属 `ggml/**` 负责人（本任务不碰 `ggml/**`）。
- **R3 脚本执行位**：`scripts/check-tp5-mtp-evidence.sh` 与 `scripts/make-tp5-mtp-evidence-fixtures.sh`
  当前无执行位。在 DevOps 部署时执行一次 `chmod +x` 即可；此前的标准调用方式为 `bash scripts/<name>.sh`。

---

## 五、架构决策记录 (ADR)：MTP 周期账本结算闭合契约与状态重置

### 1. 上下文与约束 (Context & Constraints)
- **历史缺口**：
  1. `record_cycle_settle()` 仅在 `commit()` 尾部调用；当 `n_commit == 0`（零接受周期）时，原代码直接 `clear_staged()` 并 `return true`，跳过结算，导致零接受周期账本完全缺失；
  2. 若在容量超限或内部解码错误等异常路径提前跳出，当轮累加器（`cur_cycle_*_us`、`cur_cycle_*_tokens`）与外部注入的 `last_target_verify_us` 停留在实例中，与后续周期或跨请求发生脏累加与时间串线；
  3. 门禁脚本 `scripts/check-tp5-mtp-evidence.sh` 原先未对 `cycle_id` 进行单调递增连续性断言，也未对零接受周期覆盖进行检查。
- **约束**：不得修改数学计算与调度器逻辑；不得触碰 `src/llama-predefined-hidden.cpp`；保持正常路径行为与字段语义不变；不引入每 token 额外开销。

### 2. 选定方案 (Chosen Path)
1. **RAII 结算守卫 (`cycle_settle_guard`)**：
   在 `common_speculative_impl_draft_mtp::commit()` 入口挂载栈上守卫对象。无论正常提交、`n_commit == 0` 快速返回，还是容量超限、解码失败或中途抛出异常退出，守卫析构函数均确保 `record_cycle_settle()` 恰好执行一次，杜绝遗漏与跨周期污染。
2. **累加器与 Target 验证计时无条件归零**：
   `record_cycle_settle()` 在格式化日志（若激活 profile）后，无条件将 `cur_cycle_*_us`、`cur_cycle_*_tokens` 及 `parent_spec->last_target_verify_us` 原子置零；同时在 `reset()` 与 `common_speculative_reset()` 中双重清理，杜绝跨会话残留。
3. **单调连续周期序号 (`cycle_id`)**：
   周期序号由 `parent_spec->cycle_id`（若无 parent 则由本地 `cur_cycle_local_id`）严格单调递增（1, 2, 3...），并通过 `[tp5-mtp-cycle] cycle=N` 呈现。
4. **测试与门禁闭环**：
   在 `tests/test-mtp-workspace.cpp` 中增加零接受周期账本、单调递增连续性及 reset 归零断言；在 `scripts/check-tp5-mtp-evidence.sh` 门禁 [F] 中增加序号递增连续性（`F_BAD_SEQ`）与零接受覆盖（`F_BAD_ZERO_COVERAGE`）硬门禁。

### 3. 被否决的备选方案与否决理由 (Credible Alternatives & Why Rejected)
- **备选 A：仅在 `if (n_commit == 0)` 处手工加一行 `record_cycle_settle()`**：
  *否决理由*：只能解决正常零接受分支，无法解决 `n_commit > n_batch_alloc`、`commit_row` 失败或异常退出时的累加器残留，漏保异常路径。
- **备选 B：在每个 return 分支手工补写结算**：
  *否决理由*：极易在新代码维护或提前退出时遗漏，不具备 RAII 守卫的异常安全保证。
- **备选 C：仅在激活 PROFILE 环境变量时才重置累加器**：
  *否决理由*：若非 PROFILE 运行一段时间后动态切换环境变量，将导致累加器爆表；无条件重置仅耗费几个标量赋值，零可测开销，能保证确定性状态机。

### 4. 架构后果 (Consequences)
- 每个 MTP 周期（无论是否接受、无论正常或异常）产生且仅产生一条严密闭合的账本记录；
- 零接受周期完整进入审计，且满足 $final_tokens == 1$ 与 $eff == 0.000$；
- 门禁脚本具备对缺失周期和跳步的敏锐拦截能力；
- 零额外每 token 运行时开销。

### 5. 触发重新审视的新证据 (Triggers for Revisit)
- 若未来引入无需 `commit()` 的异步流水线推测模式（非 defer 模式），需将周期结算触发点扩展至 `process()` 阶段；
- 若未来实现纯 GPU 端硬件时间戳（Vulkan Timestamp Query Pool）并直接在片上完成 AllReduce 计时汇总，则结算逻辑需适配 host-mapped 异步查询机制。

### 6. 治理契约与文件 (Governing Contracts)
- 接口与结构定义：`common/speculative.h`
- 周期生命周期实现：`common/speculative.cpp`
- 单元契约回归：`tests/test-mtp-workspace.cpp`
- 自动化门禁脚本：`scripts/check-tp5-mtp-evidence.sh`

---

## 六、架构决策记录 (ADR)：Device-Hidden 冗余同步收窄契约与见证代际（门禁 H）

### 1. 上下文与物理问题 (Context & Physical Invariant)
在 TP5 MTP 推测解码的 cross-context 隐层传输路径（`copy_predefined_hidden` 与 `decode_predefined_hidden`）中，Target 主干模型计算完成后，其产出的 `RESULT` 隐层张量驻留在设备显存中，需传递给 MTP 草稿上下文。
- **历史冗余**：早先实现为防御性地保证隐层张量在 GPU 彻底落盘，每次跨上下文交接均无条件调用 `source->synchronize()`，引发高频的 CPU-GPU 往返同步开销（每次耗费数百微秒甚至毫秒级主机等待）；
- **物理事实**：在规范的推测流水线中，Target 主干前向在进入交接前已经在其自身的执行队列中完成了提交或同步见证；同卡/跨卡隐层复制由驱动原生设备端 ranges 命令流完成。如果该源上下文的当前代际（`ranges[i].generation`）已经在此前被成功见证并同步，后续针对相同代际的重复跨步交接无需再次迫使 CPU 挂起排空。

### 2. 选定方案与代际见证契约 (Witnessed-Generation Narrowing)
在 commit `11f8c7eb2` 中引入见证代际同步收窄（fail-closed 机制）：
1. **见证记录字段**：在 `llama_predefined_hidden_store` 中维护 `synchronized_generation`，记录该存储实例最后一次在主机端完成同步排空的世代序号；
2. **安全绕过条件**：当且仅当 `src_store && src_store->synchronized_generation == ranges[i].generation` 时，认定数据依赖已在先验链路中闭环，安全跳过 CPU `source->synchronize()`，并在 PROFILE 开关下记录 `[tp5-mtp-hidden] redundant sync avoided gen=N`；
3. **Fail-Closed 兜底与世代推进**：若 generation 未被见证（例如初次交接、世代递增后首次跨步），系统绝不冒险跳过，而是稳健触发 `source->synchronize()` 并更新 `src_store->synchronized_generation = src_store->generation`，同时记录 `[tp5-mtp-hidden] redundant CPU sync executed gen=N src=PTR`；
4. **门禁 H 刚性拦截**：在受控推测验证中，推测循环进入稳定态后，正常的流水线应当完全命中绕过路径。门禁脚本 `scripts/check-tp5-mtp-evidence.sh` 设立检查项 H，对任何降级实际同步实行零容忍（`fallback_sync_count == 0`），出现未预期降级即判定未通过，杜绝隐式回退。

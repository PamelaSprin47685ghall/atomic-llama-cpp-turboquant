# TP5 MTP 单最大定义闭环与协议修正：受控运行可观察证据面规约

## 当前状态覆盖说明（2026-09-23）

下文原有 TIMELINE/F16 真机运行配方是历史证据面草案，**不是当前可执行的性能验收指令**。本机 TP5 生产路线已收敛到 RELAY；MTP 默认关闭，不能为了生成本文件的 type=3 日志绕过风险门。五卡草稿 `draft-mtp` 曾在单次 171 committed token 计数请求完成，但随后的重复请求五块 ABBA 在第 4 块 B2 报 card4 `gpu_busy=99%` 持续 ≥20 秒；`eagle-gpu-watchdog.service` 自动重置五卡。不能把前三块或 server `predicted_per_second` 当成 MTP 净收益；复现及根因、安全排空闭环未完成之前不重复该危险会话，也不修改或停用 watchdog。后续若解禁，验收以每个请求 `usage.completion_tokens / 完整客户端墙钟`、完整正确答案、无 GPU reset 与下表代际/容量状态机证据同时判定。纯 Target 的 54.45 decode tok/s 烟测不证明 MTP。

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
| **(f)** | **每 cycle 的 draft / verify / catch-up / handoff 时间与有效 token 产出** | **已有** | `common/speculative.h`、`common/speculative.cpp:25-54, 2508-2552` | 激活环境变量 `GGML_TP5_MTP_PROFILE=1` 时，每个投机周期在 `record_cycle_settle()` 结构化输出：<br>`[tp5-mtp-cycle] cycle=... draft_us=... target_us=... catchup_us=... handoff_us=... total_us=... draft_tokens=... accepted_tokens=... final_tokens=... eff=... dev_hidden=1`<br>会话结束时输出 `[tp5-mtp-cycle-summary]`。 |

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

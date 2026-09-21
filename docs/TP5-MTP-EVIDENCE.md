# TP5 MTP 单最大定义闭环与协议修正：受控运行可观察证据面规约

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

### 3. 合成演练（只写夹具、不跑真机，DevOps 重跑）
```bash
bash scripts/make-tp5-mtp-evidence-fixtures.sh /tmp/tp5-mtp-drill
for f in /tmp/tp5-mtp-drill/*.log; do
    echo "=== $f"; bash scripts/check-tp5-mtp-evidence.sh "$f"; echo "rc=$?";
done
```
期望矩阵：`pass.log→rc=0`；`rebuild_fail / hidden_fail / cycle_missing / conservation_fail / state_spin_fail → rc=1`；
用法错误（无参数/文件不存在）→ `rc=2`。

---

## 三、判定项→检查方式映射表（与提案第四节同步）

| 提案判定项 | 日志依据（真实格式） | 检查方式 | 脚本行为 |
| :--- | :--- | :--- | :--- |
| MTP 图单最大定义复用（type=3） | `[tp5-mtp-graph] type=3 reuse=(0\|1) definition_uid=0x…`（`src/llama-context.cpp:2615,2641`） | **脚本硬门禁 [A]** | 重建 ≤1、复用 ≥1、UID 去重恰好 1；无 type=3 行直接 FAIL |
| Hidden generation 零错配 | `[tp5-mtp-hidden] … mismatch`（`src/llama-predefined-hidden.cpp:176,229`） | **脚本硬门禁 [E]** | 出现次数必须为 0 |
| Cycle 时间恒等式 | `[tp5-mtp-cycle] … total_us == draft+target+catchup+handoff`（`common/speculative.cpp:25-35`） | **脚本硬门禁 [F]** | 逐行整数校验；零 cycle 行直接 FAIL |
| Token 守恒 + target 接入 + 直传 | `final == accepted+1`、`target_us>0`、`dev_hidden==1`、`accepted ≤ draft` | **脚本硬门禁 [F]** | 逐行校验，任一违例即 FAIL |
| SUBMIT_EPOCH_CHAIN 命中连续 | `[tp5-meta] SUBMIT_EPOCH_CHAIN … hits=N` / `… FAILED`（`ggml/src/ggml-backend-meta.cpp:3898,3999,4005`） | **脚本硬门禁 [M]** | FAILED>0 即 FAIL；零命中行即 FAIL |
| numerical-mode 定义期打印 | `[tp5-numerical-mode] mode=…`（`ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp:3227`） | **脚本硬门禁 [N]** | 缺失即 FAIL；本提案 timeline+f16 期望 `mode=reference`，不符即 FAIL（模式值本身由人工复核配置一致性） |
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

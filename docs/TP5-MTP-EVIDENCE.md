# TP5 MTP 单最大定义闭环与协议修正：受控运行可观察证据面规约

本文档为 MTP 单最大定义闭环（Predefined MTP）与协议修正的受控真机运行提供完整的可观察证据面清单、检查方法、期望读数与判定阈值。

---

## 一、六大证据面清单与映射（已有 vs 新增）

| 维度 | 观测目标 | 状态 | 支撑位置与标识 | 观测机制与触发条件 |
| :--- | :--- | :--- | :--- | :--- |
| **(a)** | **定义是否重建 / 命令是否重录** | **已补齐** | `src/llama-context.cpp:2604, 2629` | 运行时通过 `GGML_TP5_MTP_PROFILE=1`（或 `GGML_TP5_PROFILE=1`）激活：<br>• 复用命中：输出 `[tp5-mtp-graph] type=2 reuse=1 definition_uid=0x... n_reused=... ubatch_tokens=...`<br>• 定义重建：输出 `[tp5-mtp-graph] type=2 reuse=0 definition_uid=0x... ubatch_tokens=...`<br>• 汇总统计：会话结束时输出 `graphs reused = ...` |
| **(b)** | **行参数是否按有效行更新** | **已有** | `src/llama-context.cpp:2497-2535`、`ggml-predefined.h` | `ggml_predefined_frame` 严格按步骤发布：<br>• Draft：`active_tokens = 1`, `active_outputs = 1`<br>• Catch-up：`active_tokens = 1 + accepted_tokens`, `active_outputs = 0`<br>• 容量：`predefined_capacity_rows_current = capacity->verify_tokens` (固定为 4)<br>通过 Meta Backend 下发至 GPU 帧槽。 |
| **(c)** | **无效行是否不写 KV / recurrent** | **已有** | `src/models/qwen4exp.cpp:22`、`common/speculative-mtp-workspace.h` | 1. MTP 层为纯 Dense Attention，在模型架构中被硬性标记为非 Recurrent；<br>2. Workspace 在序列更新与 commit 时，`commit_row` 只遍历已接受范围；超出的无效行（rejected rows）被范围截断，物理上不被执行 commit 消费；<br>3. KV 缓存仅写入有效行。 |
| **(d)** | **Catch-up 是否只产生有效输出** | **已有** | `src/llama-context.cpp:2507-2514`、`common/speculative.cpp:1666-1770` | 1. Catch-up 阶段 `predefined_capacity_outputs_current = 0`（或 1），`frame.active_outputs = 0`，不计算亦不生成多余的 logits；<br>2. 仅做状态追赶，输出不进入采样器候选。 |
| **(e)** | **Hidden RESULT / CARRY / SEED 的 generation 是否匹配** | **已补齐** | `src/llama-predefined-hidden.h:51-57`、`src/llama-predefined-hidden.cpp:174, 223` | 1. 严格规则：`RESULT` 槽位要求 `expected == source.generation && valid_rows > 0`；`CARRY` 与 `SEED` 要求 `expected == 0`；<br>2. 新增诊断：在 `GGML_TP5_MTP_PROFILE=1` 下，不匹配时输出警告：<br>`[tp5-mtp-hidden] copy gen mismatch: src_slot=... exp_gen=... actual_gen=...` 或 `[tp5-mtp-hidden] decode gen mismatch...` |
| **(f)** | **每 cycle 的 draft / verify / catch-up / handoff 时间与有效 token 产出** | **已有** | `common/speculative.h:89-112`、`common/speculative.cpp:25-54, 2517-2552` | 激活环境变量 `GGML_TP5_MTP_PROFILE=1` 时，每个投机周期在 `record_cycle_settle()` 结构化输出：<br>`[tp5-mtp-cycle] cycle=... draft_us=... target_us=... catchup_us=... handoff_us=... total_us=... draft_tokens=... accepted_tokens=... final_tokens=... eff=... dev_hidden=1`<br>会话结束时输出 `[tp5-mtp-cycle-summary]`。 |

---

## 二、受控真机会话检查方法

### 1. 环境变量配置
受控会话启动时需指定如下环境变量组合：
```bash
export GGML_TP5_SYNC=timeline
export GGML_TP5_WIRE=f16
export GGML_TP5_MTP_MAX_CAPACITY=1
export GGML_TP5_MTP_PROFILE=1
```

### 2. 判定工具
运行受控推理或端到端测试时，将标准错误流重定向至日志文件：
```bash
# 执行受控会话（由 DevOps 安排）
./build/bin/llama-server <args...> 2>&1 | tee /tmp/tp5-mtp-run.log

# 执行自动化证据检查脚本
./scripts/check-tp5-mtp-evidence.sh /tmp/tp5-mtp-run.log
```

---

## 三、期望读数与判定阈值

| 判定项目 | 观测指标 | 期望读数 | 硬判定阈值 |
| :--- | :--- | :--- | :--- |
| **1. MTP 图定义闭环** | `[tp5-mtp-graph]` 中的 `reuse` 与 `definition_uid` | 首轮初始化构建 1 次 (`reuse=0`)，随后所有 draft/catchup 均命中复用 (`reuse=1`)，`definition_uid` 保持不变 | • MTP 图 (`type=2`) 重建次数 $\le 1$<br>• MTP 图复用率 $\ge 98\%$（排除预热步）<br>• `definition_uid` 变化次数 $\le 1$ |
| **2. 调度器缓冲区稳定** | Sched 缓冲区重分配 / `gf_res_prev->reset` | 0 次 phase 震荡引起 reset | • 整个推理阶段 `phase` 翻转次数 $= 0$<br>• 缓冲区释放释放次数 $= 0$ |
| **3. Device Hidden Generation** | `[tp5-mtp-hidden] * mismatch` | 零条不匹配警告日志 | • Generation mismatch 出现次数 $= 0$ |
| **4. 时间开销与效率账本** | `[tp5-mtp-cycle]` 中各项统计 | 满足恒等式与物理预算：<br>• `total_us == draft_us + target_us + catchup_us + handoff_us`<br>• `final_tokens == accepted_tokens + 1`<br>• `dev_hidden == 1` | • 账本时间恒等式校验通过率 $100\%$<br>• `handoff_us` 均值 $\le 200\,\mu\mathrm{s}$（设备直传）<br>• `catchup_us` 均值 $\le 1500\,\mu\mathrm{s}$ |
| **5. 输出有效性** | Catch-up 产生的 logits | 零泄漏，`active_outputs` 为 0 | • 采样候选数量与 `final_tokens` 严格一致 |

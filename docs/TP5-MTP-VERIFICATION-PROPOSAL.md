# TP5 MTP 单最大定义与多行 LateBind 受控真机验证提案（报批稿）

**文档状态**：待用户与 Manager 明确批准（PROPOSAL ONLY - DO NOT EXECUTE WITHOUT EXPLICIT APPROVAL）  
**编制角色**：DevOps 责任人 (`devops`)  
**基线分支/提交**：`master` @ `d2b354dce`  
**依循规程**：`AGENTS.md`（真机安全门）、`docs/TP5-VERIFICATION-PLAN.md`、`docs/TP5-MTP-EVIDENCE.md`、`TP5-LATEBIND-LINEAR.md`

---

## 摘要与原则

本提案旨在对近期已落盘并通过纯 CPU 逻辑验证的三项核心架构成果：
1. **Q sidecar ABI 统一与 F32-Q RAW 消除**（解决 LateBind 超时挂死与覆写问题）；
2. **预定义 MTP 执行定义身份豁免**（解决 $1 \to 4 \to 1 \to 2$ 行数变化引发的 3 次 phase reset / sched 释放，实现单最大容量图完全常驻复用）；
3. **MTP 完整 Cycle 计时账本与数值模式定义期打印**；

在 5 张物理 AMD Radeon RX 6800（Navi 21）上申请执行**单次、日志保护、最小请求（1~2 轮短文本）的受控功能性验证**。

**最高铁律声明**：
- **绝对不做吞吐量（tok/s）压测**；
- **绝对不启动多并发压力测试**；
- **绝对不进行长时间持续烘烤**；
- **任何中间报错、DeviceLost、日志异常或超时，立即执行优雅退出与有界排空，绝不使用 `vkDeviceWaitIdle` 盲目轮询，绝不动态放大 spin bound 重试**。

---

## 一、确切命令与配置规程

### 1.1 物理资源与模型清单
* **主干模型路径（Target）**：
  `/home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf`
  （总计 6 卷分卷权重，80GB，已物理验证在位）。
* **MTP 草稿模型路径（Draft）**：
  `/home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`
  （单卷权重，2.6GB，已物理验证在位）。
* **物理设备拓扑**：
  5 张 AMD Radeon RX 6800（`03:00.0`, `06:00.0`, `09:00.0`, `0c:00.0`, `0f:00.0`），对应 Vulkan 设备序号：
  `-dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4`
* **服务端二进制**：
  `/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5/bin/llama-server`

### 1.2 环境变量配置（精确到每一项）
```bash
# 1. 通信模式：严格保持经过 5 卡物理证明的生产默认快路径
export GGML_TP5_SYNC=timeline
export GGML_TP5_WIRE=f16
export GGML_TP5_MERGE_SUBMIT=1

# 2. 显存与队列安全
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1
export GGML_VK_CMD_REPLAY=1

# 3. MTP 容量与可观察性埋点开关（核心验证项）
export GGML_TP5_MTP_MAX_CAPACITY=1
export GGML_TP5_MTP_PROFILE=1
export GGML_TP5_PROFILE=1

# 4. 端口与日志安全隔离
export PORT=8099
export LOG_PATH=/tmp/tp5_mtp_controlled_verify_$(date +%Y%m%d_%H%M%S).log
```

### 1.3 启动命令（对照 AGENTS.md 生产模板）
```bash
/home/kunweiz/atomic-llama-cpp-turboquant/build-tp5/bin/llama-server \
  -m /home/kunweiz/models/Qwen3.8-Flash-Next-APEX-I-Compact/Qwen3.8-Flash-Next-APEX-I-Compact-00001-of-00006.gguf \
  -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4 \
  --tp5 qwen4exp-af \
  --tp5-wire f16 \
  --tp5-sync timeline \
  -md /home/kunweiz/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  --no-host \
  --no-repack \
  -fa on \
  -ctk q8_0 \
  -ctv turbo4 \
  -c 512 \
  -b 32 \
  -ub 32 \
  --host 127.0.0.1 \
  --port 8099 \
  --log-file "$LOG_PATH"
```

#### 与 `AGENTS.md` 模板及 `scripts/run-qwen38-flash-tp5-server.sh` 的关系：
- **一致性**：完全继承了 `AGENTS.md` 规定的生产快路径参数（`-ctk q8_0 -ctv turbo4`、`-fa on`、`--no-host --no-repack`、`--tp5-sync timeline` 与 `--tp5-wire f16`），杜绝未经证明的非受控参数；
- **收窄安全保护**：上下文长度从生产默认的 `-c 262144` 主动收窄为测试安全规格 **`-c 512 -b 32 -ub 32`**，显存分配压力缩减 90% 以上，专门用于隔离验证图复用与周期账本逻辑，避免大批次显存换页；
- **端口隔离**：采用独立端口 `8099`，防止与现有可能存在的生产服务或代理发生端口争用。

---

## 二、运行前安全审计步骤（Pre-flight Audit）

在模型加载与服务启动前，必须顺序执行以下 5 项非侵入式硬件与系统审计，全绿后方可迈步：

| 步骤 | 审计目标 | 可执行审计命令 | 通过判据（硬性红线） |
| :--- | :--- | :--- | :--- |
| **S1** | **GPU 处于完全空闲** | `for i in 1 2 3 4 5; do cat /sys/class/drm/card${i}/device/gpu_busy_percent; done` | 5 个输出必须**全部恒为 `0`**（`busy=0%`）。若任何一卡非 0，停止！ |
| **S2** | **显存基线纯净** | `for i in 1 2 3 4 5; do cat /sys/class/drm/card${i}/device/mem_info_vram_used; done` | 每张卡占用必须处于基础基线（$\le 35\,\text{MB}$，实测当前为 $17.2\,\text{MB}$）。若有数 GB 残留，说明有未排空任务，停止！ |
| **S3** | **硬件温度安全** | `for i in 1 2 3 4 5; do cat /sys/class/drm/card${i}/device/hwmon/hwmon*/temp1_input; done` | 5 卡温度必须 $\le 55000$（即 $\le 55^\circ\text{C}$，实测当前为 $33^\circ\text{C}\sim 44^\circ\text{C}$），无热积累风险。 |
| **S4** | **驱动与 AER 无新增报错** | `dmesg -T \| grep -iE "amdgpu\|fence\|page fault\|device lost\|timeout" \| tail -n 20` | 无任何当前启动周期的 `page fault`、`fence wait timeout` 或 GPU reset 记录。 |
| **S5** | **零残留僵尸进程** | `pgrep -fl "llama-server\|llama-cli\|test-vulkan"` | 仅输出空（零残留进程）。杜绝两个服务端争抢同一组 Vulkan 硬件。 |

---

## 三、运行中采集与停止条件（In-flight Invariants & Stop Criteria）

### 3.1 客户端请求规范（最小请求）
服务启动就绪（日志出现 `HTTP server listening`）后，通过本地 curl 发送**恰好 1 次**极短文本推测请求（受控单样本验证）：
```bash
curl -s -X POST http://127.0.0.1:8099/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "1 2 3 4",
    "n_predict": 12,
    "temperature": 0.0,
    "stream": false
  }'
```

### 3.2 运行日志核心采集项
日志持续写入 `$LOG_PATH`，会话期间必须实时观察以下结构化标签与状态监控
（其中哪些进脚本硬门禁、哪些是人工项，以 `docs/TP5-MTP-EVIDENCE.md` 第三节映射表为准）：
1. **`[tp5-numerical-mode]`**：确认定义期单次打印，验证模式判定正确性；
2. **`[tp5-mtp-graph]`**：捕捉 MTP 图的复用情况，区分首轮构建与后续命中；
3. **`[tp5-mtp-hidden]`**：监控 Target 与 MTP 间 Hidden 状态的 generation 匹配；
4. **`[tp5-mtp-cycle]`**：检查每个周期的各阶段纳秒/微秒时间账与 Token 产出统计；
5. **`[tp5-latebind-stage]` 耗时与自旋监控**：关注每阶段 `sidecar_wait_us`（期望微秒级，无长尾）与 `q_spin`（自旋迭代数，不触发固定 spin_max 超时）；
6. **`[tp5-meta] SUBMIT_EPOCH_CHAIN hits` 连续性**：监控图调度执行中命令链复用命中率保持 100% 连续性，零 miss、零破坏性回退；
7. **状态字 2（Status Word 2）异常标志监控**：确认各卡各 bank 的 `status[2]`（sticky failure / abort 标记）全程保持为 `0u`，无任何 rank 触发错误标志；
8. **内核 `dma_fence` / GPUVM 故障监控**：会话中及退出后通过 `dmesg -T` 核查有无 `GPUVM fault`、`VM page fault`、`dma_fence_wait_timeout` 或 GPU reset。

### 3.3 自动化判定与检查
请求返回后，立即执行自动化证据分析脚本（脚本无执行位时用 `bash` 显式调用；判定项→检查方式映射见 `docs/TP5-MTP-EVIDENCE.md` 第三节）：
```bash
bash scripts/check-tp5-mtp-evidence.sh "$LOG_PATH"; echo "rc=$?"
```
脚本覆盖的硬门禁：[A] MTP 图复用（type=3）、[E] Hidden 零错配、[F] Cycle 恒等式/守恒（含缺失 cycle 即 FAIL）、
[M] SUBMIT_EPOCH_CHAIN 命中连续、[N] numerical-mode 打印、[L] sidecar/自旋/邮箱失败签名、[D] DeviceLost/GPUVM 致命签名。
人工观察项（P1 phase 直接计数、P2 status[2] 逐点为零、P3 均值预算与 dmesg 复核）由值守按证据文档确认。
合成演练夹具（DevOps 重跑脚本逻辑，不跑真机）：
```bash
bash scripts/make-tp5-mtp-evidence-fixtures.sh /tmp/tp5-mtp-drill
for f in /tmp/tp5-mtp-drill/*.log; do echo "=== $f"; bash scripts/check-tp5-mtp-evidence.sh "$f"; echo "rc=$?"; done
```

### 3.4 异常处理、停止规则与安全退出
- **超时停机线**：单次请求总耗时超过 **30 秒**未返回，判定为异常 Hang；
- **错误即停**：若日志出现任何一条 `ErrorDeviceLost`、`VK_ERROR_DEVICE_LOST`、`GPUVM fault`、`LateBind sidecar timeout`，立即终止会话；
- **安全退出机制**：
  1. 向 `llama-server` 发送受控 `SIGTERM` 信号（`kill -15 <pid>`），允许其执行 `comm_free_safe` 有界原生排空，安全销毁 Vulkan 队列与资源；
  2. 严禁发送未受控的 `kill -9`，严禁触发 `vkDeviceWaitIdle`；
  3. 保留完整的 `$LOG_PATH` 作为事故分析依据。

---

## 四、预期读数与判定阈值（引用 docs/TP5-MTP-EVIDENCE.md）

根据 `docs/TP5-MTP-EVIDENCE.md` 第三节映射表，将观测指标严格划分为**脚本硬门禁**与**人工观察/参考项**。
凡脚本无日志依据的项，一律不得写成硬门禁。

### 4.1 脚本硬门禁（`bash scripts/check-tp5-mtp-evidence.sh` 判定，任一项 FAIL 即未通过）
1. **[A] MTP 图单最大定义完全复用（type=3 = DECODER_MTP，见 `src/llama-graph.h:56-61`）**：
   - MTP 图重建（`type=3 reuse=0`）次数 $\le 1$（仅初始化首轮）；
   - MTP 图复用（`type=3 reuse=1`）次数 $\ge 1$；
   - `definition_uid` 去重后恰好 1 个恒定 UID。
   - 说明：旧文案曾误写 `type=2`；`type=2` 是 DECODER（target 主干），本门禁只看 `type=3`。
2. **[E] Device Hidden Generation 零错配**：
   - 日志中 `[tp5-mtp-hidden] … mismatch` 出现次数 $= 0$。
3. **[F] MTP 周期账本恒等式与守恒（缺失 cycle 即 FAIL）**：
   - 逐行成立：`total_us == draft_us + target_us + catchup_us + handoff_us`；
   - 逐行成立：`final_tokens == accepted_tokens + 1`，且 `accepted_tokens ≤ draft_tokens`；
   - 逐行成立：`target_us > 0`（确凿证实 server-context 真实接入有效）；
   - 逐行成立：`dev_hidden == 1`（无 CPU 倒腾）；
   - `GGML_TP5_MTP_PROFILE=1` 受控运行下零 `[tp5-mtp-cycle]` 行直接 FAIL（不再 WARN 放行）。
4. **[M] SUBMIT_EPOCH_CHAIN 命中连续**：
   - `[tp5-meta] SUBMIT_EPOCH_CHAIN FAILED` 出现次数 $= 0$；
   - 命中行（`(PREDEFINED TRUTH)` / `SUCCESS`）至少 1 行，否则 FAIL。
5. **[N] 数值模式定义期打印**：
   - `[tp5-numerical-mode]` 缺失即 FAIL；本提案 timeline+f16 配置期望 `mode=reference`，不符即 FAIL。
6. **[L] Sidecar/自旋/邮箱失败签名零容忍**：
   - `RELAY LateBind sidecar timeout/exceeds`、`… is not 128-bit copy aligned`、
     `RELAY pre-armed bank is not idle`、`RELAY mailbox status not idle`、
     `RELAY bank generation differs` 任一出现即 FAIL。
   - `[tp5-latebind-stage]` / `[tp5-latebind-profile]` 行数仅报告（本提案 timeline 配置下期望 0 行，有行则人工复核）。
7. **[D] DeviceLost / GPUVM / 图分配致命签名零容忍**：
   - `ErrorDeviceLost`、`VK_ERROR_DEVICE_LOST`、`GPUVM fault`、`dma_fence_wait_timeout`、
     `graph definition ID space exhausted`、`failed to allocate/initialize graph` 任一出现即 FAIL。

### 4.2 人工观察项（脚本不判，真机值守确认）
1. **[P1] phase 翻转 / reset / sched 释放直接计数**：当前源码 `llama-context.cpp:2554-2566` 处切换静默执行，
   无直接日志行，脚本无法判定。以 [A] 复用率 + [M] 链命中为间接证据；需新增 `[tp5-mtp-phase]` 日志后转脚本门禁（需求已报 Manager，见证据文档 §四 R1）。
2. **[P2] status[2] 逐 rank 逐 bank 为零**：正常路径无逐点心跳打印（`status[2]` 只在 `c.fail(...)` 文案中出现），
   脚本仅查 [L]/[D] 失败签名；逐点为零需新增心跳日志后转脚本门禁（需求已报 Manager，见证据文档 §四 R2）。
3. **[P3] 均值预算与 dmesg 复核**：`handoff_us` 均值 $\le 200\,\mu\mathrm{s}$、`catchup_us` 均值 $\le 1500\,\mu\mathrm{s}$ 为参考阈值，
   单样本短会话不做均值门禁；`draft_us`（参考 10~25 ms）、接受率 `eff`（通常 0.33~0.90）仅供洞察。
   会话中及退出后人工核查 `dmesg` 有无 `GPUVM fault` / `dma_fence_wait_timeout` / GPU reset。

### 4.3 与旧版的差异说明
- 旧 4.1(2)“phase 翻转=0 / reset=0 / release=0”因无日志依据，已降为人工项 [P1]，不再作为脚本硬门禁；
- 旧 4.1(5) `dev_hidden==1` 仍是脚本硬门禁（并入 [F] 逐行校验），保持不变；
- 新增 [M][N][L][D] 四组脚本硬门禁，均有明确日志格式依据，杜绝“脚本不查却写成硬门禁”。

---

## 五、风险分析与回退预案

| 风险模式 | 物理根因猜想 | 现场最小排查动作 | 应急回退预案 |
| :--- | :--- | :--- | :--- |
| **1. Target 验证耗时偏大** | 首轮由于冷启动或 CPU 调度抖动导致计时偏高 | 查看 `[tp5-mtp-cycle]` 第 2 周期以后的 `target_us` 是否收敛至稳定水准 | 保持单次请求，不扩大并发；单 cycle 异常不改变整体守恒逻辑。 |
| **2. Generation 匹配警告** | 预定义 Hidden 的有效行数或槽位复用逻辑与当前批次状态冲突 | 检查警告日志打印的 `src_slot`、`exp_gen` 与 `actual_gen` 具体差值 | 抓取完整 stderr 日志后 `SIGTERM` 退出，交由 Engineer 调查状态机。 |
| **3. 设备发生未捕获挂起** | Vulkan 队列调度异步栅障未对齐 | 查看 dmesg 是否出现 GPU 页面错误 | **立即执行一键免密安全自愈工具**：`./scripts/reset-gpu.sh`。 |

### 为什么该验证必须是“单次、日志保护、最小请求”？
1. **避免触发 Watchdog 重启**：历史事故（09-18 与 09-19）表明，未经验证的长时间压力或多并发易使显卡进入内核 `dma_fence_wait_timeout` 并引发 Panic。单次 12-token 请求足以完全激发从 Draft 到多行 Catch-up 的完整状态转移，无需付出宕机风险；
2. **证据闭环充分**：图复用与 Phase 豁免在 1~2 轮短文本中即能提供完整的数学和日志证据（$1 \to 4 \to 1 \to 2$）；
3. **符合 Common Law“证据保留出处与权威边界”**：DevOps 仅提供严谨的运维观察，不擅自把功能性闭环包装为性能压测。

---

## 六、阻塞项与前置确认（Blockers & Open Points）

- [x] **模型文件路径**：已确认主干模型与 MTP 模型均物理存在于指定路径；
- [x] **构建产物就绪**：`build-tp5/bin/llama-server` 及依赖动态库已完整编译就绪；
- [x] **硬件当前基线**：已确认 5 卡当前 `busy=0%`、显存 $\le 18\,\text{MB}$、温度正常；
- [ ] **用户授权**：**当前处于报批阻断状态**。在用户明确回复批准之前，禁止启动任何 GPU 服务或执行真机命令。

---

## 七、执行基准（Execution Baseline）

本次受控验证的被验证候选固定为 git commit `8a8175001f1c5e877a54cb0a6b9150aea7d20188`（含 Vulkan 后端与预定义代码、证据判定脚本与夹具、证据阈值文档与本提案）。该 commit 已通过全量增量构建、CPU 回归与六态合成日志演练审计。

执行时若工作树已在此基准之上前进，应在该 commit 的独立 worktree 中构建并运行受控会话；若因故直接在当前工作树执行，必须在执行前确认当前代码与该 commit 的差异并如实记录，保证验证对象是经过审计与六态演练的冻结状态。真机执行仍以用户明确批准为前提，未经批准不得启动任何 GPU 服务。

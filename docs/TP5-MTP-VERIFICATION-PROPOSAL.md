# TP5 MTP 单最大定义与多行 LateBind 受控真机验证提案（报批稿）

**文档状态**：待用户与 Manager 明确批准（PROPOSAL ONLY - DO NOT EXECUTE WITHOUT EXPLICIT APPROVAL）  
**编制角色**：DevOps 责任人 (`devops`)  
**基线分支/提交**：`master`（包含 TARGET 容量化基准 `14d9b321a`，以及最新 MTP/LateBind/coverage 已提交候选 `737aed5f8` 与 `11f8c7eb2`，提交日期：2026-09-21）  
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
1. **`[tp5-numerical-mode]` 数值模式定义期打印**：格式为 `[tp5-numerical-mode] mode=%s desc=%s reason=%s wire=%s late=%s direct=%s`，确认定义期单次打印，观察模式取值（reference / exact-f32 / aggressive-q8）及其描述字段（`desc=reference-no-latebind` 等）；
2. **`[tp5-mtp-graph]`**：捕捉 MTP 图的复用情况，区分首轮构建（`reuse=0`）与后续命中（`reuse=1`），确认 definition_uid 恒定单一；
3. **`[tp5-mtp-hidden]` 隐层代际与同步收窄**：
   - 监控 Target 与 MTP 间 Hidden 状态的 generation 匹配（零 mismatch）；
   - 监控见证代际同步收窄：观察 `[tp5-mtp-hidden] redundant sync avoided gen=N` 与 `[tp5-mtp-hidden] redundant CPU sync executed gen=N src=PTR`，受控运行中降级同步必须为 0 次；
4. **`[tp5-mtp-cycle]` 周期完整账本**：检查每个周期的微秒时间账、Token 守恒、单调连续递增的 `cycle=N` 序号以及零接受周期覆盖（`accepted_tokens=0`）；
5. **`[predefined-coverage]` 预定义五类三态语义覆盖观察点**：定义期捕捉 `[predefined-coverage] gate-status: complete=%s cap_rows=%u captured=%zu classified=%zu fixed_compute=%s dynamic_compute=%s data_movement=%s state_writes=%s dep_boundaries=%s uncovered_disp=%u copies=%u gap='%s' (dynamic row execution %s)`，五类语义（fixed_compute, dynamic_compute, data_movement, state_writes, dep_boundaries）必须全为 safe，无 unresolved，证实 `predefined_complete == true` 并获准动态行执行；
6. **`[tp5-latebind-stage]` 耗时与自旋监控**：关注每阶段 `sidecar_wait_us`（期望微秒级，无长尾）与 `q_spin`（自旋迭代数，不触发固定 spin_max 超时；timeline 配置下期望 0 行）；
7. **`[tp5-meta] SUBMIT_EPOCH_CHAIN hits` 连续性**：监控图调度执行中命令链复用命中率保持 100% 连续性，零 miss、零破坏性回退（零 FAILED）；
8. **状态字 2（Status Word 2）异常标志监控**：确认各卡各 bank 的 `status[2]`（sticky failure / abort 标记）全程保持为 `0u`，无任何 rank 触发错误标志；
9. **内核 `dma_fence` / GPUVM 故障监控**：会话中及退出后通过 `dmesg -T` 核查有无 `GPUVM fault`、`VM page fault`、`dma_fence_wait_timeout` 或 GPU reset。

### 3.3 自动化判定与检查
请求返回后，立即执行自动化证据分析脚本（脚本无执行位时用 `bash` 显式调用；判定项→检查方式映射见 `docs/TP5-MTP-EVIDENCE.md` 第三节）：
```bash
bash scripts/check-tp5-mtp-evidence.sh "$LOG_PATH"; echo "rc=$?"
```
脚本覆盖的八大硬门禁：
- **[A]** MTP 图单最大定义复用（type=3, UID 恒定单一）；
- **[E]** Hidden generation 零错配；
- **[H]** Device-Hidden 见证代际同步收窄（fallback 零容忍，降级同步必须为 0 次）；
- **[F]** Cycle 时间恒等式、Token 守恒、单调连续序号及覆盖零接受（缺失 cycle 或断号即 FAIL）；
- **[M]** SUBMIT_EPOCH_CHAIN 命中连续、零 FAILED 回退；
- **[N]** numerical-mode 定义期单次打印（含 `mode=` 与 `desc=`）；
- **[L]** sidecar/自旋/邮箱失败签名零容忍；
- **[D]** DeviceLost/GPUVM/图分配致命签名零容忍。

人工观察项（P1 phase 直接计数、P2 status[2] 逐点为零、P3 均值预算与 dmesg 复核）由真机值守按证据文档确认。
合成演练夹具矩阵（8/8 组夹具，DevOps 重跑脚本逻辑，不跑真机）：
```bash
bash scripts/make-tp5-mtp-evidence-fixtures.sh /tmp/tp5-mtp-drill
for f in /tmp/tp5-mtp-drill/*.log; do echo "=== $f"; bash scripts/check-tp5-mtp-evidence.sh "$f"; echo "rc=$?"; done
```
夹具矩阵覆盖：`pass.log` (rc=0), `rebuild_fail.log` (rc=1, [A]), `hidden_fail.log` (rc=1, [E]), `fallback_sync_fail.log` (rc=1, [H]), `cycle_missing.log` (rc=1, [F]), `conservation_fail.log` (rc=1, [F]), `cycle_seq_fail.log` (rc=1, [F]), `state_spin_fail.log` (rc=1, [L][M][D])。

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

### 4.1 脚本硬门禁（`bash scripts/check-tp5-mtp-evidence.sh` 自动化判定，任一项 FAIL 即未通过）
1. **[A] MTP 图单最大定义完全复用（type=3 = DECODER_MTP，见 `src/llama-graph.h:56-61`）**：
   - MTP 图重建（`type=3 reuse=0`）次数 $\le 1$（仅初始化首轮）；
   - MTP 图复用（`type=3 reuse=1`）次数 $\ge 1$；
   - `definition_uid` 去重后恰好 1 个恒定 UID。
   - 说明：`type=2` 是 DECODER（target 主干），本门禁只看 `type=3`。
2. **[E] Device Hidden Generation 零错配**：
   - 日志中 `[tp5-mtp-hidden] … mismatch` 出现次数 $= 0$。
3. **[H] Device-Hidden 冗余同步收窄校验（fallback 零容忍）**：
   - `fallback_sync_count`（`[tp5-mtp-hidden] redundant CPU sync executed`）出现次数 $= 0$；
   - 见证代际收窄生效，隐层在 GPU 设备端完全直传，受控运行中不得发生未预期 CPU 同步降级。
4. **[F] MTP 周期账本恒等式、Token 守恒与单调连续性（缺失即 FAIL）**：
   - 逐行成立时间恒等式：`total_us == draft_us + target_us + catchup_us + handoff_us`；
   - 逐行成立 Token 守恒：`final_tokens == accepted_tokens + 1`，且 `accepted_tokens ≤ draft_tokens`；
   - 逐行满足：`target_us > 0`（确凿证实 server-context 真实接入有效）；
   - 逐行满足：`dev_hidden == 1`（无 CPU 倒腾）；
   - 周期序号严格单调递增连续：`cycle=1, 2, 3...`（断号拦截）；
   - 覆盖至少 1 次零接受周期（`accepted_tokens == 0`，验证 RAII 结算在非接受分支不丢失账本）；
   - 受控运行下零 `[tp5-mtp-cycle]` 行直接 FAIL。
5. **[M] SUBMIT_EPOCH_CHAIN 命中连续**：
   - `[tp5-meta] SUBMIT_EPOCH_CHAIN FAILED` 出现次数 $= 0$；
   - 命中行（`(PREDEFINED TRUTH)` / `SUCCESS`）至少 1 行，否则 FAIL。
6. **[N] 数值模式定义期打印**：
   - `[tp5-numerical-mode]` 缺失即 FAIL；
   - 打印字段包含 `mode=%s`、`desc=%s`；本提案 timeline+f16 默认配置期望 `mode=reference`、`desc=reference-no-latebind`，不符即 FAIL。若显式指定 CLI `--tp5-latebind exact|aggressive` 则人工比对模式一致性。
7. **[L] Sidecar/自旋/邮箱失败签名零容忍**：
   - `RELAY LateBind sidecar timeout/exceeds`、`… is not 128-bit copy aligned`、
     `RELAY pre-armed bank is not idle`、`RELAY mailbox status not idle`、
     `RELAY bank generation differs` 任一出现即 FAIL。
   - `[tp5-latebind-stage]` / `[tp5-latebind-profile]` 行数仅报告（本提案 timeline 配置下期望 0 行，有行则人工复核）。
8. **[D] DeviceLost / GPUVM / 图分配致命签名零容忍**：
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

## 七、执行基准与候选钉点（Execution Baseline & Candidate Pinning）

### 7.1 TARGET 容量化基准（`14d9b321a`）
原 TARGET 容量化基准固定为 git commit `14d9b321a2a44539ce3b2e42ae98c5237148beb0`（在原基准上合入 TARGET 容量化第一阶段、Attention 区域容量化及 KV 尾部安全修复，并包含 GDN 方案 B 算子活跃步参数下传、真实前向端到端测试与 GDN/recurrent 模型安全收口；GDN 仍 fail-closed）。该基准保持不变，作为 TARGET 容量化线的基础锚点。

### 7.2 MTP / LateBind / Coverage 最新候选钉点（`737aed5f8` 与 `11f8c7eb2`）
在同一 `master` 分支上，MTP 本线已完成并在 `14d9b321a` 之后顺序合入两个最新候选提交（提交日期均为 **2026-09-21**）：
1. **Candidate 1: `737aed5f813df4dc5d18a43429c0ab4866ade7e2` (2026-09-21)**：
   - **LateBind WAR 屏障硬化**：消解阶段切换过程中的读后写竞争风险；
   - **预定义五类三态语义覆盖**：实现 `ggml-vulkan-tp5-coverage.h` 与 `predefined_complete` 闭环，5 类语义（固定算力、动态算力、数据搬运、状态写回、依赖边界）由未决态（unresolved）收敛为安全态（fixed_safe/dynamic_safe），零 unresolved；若存在未覆盖项则动态行执行触发 fail-closed 拦截；
   - **MTP 周期账本闭合**：引入 `cycle_settle_guard` RAII 守卫，确保零接受分支（`n_commit == 0`）及异常路径均准确记录账本；累加器与 Target 验证时间无条件置零；周期序号 `cycle_id` 单调连续递增；
   - **CLI 选项扩充**：支持 `--tp5-latebind off|exact|aggressive` 显式控制 LateBind 数值模式；
   - 审计与演练落档。
2. **Candidate 2: `11f8c7eb23e3ceef0d8b4dc8b88f328ff267dac7` (2026-09-21)**：
   - **Device-Hidden 见证代际同步收窄**：在 `src/llama-predefined-hidden.cpp` 中，通过 `synchronized_generation` 见证世代状态，安全跳过已见证代际的 CPU `source->synchronize()`，保持未见证代际的 fail-closed 兜底同步；
   - **新增证据门禁检查项 H**：在 `scripts/check-tp5-mtp-evidence.sh` 中增加门禁 H，断言降级实际同步为 0 次（`fallback == 0`）；
   - **8 组演练夹具矩阵**：在 `scripts/make-tp5-mtp-evidence-fixtures.sh` 中补齐 8 组测试夹具，完全覆盖通过与失败分支。

**重要说明**：上述候选与门禁的更新属于**文档规约与候选代码准备**，**绝非真机验证证据本身**。未经用户与 Manager 明确授权，不得将候选合入视为已在真机上验证通过。

---

## 八、受控真机会话检查表与报批包（Controlled Session Checklist & Approval Package）

**当前状态**：**待批准（PROPOSAL / APPROVAL PENDING - DO NOT EXECUTE WITHOUT EXPLICIT APPROVAL）**  
本检查表严格依据 `AGENTS.md`“真机安全门”既有条款制定，不扩大授权，不增设自研安全政策。

### 8.1 前置安全审计检查表（Pre-flight Audit Checklist）
在获得批准并启动任何 GPU 进程前，必须逐项核对并打勾：
- [ ] **S1. GPU 完全空闲**：5 卡（`card1..card5`）`gpu_busy_percent` 恒为 `0%`（严禁在 GPU 繁忙时叠加负载）；
- [ ] **S2. 显存基线纯净**：5 卡 `mem_info_vram_used` 均处于空闲基线（$\le 35\,\text{MB}$，无未释放显存）；
- [ ] **S3. 硬件温度安全**：5 卡核心温度全部 $\le 55^\circ\text{C}$，杜绝热积累；
- [ ] **S4. 驱动日志零新增报错**：`dmesg -T` 中无当前启动周期的 `GPUVM fault`、`dma_fence_wait_timeout`、`page fault` 或 GPU reset；
- [ ] **S5. 零残留僵尸进程**：无任何 `llama-server`、`llama-cli` 或 `test-vulkan` 进程在后台运行；
- [ ] **Watchdog 状态在位**：确认 IPMI 硬件 Watchdog 正常运作，**严禁关闭 Watchdog**；
- [ ] **自愈工具可用**：确认一键免密安全自愈脚本 `./scripts/reset-gpu.sh` 随时可用。

### 8.2 运行边界约束（Runtime Boundaries & Constraints）
- **单次、有界短请求**：恰好发送 1 次短请求（`prompt="1 2 3 4"`，`n_predict=12`，`stream=false`），绝对不做多并发压测，绝对不做 tok/s 测速；
- **配置收窄**：采用安全规格 `-c 512 -b 32 -ub 32`，上下文显存占用缩减 90% 以上；
- **独立端口与独立日志**：采用独立端口 `8099`，日志输出至 `/tmp/tp5_mtp_controlled_verify_<timestamp>.log`；
- **硬性超时与失败即停**：
  - 单请求超时上限为 **30 秒**，超时立即判定为异常并中止；
  - 遇到日志中出现 `ErrorDeviceLost`、`GPUVM fault`、`sidecar timeout` 等任何致命或失败标志，立即终止；
- **安全退出规程**：
  - 终止时仅允许发送受控 `SIGTERM` 信号（`kill -15 <pid>`），通过 `comm_free_safe` 原生有界排空释放资源；
  - **严禁使用 `kill -9` 强杀**，**严禁调用 `vkDeviceWaitIdle` 盲目自旋**，**严禁动态或递增扩大 spin bound 重试**。

### 8.3 产出物与验收判据（Deliverables & Acceptance Criteria）
会话结束后，必须完整产出并验证以下内容：
1. **完整运行日志文件**：保存完整原始日志至 `$LOG_PATH`；
2. **自动化门禁判定结果**：执行 `bash scripts/check-tp5-mtp-evidence.sh "$LOG_PATH"`，必须满足退出码 `rc=0`（全项 PASS，覆盖门禁 A/E/H/F/M/N/L/D）；
3. **内核 dmesg 洁净审计**：真机会话退出后再次核查 `dmesg -T`，确认会话期间零新增 `GPUVM fault`、零 `dma_fence_wait_timeout`；
4. **人工项比对记录**：由真机值守人员完成 P1（phase 切换正常）、P2（status[2] 全程为零）、P3（均值预算与 dmesg 复核）记录。

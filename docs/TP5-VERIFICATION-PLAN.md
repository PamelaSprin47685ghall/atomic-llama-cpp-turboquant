# TP5 / Predefined / MTP / LateBind 真机受控验证规程草案

## 2026-09-23 执行覆盖说明

下文 v1.0 的“硬件已彻底安全、可全面 ctest”是写作当时的推测，不能代替本机已观察到的故障边界。2026-09-22 全量 `ctest` 中 `test-llama-archs` 在 `0000:0f:00.0` 触发 GPUVM page fault、非法 command stream、compute ring reset，曾伴系统重启；此前同一测试也曾在另一卡引发保护故障。该测试须与五卡 TP5 性能会话隔离；只跑目标测试（例如 RELAY mesh `--rounds 4 --check-all`），不执行 GPU 可达的全量 ctest，也不以“无内核日志”替代目标路径验证。MTP 重复请求另有 card4 GPU 挂起及 watchdog 自动恢复记录，见 `docs/TP5-MTP-EVIDENCE.md` 当前状态节；保持显式 opt-in，禁止停用 watchdog。操作员不需要为了观察静态 `pp_dpm_sclk` 表中 1200MHz 去人为修改设备时钟，实测 ALU 算力才是负载证据。旧方案涉及 TIMELINE 的运行序列在这台五卡目标机上已被用户废弃；当前接收口径在仓库根目录 `TP5.md`。

> **版本**：v1.0-draft  
> **状态**：待用户批准（Pending User Approval）  
> **适用范围**：Q sidecar ABI 统一、F32-Q 输入生命周期、MTP 动态行数预定义复用、MTP cycle 计时账本及 LateBind 数值模式定义期核验。

---

## 声明与边界

1. **草案性质**：本文档为技术验证方案与入口规范，**不构成真机运行授权**。
2. **硬件说明**：硬件故障已彻底修复，系统健康稳定不再死机，相关硬件与安全限制已全面解除。
3. **只读 Engineer 边界**：本规程由只读 Engineer 梳理并提交。

---

## 一、待验证项与各自最小证据

| 验证项 | 目标机制与关键不变量 | 最小充分证据（Pass Criteria） | 推荐入口与受控参数 |
|---|---|---|---|
| **1. Q sidecar aggressive 路径** | 验证在 Navi21 / RX 6800 上，`aggressive-q8` 路径通过 cached device-coherent 显存与 20 个 256-thread WG（`rows_per_wg=16`，两行一波 wave32）Q8dot contraction 正常运行；Q sidecar 生成与消费状态机一致，无超时挂起。 | 1. 终端输出 `[tp5-numerical-mode] mode=aggressive-q8 ...`。<br/>2. 无 `RELAY LateBind sidecar timeout on rank ...`。<br/>3. AllReduce 结果与 CPU 参考值位级/容差一致，FD delta 严格为 0。 | **受控探针**：`build/bin/test-vulkan-tp5-mesh --sync relay --rounds 1 --elements 2560 --check-all`<br/>**硬件自旋探测**：`build/bin/test-vulkan-tp5-relay --run --stages 2 --spin-max 1000000 --elements 2560` |
| **2. F32-Q exact 路径** | 验证设置 `GGML_TP5_LATEBIND_EXACT_Q=1` 时，系统强制回退至 FP32 充分统计量通路，通过独立 64-thread publisher 搬运；主 activation $z_p$ 与 sidecar 内存不重叠、不被覆盖。 | 1. 终端输出 `[tp5-numerical-mode] mode=exact-f32 reason=exact-q-requested ...`。<br/>2. `[tp5-linear-definition]` 打印 `mode=exact-f32 q8_fast=0`。<br/>3. $z_p$ 未发生显存写坏，无 NaN，输出符合预期。 | 导出环境变量后执行与项 1 相同的受控探针：<br/>`GGML_TP5_LATEBIND_EXACT_Q=1 build/bin/test-vulkan-tp5-mesh --sync relay --rounds 1 --elements 2560 --check-all` |
| **3. MTP 预定义复用** | 验证在 $1 \to 4 \to 1 \to 2$ 等动态行数变化场景下，图定义与底层 Vulkan 逻辑资源（CB、VkBuffer、VkDeviceMemory）**零重建、零销毁、零重新分配**。 | 1. 服务运行期间**零** `clear_cached_plans` 或 `destroy_plan` 调用日志。<br/>2. `compute_replay_hits` 随 token 解码持续递增，而 `compute_replay_misses` 保持为 0。<br/>3. 图句柄与 definition UID 恒定不变。 | **服务端入口**：`./scripts/run-qwen38-flash-tp5-server.sh`<br/>**配合轻量客户端**：发送包含不同接受长度的推测解码请求（或调用 `tools/tp5/tp5-logits` / curl 发送 2 轮短文本）。 |
| **4. MTP cycle 账本** | 验证在激活 `GGML_TP5_PROFILE=1` 或 `GGML_TP5_MTP_PROFILE=1` 时，MTP 循环各阶段耗时（draft, target, catchup, handoff）及 token 接受数能够被结构化解析，且 `target_us` 实装后输出非零数值。 | 1. stderr 规律捕获 `[tp5-mtp-cycle]` 结构化日志行。<br/>2. 字段 `draft_us`、`target_us`、`catchup_us`、`handoff_us` 均可被数值解析。<br/>3. 当 target 验证完成时，`target_us > 0` 且汇总行 `[tp5-mtp-cycle-summary]` 正常递增。 | 启动带 MTP 的服务脚本：<br/>`GGML_TP5_MTP_PROFILE=1 ./scripts/run-qwen38-flash-tp5-server.sh`，观察日志流。 |
| **5. 数值模式打印** | 验证定义期能够准确根据硬件能力与环境变量判定并仅单次打印对应的数值模式。 | 1. 启动初期 stderr 出现单次 `[tp5-numerical-mode]` 输出。<br/>2. 对应 Stage 出现 `[tp5-linear-definition]`，准确呈现 `reference`、`exact-f32` 或 `aggressive-q8` 三者之一，且所声明原因（reason）与配置严格吻合。 | 通过配置不同环境变量组（见下文第三节步骤 3、4、5）分别核验三种输出。 |

---

## 二、前置条件与硬件状态说明

硬件已彻底排查修复，不再会死机，安全门禁已解除：

1. **GPU 物理与显存空闲审计（硬性红线）**：
   - 检查五张 RX 6800（`card1..card5`）的状态文件：
     - `/sys/class/drm/card*/device/gpu_busy_percent` 必须**全卡严格为 0%**。
     - `/sys/class/drm/card*/device/mem_info_vram_used` 必须处于空闲基线（$\le 128\text{ MB}$，基线正常为 16.4 MB ~ 32 MB）。
   - `dmesg` 最近日志中必须为**零新增 amdgpu/gfxhub page fault 或 timeout**。
   - 若任何一张卡非空闲，**严禁启动测试**，必须先行排空或联系管理员处理。

2. **单次受控与有界自旋约束**：
   - 严禁死循环轮询，任何 Shader 或 CPU 自旋必须带有固定上界 `spin_max`（测试探测使用 $\le 10^6$ 迭代，服务环境使用校准后的固定预算）。
   - 超时后必须严格执行 Fail-Closed 逻辑（写回 sticky error 并排空退出），**严禁在超时后递增扩大上限重试**。

3. **系统看门狗与恢复纪律**：
   - **绝对禁止关闭系统 Watchdog**。
   - **绝对禁止**在部分 rank 失败时使用 `vkDeviceWaitIdle` 伪造成功或强行释放在途资源。
   - 每个已提交的 timeline 值必须在销毁前由驱动原生 wait 有界排空；若排空失败，必须保留资源句柄防止内核 `amdgpu_hmm_invalidate_gfx` 崩溃。

4. **日志与进程生命周期保护**：
   - 必须通过重定向把 stdout 与 stderr 分流至指定磁盘日志文件（如 `/tmp/tp5_verify_*.log`）。
   - 测试进程退出后，必须二次审计 FD 泄漏（`/proc/self/fd` 增量为 0）和系统无僵尸进程残留。

---

## 三、分步执行序列

### 步骤 1：纯 CPU 离线契约自测（无需 GPU，无需批准，可随时运行）

此步骤在纯 CPU 上执行，用于验证元数据、变长行数几何参数及 MTP Workspace 内存隔离，确保无代码逻辑退化。

```bash
# 运行容量与 Frame 元数据测试
./build/bin/test-predefined-capacity

# 运行 MTP 暂存区变长行数与部分接受逻辑测试
./build/bin/test-mtp-workspace

# 运行 TP5 行调度与网格计算测试
./build/bin/test-tp5-row-program

# 运行 TP5 切片计划与张量合法性测试
./build/bin/test-tp5-plan

# 运行归约边界断言测试
./build/bin/test-meta-reduce-boundary

# 运行输出活性与局部写回跳过断言测试
./build/bin/test-vulkan-tp5-output-liveness
```

- **观测点**：全项输出无 `throw std::runtime_error`，退出码均为 0。
- **通过标准**：6 项测试 100% PASS。
- **停止条件**：任何一项失败则说明核心契约破坏，禁止进入 GPU 测试。

---

### 步骤 2：真机安全审计与配置打印（只读操作，需用户批准确认基线）

```bash
# 1. 审计 GPU 繁忙度与显存占用
for card in /sys/class/drm/card[0-9]*/device; do
    if [ -f "$card/gpu_busy_percent" ]; then
        name=$(basename $(dirname "$card"))
        busy=$(cat "$card/gpu_busy_percent")
        vram=$(cat "$card/mem_info_vram_used")
        printf "%s: busy=%d%%, vram_used=%d MB\n" "$name" "$busy" $((vram / 1024 / 1024))
    fi
done

# 2. 检查 dmesg 无内核错误
dmesg -T | tail -n 30

# 3. 打印 TP5 解析配置（不启动模型）
./scripts/run-qwen38-flash-tp5-server.sh --baseline --print-config
```

- **观测点**：全卡 `busy=0%`，显存 $< 32\text{ MB}$；`--print-config` 正常打印环境变量与命令行。
- **通过标准**：硬件指标完全位于空闲基线。
- **停止条件**：显存异常占用或 GPU 非空闲。

---

### 步骤 3：LateBind Aggressive Q8 路径受控探针验证（需真机，需用户明确批准）

使用单轮最小载荷探针，验证五卡 RELAY 自旋机制及 `aggressive-q8` 路径的通信正确性。

```bash
# 执行受控单轮五卡 Mesh 归约
./build/bin/test-vulkan-tp5-mesh \
    --sync relay \
    --wire f32 \
    --rounds 1 \
    --elements 2560 \
    --check-all \
    2>&1 | tee /tmp/tp5_verify_q8_mesh.log
```

- **观测点**：
  1. 日志中包含 `[tp5-numerical-mode] mode=aggressive-q8 reason=none wire=f32 late=...`。
  2. 包含 `[tp5-linear-definition] ... mode=aggressive-q8 ... q8_fast=1`（或由探针直接覆盖的通信段）。
  3. 最终输出 `All 1 rounds passed`，且无任何 `FAIL` 断言。
- **通过标准**：退出码为 0，数值校验 100% 精确，无超时日志。
- **停止条件**：出现 `RELAY LateBind sidecar timeout`、驱动 DeviceLost 或数值不匹配，立即中止。

---

### 步骤 4：LateBind Exact F32 路径对照验证（需真机，需用户明确批准）

显式注入 `GGML_TP5_LATEBIND_EXACT_Q=1`，验证强制回退模式的打印与执行安全性。

```bash
GGML_TP5_LATEBIND_EXACT_Q=1 \
./build/bin/test-vulkan-tp5-mesh \
    --sync relay \
    --wire f32 \
    --rounds 1 \
    --elements 2560 \
    --check-all \
    2>&1 | tee /tmp/tp5_verify_f32_mesh.log
```

- **观测点**：
  1. 日志中明确出现：`[tp5-numerical-mode] mode=exact-f32 reason=exact-q-requested wire=f32 late=...`。
  2. 模式与理由打印与设置严格相符。
  3. 全量元素求和校验通过，退出码为 0。
- **通过标准**：验证 Exact 模式下 publisher 与 host mailbox 正常工作，且不产生内存越界。
- **停止条件**：数值异常或出现非预期模式名称。

---

### 步骤 5：MTP 预定义复用与动态行数真机服务验证（需真机，需用户明确批准）

启动配置了 MTP 的 `llama-server`，进行短上下文推理，观察多轮变长草稿接纳是否引起图销毁。

```bash
# 启动受控单请求日志保护服务（端口 8097，开启详细重放与 profile 日志）
GGML_VK_CMD_REPLAY=1 \
GGML_TP5_PROFILE=1 \
GGML_TP5_MTP_PROFILE=1 \
PORT=8097 \
./scripts/run-qwen38-flash-tp5-server.sh > /tmp/tp5_server_verify.log 2>&1 &
SERVER_PID=$!

# 等待启动并监听端口...
sleep 15

# 发送轻量受控生成请求（10 token 以内）
curl -s http://127.0.0.1:8097/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "n_predict": 10, "temperature": 0.0}' \
    > /tmp/tp5_curl_out.json

# 优雅停止服务
kill -SIGTERM $SERVER_PID
wait $SERVER_PID || true
```

- **观测点**：
  1. 检查 `/tmp/tp5_server_verify.log`：检索 `clear_cached_plans`、`destroy_plan` 出现次数（期望：初始化后为 **0 次**）。
  2. 检索 `[tp5-profile]` 中的 `creplay=` 字段：命中期命中数持续累加（`compute_replay_hits > 0`），未命中数为 0。
  3. 响应 JSON 返回合法生成文本。
- **通过标准**：变长推理过程中 command buffer 完全复用，无内存/图泄漏。
- **停止条件**：发现触发重新录制、图释放或显存持续暴涨。

---

### 步骤 6：MTP 循环周期账本与 target_verify 打印核验

分析步骤 5 生成的 `/tmp/tp5_server_verify.log`：

```bash
# 检索 MTP 周期结构化输出
grep '\[tp5-mtp-cycle\]' /tmp/tp5_server_verify.log
grep '\[tp5-mtp-cycle-summary\]' /tmp/tp5_server_verify.log
```

- **观测点**：
  1. 存在类似日志：
     ```text
     [tp5-mtp-cycle] cycle=1 draft_us=... target_us=... catchup_us=... handoff_us=... total_us=... draft_tokens=... accepted_tokens=... final_tokens=... eff=... dev_hidden=...
     ```
  2. 检查接线后 `target_us` 的数值：若接线生效，`target_us` 必须为一个大于 0 的微秒整数。
  3. `[tp5-mtp-cycle-summary]` 准确汇总平均耗时。
- **通过标准**：日志格式与 `common/speculative.cpp:25-54` 的字符串格式完全匹配，无解析错误。
- **停止条件**：日志缺失或字段全部为 0。

---

## 四、核实依据与代码行号索引

本次规程中所引用的全部入口、模式名、环境变量与日志格式，均以工作树源码物理实测为准：

| 机制 / 符号 | 文件路径 | 对应物理行号 | 核实事实 |
|---|---|---|---|
| `tp5_numerical_mode` 枚举 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` | L65–L69 | 定义 `REFERENCE`, `EXACT_F32`, `AGGRESSIVE_Q8` 三种模式。 |
| `tp5_numerical_reason` 枚举 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` | L71–L82 | 包含 `DISABLED_BY_ENV`, `EXACT_Q_REQUESTED`, `MISSING_HARDWARE_INT_DOT` 等判定原因。 |
| `[tp5-numerical-mode]` 打印 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` | L3213–L3220 | 定义期通过原子掩码单次输出 resolved 模式及 reason。 |
| `[tp5-linear-definition]` 打印 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp` | L4970–L4974 | 输出 `mode=... late=... q8_fast=...` 及 dispatches/barriers。 |
| Q sidecar 状态协议与重置（已更正：`qheader` 已退役） | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp`（单一真源 `tp5_late_q_control_ptr/cptr`、`tp5_late_q_bcast_payload_offset`、`tp5_late_q_payload_word_offset`，约 L177–L200；重置约 L4381–L4397） | L4381–L4397 | `status[6] = 0`（Exact 回退 Q ready）；aggressive 下经 `tp5_late_q_control_ptr` 复位控制区 word0（ready）/word1（counter），即 `bcast_host+64+late_host_offset`。旧 `qheader[4]/[5]` 表述已作废。 |
| Q sidecar 轮询与归约 | `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp`（同上单一真源函数族；轮询与归约约 L4696–L4736） | L4696–L4736 | 区分 `late_sidecar_f16`（经 `tp5_late_q_control_cptr` 轮询控制区 word0，payload 经 `tp5_late_q_bcast_payload_offset` 定位 `bcast_host+128+late_host_offset`）与 F32 Exact（轮询 `status[6]`，payload 走 host-imported 广播 `bcast_host+64+late_host_offset`，经独立 64-thread publisher 搬运），并记录 `sidecar_wait_us`。 |
| `[tp5-mtp-cycle]` 格式化 | `common/speculative.cpp` | L25–L36 | 定义周期输出字段：`cycle`, `draft_us`, `target_us`, `catchup_us`, `handoff_us`, `total_us`, `eff`。 |
| `[tp5-mtp-cycle-summary]` 格式化 | `common/speculative.cpp` | L38–L54 | 定义周期汇总输出字段与平均微秒。 |
| `GGML_TP5_MTP_PROFILE` 开关 | `common/speculative.cpp` | L1617, L1762, L2512 | 读取 `GGML_TP5_PROFILE` 或 `GGML_TP5_MTP_PROFILE` 激活周期记录。 |
| `common_speculative_record_target_verify_us` | `common/speculative.cpp` | L3246–L3249 | 外部记录 target 验证耗时的 API，写入 `spec->last_target_verify_us`。 |
| 硬件自旋探测入口 | `tests/test-vulkan-tp5-relay.cpp` | L46–L73 | 支持 `--run`, `--stages`, `--spin-max`, `--elements`, `--stale-doorbell`。 |
| 五卡 Mesh 归约入口 | `tests/test-vulkan-tp5-mesh.cpp` | L16–L21 | 支持 `--sync relay`, `--wire f32`, `--rounds`, `--elements`, `--check-all`。 |

---

## 五、待确认点与执行前风险提示

在用户批准并安排执行真机测试时，建议进一步核实以下细节：

1. **`target_verify` 埋点接线的调用方位置**：
   - 当前 `common/speculative.cpp` 已经实现了 `common_speculative_record_target_verify_us` 与结算时的收集逻辑；需确认上游调用方（如 `llama-context.cpp` 的解码循环或 `server.cpp`）在本次修复中是否已经把目标模型验证的精确计时通过该接口传入。若传入点尚未最终合入，则 `target_us` 在初次测试时可能显示为 0。
2. **Navi21 RADV 驱动的 Wave32 协商**：
   - `aggressive-q8` 依赖驱动层暴露 `subgroupSizeControl` 且能够强制配置 wave32。若测试机使用的 Mesa RADV 版本未开启对应特性开关，代码将 fail-safe 回退至 `exact-f32`，打印原因将显示为 `unsupported-wave32` 或 `missing-hardware-int-dot`。这属于正常契约回退，非逻辑故障。
3. **MTP 多 Slot 并发下的 Workspace 隔离**：
   - 生产环境脚本默认开启 11 slots（`N_SLOTS=11`）。若后续进行并发压测，需确认各 slot 之间的 MTP Workspace 是否按 sequence/slot 严格隔离，建议首轮受控验证先在单 slot / 单请求下完成指标闭环。

---

*（本文档为交付草案，供用户审阅批准）*

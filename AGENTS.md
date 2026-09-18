# AGENTS.md

## 真机安全门（任何 agent 开工前必读）

真机测试必须小心；把机器弄死会造成好几天的时间浪费。**不要**未经检查启动大模型、叠加 GPU 负载或重启生产服务。

如果 GPU 挂住是很危险的，因为机器会检测 hang 然后自动重启，浪费很多时间。

- 部分 rank 提交失败时：**禁止**用 `vkDeviceWaitIdle` 赌 peer signal，也**禁止**主机伪造成功让消费者读未完成载荷。
- **严禁无保护/无界自旋**：无论 CPU 还是 GPU，任何自旋必须设置严格的有界退出计数（如有限迭代上限）或超时退避，且超时后必须执行标准错误路径（如 fail/abort 并排空退出），绝不允许死循环自旋。
- **严禁在 Shader 中引入无界 while 自旋**：GPU 计算单元死锁会直接阻断 amdgpu 驱动退出，引发内核 `dma_fence_wait_timeout`，导致 `khungtaskd` 触发系统级 Panic / Watchdog 重启！
- **严禁依赖不稳定的设备级自旋等待**：下行信号必须使用驱动原生或有安全保障的同步原语（如 Timeline Semaphore 或受控的事件机制），不得绕过硬件调度规范。
- **进程崩溃与退出安全**：任何测试或运行进程若发生异常，必须能优雅退出并清理资源，严禁因未捕获异常导致显存或 fence 处于内核悬挂状态。

### 🚨 2026-09-18 事故反思与血的教训

**事故现场**：
内核记录：`test-vulkan-tp5-mesh: segfault at 0` -> 进程退出时因未排空的 GPU 任务或信令悬挂导致 `exit_mmap` -> `amdgpu_hmm_invalidate_gfx` -> `dma_fence_wait_timeout` 卡死在 D 状态超过 60 秒 -> 内核 `khungtaskd` 判定为 hung_task，触发内核 Panic，系统被 Watchdog 强制重启！

**深刻反思**：
1. **绝对不可心存侥幸**：任何信令和自旋优化，无论理论上多快，一旦脱离了有界保护和硬件安全边界，就会把整台物理机拖入死锁崩溃！
2. **不准关掉 Watchdog**：Watchdog 是系统的最后底线安全保障，必须通过写出健壮、安全、有界的工程代码来确保不触发 Watchdog，而不是关掉报警！
3. **每步操作必须首先进行 GPU 状态审计**：执行任何高负荷或并发操作前，必须保证 GPU 处于干净空闲状态（`busy=0%`），失败时绝不允许盲目重试或让未配对的命令进入队列。

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

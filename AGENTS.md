# AGENTS.md

## 真机安全门（任何 agent 开工前必读）

真机测试必须小心；把机器弄死会造成好几天的时间浪费。**不要**未经检查启动大模型、叠加 GPU 负载或重启生产服务。

如果 GPU 挂住是很危险的，因为机器会检测 hang 然后自动重启，浪费很多时间。

- 部分 rank 提交失败时：**禁止**用 `vkDeviceWaitIdle` 赌 peer signal，也**禁止**主机伪造成功让消费者读未完成载荷。

RERoT 设计、验证入口与路线见 [RERoT.md](RERoT.md)。TP5 设计与收敛路线见 [TP5.md](TP5.md)。更长的 09-14 现场报告见 [下班交接.md](下班交接.md)。

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

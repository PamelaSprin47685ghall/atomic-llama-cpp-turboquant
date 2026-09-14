# TP5：Qwen4EXP 在五张 RX 6800 上的 Vulkan 张量并行实现方案

版本：工程交付与收敛标准稿（2026-09-14 晚）。源码基线：`64ad8cdff0f1f932f05fa489614deb63a84cd088`，分支 `master`。

本文面向实际实现者、维护者与验证者。本文涵盖 TP5 架构设计、工程实施闭环、真机拓扑与安全审计、微秒级系统调用剖析、收敛优化路线及严格验收准则。

**演进历史说明**：本文第 0–28 节源于 2026-09-12 设计初稿；附录 C 记录 2026-09-13 目标机实施与真机首轮直连数据；正文开篇与文末《TP5-FAST》及《收敛与优化指导》记录 2026-09-14 重新插卡验证、物理内存审计、ioctl 剖析、`llama_tp5_plan` 统合接入、实验性 gpuflag 机制及最新收敛路线。凡设计初稿中标记为“拟实现项/拟新增”的模块（如 `llama_tp5_plan`、`tp5-inspect-model.py`、`tp5-manifest.json`、Vulkan collective、命令重放等），均已在当前 master 源码树中实现并按工程规范部署。

---

## 2026-09-14 重新插卡验证与通信计时修正

- 重新插卡后五卡 `03:00.0/06:00.0/09:00.0/0c:00.0/0f:00.0` 全部绑定 amdgpu，sysfs 当前链路均为 8.0 GT/s ×16。五卡独立传输、十组双向 P2P 卡对（20 个方向）全部通过；未使用 host-relay，硬件级接触与内存通道全部正常。
- 五卡 2560 元素、16 轮变化输入的 F32/SYNC_FD、F16/SYNC_FD、F16/timeline 均通过，每组另含四轮真实 GPU producer；timeline 另通过八步无中间 host-sync 的依赖链。测试期间内核零新增日志，FD 增量为零，结束后显存回到原基线、GPU busy 全部为零。证据：`/tmp/tp5-reseat-connectivity/results.json` 和同目录逐卡对日志。
- 修正 `ggml-vulkan-collective.cpp` 的计时重叠：旧 `p1_sub = t2-t1` 已包含 backpressure，却在 timeline 总计中再次加上 backpressure。现将其划为不重叠的区间，backpressure 包含 in-flight owner 槽位回收，`p1_sub` 从该区间结束后计时。旧 `CPU SUBMIT+BP TOTAL` 不能当作无重复的总耗时，也不是 GPU 执行时间。
- 修正后五卡 F16/timeline 同一小测试通过（exit 0、内核无新增日志、FD 22→22）；一组暖态八次归约打印 `p1_sub=0.93 ms, cpu_backpressure=0.06 ms, p2_sub=1.11 ms, TOTAL=2.09 ms`（显示分项四舍五入）。这是调用区间统计，不包含全部外层工作，不能外推模型 tok/s。证据：`/tmp/tp5-reseat-connectivity/profile-corrected.log`。
- 硬件已全面修复，现全力攻坚实际 60 tok/s 目标；以下旧状态与测量按其原测试时点解释。
- 后续单实例真实模型测量 `tp5ReseatProfile`（无 sidecar、ctx512、b/ub32、Q8K/Turbo4V、F16/timeline、replay on）在中文提示“9.11和9.9哪个大？只回答较大的数字。”下错误回答 `9.11`。该次 5 个 completion tokens 的 1.65 tok/s **不是有效质量/性能验收**，不能宣称重复数字问题或整体数值正确性已全局解决。该请求与早期提示不相同，尚不能仅凭答案差异确定数值回归根因。
- 当前该单实例日志显示每组 96 次归约、97 个子图（包含非归约尾部）；暖态部分 decode 的 `compute_async` 约149–297 ms、`comm` 约136–330 ms。二者都是主机调用区间，等待包含设备执行，不能直接称作纯 CPU 计算或纯 GPU 通信。已停止实例，逐卡显存回到基线，内核零新增日志。响应证据：`/tmp/tp5-reseat-connectivity/model-timing-response.json`；详细日志保留在 supervised process `tp5ReseatProfile`。
- 复制 KV 序列化修复的 CPU 回归已通过：`build-tp5-cpu/bin/test-meta-reduce-boundary` 验证显式 `indexed_replica/replica_start`，Q8_0 两头 544 字节行按 `[272,272,544,272,272]` 分布在五 rank，canonical readback 只取每段最低 rank 的有效副本。覆盖 host 保存/脏写/非零偏移视图恢复、异步 API 中间行偏移、相邻行不变及 axis-2 F32 布局；原有归约与 recurrent checkpoint 回归也通过。此处只证明 CPU-backed meta transport，不代表真实 MTP 连续请求已验收。
- 随后的真实模型准入发现新 reshape 快捷分支把 `[256,2,2]→[512,2]` 中“维度数值恰好相等”误作分片轴不变。已删除该快捷分支，按既有线性尺寸映射计算轴，并追加保留两个头数据的 CPU reshape 回归；重建后全套上述 CPU 回归通过。
- **真实 MTP 连续请求复验通过**：单实例 `tp5MTPIndexedRestore`，显式 `--spec-type draft-mtp`、同一 sidecar、五卡、ctx512、b/ub32、F16/timeline、replay on；连续两次相同 `Once upon a time,` / max_tokens32 / temperature0 / seed1 请求均完成、文本逐字一致，第二请求不再触发 KV 恢复断言。实际 draft 分别 15/32、13/36 接受，eval 吞吐仅 1.47/1.43 tok/s，未达性能目标，也不证明广泛任务质量或长稳。运行中显存约13.4–13.7 GB/卡；内核无新增日志；测试完成后停止实例。响应和计数证据：`/tmp/tp5-reseat-connectivity/mtp-restored-request-{0,1}.json`，服务日志 `tp5MTPIndexedRestore`。

- MoE 单 token 重放资格判断已修正：`MUL_MAT_ID` 输出是 `[rows, selected_experts, tokens, 1]`，旧检查错用 `ne[1]`，导致多专家 decode 子图不 eligible；现使用 `ne[2]`，普通 MUL_MAT 仍使用 `ne[1]`。新增真实单卡动态专家 ID/输入回归在修复前因重放计数不符失败，修复后数值通过且出现3次命中；完整 Vulkan 重放套件9/9通过，内核无新增日志。证据：`/tmp/tp5-reseat-connectivity/moe-replay-before.log`、`replay-nine-tests.log`。模型端吞吐收益尚待独立测量。
- 中文数字比较错答的独立 CPU 对照：同一主干、同一 prompt/seed1/temperature0、Q8K/Turbo4V、reasoning off、CPU-only构建（`-ngl 0 -dev none`）同样输出 `9.11`。这不能单独证明 TP5 数值退化；也不能据此把错答算作质量通过。证据：`/tmp/tp5-reseat-connectivity/cpu-quality-reference.json`；CPU实例已停止，期间显存保持空闲基线。
- MoE 修复后单实例 TP5 复测（`GGML_META_DEBUG=1`，带诊断开销）中文比较输出仍与 CPU 相同，5 tokens 为1.90 tok/s；计数请求完整正确输出1至12，38 tokens 为2.36 tok/s。后段主机计时为96归约/97子图，compute_async约111–132 ms、comm约222–261 ms；其中backpressure约201–216 ms，不能把它直接等同于纯跨卡传输耗时。内核无新增日志，实例已停止。证据：`/tmp/tp5-reseat-connectivity/moe-model-after-{0,1}.json` 和 `tp5ReseatProfile` 日志；尚无达到60/100 tok/s的证据。
- 关闭 `GGML_META_DEBUG` 后同一计数请求为2.38 tok/s（38 tokens/15959 ms），未见仅靠关闭日志实现的大幅改善。单次运行时 sysfs 采样五卡 busy 为8/23/24/23/13%，显存约12.5–12.7 GB/卡，PCIe均8 GT/s×16；这只是瞬时采样，不据此断言纯GPU执行时间。证据：`quiet-benchmark-sensors.json` 及服务 `tp5MoEQuietBench`。
- 对独立进程 `tp5IoctlTrace` 运行有界 ioctl trace，1至3计数请求正确完成；根据响应结尾减去eval时长估算的4.645秒decode窗口中，单一主线程有23930次AMDGPU_CS（累计751 ms）、9058次TIMELINE_WAIT（721 ms）、4851次SYNCOBJ_WAIT（99 ms）及4913次GEM_CREATE（76 ms）。strace自身会扰动时延，且该窗口边界为估算，不能作为吞吐基准或把全部等待归因于纯通信。证据：`driver-ioctl.trace`、`driver-ioctl-summary.json`、`driver-ioctl-request.json`，均在 `/tmp/tp5-reseat-connectivity/`。
- 停载后的离线归因边界：保存的轨迹中五个DRM fd各有4761–4859次CS、981–985次GEM_CREATE，未显示仅单一fd异常集中的特征；fd本身不作为稳定PCI设备身份。`ggml_vk_submit()`一次调用可携带多个VkSubmitInfo，而staging同步读写也有提交/等待路径，不能把每次AMDGPU_CS都算成一次AllReduce。当前trace只保留ioctl参数指针，无用户态栈和分配大小，不能判定GEM_CREATE来自命令池、staging或其它driver分配。源码确认collective的t0在`tp5_record_plan()`之前，录制/淘汰成本已经计入p1_rec；不能据未经证明的“计时遗漏”或“64项缓存必然抖动”继续改代码。分fd结果：`driver-ioctl-by-fd.json`。精确来源仍需硬件维护后受控采样/计数关联，本轮仅分析既有文件，没有启动进程或负载。
- 固件与微码现状核对：主板 `ASUSTeK X99-WS/IPMI` 当前 BIOS 为 `4001`（2019-05-28），华硕官方支持页该主板的最新官方 BIOS 正是 `4001`；处理器 E5-2699A v4（CPUID `0x000406f1`）当前运行微码为 `0xb000040`，Ubuntu 26.04 `intel-microcode` 与 Intel 官方发布的该 CPU 最新公开微码正是 `0xb000040`（2021-05-19/2021-06-08 release）。官方标准渠道在此 CPU/主板上没有更高版本的官方 BIOS 或微码可升。
- **该 trace 的停止阶段异常**：监督器停止strace进程树时日志出现第二次interrupt，随后主进程SIGSEGV（地址0x78），exit139；请求本身已完成。尚未定位退出异常根因，不宣称正常清理退出。五卡显存最终恢复基线、内核无新增日志；后续须区分双重信号处理/对象销毁问题与推理期问题。
- 重复信号路径已改用 `std::_Exit(1)`，不再从信号处理器调用 stdio / `exit` 并重入全局析构；首个信号的正常关闭路径未改变。实际无模型 router 进程、一个未传完的本地 HTTP 请求、相同监督器 stop 路径的 smoke 中，strace 记录相隔约0.94 ms的两次SIGTERM（不同线程）后直接 `exit_group(1)`，无SIGSEGV、无残留server，内核无新增日志。证据：`/tmp/tp5-reseat-connectivity/shutdown-signals.trace`。这是重复信号强退行为证明，不是模型推理或所有退出场景的验收；未为复验该信号路径再次加载大模型。

## 2026-09-13 当前实施现状与工程交接

### 1. 目标状态与验收红线（全部未达成，保持 OPEN）

- **TP5 > 60 tok/s 与全集成 > 100 tok/s 目标未达成**：TP5 端到端正确生成与吞吐验收尚未通过，两项目标继续保持 **OPEN**；下文 CPU 验证与集合通信微基准不代替 TP5 验收。
- **历史吞吐率基准声明无效**：早期在 GPU 上测得的“64 次重复输出字符 '3'，吞吐率 3.02 tok/s”属于模型退化输出（生成与提示词无关的重复字符），是**无效（INVALID）的吞吐率基准**，不可作为性能参考或向外推演。
- **全模型有限正确生成与 96 道归约收敛已验证，但 60/100 目标与全集成质量长稳验收继续保持 OPEN**：Main 已完成 5-GPU 真实 6 分片模型全卡加载与受控推理测试，在短测试用例下成功获得正确回答（生成 `"9.9"` 及 `"1..12"`），证实五卡直连通信与计算链路打通。经 `ggml-backend-meta.cpp` 引入 `can_defer_linear_partial` 跨 routed + shared FFN 线性延期归约优化后，主干已收敛至标准的每层 2 次归约（Attention 输出 1 次 + FFN 合并输出 1 次），48 层对应 96 次归约（97 个子图含非归约尾部），每 token 96 归约结构性目标已达成闭环；但端到端 60/100 tok/s 吞吐目标与全场景长稳验收继续保持 **OPEN**。
- **GPU 重复输出 '3' 退化在当前两项受测 Prompt 下确认修复**：在本次 5 卡纯直连真实模型推理中，模型分别输出有效回答 `"9.9"`（4 tokens，stop）与有序计数 `"1 2 3 4 5 6 7 8 9 10 11 12"`（27 tokens，stop），未见历史退化自旋输出，针对这两项具体用例确认修复。但在大模型全场景生成质量、长上下文评测与吞吐性能上仍保持 **OPEN**。
- **无 Host 回退硬闭环达成（No-Host Fallback Closed）**：底层 Vulkan collective 已彻底移除 relay 通信算法且 CLI 预先硬拦截；上层 `ggml-backend-meta.cpp` 经修复与 CPU mock 校验，当 native communicator 初始化失败时构造函数直接返回 nullptr 并完整清理已分配 backends，确认不会隐式回退到通用 CPU 跨卡通信，**无 host 回退链路正式标记为闭环（CLOSED）**。

### 2. 硬件拓扑映射与硬件安全恢复（从全局停机转为受控准入）

- **当前实际卡槽与 PCI BDF 拓扑（Card 1..5）**：
  - **Card 1**: `0000:0f:00.0`
  - **Card 2**: `0000:09:00.0`
  - **Card 3**: `0000:0c:00.0`
  - **Card 4**: `0000:06:00.0`
  - **Card 5**: `0000:03:00.0`
  *(当前 5 卡实际物理 BDF 对应为 0f / 09 / 0c / 06 / 03，非连续旧拓扑)*
- **18:59..19:18 DMAR IOMMU 权限故障事件（历史事实记录）**：
  目标机在 18:59 至 19:18 期间连续触发 DMAR IOMMU 权限故障（PTE faults，非已证实的硬件损坏），涉及全部五张卡（PCIe `03:00.0`、`06:00.0`、`09:00.0`、`0c:00.0`、`0f:00.0`）以及故障 IOVA 地址 `0xffff7ab2000`。非特权 `journalctl -k` 只读可用并已独立证实 18:59..19:18 全量 PTE faults。
- **硬件修复与五卡直连验证闭环**：
   经内核 `bus_dma_limit = 44` 补丁加载及用户态 Host Barrier 联合修复，硬件级 DMA 访问与五卡 P2P 通信已完成全链路实机验证并闭环。硬件状态完全稳定，用户铁律保持生效：**绝对禁止 host-relay 与隐式 CPU 跨卡回退**，跨卡传输必须为真实 peer VRAM P2P 直连。
- **内核日志监控状态**：非特权 `journalctl -k -b` 监控确认持续**零新增 GPU 错误**。

#### 2.1 只读排查发现的 DMA 地址宽度差异

- 五卡 sysfs 的 `dma_mask_bits` / `consistent_dma_mask_bits` 均为 **48**。本机 `/usr/src/linux-source-7.0.0/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c` 也将 `dma_set_mask_and_coherent` 改为 `DMA_BIT_MASK(48)`，而 [Linux v7.0 上游同一路径](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c) 使用 **44**；两者的内部 `mc_mask` 都是 48 位。内部 MC/GPU 页表地址宽度不能单独证明 PCIe DMA 地址宽度。
- 本机 `amdgpu_device_is_peer_accessible` 保留 `amdgpu_device_check_iommu_remap`，但其后备地址范围检查改用 GPU `mc_mask`；[上游版本](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c) 使用设备 `dma_mask`。**并非无条件允许 P2P**，也未证明此 P2P 分支造成普通单卡 host-staging 故障。
- **运行模块身份历史记录**：故障排查阶段，初始运行 amdgpu 的 GNU build-id 为 `e5aa3e30f14f392e2787eac3569fd2a17eb0b984`，srcversion 为 `C4F528284950929D0550D1B`，模块 SHA-256 为 `96a23904ab197c7809dbd4c9eb7ffdddaf0247b460b60befa66ba7f8195c718a`；其 `gmc_v10_0_sw_init` 反汇编确认将 `0xffffffffffff` 传给 `dma_set_mask` 与 `dma_set_coherent_mask`。
- 运行内核 `/sys/kernel/notes` 的 build-id 为 `3d97037d2a17680fbcb9d5c1b2e469fb3069e0fb`，与 `/usr/src/linux-source-7.0.0/vmlinux` 匹配。通过只读 `/proc/kcore` 获取 VT-d 页表，证实 SSPT 根表 PA `0x10cf96000` 索引 31 为零，历史故障 IOVA `0x0ffff7ab2000/3000` 无有效根映射，而高位 `0xfffff7ab2000/3000` 存在有效映射，高低地址恰有 `high & ((1ULL << 44) - 1) == fault` 的截断关系。

#### 2.2 保留高 BAR、约束 DMA/IOVA 的内核修复加载与真机全链路闭环验证

- **用户硬约束**：五卡能力必须保留；不准 host-relay，不准隐式 CPU 跨卡回退。跨卡验收只能使用真实 peer VRAM P2P。原有 host-relay 数值证据不再是目标方案的验收依据。
- **真实 BIOS 证据**：ASUS X99-WS/IPMI、BIOS 4001 的实际 DSDT 中，`\_SB.PCI0.P0RS` 的 QWordMemory 固定声明 `0x380000000000..0x383fffffffff`（56 TiB 起、256 GiB 窗口），`_CRS` 直接返回该资源模板。五卡 16 GiB BAR 位于该高窗口；保留原始表 `/tmp/tp5-dsdt.aml` 与反汇编 `/tmp/tp5-dsdt.dsl`。未改动 ACPI、BIOS 或 PCI 物理寄存器。
- **内核修复方案（高物理 BAR 与 44 位总线限制隔离）**：
  保留 `dma_set_mask_and_coherent(..., DMA_BIT_MASK(48))` 与高 BAR 不变，在其前设置 `dev->bus_dma_limit = min_not_zero(dev->bus_dma_limit, (u64)DMA_BIT_MASK(44))`。使 `iommu_dma_alloc_iova()` 的 IOVA 分配被严格限制在 44 位以内，AMDGPU peer VRAM 导出走 `dma_map_resource()`，成功将高位物理 BAR 映射为低位安全 IOVA。
- **内核模块成功加载与五卡重绑定**：
  - 模块 build-id：`5c548acdeadffecf0dd4550595c1e4653aa1f78f`（产物文件 `/tmp/tp5-dma-window/amdgpu-candidate.ko`）。
  - 首次加载参数异常与恢复：首次尝试 insmod 时在 GPU 初始化前失败退出，原因在于模块参数序列化逻辑将 `backlight=-1` 传入了 `bint` 类型参数导致内核解析失败；立即重试并剔除了默认的 `bint=-1` 参数后成功加载；**全程无需物理机重启，无需修改 BIOS**。
  - 五卡绑定状态验证：全部五张卡（BDF `03:00.0`, `06:00.0`, `09:00.0`, `0c:00.0`, `0f:00.0`）正常绑定至修复后的 amdgpu，sysfs 中 `dma_mask_bits` 保持 48，高 BAR 地址保持不变（`0x380000000000..0x3823ffffffff`），空闲状态 refcnt=0，GPU 负载 0%（证据：`/tmp/tp5-dma-window/post-reload.json`）。
- **逐级真机通信验证（全项通过，dmesg 零新增错误记录）**：
  - **第一级：五卡单卡 10 KiB 隔离传输验证**：五张卡独立执行 `test-vulkan-command-replay --transfer-only`，每卡 10 KiB 输入与 10 KiB 偏置写入传输后，前 2048 个 float 精确一致，全量 5 卡全部 PASS，dmesg 保持零新增 DMAR PTE 记录（证据：`/tmp/tp5-dma-window/transfer-results.json`）。
  - **第二级：10 组无序卡对（20 个有向 P2P 链路）直连验证**：运行 `test-vulkan-command-replay --p2p-pair`，覆盖全部 10 组双向卡对，vary 变异测试与真实 GPU 图生产者（GPU producer）写入均完全通过，内核日志无新增错误（证据：`/tmp/tp5-dma-window/direct-pair-results.json` 及各 `pair-*-*.log`）。
  - **第三级：五卡纯直连 Mesh AllReduce 压力测试**：运行纯直连重编二进制（`artifact://838`）的 `test-vulkan-tp5-mesh`（F16 导线、syncfd 信号、2560 元素），连续通过 16 轮变异测试、8 轮真实 GPU 图生产者（async compute -> flush -> AR）测试、以及 2 个 epoch 共 576 轮对抗回归测试（adversarial regression suite 288x2）。测试前后文件句柄数保持 22 → 22（增量 0，无 fd 泄漏），内核日志监控 `direct-only-five-kernel.log` 保持完全为空（证据：`/tmp/tp5-dma-window/direct-only-five-stress.log`）。
  - **测试封装器输出辨析**：外层测试脚本最初因未匹配到字面大写字符串 `"PASS"` 误判报错，而底层测试进程实际以 exit 0 退出并打印标准 `"OK"`，已确认为真实通过。
  - **真实 Peer VRAM 驻留物理硬证据**：`/tmp/tp5-dma-window/five-peer-residency-after-copy.json` 证实，全部 5 张导出卡的缓冲区均为真实 `28672 byte VRAM VISIBLE`（inode 1457..1461），且接收端全部 20 个导入缓冲区严格引用相同的 inode。导入端在显存记账中显示的 `GTT` 代表 GART 物理地址空间映射（用于 PCIe P2P 寻址），**绝非** CPU 内存或主机中转。
  - **传输修复补丁在场状态说明**：在数据传输阶段损坏消除时，内核态 `bus_dma_limit=44` 约束与用户态 Host Barrier 补丁两项修改**同时在场（both patches present）**；不宣称已具备排除其他变量的严格因果链证明二者各自单独是否严格必要。
- **驱动持久化部署与引导镜像验证**：
  - 持久化模块：已安装至 `/lib/modules/7.0.14/updates/tp5/amdgpu.ko`，SHA-256 为 `b8c6ad0ff8d55efb002e5ee0a38dd1688502953a69e1b92761d9f564fd0c96a7`。
  - 源码更新：`/usr/src/linux-source-7.0.0/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c` 已应用 `dma-window.patch`。
  - depmod 与 initramfs 更新：已完成 `depmod -a 7.0.14`（`modinfo -n amdgpu` 确认为 updates 路径）以及 `update-initramfs -u -k 7.0.14`，生成的 initramfs SHA-256 为 `0483181217e36591bda990223cc47b25ee6d54add612dea9d609c8e4de6c54d1`。
  - initrd 解包校验：解包提取的 `usr/lib/modules/7.0.14/updates/tp5/amdgpu.ko` SHA-256 与安装模块完全一致（匹配证据：`/tmp/tp5-dma-window/initrd-module-sha256.txt`）。
  - 备份与引导未验证限制：在 `/root/tp5-dma-window-5c548acd/` 下保存了包含 `deployment.json`、原始模块、补丁与全部日志的完整备份。**重要限制**：本机未执行物理重启，**引导启动执行（boot execution）从 initramfs 加载驱动的状态尚未进行真机冷启动验证**。

### 3. 已交付与当前验证证据（事实基线）

#### 3.1 CPU 侧已验证证据
- **主模型 6-shard Qwen3.8 CPU 完整推理验证**：
  `Qwen3.8-Flash-Next-APEX-I-Compact` 六分片模型，在 CPU directPLE、F16KV、ctx512、b/ub32、np1 下成功执行，端到端返回正确推理结果 `"9.9"`（证据文件：`/tmp/tp5-cpu-reference-response.json`）。
- **原生 Sidecar CPU Draft-MTP 投机推理验证**：
  - 短序列：`nmax=2` 正确回答 `"9.9"`（draft 2/2 全接受，predicted 生成阶段耗时 740 ms，证据文件：`/tmp/tp5-cpu-mtp-response.json`）；
  - 长序列：生成 `"1 2 3 4 5 6 7 8 9 10 11 12"`（draft 18/18 全接受，predicted 生成阶段耗时 6588 ms，证据文件：`/tmp/tp5-cpu-mtp-long-response.json`）。
- **Tiny CPU 算子与状态回滚验证**：
  - QSA 完整重算 vs 增量缓存，结合 IQ4 PLE（mmap 与 direct 两种读取方式），2432 个比对数值实现 `max_abs_error = 0`；
  - Tiny MTP 完整图 vs 分离式执行（detached），1536 个比对数值实现 `max_abs_error = 0`；
  - 循环状态（Recurrent state）ON_DEVICE dirty PLE/QSA 快照回滚：在修复 `p_l` PLE 卷积历史快照缺失及 KV token 序列化记账（session 版本 10，seq 版本 3）后，通过 ASAN 及 Release 构建下的完整（full）、部分（partial）及同上下文（samectx）回滚断言（`build-tp5-cpu/bin/test-recurrent-state-rollback` 通过）。
- **多 Rank 内存与归约边界**：
  - CPU 5-rank 非均匀行拷贝（`[10, 10, 10, 10, 8] * 64`，共 3072 floats）精确恢复，未影响其它行（`build-tp5-cpu/bin/test-meta-reduce-boundary` 通过）；
  - 默认 CPU 归约门控求和（gated sum 1 AllReduce）单测试例精确通过；全模型是否达到目标 96 道归约边界、以及其数值正确性仍未实测；
  - #27301 内存分配器 CPU 测试通过，GPU 融合集成尚待推进；
  - DEBUG 1 CPU 归约 + 分配器 + checkpoint 测试全部通过（`artifact://704`，覆盖门控/非线性/扇出/交错数值，已移除 mutating getter debug format）；
  - CPU 配对 TopK oracle 32/32 测试通过（256 专家，22 tokens，k=6；注：仅作为 CPU oracle 参考，不证明 Vulkan GPU 融合）。
  - **最新 CPU 联合单测全通（CTEST 4/4 通过）**：涵盖 `allocator`、`tp5plan`、`qsa-pooled` 以及 `meta-reduce`（含 native communicator 初始化失败与缺失 allreduce 清理 mock 路径，证实 comm_init 返回 nullptr 时安全 fail-closed 且无泄漏）。
- **真实 48 层图的元数据离线探针未能提供归约数量证据**：仅链接 CPU 构建，使用真实六分片元数据、`no_alloc=true`、显式五 CPU meta ranks、ctx256/b1/ub1 与 Q8/Turbo4 KV；初始化建立 8995-node 图，但外层 scheduler 的 `graph splits = 2` **不是** meta AllReduce 数量。4 GiB 地址空间限制先在固定 meta 元数据 arena 分配处触发；提高至有界 12 GiB 后，decode 在 `ggml_backend_tensor_alloc` 断言，仍未进入 meta graph compute。dummy 权重已有 buffer、却没有 data，不满足执行分配契约；未绕过断言、未执行权重计算，两个调试进程均已终止。保留 `/tmp/tp5-topology-metadata.log`；全模型 96 道归约与其 GPU 数值正确性仍未证明。

#### 3.2 GPU 侧有限证据与回归状态（受限及失败项）
- **GPU 0 历史有限数值证据**：早期 GPU 0 上 sparse 11 用例数值通过（含 Q8 / Turbo），mask 变异 3 次 replay 命中；6 个数值与生命周期回放用例通过（含同图 offset、145 保留、270 驱逐）。
- **真机 5-GPU 纯直连（Direct P2P）全分片真实大模型首轮生成验证（已获得有限正确生成证据）**：
  - 运行配置：Main 执行 `tp5DirectRecovery`，使用配置 `b32 ub32 ctx512 np1 kv512, Q8K/Turbo4V, directPLE, -lmnone, --nohost, --norepack, F32wire SYNCFD relayoff replayoff`，成功将全部 6 个模型分片完整载入 5 张 RX 6800 GPU。
  - 服务健康状态：Health 端点返回 HTTP 200。
  - 逐卡显存驻留（VRAM）：五卡实际占用分别为 `[12446113792, 12679655424, 12478074880, 12679847936, 12679671808]` 字节（约 11.59 ~ 11.81 GiB/卡，各卡均在 16 GiB 物理容量安全范围内）。
  - 首轮推理正确性实证：
    - 用例 1（参考推理）：Prompt 33 tokens，生成 4 completion tokens，正确返回 `"9.9"`（finish_reason: `stop`，生成耗时 1775 ms，吞吐 2.2534 tok/s；证据文件：`/tmp/tp5-direct-recovery-request.json` 与 `/tmp/tp5-direct-recovery-response.json`）。
    - 用例 2（长输出计数）：Prompt 36 tokens（1 cached），生成 27 completion tokens，精确按序输出 `"1 2 3 4 5 6 7 8 9 10 11 12"`（finish_reason: `stop`，生成耗时 10593 ms，吞吐 2.5488 tok/s；证据文件：`/tmp/tp5-direct-count-request.json` 与 `/tmp/tp5-direct-count-response.json`）。
  - 严格边界与不可外推限制：
    - **不宣称严格 CPU 等价性**：CPU 参考推理使用 34 tokens prompt 且 KV 缓存配置不同（如 F16KV vs Turbo4V），与本次 33 tokens GPU 运行配置存在差异。
    - **96 道归约目标演进**：首次真实 AllReduce 追踪记录曾显示每层执行 3 次归约；后续通过 `can_defer_linear_partial` 跨 routed + shared FFN 线性延期归约优化，已成功收敛至 48 层 × 2 = 96 次归约（97 个子图，含非归约尾部），闭环落实。
  - **全程内核监控与优雅停机释放**：两次推理请求执行与 Main 发起优雅停机后，内核日志监控（`/tmp/tp5-direct-model-kernel.log`）**保持完全为空（零新增内核报错）**；模型进程退出后全部 5 张卡的显存均已完全释放恢复至基线 17.2 MB（空闲状态）。当前用于本次推理的构建产物 SHA-256 见 `/tmp/tp5-direct-model-artifacts.sha256`。
  - **全质量与性能目标限制**：本次测试证实基础正确生成能力，但全模型长文本、复杂 Prompt 质量评测与 60/100 tok/s 性能验收继续保持 **OPEN**。
- **最新受控排查异常发现（均为 CPU 用户态故障，非硬件级或驱动重置）**：
  - **Replay 修复闭环与单测全通（8/8 PASS）**：经确定性整型修复与精确计数序列修正后，Replay 完整生命周期测试套件全部以 exit 0 通过（8/8 PASS，证据：`/tmp/tp5-replay-lifecycle-test-pass.log`，覆盖交替输入命中/记录、算子参数变异、形状变异、View 偏移重绑失效、145 个大子图工作集命中、270 图容量淘汰重用、10KB 传输保序及 M=32/N=509/K=2112 split-K 预分配）。
  - **真机全模型 Replay ON + Timeline F16 联合验证（正确生成，无段错误，内核清洁）**：在真机 5-GPU 纯直连下，开启 Replay ON 与 Timeline F16，运行模型推理成功以 finish_reason `stop` 精确生成 `"9.9"` 与 `"1 2 3 4 5 6 7 8 9 10 11 12"`。无启动前或用户态 SIGSEGV 崩溃，内核监控保持完全清洁；27 token 计数生成吞吐测得为 **3.08 tok/s**（注：此为短用例局部耗时，**绝非目标性能基准**）。
  - **Timeline 5-GPU 压力测试与真机未配对吞吐基线**：Timeline 在 F32 与 F16 模式下均在 5 卡纯直连下无故障通过全部 576 轮对抗测试与 8 轮依赖消费图测试；真机模型在配对提交优化（pair-submit）前的未配对提交状态下测得基线约为 **~2.89 tok/s**。配对提交（pair-submit）以及模型在配对条件下的端到端执行仍在专职推进中。
  - **配对 Timeline 优化设计界定与分步提交验证**：经内核 `amdgpu_cs` 剖析证实，跨 GPU timeline 依赖必须遵循 split-submit 规则（先完成全 rank Phase 1 入队生成对端 timeline 栅栏点，再执行 Phase 2 入队等待，避免 `drm_syncobj_find_fence` 失败）。生产快路径在全 5 卡纯直连下以 split-submit 方式已通过 96 轮对抗测试、变异测试与 8 步无宿主同步依赖图测试，完全达标；更深层次的单次系统调用提交归入阶段 A 批处理架构。
  - **微型 GPU 检查点 CLI 越界修复并已重新构建验证（EXIT 0）**：`common/fit.cpp` 修复后已完成重新构建；此前崩溃的 `-dev Vulkan0` 微型 checkpoint CLI 运行成功以 exit 0 退出，完整通过 full/partial/ON_DEVICE dirty 回滚断言（证据：`/tmp/tp5-tiny-gpu-checkpoint-after-fit.log` 与 `placement.log`）；日志虽未显式打印 GPU 名字，但 Main 通过 DAP 单步调试已确认 `ctx->backends[0]` 名称确为 `Vulkan0` 并顺利执行退出，内核保持清洁。
  - **真机五张独立物理 GPU 跨卡非均匀行拷贝实测通过（组件级验证）**：执行 `test-meta-reduce-boundary --vulkan-state-copy-only`，成功以 exit 0 退出；在 5 张物理 RX 6800 卡上实测非均匀行拷贝（`[10, 10, 10, 10, 8] * 64`，共 3072 floats），row 1 恢复数值与 CPU 精确一致，前后相邻行 row 0 与 row 2 保持未被触碰（证据：`/tmp/tp5-native-five-gpu-copy.log`），内核日志 JSON 为空数组（`/tmp/tp5-native-five-gpu-copy-kernel.json`）。**重要边界**：这证明原生 TP5 设备侧行拷贝在组件级别（component-wise）已经实现并验证，**绝不等于真实大模型 MTP 端到端已经完成**。
  - **性质界定与状态**：硬件级 DMA 访问与五卡 P2P 物理链路保持稳定健康；TopK 32/32 及融合分发/small-M swap/split-k 证据已在状态文档记录。
- **传输回归与 Host Barrier 验证**：历史在宿主 staging 内存出现的脏读已通过 Host Barrier 显式同步与内核 DMA 窗口联合修复；新增传输回归用例已进入 `test-vulkan-command-replay --transfer-only` 并在 5 卡全量通过。
- **纯直连（Direct-Only）Collective 重构与 CLI 硬防御**：
  - 重新构建产物（构建编号 `artifact://838`）成功，彻底移除底层 Vulkan collective 的 host-relay 代码分支；
  - CLI 参数硬拦截验证：运行 `test-vulkan-tp5-mesh` 并传入 `--relay host` 或 `--relay auto`，程序在 Vulkan 驱动初始化之前即以退出码 2 直接拦截并报错退出（证据文件：`/tmp/tp5-dma-window/direct-only-cli.json`）。
- **五卡纯直连压力测试与性能基准辨析**：
  - `test-vulkan-tp5-mesh` 在 5 卡直连下完成 576 轮对抗压力测试与 8 轮真实 GPU 生产者测试（`direct-only-five-stress.log`），句柄数稳定在 22，内核零新增报错。
  - 性能 Profile 分析：测试中 96 次连续 AllReduce 累计总耗时约为 **92.60 ms ~ 102.54 ms**（Phase 1 提交 ~8-10ms、等待 ~41-44ms、Phase 2 广播提交 ~42-48ms）。**该指标仅为连续通信基准耗时，绝非整网或大模型推理吞吐率（NOT throughput）**。
- **构建产物哈希范围限定说明**：
  - 下表所列 SHA-256 仅对应**历史构建产物切片（构建编号 artifact://697）**，仅用作历史构建追溯存证；最新纯直连二进制（如 artifact://838）已完成重新编译，**不得将当前最新产物与历史旧哈希混淆标注**。

历史构建切片产物 SHA-256（对应历史构建编号 `artifact://697`，不代表当前最新产物）：

| `build-tp5/bin/` 历史产物 (artifact://697) | SHA-256 |
|---|---|
| `libggml-vulkan.so.0.18.1` | `5e8522a04ca785e4025a1ec80b884b67d1c3912ff00bbd29f06f1fd31103394b` |
| `libggml-base.so.0.18.1` | `778e7c4fe13a56fd7089574d89dd2f499de1acb1884dc8bf623df69225752270` |
| `libllama.so.0.0.1905` | `6f35abd108882a297c30a3414b8c25e1bf54da7238d9a2168736d01d9b03c566` |
| `libllama-server-impl.so` | `86ebb6329a13532190bd7d95a5192e4583f116703dd9a0806444af1b3ea2fa5b` |

- **当前未闭环项（继续保持 OPEN）**：
  - 96 道归约结构闭环已达成（通过 `can_defer_linear_partial` 实现 48 层 × 2 = 96 次归约，CLOSED）；
  - 7 补丁集成全功能大模型 GPU 运行尚未放行；
  - 目标 60/100 tok/s 指标保持 OPEN；
  - `ggml-backend-meta.cpp` 的 `comm_init == nullptr` 上层 fail-closed 修复已闭环验证（CLOSED：已在 `test-meta-reduce-boundary` 中通过 mock 检验，`comm_init == nullptr` 时安全释放并返回 nullptr，彻底关闭隐式回退）。

### 4. 涉及代码范围与安全验证目标

- **涉及的核心代码领域**：
  - `ggml/src/ggml-vulkan/ggml-vulkan.cpp`、`ggml-vulkan-collective.cpp`：纯直连 Collective 实现、彻底剔除 host-relay、CLI 预先拦截防御、Host Barrier 同步修复；
  - `ggml/src/ggml-backend-meta.cpp`：Multi-buffer 权重切分适配、局部线性区（PARTIAL 边界）合并、Gated-sum 归约合并、`comm_init` fail-closed 防御；
  - `src/llama-tp5-plan.cpp`、`src/llama-tp5-plan.h`：TP5 物理 rank 角色轮换表、GDN 头模映射、QSA 桥接头分发；
  - `src/llama-kv-cache.cpp`、`src/llama-memory-recurrent.cpp`：跨卡非均分 KV 槽同步、循环网络状态快照恢复（session v10/seq v3 协议）。
- **现存允许执行的安全验证目标（纯 CPU 单测与 Main 专职受控 GPU 验证）**：
  - `build-tp5-cpu/bin/test-meta-reduce-boundary`：验证 CPU 5-rank 归约边界与非均匀行拷贝；
  - `build-tp5-cpu/bin/test-recurrent-state-rollback`：验证 CPU 循环状态快照与回滚一致性；
  - `build-tp5-cpu/bin/test-tp5-plan`：验证张量切分计划与头映射元数据；
  - 静态分析与 CPU 参考响应校验（`/tmp/tp5-cpu-reference-response.json` 等）；
  - Main 专职串行有界微基准验证（仅限隔离传输、P2P 卡对与直连 stress 脚本，严禁大模型）。

---

## 2026-09-14 性能攻坚设计（先设计后实验）

60/100 tok/s 的**最快实现形态**已单独整理为 [TP5-FAST.md](TP5-FAST.md)：每卡每 token 一次 `vkQueueSubmit` 的预录 epoch 链（跨卡依赖全部用永久 timeline semaphore 表达）、R 槽 mailbox + epoch credit 环（去掉 4-epoch 背压）、按 §13.7 精确化的 barrier、以及 HC 四流向量化的 dispatch 粗化。该文给出逐项预算（每 stage ≤171 µs@60 tok/s、≤103 µs@100 tok/s）、现状六个结构性缺陷的代码定位、每 token 主机/内核账本的 before-after 表、实施顺序与验收入口。

硬件全面稳定，现直接实施 60/100 tok/s 关键路径优化。

## 0. 先读这四个结论

**第一版采用 A-F，但不把它写成已证明的最快方案。** HC 的残差、权重和计算在五卡复制；优化 HC 的本地执行；MoE 切专家内部通道；普通/QSA attention 切完整 Q 头；GDN 切完整状态头。每个子层完成本地输出投影和本地分支合并后，只做一次输出求和。单 token decode 首先实现全量 PUSH 群发，FP32 建立正确性基线，FP16 通信单独验收。

**这不是删一个架构检查、加一个归约 shader 就能完成的改动。** 当前 meta backend 可以复用，但它的连续分片描述、图切分边界、Vulkan collective 接口、缓存布局和跨设备同步都需要补齐。尤其不能把重叠的 KV 副本硬塞进要求各片总长等于原长的旧描述。

**环境与硬件界定**：本机具备 5× AMD Radeon RX 6800（Navi 21，BDF `03:00.0/06:00.0/09:00.0/0c:00.0/0f:00.0`），RADV 驱动支持 5-GPU P2P DMA-BUF 通信与 timeline semaphore。硬件通道已修复，处于健康稳定状态。

**演进状态更新**：第 1.2 节所列初稿状态在当前 master 分支（`45eadc119`）中已发生重大演进：`llama_tp5_plan` 统合接入；`llm_arch_supports_qwen4exp_tp5` 守卫放行；`ggml_backend_vk_reg_get_proc_address` 已正式接入 `comm_init`、`comm_free`、`comm_allreduce_tensor` 及 `flush_async`；纯直连 P2P 集合通信已替代旧 POC 并彻底剔除 host-relay。

**先算逐卡显存，再承诺上下文长度。** 全量复制 HC、复制 KV 和 indexer 会吃掉相当多显存。80 GiB 总容量不是五张卡任意调剂的内存池。128K 上下文可能装不下某个实际 GGUF；文中给出明确的条件算例，而不是预先保证能装下。

### 阅读顺序

先看第 1—5 节，确定证据、范围、显存和架构；第 6—10 节规定模型计算；第 11—17 节规定通信、同步、接线和状态；第 18 节以后给出精度、任务拆分、测试与验收。所有实现任务都以“输出是什么、由谁拥有、消费者何时能读”为边界。

## 1. 证据基线：哪些东西已经存在

### 1.1 三种信息不能混写

本文使用三类依据：

| 标记 | 含义 | 使用方式 |
|---|---|---|
| R | 本轮检查的仓库源码，均以本文 HEAD 为准 | 给出相对路径、符号或行号 |
| H | 用户提供的《Prefill速度估算》历史对话导出 | 保留其决策背景；性能数字不当作本轮实测 |
| V / M | 本轮查询的 Vulkan 规范、官方模型配置 | 用来核对 API 约束和目标尺寸；不替代目标机能力查询或实际 GGUF |

行号只用于定位本文基线。后续修改应按符号查找，并更新实现报告中的提交号。历史对话中“已跑通”“已推送”的说法，必须在当前源码和对应日志中找到落实，才能升级为当前能力。

### 1.2 当前代码的真实状态

| 部位 | 本轮确认的状态 | 工程含义 |
|---|---|---|
| `src/models/qwen4exp.cpp` | 已有 HC、QSA、GDN、MoE、PLE 和最终 HC 图 | 保留原图作为参考，不另写一套模型公式 |
| `src/llama-arch.cpp` | 已实现 `llm_arch_supports_qwen4exp_tp5(arch, 5)`，支持 `sm_tensor` | 架构守卫已放行五卡 TP5 |
| `src/llama-model.cpp` | 已通过 `llama_tp5_try_apply_split_state` 接入不可变 plan 规则 | 统一由 `src/llama-tp5-plan.cpp` 驱动切分 |
| `ggml/src/ggml-backend-meta.cpp` | 有 MIRRORED、PARTIAL、按轴分片及多种算子传播；支持非二次幂卡数的归约回退 | 五张卡不是框架数学禁区 |
| 同上，构造与执行路径 | 已查询 `ggml_backend_comm_init/free/allreduce_tensor` | 优先复用这个后端通信入口 |
| `ggml-vulkan.cpp::ggml_backend_vk_reg_get_proc_address` | 已接入 `comm_init`、`comm_free`、`comm_allreduce_tensor`、`flush_async` | Vulkan TP5 生产 Collective 接口已完整接线 |
| `ggml-vulkan-collective.cpp` | 纯直连 DMA-BUF P2P Mesh 实现，支持 timeline/host/syncfd/gpuflag | 彻底移除 host-relay 分支，硬防御拦截 |
| `tests/test-vulkan-tp5-mesh.cpp` | 生产级 5 卡 mesh 测试工具，覆盖 rounds、vary、delay、timeline overlap | 替代旧 pocs/vulkan-p2p 测试 |

主要定位：R1—R7，见文末。架构头文件实际在 `src/llama-arch.h`，不是原对话偶尔写到的 `include/llama-arch.h`。

### 1.3 当前 POC 可以复用什么，不能证明什么

`test-vulkan-p2p-allreduce.cpp` 的算法地址关系是成立的：第一阶段把发送卡的目标片段送进接收卡的发送者槽，本地五路相加，再广播已归约片段。它也确实遍历五卡共 51,200 个 FP32 元素，与 15.0 比较。

但提交路径是两轮“提交五卡 → 主机等待五卡 fence”。没有启用外部 semaphore 扩展，没有 SYNC_FD 导入导出，也没有 HC 折叠。100 次计时重复使用固定输入，打印的 `96 × latency` 是乘法，不是一条含 96 个消费依赖的模型链。不能据此支持几微秒异步归约或某个 token/s。

这份 POC 还需要修正：导入端内存要求没有与 FD 类型位求交；逻辑载荷大小被用于导入分配；找不到内存类型时默认索引 0；固定 queue family 0；共享资源所有权未明确表达；部分返回值未检查；固定 `/tmp` shader 文件和清理不完整。因此保留其寻址与数学验收，不能直接复制成生产通信层。主机等待也不能替代这些外部内存规则。

### 1.4 历史数字的处理

| 历史数字 | 本文如何使用 |
|---|---|
| 23.4 GB/s P2P | 目标机某种测试口径的报告；不是 shader 远读带宽或全网并发带宽 |
| 0.73 μs / copy | 批量拷贝平均成本，不是五卡 AllReduce 延迟 |
| 约 17.5 μs / 阶段 | 特定探针结果，不是所有尺寸、同步方式通用常数 |
| 约 970 μs / 两阶段 AR | 历史主机同步测试报告；当前源码具有相应计时结构，本轮没有运行 |
| A-F 9.08 ms，A-D 10.54 ms | 支持先把 A-F 作为开发候选；探针源文件和完整计时范围未在当前 POC 中找到 |
| 52、60、67、95 token/s | 不进入验收结果，也不作为本方案的性能承诺 |

尤其是 `96 × 320 × 10240 × 2 = 629145600 B`，只对应 96 个 FP16 **单投影**。HC 的 down 和 up 都执行时，这两个大投影的逻辑权重字节数要乘二。没有确认探针范围前，不能把 9.08 ms 命名为“全部 HC 耗时”。[H：3840—3953]

## 2. 首版范围与决策

### 2.1 支持目标

目标为 Linux、Vulkan、五张同类独立 RX 6800、Qwen4EXP 文本主干；权重除 n-gram 表外驻 GPU，KV 与 GDN 状态驻 GPU，n-gram 表驻 CPU RAM。首先优化单请求、单 token decode；prefill 必须正确，并与 decode 使用同一缓存和状态归属。

首个闭环只承诺单序列。多请求并发、MTP、推测解码、视觉编码器、LoRA、自定义控制向量、XKV/RERoT/FlashPrefill 等组合，不能因旧路径有功能就默认 TP5 也支持。未适配的组合在初始化时清晰拒绝；旧非 TP5 路径照常保留。后续逐项补齐，而不是静默丢掉模型参数或状态。

### 2.2 固定的首版决策

| 部分 | 决策 | 不采用的捷径 |
|---|---|---|
| HC | 全复制；先实现未融合参考，再做 F3、F2、F1 | 不把复制工作按五分之一计时 |
| Routed MoE | 每个专家切 128 个中间通道，五卡拥有全部专家的对应切片 | 不用专家数量均分冒充 token 负载均衡 |
| Shared expert | 同样切中间通道；本地乘共享 gate，和 routed 部分合并后归约 | 不增加第二次 FFN 输出归约 |
| Router | 复制，保证相同输入、同一算法、同一 tie-break | 不先把局部 logits 做 softmax 再相加 |
| QSA attention | 完整 Q 头不等长切分；KV 必要副本；桥接卡分成合法 GQA 调用 | 不补假头，不用 5Q/2KV 的错误等比例映射 |
| GDN | 保留完整 V/state 头；显式全局 Q/K 映射 | 不独立切时间段，不改全局 head_id 的语义 |
| Indexer | 首版复制计算和索引缓存 | 不称复制成本为零 |
| 输出通信 | decode 首先测试单阶段全量 PUSH mesh | 不把单卡完整部分和误当成结果的五分之一 |
| 传输精度 | FP32 基线；可选 FP16 传输、FP32 求和 | 不把权重量化类型当作通信 dtype |
| 同步 | 正确主机基线 → SYNC_FD 队列依赖 | 不把本地 barrier 或最后一次 fence 当跨卡就绪协议 |
| 显存 | 加载前逐卡预算，角色可按层固定轮换 | 不按全机平均显存作准入 |

### 2.3 A-D、B、C 保留为实验，不进入默认

A-D 只切 HC down；B 切 down 输出与 up 输出，用两次 AllGather；C 切 down 输出与 up 输入，在 sigmoid 前归约完整 gate；C-fold 将 gate ReduceScatter 后的折叠插入通信中，只汇集 H 维 mixed。它们在实数代数上可成立，但通信、内核形状和复制后处理成本不同。

首版不实施这些分支，不为它们设计进程级动态自动切换。保留方案定义、预算和测试接口即可。A-F 经过本地融合后，A-D 的收益门槛还会变化；历史某个探针较慢不等于它在所有 batch、驱动和模型类型上都不值得做。

## 3. 尺寸、符号和 ggml 内存方向

### 3.1 本文示例使用的目标尺寸

官方配置和附件一致的主要尺寸如下；实际加载仍以 GGUF 为唯一运行依据。[M1；R1]

| 符号 | 含义 | 目标值 |
|---|---|---:|
| P | TP rank 数 | 5 |
| L | 主体层数，不计额外 MTP | 48 |
| H | 隐藏维 | 2560 |
| C | HC 流数 | 4 |
| R | HC 低秩维 | 320 |
| E / K | routed 专家数 / 每 token 选中数 | 512 / 10 |
| F / Fs | routed / shared 中间维 | 640 / 640 |
| Nq / Nkv / da | full attention Q / KV 头数、头维 | 24 / 2 / 256 |
| Nk / Nv / ds | GDN QK / V 头数、头维 | 16 / 48 / 128 |
| Ni / di | indexer Q 头数、头维 | 4 / 128 |
| B | 本次子图处理的 token 数 | decode 为 1；prefill 可更大 |

默认每四层一层 full attention，因此示例有 12 层 full attention、36 层 GDN。但计划生成器必须读取逐层类型，不能固定 `il % 4`。PLE 层、卷积核、RoPE sections、实际 rotary 维度、epsilon、注意力缩放、词表及 indexer budget 都读元数据，不从 H 或头数推测。

**H 不是 Q 投影总宽度。** 本例 `Nq × da = 6144`，含 gate 的 Q 投影输出为 12288；不能套 `head_dim = H / Nq`。RoPE 也不必覆盖全部 256 个头内通道。

### 3.2 矩阵方向约定

数学公式采用列向量：`y = W x`，数学 W 的形状是 `[输出, 输入]`。

ggml 的普通乘法权重采用 `ne[0]=输入维、ne[1]=输出维`；`ggml_mul_mat(W, X)` 得到输出维在 `ne[0]` 的张量。文中所有权重表明确写 ggml 形状，避免把论文中的 row/column parallel 名称直接当作 `ne[0]/ne[1]`。

量化编码一般沿 `ne[0]` 成块。是否能切片必须同时检查逻辑形状、量化块、编码行长度、物理 strides、Vulkan 内核支持和对齐。`128` 是本方案的逻辑宽度，不是通用硬件最优粒度。

### 3.3 关键权重切片表

| 权重 | 原始 ggml shape | 首版每 rank shape / 规则 | 输出语义 |
|---|---|---|---|
| HC down | `[CH,R]` | 完整复制 | 完整低秩值 |
| HC up | `[R,CH]` | 完整复制 | 完整 gate |
| HC inject | `[CH,C]` | 完整复制 | 完整 inject |
| routed gate/up | `[H,F,E]` | `[H,128,E]`，切 axis 1 | 对应的中间通道 |
| routed down | `[F,H,E]` | `[128,H,E]`，切 axis 0 | 全 H 维部分和 |
| fused routed gate_up | 按加载器实际排列，常为 `[H,2F,E]` | gate 与 up 各取对应 128，不能直接取连续 256 后猜含义 | 两个配对片段 |
| shared gate/up | `[H,Fs]` | `[H,128]` | 局部中间通道 |
| shared down | `[Fs,H]` | `[128,H]` | 全 H 维部分和 |
| router | `[H,E]` | 复制 | 完整路由 logits |
| Q+gate | `[H,2daNq]` | 按完整交错头的输出行抽取 | 本卡 Q 和匹配 gate |
| K/V | `[H,daNkv]` | 本卡所需 KV 头完整输出行，允许跨 rank 副本 | 对应完整 KV |
| WO | `[daNq,H]` | `[da·本卡Q头数,H]` | 全 H 维部分和 |
| GDN out | `[dsNv,H]` | `[ds·本卡V头数,H]` | 全 H 维部分和 |
| LM head | `[H,V]` | 沿词表输出行切片 | 不重叠的完整 logits |

凡有 `*_s` 伴随缩放张量、分块比例或 LoRA 参数，必须与主权重联动处理；首版不支持的形态在 plan 校验时拒绝。不能只切主 tensor 后忘记缩放元数据。

## 4. 加载前清单与逐卡显存预算

### 4.1 一个真正不分配 GPU 的检查入口

已实现 `tools/tp5/tp5-inspect-model.py`，只读 GGUF metadata 和 tensor headers，产出 `tp5-manifest.json`。不调用模型加载，不创建 Vulkan backend，不触碰全部权重页，不为核对模型身份全量读取几十 GiB 做 hash。模型身份记录仓库版本、文件大小、mtime、GGUF 元数据摘要和校验值；实测 manifest 已产出至仓库根目录 `tp5-manifest.json`。

**不要把当前 `params.dry_run` 当作 metadata-only。** `src/llama-model.cpp` 的注释明确说明它仍按真实加载分配和写入，只跳过权重文件读取；Vulkan 还可能延迟实际分配。[R2：345—354]

manifest 对每个 tensor 记录：名称、语义枚举、层号、`ne[]`、类型、量化块、编码 row_size、逻辑字节数、实际 shard 字节数、每 rank 的源范围/目标范围、复制原因、重排方式、伴随 scale 和计划消费者。缓存另记头映射、容量、dtype、padding、物理 strides、序列槽数和快照份数。

全局记录 `schema_version`、源码 HEAD、模型元数据摘要、rank 到设备 UUID/PCI BDF 的绑定、角色轮换表、通信精度、状态格式版本和 plan hash。manifest 不是测速日志，也不能写未查询到的显存空闲量。

### 4.2 准入公式

对物理 rank i：

\[
M_i=M_{W,shard,i}+M_{W,replica,i}+M_{KV,i}+M_{idx,i}
  +M_{GDN,i}+M_{PLEstate,i}+M_{activation,i}+M_{scratch,i}
  +M_{comm,i}+M_{runtime,i}+M_{reserve,i}.
\]

必须 `M_i <= 当前可用的设备预算_i`，而不是只满足五卡总和。reserve 由目标机实际运行需求给出，不把设备标称容量全部用满。报告静态预算与运行峰值两列；原始权重映射、重排临时副本、pipeline/descriptor 和驱动开销不能藏在“其他约零”中。

### 4.3 两个必要的容量算例

96 个主体 HC 的两个大投影，假设权重均为 FP16：

\[
96\cdot 2\cdot10240\cdot320\cdot2
=1.171875\ \mathrm{GiB/rank}.
\]

最终 HC 的两个大投影再加 12.5 MiB/rank；norm、inject、其他权重另计。这里的“五卡复制”确实会在五卡各占一份，不是只在某张主卡保留。

所有 routed 专家的三个投影共有：

\[
48\cdot512\cdot3\cdot2560\cdot640
=120795955200\text{ 个权重元素}.
\]

理想无开销 4 bit 且完全均分，也需要 11.25 GiB/rank。假设 gate/up 每 256 元素编码为 136 字节、down 每 32 元素编码为 18 字节，则这部分变成 12.1875 GiB/rank。**这是明确指定编码密度的条件算例，不是本轮发现实际 GGUF 采用了这些类型。** 实际编码字节数必须由 `ggml_row_size` 等类型规则计算。

仅用后一算例加 HC 与固定桥接卡 128K FP16 KV，即 `12.1875 + 1.171875 + 3 = 16.359375 GiB`，已经超过单卡 16 GiB，还没有放其他张量。因此不能预先保证 IQ4 名称的某个文件在 128K 可以运行。

### 4.4 KV 副本与角色轮换

本方案每个 full attention 层的 Q 角色为 `[5,5,5,5,4]`，KV 实例数为 `[1,1,2,1,1]`。12 个 full attention 层、单序列 128K、K/V 均 FP16、头维 256 时：

\[
M_{KV}=N_{layers}\cdot T_{ctx}\cdot2_{K,V}\cdot N_{KV}\cdot256\cdot2_{bytes}.
\]

原来全机约 3 GiB；采用副本后全机约 9 GiB。固定角色会使桥接卡占 3 GiB，其余各 1.5 GiB。

建议为第 a 个 full attention 层使用固定映射：

\[
physical\_rank=(logical\_role+a)\bmod5.
\]

12 层后各物理卡持有的 KV 头实例为 `[14,14,15,15,14]`，相应容量为 `[1.75,1.75,1.875,1.875,1.75] GiB`。总量不变，最高单卡 KV 从 3 GiB 降到 1.875 GiB。这是本文的容量平衡设计；**不能把它记作每个同步阶段的耗时下降**。

角色表加载时固定，并写入 plan 和状态版本。prefill/decode 必须一致，不得每 token 轮换；分片权重、KV、WO 和 global head 映射同步轮换。测试版支持固定角色，用来排查轮换导致的绑定错误。

在前述条件量化例中，使用 1.875 GiB KV 后三项仍有 15.234375 GiB/rank。其余权重、indexer、工作区可能让它继续超限。显存不足的正确处理是降低已声明的上下文/工作区、采用已经验证的其他缓存格式或换权重；不是偷偷把权重退回 CPU 后继续标成纯 GPU。

### 4.5 其他不能漏掉的容量

GDN 基础状态按 FP32、36 层、48 个 128×128 状态矩阵计，全机为 108 MiB/序列，按头分片后分摊；共享状态、快照、hand echo 和 sequence slots 另计。卷积历史不是这个矩阵的一部分。

若 indexer 全量 token key cache 为 128 维 FP16，12 层、128K 的复制缓存约为 **384 MiB/rank**；FP32 则加倍。实际当前缓存类型、长度、padding 和 pooled 工作区应从内存构造器读取；不能仅因 indexer 权重小就忽略缓存。

HC 单个残差 FP32 是 40 KiB/token，B=1024 时是 40 MiB/rank。96 阶段固定 inbox 在 decode 只需约 4.69 MiB/rank，但若机械乘 B=1024 就接近 4.69 GiB/rank，prefill 必须换有界复用方案，见第 17 节。

## 5. 整体架构：复用框架，但扩展真正缺失的契约

### 5.1 分成五层

```text
GGUF metadata / tensor headers
        ↓
llama_tp5_plan：语义、源切片、头映射、状态归属、预算
        ↓
Qwen4EXP 逻辑图 + 明确的子层/HC/副作用边界
        ↓
meta backend：每 rank 子图、局部布局、PARTIAL 合并、collective 计划
        ↓
Vulkan backend：现有矩阵/attention/GDN + 新 collective + HC fusion
        ↓
既有 VkDevice/queue/buffer 生命周期；DMA-BUF mailbox；SYNC_FD
```

模型层不调用 Vulkan，也不操作 FD。通信层不知道“第几层 Qwen 的专家”。执行层收到的是已经校验的 tensor layout、collective 描述、ready 依赖和 consumer recipe。

### 5.2 旧 meta 描述的真实限制

`ggml_backend_meta_split_state` 使用一个 axis、每段每卡长度和重复次数；当前校验要求所有片段总长等于原 tensor 轴长。它可以表达很多连续分片，却不能直接表达“KV0 同时放三卡、KV1 同时放三卡”或任意 gather 后的 GDN 头顺序。[R3；R4]

因此需要一个兼容旧接口的显式布局扩展，而不是去掉 `ne_sum == tensor->ne[axis]` 的断言。去掉断言只会使加载偏移、view 和缓存写入的语义更模糊。

实现方案：在 `ggml-backend-meta.cpp` 与 `src/llama-tp5-plan.cpp` 中引入显式 `indexed_replica` / `replica_start` 元数据与 `llama_tp5_tensor_plan`，兼容旧 `split_state` 接口的同时支持复制头精准切片与最低 rank 规范合并。其概念结构如下：

```cpp
// 设计草图；不要把此草图当作现有 ABI。
struct ggml_meta_span {
    int32_t source_axis;
    int64_t source_begin;
    int64_t length;
    int64_t destination_begin;
};

enum class ggml_meta_value_kind {
    mirrored,        // 每卡同一份完整值
    disjoint,        // 最终值的不相交片段
    partial_sum,     // 每卡均覆盖完整输出域，但只是部分贡献
    indexed_replica  // 显式源范围/头编号，允许有声明的副本
};

struct ggml_meta_rank_layout {
    ggml_meta_value_kind kind;
    std::vector<ggml_meta_span> spans;
    std::array<int64_t, GGML_MAX_DIMS> local_ne;
    std::vector<int32_t> global_heads;
    uint64_t contribution_domain; // 哪些乘法贡献组成这个部分和
};
```

实际接口应放在内部边界，并给出版本与生命周期。旧 backend 不用新接口时保持原有路径。布局扩展不只是存一份 vector：权重装载、数据上传、tensor view、reshape、cache get/set、FA/GDN lowering、回读重建都要消费同一个描述。

### 5.3 图切分不能过早归约

当前 meta 源状态查询使用 `assume_sync=true`，并据此建立同步子图。不能假定 routed down 和 shared down 的 PARTIAL 会自然在卡内合并后才归约。[R3：约 790—1064、1815 以后]

应引入明确的局部线性区域：相同贡献域的 `PARTIAL + PARTIAL` 仍是 PARTIAL；`PARTIAL × MIRRORED` 的逐元素已知权重仍可保留 PARTIAL；不同中间通道的完整分片不能混成 PARTIAL。遇到非线性、归一化、需要完整输入的 router，或者区域指定的输出边界，才归约。

尤其不能把 `residual + partial` 在五卡各做完后直接求和，那会把残差加五次。HC combine 必须消费已归约的 y，或者由专用融合消费五份 y_i 后只加一次 residual。

对 Qwen4EXP 首版，三个输出边界明确命名：`attn_output`、`linear_attn_out`、合并 routed/shared 后的 `ffn_out`。48 层主体应产生 96 个逻辑输出归约点；数量由实际图生成并断言，而不是计时器里写死 96 掩盖额外归约。

### 5.4 不启动第二套 Vulkan 设备

POC 自己创建五个 VkDevice 适合独立测试；生产路径必须使用现有 backend 的设备、队列、分配器、锁和提交状态。另建 VkDevice 会使权重/缓存句柄不可直接复用，打乱显存预算和同步归属。

`ggml-vulkan.cpp` 内有大量私有类型和 static 函数。拆分新 `.cpp` 时，先提取窄的内部桥接接口，不把整个 `vk_device_struct` 暴露到 llama 模型层。桥接只提供设备能力、tensor 子缓冲定位、生产者提交/完成事件、待消费等待、mailbox 管理和受锁保护的队列提交。

## 6. MoE：专家内 TP 的数学与程序设计

### 6.1 路由不变，切专家内部

给定完整输入 x，使用现有 router 得到选中专家集合 E(x) 和权重 p_e。保留当前 softmax、选中权重归一化、`expert_weights_scale` 及 tie-break 行为，不另写近似 router。

单个专家：

\[
z_e=\operatorname{SiLU}(G_ex)\odot(U_ex),\qquad y_e=D_ez_e.
\]

把 F=640 沿中间通道分为 `I_i=[128i,128(i+1))`：

\[
z_{e,i}=\operatorname{SiLU}(G_e[I_i,:]x)\odot(U_e[I_i,:]x),
\quad y_{e,i}=D_e[:,I_i]z_{e,i}.
\]

于是 `y_e = sum_i y_{e,i}`。每卡中间结果是 128 维，但 down 输出是完整 H 维的部分和，不是 512 维结果片段。

共享专家还必须乘同一个输入产生的标量门：

\[
a_s=\sigma(w_s^Tx),\quad
y_i=\sum_{e\in E(x)}p_e y_{e,i}+a_s y_{shared,i},\quad
y=\sum_i y_i.
\]

共享 gate 的 sigmoid 不能漏掉；它在当前 `build_layer_ffn` 中确实存在。[R1：988—1032]

### 6.2 实现落点

plan 为 `ffn_gate_exps/up_exps/down_exps` 和 `ffn_*_shexp` 生成配对切片；修改 `llama_meta_device_get_split_state` 的架构专用适配，不能只依赖当前不覆盖所有 shared 命名的通用 regex。

继续使用 `build_moe_ffn` 和 `GGML_OP_MUL_MAT_ID` 的设备端专家选择。不允许每个专家一个 CPU 分支、每层将专家 ID 回读再重新构图，也不允许每专家 AllReduce。128 宽是否适合已有 Vulkan grouped kernel，应以真实 token 分布做小基准。

本地 routed 和 shared 输出保持 PARTIAL，先合并，再送一次公共 collective。若 debug 需要查看二者，应写调试副本，不为查看副本提前归约。

### 6.3 量化切片的合法性

gate/up 切的是输出行，整行输入长度仍是 H；down 切的是编码输入轴，起点与长度都必须满足实际块大小。读取同一个 GGUF 的每个 down 类型，不能根据文件名中的 IQ4 推断。

若 down 的量化块为 32，则 128 通道包含四个完整块；若实际要求 256 且不支持分块抽取，则这条 128 切法不能直接启用。首版明确报错，不静默反量化为 FP16、不截半个 block、不补假通道。后续可设计重新编码或不同分片粒度，但要重新做显存与精度验收。

重排按完整编码单元拷贝；同时处理专家轴 `ne[2]`、`nb[2]` 和 fused gate/up 的两个段。每个专家各自的 I_i 相同，不能把 512 个专家 flatten 后切成五段。

### 6.4 验收

先用固定路由检查每个专家的局部和，再接真实 router。构造所有 token 都选同一个专家的热点输入，确认五卡仍共同计算该专家；构造不同 batch 的路由，验证专家槽复用和累加清零。记录真实 dispatch 数、MUL_MAT_ID 形状和 fallback；均分 FLOPs 不等于 kernel 时间降低到五分之一。

## 7. QSA attention：头映射、缓存与输出投影

### 7.1 不补头的角色表

原始映射为 `kv(q)=floor(q/12)`。逻辑角色如下，之后才按第 4 节映射到物理卡：

| 角色 | Q 头半开区间 | 所需 KV | 本地 attention 调用 |
|---:|---|---|---|
| 0 | `[0,5)` | KV0 | 5Q / 1KV |
| 1 | `[5,10)` | KV0 | 5Q / 1KV |
| 2 | `[10,15)` | KV0、KV1 | `[10,12)` 的 2Q/1KV；`[12,15)` 的 3Q/1KV |
| 3 | `[15,20)` | KV1 | 5Q / 1KV |
| 4 | `[20,24)` | KV1 | 4Q / 1KV |

桥接卡的两次调用是为了使用现有等比例 GQA kernel，不是把原模型改成局部 5Q/2KV。后续显式 `q_to_kv[]` 融合 kernel 可以减少调用次数，但先不作为正确性闭环的前置条件。

Q 头最大数量相对理想 4.8 头是 `5/4.8-1 = 4.1667%`。这只是 Q 头数量的最大尾差，不是全模型统一性能系数。

### 7.2 Q 与 gate 的行定位

当前 Q 投影的每个全局头存 `[q_d通道, gate_d通道]`，然后下一个头。全局 q 头对应输出行：

```text
Q rows    = [2*da*q,     2*da*q + da)
gate rows = [2*da*q+da,  2*da*(q+1))
```

取整头时两段一起复制；归一化和 RoPE 只应用到 Q，不应用到 gate。attention 输出乘 `sigmoid(gate)` 后，使用 WO 中对应全局头通道的输入列。若本地调用先分两组再拼接，拼接顺序与 WO 的本地列顺序必须同表生成。

### 7.3 缓存必须与计算一起分片

各卡本地持有需要的 K/V 投影权重并更新副本，避免每步远程搬历史 KV。每层每个本地唯一 KV 头的 cache 写入只能执行一次；桥接角色的两个 attention 调用共享已经写好的 cache，不能各自再推进 cache cell。

所有 rank 使用同一 logical cell/position/sequence ID。局部物理头维可不同，但 token 轴必须一致。修改点包括 `llama-kv-cache` 的分配、get/set view、索引输入和状态读写；只改 QKV 权重不会自动让缓存变成本文布局。

保存计划中的 `(layer, global_kv_head) -> local_head` 映射。跨卡副本在 debug 中比较内容；生产不为此增加逐层回读。若不同 rank 的复制输入发生数值漂移，先解决通信表示和 router/indexer 确定性，不能容忍副本长期各算各的。

### 7.4 QSA 的计算保持当前语义

按当前 `build_qsa_top_k` 保留 indexer 的 query/key 归一化、RoPE、块均值、因果限制、ReLU/头合并及选择规则。索引选择属于非线性离散决策，不能把四个 indexer 头分到四卡后只各取本地 top-k 再拼起来。

本轮源码中的 `build_attn_qsa` 根据选中 cell 构造 mask，再调用 `build_attn_mha`；`build_qsa_top_k` 为绕过 Vulkan TOP_K 的 k 范围使用 `ggml_argsort_top_k`。这条路径应描述为“QSA 功能图”，不能自动称为已经跳过未选 KV 的稀疏 Flash Attention。[R1：525—760]

首版在各卡复制 indexer，保留相同 top-k 和 cache cell 对应关系。耗时和容量都要单列。增量 pooled-key cache、稀疏 kernel、索引广播属于后续优化，不能与 TP5 正确性改动混在同一验收里。

### 7.5 正确性检查

用每个全局 Q/KV 头具有不同数据签名的输入检测错映射；只用相同头值会漏掉这类错误。测试桥接角色、全因果与尾块、不整齐 context 长度、单次及分块 prefill、连续 decode。对照 attention 的 pregate、gate 后输出、WO 部分和、KV 新写行与 indexer 选择。

## 8. GDN：完整状态头与全局取模映射

### 8.1 切片表

逻辑 V/state 头范围先取 `[0,10)、[10,20)、[20,30)、[30,40)、[40,48)`。每个头包含完整的 ds×ds 递归状态、V、z、alpha、beta、norm 输出和对应的 ssm_out 输入通道。

选择偶数头数使 `ds × 头数` 适合 256 元素量化块的一类布局；这仍要以实际 down/out 编码校验。若未来改为 `[10,10,10,9,9]`，先证明量化和内核均支持，不能只按头数尾差决定。

### 8.2 Q/K 的映射是全局头号取模

当前 Qwen4EXP 图使用与 Qwen3.5 相同的重复模式；Vulkan `gated_delta_net.comp` 也明确使用 `head_id % neq1`。[R1：849—986；R8：108、417]

因此本例的全局 V 头 h 使用 Q/K 头 `h % 16`。这不是 `floor(h/3)`。当本地 V 头从全局 10 开始时，本地编号 0 应读取全局 QK10，不能再读 QK0。

首版采用预排列法：将每卡所需 Q/K 及对应卷积通道，按本地 V 头顺序排列为相同数量的本地 Q/K 头，使本地 kernel 的一一匹配仍然正确。或者在降层时显式提供 map；两者选择一种作为默认，不同时隐藏两套规则。

预排列增加 Q/K 投影与卷积历史副本。全机原先 16 个 QK 头可能变为 48 个实例；`Q+K+V` 的通道总数从 10240 增至 18432。它不增加跨卡状态交换，但增加本地权重、卷积和访存成本，必须进入预算和 profile。

### 8.3 递归数学契约

用每个头的状态矩阵 S 表示从 key 维到 value 维的映射，约定 S 为 `[key,value]`。令 q、k 是当前图完成卷积、SiLU、L2 normalization 及原路径要求的缩放后的向量，v 为 value，`b=sigmoid(beta_linear)`，`g=ssm_a*softplus(alpha_linear+dt_bias)`。单序列的等价递归可写为：

\[
\bar S_t=e^{g_t}S_{t-1},\quad
u_t=v_t-\bar S_t^T k_t,\quad
S_t=\bar S_t+b_t k_tu_t^T,\quad
o_t=S_t^Tq_t.
\]

这里不另增加缩放因子：query 缩放由现有 recurrent builder/kernel 的契约决定，并在参考测试中核对。实际 ggml 状态存储的轴方向可能与数学写法转置，必须通过 `ne/nb` 定位，不能直接按公式解释内存。

各头状态独立，所以完整头切片不需要跨 rank 交换 S。但同一头沿 token 时间有递归依赖；把不同时间段随意分给五卡不成立。prefill 的 chunk 算法也必须与同一递归等价，并连续接上前一 ubatch 的状态。

输出阶段使用当前 Qwen4EXP 的 `RMSNorm(o) × sigmoid(z)`，**不是其他架构常见的 SiLU(z)**。之后本地 ssm_out 产生全 H 维部分和，再归约一次。

### 8.4 状态对象与副作用

`cache_r` 卷积历史、`cache_s` 递归状态及 PLE history 不混为一个平面切片。使用 plan 为每类状态建立独立映射。原图有共享状态和 hand echo 路径；首版未适配时应拒绝这些模式，而不是强行走普通状态指针。

GDN 更新具有副作用。collective 出错后，不能重新执行整个子图来“回退”，否则状态可能推进两次。执行器必须知道提交前、已经提交、已完成和状态已发布的区别，失败处理见第 16 节。

验收先比较单头单步，再比较多步和分块 prefill；故意让每个全局头的初始 S 不同，验证切片、序列 copy/reset、保存恢复和角色轮换。只检查最终文本不能定位状态映射错误。

## 9. HC：先固定数学，再做 F1 / F2 / F3

### 9.1 原图的完整公式

对每个 token，残差是 `x[c,d]`，不是一条共同归一化的 10240 维向量。设 gamma 是 GGUF 中已经折叠好的 norm weight：

\[
r_c=\left(\frac1H\sum_{d=0}^{H-1}x_{c,d}^2+\epsilon\right)^{-1/2},
\qquad n_{c,d}=\gamma_{c,d}x_{c,d}r_c.
\]

将 n 按原图流优先顺序展平为 n_flat：

\[
\ell=\operatorname{SiLU}(D n_{flat}/C),\qquad
g=U\ell,\qquad
m_d=\frac1C\sum_c n_{c,d}\sigma(g_{c,d}).
\]

独立的注入分支为：

\[
j=J n_{flat},\qquad a_c=2\sigma(j_c/C).
\]

子层输出 y 完成后：

\[
x'_{c,d}=x_{c,d}+a_c y_d.
\]

这些分别对应 `build_hc_mix`、`build_hc_combine`。转换器已将 gamma 折叠为 `(1+w)`，融合不能再加一。代码使用 `build_lora_mm`，所以也不能在开启 LoRA 时擅自当作纯 W·x。[R1：266—335]

数学上可以按低秩维切 D 的输出和 U 的输入，得到 `g=sum_i U_i·SiLU(D_i n/C)`，但它要求在 sigmoid 前归约。A-F 不采用此通信布局。将输入和输出都除以五而不补交叉项，或把各卡 sigmoid 后的值相加，都不是原模型。

### 9.2 A-F 的数据路径

```text
完整 HC 残差（各 rank 复制）
        ↓
本地 grouped RMSNorm / gamma
        ↓
F2：完整 down → 1/C → SiLU；生成本子层 inject
        ↓
F3：完整 up → sigmoid → 乘 n → 四流平均
        ↓
完整 mixed[H,B]（各 rank 一致）
        ↓
本子层 TP：attention / GDN / MoE
        ↓
每 rank 的完整 partial[H,B]
        ↓
PUSH + 跨设备就绪依赖
        ↓
F1：求和 → 本子层 HC combine → 下一 HC 的 grouped RMSNorm
```

F1、F2、F3 是逻辑融合段，不强求只有三个 dispatch。一个段拆成两次高效 kernel，可能比一个寄存器溢出的“大融合”更快。先保持可对照的未融合路径，再逐段替换。

### 9.3 F3：最先实现的融合点

输入：完整 `lo[R,B]`、`n[H,C,B]`、up 权重。输出：`mixed[H,B]`。不必永久保存 `gate[CH,B]`、`sigmoid[CH,B]` 和 `gated[CH,B]` 三份中间张量。

计算单位按同一个隐藏通道 d 的四条流组织：

```text
for token t:
  for hidden d:
    mixed = 0
    for stream c in 0..C-1:
      gate = dot(U[c*H+d, :], lo[:, t])   // 完成整个 R 维点积
      mixed += n[d,c,t] * sigmoid(gate)
    output[d,t] = mixed / C
```

这是算法伪代码，不是推荐逐标量执行。Vulkan kernel 中由线程组协作读 lo、分担输出 d，并保留 FP32 累加。先试每线程或线程子组处理一组 d 的四个流输出，再比较每组多个 d；对 R=320，向量化、尾部和共享内存容量要单独处理。

权重原始编码行 `row=(c*H+d)`，行偏移用 `ggml_row_size(type,R)`，不是默认 `R*sizeof(half)`。可在加载时把完整行从 `[stream,hidden,rank]` 重排到 `[hidden,stream,rank]`，但必须保留编码单元和类型。不要为了重排改变权重量化。若存在旧布局和新布局双份驻留，把峰值列入第 4 节预算，并在转换后释放不再需要的副本。

首版至少有以下 guard：C/R/H 与 kernel 契约匹配；输入和权重类型在支持表内；无未处理 LoRA/bias/scale；strides 可解释；中间量没有其他观察者；输出 token 集合与原链一致。条件不满足则回原算子链，日志记录原因。

debug 模式允许额外写 gate 供逐项比较；性能模式不写。每次融合必须更新后端的读写依赖和 `fused_ops_write_mask`，不能只跳过图节点而让 allocator 提前复用仍在读的 n 或 lo。

### 9.4 F2：完整 down 的后处理和 inject

输入完整 n_flat，计算 320 个 down 输出，点积完成后先乘 `1/C`，再 SiLU，直接写 lo。初版保留现有 GEMV 的主体，只把后处理加进 epilogue，避免同时重写量化解码和融合。

inject 的 C 个输出也读取 n_flat。可实现共享输入的双投影 dispatch，逻辑输出为 `R+C` 个值；但是两个权重 dtype 不同或算法不合适时，保留独立 inject GEMV。实际可选择保存 raw inject 或已转换的 a，必须在类型/字段名中说明，不能让 combine 再做一次 sigmoid。

建议字段名：

```text
hc_raw_inject_current[C,B]  // 原图接口仍使用 raw 值时
hc_alpha_current[C,B]       // 已是 2*sigmoid(raw/C)
hc_alpha_next[C,B]          // 下一个子层自己的系数
```

当前子层的 alpha 必须到 F1 消费完才可覆盖。把 inject 放在 PUSH 期间计算和把 inject 融进 F2 是两种不同安排；不能同时记两份“重叠收益”。首版先选后者或独立串行分支，重叠放后续实验。

### 9.5 F1：不要一次跨越所有边界

F1 的通常输入是五份 `partial[H,B]`、当前残差 x、当前子层 alpha、**下一 HC** 的 gamma/epsilon；输出是更新残差 x' 和下一 n。先按固定 rank 顺序求 y，再完成 x' 和四个流各自的平方和与归一化。

至少定义四个路径：

| 路径 | 使用场景 |
|---|---|
| `sum_only` | collective 参考实现，或 y 有其他消费者 |
| `sum_combine` | 后面接 PLE、输出筛选或其他不能跨越的操作 |
| `sum_combine_norm` | 普通子层相邻边界，能够直接进入下一 HC |
| `norm_only` | 模型入口或 PLE 已完成之后 |

第一版先实现“公共归约写 y → 本地 combine+norm”。这一步可以复用旧 collective ABI，便于分离同步问题。第二步才实现“接收槽五路求和+combine+norm”，需要第 14 节的 fused consumer 契约。不能既在通信层求过和，又在 F1 将五份输入再求一次。

单 token 候选 kernel：每个流一个 workgroup，组内线程分担 H=2560 项，计算 x' 并累计平方和，组内规约后写 n。所有线程必须参加 barrier，越界线程使用零贡献，不能提前 return。只启动四个 workgroup 可能占用不足；同时保留多组 partial-sum 加第二个 norm dispatch 的版本，按完整 F1 耗时选择。

更新残差仍须落地，因为本子层 combine 后的残差是后续状态链的输入，也可能被图观察者使用。不能为了融合只写 n 丢掉 x'。

### 9.6 四个不能自动跨越的边界

**PLE。** 原图在该层 HC 前先执行 PLE。上一层输出归约后只能先 combine，执行 PLE，再 norm。把 norm 提前会改变输入。

**最后一层 `inp_out_ids`。** 当前图在最后一次 attention/GDN 后，先选择 cur、inject 和 residual 的输出 token，再做 combine/FFN。prefill 的 B 在这里可能从 B_all 变为 B_out。融合必须使用相同 selector，不能访问被裁掉 token 或错把 B_all 写入后续 slots。

**首层和最终 HC。** 首层 residual 由 embedding 复制 C 次，没有前一 y。最后一个 FFN combine 后还有 `hc_head_norm/down/up` 的最终 mix；它没有 inject，不能假装下一次普通子层。

**观察者和副作用。** `t_layer_inp`、`l_last`、embedding 输出、debug callback、imatrix/control vector 等可能要求中间结果存在。融合只消除确实没有独立使用者的值；不支持某个观察模式时拒绝融合，不能让观察 API 静默读错。

## 10. 入口、PLE、最终输出：96 次以外也要实现

### 10.1 Token embedding 与 LM head

推荐 plan 支持沿词表行分片的 token embedding 和 LM head：当前 token 的 embedding 行在一个 owner 上读出，分发完整 H 维到其余 rank；最终 mixed 在各卡完整复制，LM head 在各卡算本地词表区间的 logits。

这是输出轴分片，不需要对 logits 求和。首版收集完整 FP32 logits 到既有采样器，再使用原有全局 softmax/top-p、重复惩罚、grammar 和采样逻辑。不能用“每卡只给局部 top-k”代替完整接口后仍声称所有采样行为未变。

词表不整除时使用显式行范围；此轴不是权重的量化输入轴，但仍检查物理 row_size。tied embedding/output 必须保持权重共享和布局一致，不能多加载一份或错误共享不同 shard。若首个实现暂时复制 token embedding，应在 manifest 明示额外字节数，而非悄悄改变准入预算。

入口广播和 logits 收集都计入整 token 延迟，不能因为不属于 96 个子层归约就略去。不要在文档里固定词表容量的通信字节数，运行时根据实际 V、B_out 和 dtype 生成。

### 10.2 PLE 查表与历史

当前 `build_inp_ple` 生成 I32 rows，从 host 上的 `per_layer_tok_embd` 做 `get_rows`，再把结果展平；查表留在表所在 backend，只有结果跨到设备。[R1：1208—1231]

首版共享一次 CPU hash/查表结果，向五卡提供一致的 PLE embedding。结果宽度为 `ple_head_dim * ple_n_heads`，必须核对等于 PLE 投影期望的输入宽度，不能无条件写死为 H。表本身不复制到 GPU，也不因 TP5 五个子图执行五遍 token history 更新。

PLE GPU 部分首版复制，保留原图：key/value 投影；按流 RMSNorm；key 与 hidden query 的点积除以 sqrt(H)；带符号平方根与 sigmoid；value 按流加 gate；norm 后的膨胀因果卷积；SiLU；最终 `hidden + gated + conv_out`。这些操作不等于一个普通 embedding 相加。[R1：1233—1325]

卷积历史长度为 `(ple_conv_kernel-1)*ple_ngram_size`，逐序列维护。当前加载器只允许一个 PLE 层，且要求其为 recurrent 层；TP5 不绕过这些验证。EOS、序列 reset/copy、chunked prefill 和 token-history 提交必须在统一请求状态中执行一次。

### 10.3 CPU 工作并不只有 n-gram

还存在 tokenizer/输入准备、调度、SYNC_FD 操作、采样及可能的日志。所谓“权重和 KV 全在 GPU”不等于 TP5 的 CPU 时间为零，也不等于 host staging 不可能发生。性能版对非允许节点的 CPU fallback 计数并报告；无意的矩阵/attention 回 CPU 视为未达到首版目标。

## 11. 单阶段 PUSH collective 的准确契约

### 11.1 输入和输出

每 rank 提供 `partial[H,B]`，全部 H 维都是该 rank 的部分贡献。输出为每 rank 都能安全消费的：

\[
y_{d,t}=\sum_{i=0}^{P-1}partial_{i,d,t}.
\]

单阶段 mesh：每卡把自己的**完整** partial 发给其他四卡；每个接收者加自己的 canonical local contribution 后，固定顺序求和。没有第二次 AllGather。若改成发送 S/5，则已换成另一种算法，必须补齐 ReduceScatter 与 AllGather 的依赖及计时。

### 11.2 字节账本

`S = H * B * wire_bytes`。下面只算 payload，不含 padding、FD、提交和协议开销：

| 单 token decode | FP32 | FP16 |
|---|---:|---:|
| 一份完整 partial S | 10240 B = 10 KiB | 5120 B = 5 KiB |
| 向一个对端发送 | S | S |
| 每卡向四个对端发送 | 40 KiB | 20 KiB |
| 每卡接收 | 40 KiB | 20 KiB |
| 五卡总发送 | 200 KiB | 100 KiB |
| 96 个主体事件，每卡发送 | 3.75 MiB | 1.875 MiB |
| 96 个主体事件，五卡总发送 | 18.75 MiB | 9.375 MiB |

每个事件有 20 笔远端发送，96 个事件有 1920 笔；本地 self-copy 或 packing 另计。全网总发送已经把每份传输记一次，不要再加总接收来声称更大的“有效带宽”。

### 11.3 FP16 wire 必须统一本地与远端表示

```text
FP32 partial_i
    → 同一个转换 kernel 生成 canonical_wire_i
    → 本卡求和也从 canonical_wire_i 读取
    → 四份远端 PUSH 都复制 canonical_wire_i
    → 各接收者转 FP32，按 rank 0,1,2,3,4 顺序相加
```

禁止“自己用未舍入 FP32、别人读 FP16 副本”。否则各 rank 得到的 y 不一致，复制的 HC、router、indexer 和 KV 会开始分叉。求和顺序也不能按数据到达顺序决定。

### 11.4 单阶段不代表无同步

一次完整事件的边界是：`生产完成 → copy 可读 → 四路发送完成并 release → 接收端 wait/acquire → 求和/融合 → 消费完成`。只测 vkCmdCopyBuffer 的时间、或只等发送提交返回，都不是这条链的延迟。

PUSH 不要求 shader 高速读取远端显存：发送者 copy 到接收卡本地 mailbox，求和 shader 只读本地内存。远端 shader 任意访问是另一项能力，不是本方案前提。

### 11.5 对照算法

同长 S 的一般 AllReduce，可对照 ring 或直接 ReduceScatter+AllGather，后者每卡发送 `2(P-1)S/P = 1.6S`。mesh 为 `4S`，省交换依赖但多流量。不要把两者的 event 数与物理步骤混写。

对 B=1024 的 FP32 输出，S=10 MiB，mesh 每卡一次发 40 MiB；它不应自动成为 prefill 的最快选择。算法选择由消息大小和目标机完整依赖链测试决定；首版先保留数学/状态不变的替换入口，见第 17 节。

## 12. Mailbox、生命周期与显存所有权

### 12.1 逻辑地址

decode 正确性优先，每 rank 逻辑上有：

```text
inbox[epoch_slot][stage][sender_rank][token][hidden]
stage = 0 .. 2*L-1
```

同一 token 内 96 个 stage 使用不同范围。`GPU i → GPU j` 的目标固定为 `j.inbox[stage][i]`，源是 i 的完整 canonical wire，不再用两阶段 POC 的 `src_chunk=j`。

包括自身槽，FP32、B=1、单 epoch 的逻辑大小为 `96*5*2560*4 = 4.6875 MiB/rank`；FP16 为 2.34375 MiB。物理对齐和元数据另计。

### 12.2 物理对象建议

为每个有向卡对建立独立 mailbox，即 20 个远端接收对象，各对象内部装 96 个 stage 的范围；自己的一份 contribution 可以在非导出的本地 buffer。这样单一写者明确，也避免每个 stage 一个小 allocation 的数量膨胀。

第一次实现不做复杂的多发送者别名复用。每个 mailbox 由接收卡分配 device-local storage，并导出 DMA-BUF，由指定发送卡导入为可写目标。一个物理 allocation 的范围可以细分，但所有权和重用规则必须精确到有效范围；若实现只支持整 buffer 转移，则按整 buffer 生命周期组织，不能假装可独立转移其中尚在使用的片段。

stage stride 至少满足实际 storage-buffer offset alignment、copy offset/size 和所选访问方式要求；用 checked arithmetic 计算 `align_up(S,alignment)`，禁止 int32 字节偏移溢出。非连续 tensor 先通过已计时的 packing 生成规范线性 layout，不能把 `ggml_nelements*element_size` 当作任意 tensor 的连续字节数。

### 12.3 显式状态机

每个共享范围维护：

```text
AVAILABLE_TO_WRITER
  → WRITER_OWNED
  → WRITE_SUBMITTED
  → RELEASED_WITH_READY_EVENT
  → READER_ACQUIRED
  → READER_CONSUMED_AND_RELEASED
  → RECYCLABLE
```

对应数据至少包含 `epoch、stage、sender、receiver、generation、byte_range、writer_ready、reader_done`。相同地址不同 epoch 不是同一份数据。debug 可以加哨兵和 generation 校验，但不能用 shader 自旋检查一个标志来代替 Vulkan 同步。

decode 首版在整 token 的所有消费者结束后才回收 epoch，包括最后一次 y 的消费和共享范围 release。主卡已经得到 logits，不等于其他 rank 已经结束。使用两份 epoch buffer 也不自动合法，复用仍需完成依据。

### 12.4 release/acquire 与外部域

独立物理设备的外部共享不能假定一个本地 VkMemoryBarrier 已解决所有权。`VK_QUEUE_FAMILY_EXTERNAL` 代表的队列要求相同物理设备和驱动版本；`VK_QUEUE_FAMILY_FOREIGN_EXT` 可表达不同设备，但支持更严格，需要实际能力路径确认。[V6]

对本方案的有向 mailbox，初始化时先建立可供发送者取得的外部状态；发送者 acquire → write → release 到适用外部域并 signal；接收者等待对应完成事件后 acquire → read → release；下一 epoch 才重新取得。使用带 buffer、offset、size 和 queue family 的 barrier，不能把不同物理设备的 queue family 数字直接当作同一设备的两个队列编号。

这部分必须按目标驱动支持的外部内存路径写成共享资源协议并测试。能力不满足时禁用 fast path，而不是删 ownership 操作“试着跑”。普通本地 barrier、跨设备 semaphore、所有权转移各管一部分，三者不能互相替代。

## 13. SYNC_FD：提交依赖，而不是逐层等 CPU fence

### 13.1 初始化能力查询

每个目标设备查询：external memory FD、DMA-BUF handle type 对指定 buffer usage 的可导入/导出性；实际 FD 的内存类型；binary SYNC_FD 的可导入/导出性；适用的外部 queue-family ownership；compute/copy 队列；所需 synchronization2 能力。

不要只查扩展名称。不要把 device group 的 peer-memory bits 当作这条独立 VkDevice DMA-BUF 路径的证明。不要默认同型号、同驱动的不同 GPU 有相同 deviceUUID。[V1—V7]

### 13.2 导入内存的检查顺序

```text
创建导入端 VkBuffer，携带相同 handle type 与所需 usage
    → 查询目标 VkMemoryRequirements / dedicated 要求
    → vkGetMemoryFdPropertiesKHR(actual_fd)
    → compatible = buffer.memoryTypeBits & fd.memoryTypeBits
    → 核对导出 allocation 大小、目标要求、绑定范围和 alignment
    → 从交集中选合法 memoryType；无解则返回明确错误
    → 按要求建立 import / dedicated pNext 链
    → allocate / bind；成功后完成 FD 所有权转移
```

记录逻辑 payload 大小和实际 allocation 大小两个字段。不能将 S 直接当作导入 allocation 大小，也不能随便把大小取 max 后认为两个设备就一定兼容。[V7；R6]

### 13.3 一道输出边界的提交顺序

对 stage s，所有 rank 先提交本地计算、wire 转换、四路 PUSH、release 和 source_done 信号。**五个生产者提交完成以后**，主机逐个导出事件并建立消费者等待。它不等待 GPU 完成本 stage。

```text
for each rank i:
    submit(produce_i + push_to_four_peers + release, signal=done[s][i])

for each producer i:
    fd = export_once(done[s][i], SYNC_FD)
    for each consumer j != i:
        import_temporary(dup_or_signaled_minus_one(fd), wait[s][j][i])
    close_original_if_valid(fd)

for each rank j:
    submit(next_consumer_graph,
           waits=wait[s][j][all_remote_senders])
```

这是提交层伪代码，不是完整 Vulkan 实现；每个 submit 仍带准确的本地 memory barriers 和共享范围 acquire/release。自己的 contribution 依赖本卡队列顺序或相应本地 semaphore；四个导入 semaphore 管远端生产者。下一图的第一次相关读取不能早于这些 wait。

### 13.4 API 规则必须落实为代码

SYNC_FD 是 binary semaphore 的 copy-transference、临时导入载荷，不是跨物理卡永久共享的 timeline。导入带 `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`；`fd=-1` 合法，表示已完成事件，不能对它调用 dup。[V2、V4]

导出前必须已有提交的 signal operation，且它依赖的 signal 也已提交；可以是 pending，不必先等待。copy-transference 导出时不能另有 queue 正在等待这个源 semaphore。源 done 对象只用于导出，不同时当作本地其他队列的 wait 对象。[V3]

每个生产者事件导出一次，再为四个接收者复制 FD。不要对同一次 signal 连续导出四遍。每个成功导入的 FD 所有权转给 Vulkan；失败或未转移的 FD 由应用关闭。导入目标 semaphore 不能仍被未完成 queue command 使用。[V4、V5]

预先分配 `done[stage][rank]` 和 `wait[stage][consumer][producer]`，避免每层创建对象；但**每个新 epoch 的事件载荷仍要重新导出/导入**。对象复用必须等上次使用完成。生产使用 checked `dup`/`F_DUPFD_CLOEXEC` 和 RAII，不能泄漏句柄或在导入成功后双重 close。

### 13.5 预录的准确含义

可以预录每个 stage 的稳定 command buffer，复用 pipeline、descriptor 和已导入的显存对象；SYNC_FD 的载荷和 wait/signal submission 仍按本轮更新。

不能预录一个完全没有中间跨卡等待的巨大 command buffer，然后只在头尾等 fence，声称它等价于 96 道正确依赖。SYNC_FD 也不能提前代表任意未来尚未提交的 signal。第一版按 stage 提交，先保正确；降低 host 提交成本是另一个明确的优化任务。

当前 Vulkan 图提交还有节点数/FLOPs 阈值和较弱设备的超时保护。TP5 接线不能删除这些保护来追求“一次 submit”。即使每个逻辑 stage 内分多次本地提交，其最后的 release/signal 仍须覆盖真正的生产者。

### 13.6 主机成本账本

按 96 道边界直接全互连转发，每 token 有 480 次生产者事件导出、1920 次消费者事件导入，另有 FD 复制、关闭、图提交和锁操作。实际 final consumer、入口/PLE/LM head 的事件另外计数。

这些调用可能成为关键路径，必须在 end-to-end 计时范围内。没有 `vkWaitForFences` 不等于没有 CPU 参与，也不能由导入成功推出纳秒唤醒或固定的几十微秒 collective 延迟。

第一版由一个调度线程按 rank 顺序提交，保证 VkQueue 外部主机同步和可追踪性；多个线程并行提交是后续优化。不得在一条线程持有 rank0 队列锁时等待另一线程持有 rank1 队列锁的操作，统一锁顺序或不跨队列持锁。

### 13.7 本地 barriers 的模板

| 依赖 | 源 stage / access | 目标 stage / access |
|---|---|---|
| 本地输出给 copy | COMPUTE_SHADER / SHADER_WRITE | TRANSFER / TRANSFER_READ |
| 本地 copy 给求和 | TRANSFER / TRANSFER_WRITE | COMPUTE_SHADER / SHADER_READ |
| F1 写 n 给 F2 | COMPUTE_SHADER / SHADER_WRITE | COMPUTE_SHADER / SHADER_READ |
| F2 写 lo 给 F3 | COMPUTE_SHADER / SHADER_WRITE | COMPUTE_SHADER / SHADER_READ |
| device 回读给 CPU | TRANSFER / TRANSFER_WRITE | HOST / HOST_READ |

外部 mailbox 的行还要加适用的跨设备 event 和 ownership 操作，不能只套第二行。使用 stage 支持的 access bits；不要把 SHADER_WRITE 放在只有 TRANSFER 的源阶段里。非 coherent host 内存还需按 `nonCoherentAtomSize` 对齐 flush/invalidate；coherent 也不免除执行依赖。[V8]

### 13.8 真正的等待测试

独立 SYNC_FD 测试不能只导入一个已经 signaled 的 FD 后返回成功。让生产者先执行有限工作，再写本轮新数据并 signal；消费者在生产者未完成时就提交 wait 后的校验 kernel。全部数据按 index/epoch 变化，并人工延后一个生产者。

不在导出前等待 producer fence，不用无限自旋制造“延迟”，不依赖一个只能打印 VK_SUCCESS 的探针。测试必须证明消费者读取的是本轮写入值，且同步后继续生成下一轮输入。

## 14. Vulkan collective 与 F1 的具体接线

### 14.1 首先实现现有三个 registry 入口

当前 `ggml/include/ggml-backend.h:233—235` 定义：

```cpp
typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t *, size_t);
typedef void   (*ggml_backend_comm_free_t)(void *);
typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void *, ggml_tensor **);
```

在 Vulkan registry 中接入对应过程名，meta 的构造、执行和析构就可使用该路径。`comm_init` 只接受已经通过能力检查、同一 Vulkan 实现、rank 顺序确定的 backend 集合，创建共享 transport/context；`free` 等待或确认全部在途资源退休后释放。

初版 `allreduce_tensor` 只接受连续、同 shape、支持 dtype、预分配目标的输入，完成 mesh + 固定顺序本地 sum，输出仍按旧接口约定写回各 tensor。对不支持的尺寸，在**没有提交任何副作用前**返回 false 让旧 fallback 处理；性能严格模式应禁止此 fallback，返回可读错误。

异步返回 true 的含义不是“数据已经在 CPU 看来完成”，而是“后续同 backend 的合法消费者必定通过已建立的依赖读到结果”。必须把外部 wait/acquire 安装到消费队列或当前提交链。单纯发出 PUSH 就返回 true 是错误实现。

### 14.2 bool 接口的错误边界

已经向部分 GPU 提交后再返回 false，会让 meta fallback 重新计算/归约已被修改的数据；对 GDN 还可能伴随状态推进。这种失败不能走旧的“unsupported”分支。

建议增加内部 v2 状态：`unsupported_before_submit`、`queued`、`failed_after_submit`，并向上转为 `ggml_status`。旧 bool 入口只用于能保持原契约的操作，错误必须进入 backend error 状态，禁止继续消费；不把运行失败伪装为“不支持”后重放。

### 14.3 完整 F1 需要一个明确的新契约

普通 allreduce 入口只知道输入输出 tensor，不知道下一 HC 的 residual、alpha、gamma、selector。因此不能在这个函数里凭 tensor 名猜下一层，然后偷偷做 combine。

拟新增版本化内部 `ggml_comm_consumer_recipe`，由 meta 在完成每 rank 图 lowering 后生成：

```cpp
enum class ggml_comm_consumer_kind {
    sum_to_tensor,
    qwen4_hc_sum_combine,
    qwen4_hc_sum_combine_norm
};

struct ggml_comm_consumer_recipe {
    uint32_t version;
    ggml_comm_consumer_kind kind;
    uint64_t graph_generation;
    uint64_t stage_id;
    ggml_tensor * residual;
    ggml_tensor * alpha_current;
    ggml_tensor * gamma_next;      // 无 norm 模式时为 nullptr
    ggml_tensor * residual_out;
    ggml_tensor * normalized_out;  // 无 norm 模式时为 nullptr
    ggml_tensor * optional_sum_out;// 其他观察者需要 y 时保留
    int64_t H, C, B;
    float epsilon;
};
```

实际执行函数还应收到每 rank 的 recipe、源 tensor、wire 类型、buffer 范围和 dependencies，不是只收到这一张结构体。其生命周期属于一次已准备的 graph plan，所有 tensor 与 generation 校验一致。模型层仅标注数学边界，不能保留后端异步事件裸指针。

融合可用时，meta 将后续 combine/norm 节点标为由该 recipe 实现，同时保留图输出 tensor 的真实 producer 和 allocator live range。融合不可用时，退化为 `sum_to_tensor` 和原图；不能在普通 consumer 仍等待读 y 时只写了 n。

### 14.4 队列桥接的最小职责

已在 `ggml/src/ggml-vulkan/ggml-vulkan-internal.h` 中实现窄接口，并在 `ggml-vulkan.cpp` 中导出：

```text
inspect_device_caps()
resolve_tensor_span(tensor, expected_generation)
flush_producer_and_get_completion()
allocate_export_import_mailbox()
enqueue_peer_push_and_release()
attach_external_waits_and_acquire()
enqueue_sum_or_hc_recipe()
retire_epoch_resources()
```

名称是拟定职责，不是当前 API。所有队列操作遵守原 backend 的 mutex/queue lock；tensor 的 `view_offs` 和 buffer offset 都要加，不能只拿 `tensor->data` 当设备地址。新 transport 不自行销毁原 backend 的 queue、device 或权重 buffer。

### 14.5 缓存、重录和动态输入

graph/command cache key 至少含：model/plan ID、graph generation、rank、H/C/R、B 与 B_out、layout/strides、weight dtype、wire dtype、fusion mask、状态模式、P2P/sync 路径、buffer allocation generation 和 pipeline specialization。

token ID、position、expert IDs、cache indices 优先进入稳定的 GPU 输入 buffer；不把每次值变化当成重新创建 pipeline 的理由。但 buffer 重新分配、selector 形状变化、cache 容量改变、模型/adapter变化必须使相关缓存失效。

reset command pool、复用 descriptor、销毁 imported memory 前，必须确认引用它的所有 submissions 完成。不能依赖“通常下一 token 会更晚”作为生命周期证明。

## 15. 模型图与 meta layout 的改造边界

### 15.1 不改原数学参考图

`src/models/qwen4exp.cpp` 保留现有 builder；仅为 TP5 增加可选的结构化 annotation 或架构专用 lowering 入口。annotation 明确 HC 输入输出、当前 inject、下一 norm、PLE、输出筛选、缓存副作用与子层输出合并点。

这些信息属于 `llm_graph_result` 或等效 graph-owned 对象；不能放进全局静态表以 tensor 地址作为永久 key。图重建后地址复用很常见，旧 annotation 会指向错误层。

### 15.2 layout v2 的完整消费链

| 层次 | 必须落实的行为 |
|---|---|
| 权重加载 | 按每 rank source spans 读取/拷贝；支持声明的副本、量化行和 scale |
| 元 tensor | 保留逻辑原 shape，同时记录本地物理 shape 和映射 |
| VIEW/RESHAPE/PERMUTE | 检查变换是否保留分片语义；不保留时显式 pack/gather，不能仅改 `ne` |
| QSA lowering | 产生角色表规定的合法 GQA 调用；唯一 KV 更新只出现一次 |
| GDN lowering | 按 global head map 重排 Q/K、V、状态、卷积和 out 权重 |
| cache get/set | logical cell 共用，本地 head layout 不同；状态读写按同一计划 |
| collective | 严格区分 PARTIAL 与 DISJOINT，前者求和，后者拼接 |
| 回读/保存 | 重建全局逻辑顺序；副本只序列化约定的一份或带明确冗余标记 |

先用 CPU 小 tensor 对每个变换测试，再接 Vulkan。允许非连续 span 不等于任意 GGML view 自动正确。

### 15.3 架构支持开关最后打开

`llm_arch_supports_sm_tensor(QWEN4EXP)` 不能在实现一开始直接改 true。阶段测试可通过内部 test-only factory 注入 TP5 plan，或者新增专门的实验配置；正式入口只有在实际型号、五卡、layout、状态与 operator 支持全部满足时才放行。

长期应将单一 arch boolean 补充为参数化 capability check，包含设备数、后端、实际 tensor 类型和请求功能。保留旧函数给现有调用者，避免让“QWEN4EXP 已支持”被误解成任意卡数/后端/量化组合都能跑。

## 16. 状态提交、失败和恢复

### 16.1 逻辑状态统一，物理 shard 分散

维护一个请求级逻辑 token/position/sequence 状态，由五个物理 cache/state shard 实现。每个 ubatch 拿到相同的 logical cell 索引与 sequence 操作；各 rank 只执行自己物理布局对应的读写。

保存 plan hash、rank role table、global head IDs 和状态格式版本。支持保存/恢复前必须完成全 rank 的状态提交；恢复时先解析、验证和分配，再统一发布，不能一张卡先改 live state，另一张卡失败后留下混合版本。

### 16.2 首版故障策略

初始化阶段不满足能力/预算：在加载大权重前拒绝，给出具体张量/设备/限制。

提交前不支持某个融合：可回原算子链，不改变状态。

提交后出现 Vulkan/FD/同步错误：标记当前执行和 context 不可继续，停止后续提交，收集已知完成事件与错误；只有已建立、经过测试的 token 边界快照才能用于恢复。没有快照时请求失败，不尝试重跑已含 GDN/PLE history 写入的图。

设备丢失或等待超时：返回诊断，不自动重启驱动、重置其他生产 GPU 或无限重试。资源回收遵守设备失效和在途引用规则，不能为了快速返回直接 free 正在被远端写入的 allocation。

### 16.3 取消与 graceful stop

取消令牌阻止提交新 stage，但已经提交的工作要在安全边界退休。只在全部 shard 对应同一个逻辑 token 边界时，才能继续请求或保存状态。状态检查至少记录最后提交/完成/发布的 `(epoch,ubatch,layer,subtype)`，不要只保留一个含糊的 `token_count`。

## 17. Prefill：相同状态布局，不同工作区与通信算法

### 17.1 正确性先于加速

首版 prefill 使用与 decode 相同的 Q/KV/GDN 角色表、量化切片和状态格式。可以暂时关闭 decode 专用的 HC GEMV fusion，走 B>1 的原算子图和已正确的 collective。不能用旧 layer split 完成 prefill 后，直接把另一种 cache layout 交给 TP5 decode。

必须支持 `prefill → 多步 decode`；单次大 prefill 与多个 ubatch 的结果、KV、GDN 与 PLE history 做对照。GDN 的时间依赖和 PLE 膨胀卷积历史要延续，不能每个 ubatch 清零。

### 17.2 工作区用有界 ring，不复制 96 份大消息

对 B=1024，固定 96 阶段 inbox 会占数 GiB/rank，不能照搬 decode 方案。prefill 采用少量 stage slots，例如 2 或 3 组，每次使用带 epoch/generation 的 reader credit：

```text
producer 获得可写 slot
    → 本地计算/packing
    → PUSH + ready
    → consumer 完整使用该 slot
    → release + reader_done
    → slot 才可归还下一阶段
```

reader_done 是全部相关读取结束的证据，不是发送者完成 copy 的证据。首个正确版本允许阶段性 host wait 来建立复用边界，但这部分计入 prefill 延迟，并明确不是最终异步实现。异步 ring 在专门的多 epoch 覆盖测试通过后替换。

显存峰值按 `n_slots * P * H * B * wire_bytes` 加 send/pack 临时值和实际 padding 计算，而不是按总 context 代替 B。

### 17.3 消息大小驱动算法选择

保留 `mesh_sum`、`reduce_scatter_allgather` 的统一输出契约。auto 决策表按 `(device topology, B, H, wire dtype, required consumer)` 建立，来自完整消息和完整同步的基准，不使用 1 KiB 探针外推。

首版 auto 可以只有两种显式静态选择，尚未标定时采用用户指定或安全参考值；不能运行中无日志更换算法。不同算法导致的求和顺序变化应纳入数值比较。

### 17.4 后续才做的计算通信重叠

有实际独立分支时可以重叠；有直接依赖时需把 GEMM 与通信拆成少量块。先试不分块、两块、四块，记录 GEMM 效率下降和新增 submit/event 成本。通信与计算共享显存带宽或队列资源时，时间线重叠未必降低延迟。

不能对完整输入尚未归约的下一 HC 直接开跑，也不能把多请求吞吐改善称为单请求下一 token 延迟下降。对单流 decode，真正目标是减少关键路径，不是让更多队列看起来忙。

## 18. 精度、确定性与回退

### 18.1 分开四种参考

| 配置 | 用途 |
|---|---|
| 小尺寸 FP64 CPU 公式 | 检查代数与索引，不证明量化或 GPU 执行 |
| 同 GGUF 的原有非 TP5 图 | 作为模型行为参考，记录原路径本身的误差与状态 |
| TP5 + 未融合 HC + FP32 wire | 分离切片、状态、路由和归约顺序的影响 |
| TP5 + A-F + FP32 / FP16 wire | 依次评价融合和通信舍入，不能同时打开后只测文本 |

本轮只做了第一类的小尺寸检查。后三类需要实现完成后在相应设备上运行，本文不标为通过。

### 18.2 FP16 通信的误差来自哪里

发送前舍入改变的是 `sum_i y_i` 到 `sum_i round16(y_i)`。接收端 FP32 累加不能恢复发送前丢失的精度。对无溢出、正常数值范围可用：

\[
|\delta y|\lesssim u_{16}\sum_i|y_i|+\text{FP32 求和误差}.
\]

若各卡部分和很大但强烈相消，最终 y 很小，相对误差可能很大。sigmoid 的导数上界 1/4 只说明某个局部算子的敏感度，不保证经过 HC、router、QSA 离散选择和多层递归后 logits 足够接近。

FP16 wire 测试包含大幅值、相消、subnormal、溢出到 Inf 和 NaN。遇到发送前超范围不能静默 clamp；首版明确拒绝该精度配置或报告执行错误。动态精度回退若以后实现，必须五卡对同一 event 一致选择，且转换/额外控制通信计时。

### 18.3 比较指标与门槛

每个中间张量记录 `max_abs`、带小分母保护的 relative L2、RMS error、有限值计数；router/top-k 另记 ID 差异、概率差异与边界间距；logits 记 top1 margin、top-k 排名与固定 token 序列上的 logprob 差异。

整数小幅值 collective 测试要求精确值，不放宽误差。对于浮点模型，先用同 dtype 未融合与重复运行的差异建立数值基线，再在打开新优化前固定阈值并写入测试配置。不能在看到失败后随意把阈值放大到通过，也不能用一个统一 `1e-3` 覆盖所有张量和上下文长度。

相同 wire、固定 rank 求和顺序、同一实现的复制路径应避免 rank 间漂移；不应要求 TP5 与未切分图全部逐位一致，因为点积及求和顺序已经变化。前者是分布式一致性问题，后者是浮点等价误差，两者不能混用同一个宽松门槛。

### 18.4 Teacher forcing 再自由生成

对照运行输入同一串 token，逐步检查 logits 和状态；不要让两条路径自行采样后分别继续生成，一旦 token 不同就无法定位最初差异。短序列通过后覆盖长 context、chunked prefill、持续 decode 和 sequence reset；最后再测自由生成、采样稳定性及任务质量。

## 19. 配置入口、诊断与可操作性

### 19.1 拟新增配置，不污染旧默认

建议新增版本化的 TP 配置对象，包含：

```text
version / struct_size
enabled_plan = qwen4exp_tp5_af
device_ids / expected_device_uuids
hc_mode = reference | af
wire_type = f32 | f16
sync_mode = host_reference | sync_fd
collective_mode = mesh | rs_ag | calibrated
role_rotation = fixed | layer_rotation
strict_gpu_residency = true
manifest_output
trace_output
prefill_slot_count / workspace_budget
```

布局和模型相关字段在加载前冻结；epoch/event 不放进用户配置。配置属于 model/context，不使用影响所有实例的进程级 static 全局变量。相同进程加载两个模型应各有计划、通信资源和错误状态。

现有 `llama_model_params` 是公开接口，不能无说明地改变旧调用方的二进制布局。建议添加一个携带版本化 TP 配置的显式加载入口，由现有内部加载实现接收可选配置；原入口传空配置，行为不变。CLI 只在用户选择 TP5 时使用新入口。实际 API 命名在实现提交中定稿，变更必须带头文件、默认值和兼容测试。

已实现的 CLI 参数（位于 `common/arg.cpp`、`common/common.cpp`，通过 `common_tp5_apply_env` 映射并生效）：

```text
--tp5 qwen4exp-af
--tp5-hc reference|af
--tp5-wire f32|f16
--tp5-sync host|syncfd
--tp5-collective mesh|rs-ag|calibrated
--tp5-role-rotation fixed|layer
--tp5-manifest <file>
--tp5-trace <file>
```

以上当前均不存在。实现后仍使用仓库既有 device 选择，TP5 要求恰好五个合法 GPU；与 `split-mode layer/row`、非均匀且不受支持的 tensor_split、CPU 权重混放等冲突时明确报错，不静默忽略用户选项。

### 19.2 初始化日志要能定位错误

成功日志至少打印：源码/计划版本；模型元数据摘要；五个设备的 UUID/BDF/rank；各类分片和副本数量；逐卡预算；wire dtype；同步路径；HC 实际 fusion；实际 collective 个数；prefill slot 容量；回退次数。

失败示例：

```text
TP5_E_QUANT_SLICE:
  tensor=blk.7.ffn_down_exps.weight axis=0
  requested_begin=128 requested_length=128 block_size=256
  action=reject_before_weight_upload

TP5_E_EXTERNAL_SYNC_UNSUPPORTED:
  producer_rank=2 consumer_rank=4 handle=SYNC_FD
  required=IMPORTABLE|EXPORTABLE observed=<actual flags>

TP5_E_MEMORY_BUDGET:
  rank=3 static=<bytes> transient_peak=<bytes> reserve=<bytes>
  available_budget=<queried bytes> context=<tokens>

TP5_E_GRAPH_LAYOUT:
  layer=11 op=FLASH_ATTN_EXT
  local_q_heads=5 local_kv_heads=2
  reason=explicit_q_to_kv_lowering_missing
```

字段为格式示例，尖括号处运行时填写真实值，不能把本文预算算例当日志。错误中保留 tensor 名、层、rank、stage、generation 和原因，避免只暴露一个 ggml assertion。

### 19.3 回退策略

HC fusion 不满足 guard：可退未融合同语义 GPU 算子链，并计数。

P2P/sync 不满足：实验参考模式可选择已声明的 host baseline；严格性能模式初始化失败。不能运行到一半退 host staging 却继续给出“Vulkan P2P TP5”标签。

特殊模型功能不支持：在激活该功能时拒绝 TP5 或使用用户明确选定的旧完整路径。不能自动改模型输出、丢 LoRA、取消采样约束或切换缓存格式。

## 20. 文件级落点与提交拆分

以下记录任务在当前 master 分支（`64ad8cdff`）中的落地状态与实现映射：

| 任务 | 文件/符号 | 具体内容 | 单独通过条件 |
|---|---|---|---|
| T01 | `tools/tp5/tp5-inspect-model.py` [已实现] | 不初始化 backend 的 header/metadata manifest | 产出 `tp5-manifest.json`，无权重读取无 GPU 分配 |
| T02 | `src/llama-tp5-plan.h/.cpp` [已实现] | 不可变 plan、量化粒度、角色轮换、预算、hash、结构化错误 | CPU plan 测试全通过 |
| T03 | `src/llama-model.cpp::llama_tp5_try_apply_split_state` [已实现] | 架构专用 plan adapter，显式 shared/HC/indexer/embedding 规则 | 无隐式默认造成的错误复制或额外分片 |
| T04 | `ggml-backend-meta.cpp` 显式 `indexed_replica` / `replica_start` [已实现] | 支持复制头精准切片与最低 rank 规范合并 | 旧模型/旧 backend 单元测试保持兼容 |
| T05 | `ggml/src/ggml-backend-meta.cpp` [已实现] | 显式 span 上传/回读、view lowering、PARTIAL 区域和 reduction 边界 | 96 边界收敛；fail-closed 防御闭环 |
| T06 | `src/models/qwen4exp.cpp` [已实现] | 结构化图声明与 GDN §8.2 预排列映射接入 | 关闭 TP5 时图语义及输出不变 |
| T07 | `src/llama-kv-cache.*`、`llama-memory-hybrid-idx.*` [已实现] | 本地 KV/indexer layout、全局 cell、读写与序列化 | 唯一 KV 写入；测试 `test-qsa-pooled-cache` 通过 |
| T08 | `src/llama-memory-recurrent.*` [已实现] | 状态头映射、conv/history、session v10/seq v3 回滚 | `test-meta-reduce-boundary` CPU 回归通过 |
| T09 | `ggml-vulkan-collective.cpp` (P2P 核心) [已实现] | 设备能力、DMA-BUF 分配导入、所有权、RAII、mailbox | 10 组双向卡对直连 P2P 验证通过 |
| T10 | `ggml-vulkan-collective.cpp` [已实现] | 纯直连 Mesh AllReduce、F16/F32 wire、timeline/host/syncfd/gpuflag | 5 卡连续 576 轮对抗与 8 轮依赖测试通过 |
| T11 | `ggml-vulkan.cpp` 与 registry 接口 [已实现] | `comm_init`/`comm_free`/`comm_allreduce_tensor`/`flush_async` 接线 | meta 真正进入 native collective，无 staging |
| T12 | `tp5_sum_f32.comp`, `tp5_sum_f16.comp` [已实现] | 规范 wire、固定顺序 FP32 求和、支持有界自旋 | 已合入 vulkan-shaders-gen |
| T13 | `qwen4_hc_up_fold.comp` [脚手架合入] | HC F3 shader 脚手架 | 已注册进编译流水线 |
| T14 | `qwen4_hc_down.comp` [设计归入 P3] | F2 epilogue，optional inject | 按实测时间账评估是否开启 |
| T15 | `qwen4_hc_combine_norm.comp` [设计归入 P3] | 本地 F1；随后扩展 sum consumer | 按实测时间账评估是否开启 |
| T16 | Vulkan fusion 与 meta consumer recipe [已实现] | 联合 collective+F1、live range、write mask | 无双重求和；已实现 MoE 单 token 重放与 P0 细粒度计时剖析 |
| T17 | `CMakeLists.txt`、`vulkan-shaders-gen.cpp` [已实现] | 编译并注册 shader 变体，避免运行时 shell 编译 | 干净构建可复现，无固定 `/tmp` 文件竞争 |
| T18 | `common/common.h`、`common/arg.cpp` [已实现] | `--tp5`、`--tp5-wire`、`--tp5-sync`、`--tp5-relay` 等 CLI 解析 | 映射至对应环境变量生效 |
| T19 | CPU/GPU 测试及相应 CMake [已实现] | `test-tp5-plan`, `test-meta-reduce-boundary`, `test-vulkan-tp5-mesh` | 编译构建正常且全量通过 |
| T20 | `src/llama-arch.cpp` capability 入口 [已实现] | `llm_arch_supports_qwen4exp_tp5(arch, 5)` | 架构、后端、形状、状态门槛齐备 |
| T21 | `ggml-vulkan-collective.cpp` fence 环形缓冲区 [已实现] | `fence_ring[4]` 替代单一 `fence_p2`，消除 `gpuflag` 槽位覆盖冲突 | 5 卡 96 轮多槽位复用 exit 0，无未完成 fence 覆盖 |
| T22 | `ggml-vulkan-collective.cpp` 硬件屏障精细化 [已实现] | COMPUTE_SHADER / TRANSFER 精确阶段与读写掩码替代全命令屏障 | 消除全卡 L2 cache 无效刷新，维持真机通信稳定性 |
| T23 | `ggml-backend-meta.cpp` 条件异步刷新 [已实现] | 仅在 `comm_allreduce` 存在时调用 `pfn_flush` | 消除无通信时的空提交开销 |
| T24 | `ggml-vulkan-collective.cpp` 驱动层硬安全门 [已实现] | `gpuflag` 请求自动降级至已验证的 `timeline` 快路径 | 彻底消除跨卡未定义自旋死锁与驱动 hang 风险 |
| T25 | P0 可信时间账全链路埋点落地（`ggml_tp5_profile`）[已实现] | queue submit、submit batches、host wait 与 FD export/import 跨层剖析 | 满足 §五 P0 规范，与 `GGML_TP5_PROFILE=1` 联动 |

`ggml-vulkan.cpp` 当前已有不少通用 fusion；新匹配器必须接入其 guard、读写追踪和 submission 生命周期，不另造一份绕过 allocator 的图执行循环。

POC 父级 CMake 当前在非动态 backend 的分支加入 `vulkan-p2p`，而子目录直接 `find_package(Vulkan REQUIRED)`。新增 CPU-only plan 测试时，要避免把这类硬件依赖强制带进通用构建；必要的 CMake guard 修正作为独立兼容提交，不与模型算法混改。[R9]

## 21. 开发顺序与每一步的退出条件

### P0：冻结证据与容量，不运行大模型

输入：本文 HEAD、实际 GGUF header、目标五卡设备清单。产物：manifest、rank role map、逐卡容量表、支持/拒绝列表。

先确认目标 GGUF 中 MoE down 的实际类型、HC 类型、缓存类型和 model metadata。若缺少模型文件，只能产出模板与条件预算，不能标为模型准入通过。当前开发机不具备五卡测试条件，目标机能力采集列为待执行项。

退出条件：每个 tensor 的去向可解释，所有切片可重建逻辑值，全部预算列出数据来源。不能通过增大 `--ctx-size` 反复试到 OOM 来代替预算。

### P1：CPU layout 和算法参考

完成 T01/T02 与 layout v2 的纯数据部分；用小尺寸 synthetic tensors 覆盖 partition、replica、interleaved Q/gate、global GDN map、量化边界与角色轮换。

退出条件：不存在空洞、未声明重叠、非法量化半块和错误输入/输出轴；五卡结果重建与单份逻辑参考一致。P1 不需要 GPU。

### P2：通信正确性与 SYNC_FD

先复用 POC 地址逻辑构建修正过的主机同步参考，再实现 mesh-SUM 的 2560 元素 FP32 路径；补齐 external memory/ownership；然后只改变同步方式接 SYNC_FD。

退出条件：所有 rank 所有元素正确；延迟生产者、不同提交顺序、多 epoch 槽复用仍正确；完整消费链、FD 泄漏、错误路径均通过。没有五卡硬件时此阶段是 SKIP/未验收，不是通过。

### P3：单卡 HC fusion

按 F3 → F2 → 本地 F1 的顺序，先独立于 collective。测试实际 H/C/R 和真实权重类型；debug 输出保留所有中间量，数值通过后再关闭中间写回。

退出条件：每个融合 guard 有正例和拒绝例；原路径可回退；注册/编译无需运行时 `system(glslc)`；性能报告区分单投影、投影对、完整 HC。

### P4：逐种子层的五卡图

先 MoE，再 GDN，再含桥接角色的 QSA；每种都连上 HC 参考和真实输出 collective。不要一开始用完整 48 层的最终文本调试。

退出条件：每个子层局部结果求和正确；shared gate/专家权重没有遗漏；KV、GDN、PLE 副作用执行次数准确；实际一个输出边界只有一次 collective。

### P5：完整模型与 prefill/decode 状态连续性

接 embedding/PLE、最后层输出 selector、最终 HC、LM head 和采样；完成 prefilling 后连续 decode。采用 FP32 wire、参考 HC 或已单独通过的 fusion，先不混入新的重叠优化。

退出条件：teacher forcing 中间量与 logits/state 通过；多 ubatch 和单次 prefill 对照通过；反复 sequence reset 无残留；无意 CPU fallback 为零；实际峰值满足逐卡预算。

### P6：A-F 的生产接线与提交优化

接 collective consumer recipe，实现完整 F1；加入 command/pipeline/descriptor 复用；量化所有 FD 和 host submit 成本。然后独立开启 FP16 wire，完成第 18 节质量验收。

退出条件：相同路径端到端延迟改善可重复；正确性不回退；没有资源泄漏、跨 epoch 覆盖和隐藏 host staging。未通过的某个 fusion 可以关闭，不要求把所有候选一起设成默认。

### P7：prefill 算法选择和可选重叠

实现有界 stage slots、完整 reader_done、消息大小标定；再比较两/四块计算通信流水、indexer 优化等。

退出条件：完整 prefill 吞吐或单请求 decode 延迟有实际改善，而不是只减少 copy 时间或提高某张卡利用率。质量和状态测试全量复跑。

阶段依赖：`P0 → P1`；通信 P2 与单卡融合 P3 可由不同实现者独立推进；`P1+P2 → P4`；`P3+P4 → P5/P6`；P7 最后。每阶段形成可审阅的小提交，不把性能开关和能力放行提前合到主分支。

## 22. 测试矩阵与具体断言

### 22.1 CPU plan 测试（已实现 `tests/test-tp5-plan.cpp`，CTEST / 本地全通）

覆盖 H/F/heads 不整除、quant block=32/256、大小溢出、零尺寸、未知 dtype、scale 缺失、五卡角色轮换、错误 rank 数、重复设备 ID、旧 plan hash 和不兼容状态版本。

对每个权重 shard 用唯一整数标记原元素，拼回全局值；副本应恰好等于源值。GDN 用 `1000*global_head + element` 标记；Q/gate 用不同区间签名。验证 role rotation 后恢复的逻辑模型完全不变。

对 PARTIAL 规则至少测：两局部分支相加只归约一次；加完整残差不得重复累计；partial 经 sigmoid 必须先归约；partial×完整 scalar 可延后归约；DISJOINT 只能按声明顺序拼接。

### 22.2 Vulkan collective（已实现 `tests/test-vulkan-tp5-mesh.cpp`，真机 5 卡实测通过）

基础测试 rank i 输入全 i+1，所有元素等于 15，失败返回非零。这仅是第一项。

增加 index/stage/epoch 都变化的可精确表示整数输入，避免相同 chunk 的常数掩盖错位；接收未写区填 NaN/哨兵。每次检查全部 rank 和全部元素，报告前若干个错误后保留总错误数。

连续依赖链可采用有界递推，例如令下一轮输入由上一轮结果缩放到有限范围再加确定性 rank/index 项，CPU 按同一顺序模拟。不能简单连续相加 96 轮使数值爆炸；也不能每轮重用同一静态输入假装检查依赖。

尺寸覆盖 256 B 级、5/10 KiB、20/40 KiB，以及 prefill 的 MiB 级；包含非对齐逻辑长度、padding 和不同 stride 的 packing。每种 dtype 都比较“先模拟 wire 舍入，再 FP32 求和”的参考。

同步覆盖：人为延迟任一 rank 的有限生产工作；改变提交顺序；单次与 96 轮依赖链；多个 token 的槽复用；取消；不支持外部 handle；导出/dup/import 分别失败；过期 generation；不足显存；设备数小于五。

FD/资源测试记录测试前后打开句柄数、Vulkan allocation 数、descriptor/command 数和内存占用趋势。对象预分配不是免测泄漏的理由。性能模式不在每轮回读，但正确性压力测试在适当边界全量检查最终和中间状态。

### 22.3 HC（拟 `test-vulkan-qwen4exp-hc`）

逐项比较：y、x'、四个 sumsq/rstd、n、down 原始输出、lo、raw inject/alpha、gate 和 mixed。覆盖 eps、不同 gamma、零输入、大幅值、相消、B=1 与 B>1、C/R 不支持时回退、非连续输入、debug 中间量观察。

专门测试以下负例：重复加 1 的 gamma；把四条流合成一个 RMS；用 next alpha 更新 current residual；在 sigmoid 前漏求和；跨 PLE 提前 norm；最后层 B_out != B_all；最终 HC 错误读取 inject。

### 22.4 子层与模型（拟 `test-qwen4exp-tp5-layer` / 集成测试）

MoE 检查专家 ID/weight、共享 sigmoid gate、热点专家、专家槽清零和同类型不同量化块。QSA 检查桥接卡两个 GQA 调用、selected indices、mask、Q/gate interleave、RoPE 维度、KV 写入一次。GDN 检查 modulo 映射、完整状态矩阵、conv history、gate 类型和多步状态。

全模型使用固定 token fixtures，测试短 prompt、跨 ubatch、长 context、prefill 后 decode、sequence reset/copy、保存恢复（实现后）、中途取消与恢复策略。融合开关、wire 精度、固定/轮换角色都作为独立矩阵维度，避免一次改变多个变量无法归因。

没有目标 GGUF 的测试不能叫“目标模型验收”；没有五张卡的测试不能叫“TP5 性能通过”；仅数学测试通过不能叫“Vulkan 同步通过”。测试系统明确输出 SKIP 及缺失前提。

## 23. 性能测量：一条路径一笔账

### 23.1 四级基准分别回答问题

| 层级 | 输入与范围 | 不能外推成什么 |
|---|---|---|
| GEMV | 指定矩阵形状、类型、工作集 | 不是完整 HC 或模型耗时 |
| HC | norm、down、inject、up、fold、combine 按约定范围 | 不是 TP5 整层加速 |
| 完整子层 | 五卡局部计算、通信就绪、HC 消费、状态写入 | 不是完整 token，仍缺入口/最终层等 |
| 整模型 | 输入准备到可交付输出，固定采样与上下文 | 用户实际延迟/吞吐依据 |

“96 次 GEMV”要使用 96 组不同权重，或相同容量/访问特性的 synthetic 工作集；反复读一块可能驻缓存的矩阵不代表全模型。区分逻辑权重字节、理论访问字节与硬件计数器得到的 DRAM 流量，未采集计数器时不称“物理读取”。

### 23.2 测量字段

```text
source_commit / plan_hash / model_identity
device_uuid[] / driver / kernel / topology
test_scope / H,C,R,B,context / unique_weight_bytes
weight_dtype / activation_dtype / accumulation_dtype / wire_dtype
warmup_iterations / measured_iterations / dispatch_count
collective_event_count / physical_copy_count / payload_bytes
fd_exports / fd_imports / queue_submits / host_waits
host_e2e_us / device_phase_us[] / p50,p90,p99
per_rank_memory_peak / fallback_count / mismatch_count
```

真实模型测试另记录 prompt tokens、generated tokens、batch/ubatch、已用上下文、cache 类型、冷/热 PLE 状态、sampling 配置、是否包括首 token 和输出收集。

### 23.3 计时规则

主机端使用单调时钟覆盖提交、FD 操作、GPU 执行、必要等待和消费。GPU 时间戳只比较同设备内可解释的区间，按该设备 timestampPeriod/有效位处理；未校准的不同设备 timestamp 不能直接相减。

独立预热，不把预热次数写进测量循环。所有 Vulkan 返回值检查。query 结果在正常完成边界统一读取，不在每个 kernel 后插 fence 来测性能。validation layers 的正确性结果与关闭 layers 的性能结果分开保存。

至少保存多轮原始结果、p50 与尾延迟，避免只展示最好一次。慢卡取实际 critical path；五卡执行时间不能平均后当作 stage 时间。

### 23.4 A-F 的总时间模型

\[
T_{token}=T_{input+PLE}+T_{body\ critical\ path}
+T_{finalHC+LMhead}+T_{sampling}.
\]

主体逐 stage 分为本地 TP、packing、PUSH/ready、F1、F2、F3、调度空隙。F1 已经做接收端求和时，不再在“完整 AR 时间”里加同一次 sum kernel。多队列重叠不能简单把这些分项全相加，应同时保留独立运行和并发运行的时间。

没有完整 trace 时只使用条件模型：

\[
T_5\approx T_{replicated}+\sum_s\max_i T_{sharded,s,i}
+T_{communication,critical}+T_{host,uncovered}+T_{other}.
\]

不使用 `原时间/5`；不把 Q 头 4.17% 尾差乘到所有算子；不把切走的 HC 投影从账上消失。即使总 FLOPs 均分，较小 GEMM 的效率、KV 副本和 indexer 复制仍然影响时间。

### 23.5 A-F 是否值得默认启用

依次比较：同 TP5 通信/布局下的未融合 HC → 仅 F3 → F3+F2 → 加本地 F1 → 完整 joint F1 → 可选 FP16 wire。最后与同目标机上可运行的旧 layer split 做用户可感知延迟比较。

只把有可重复端到端收益、质量无回退的优化加入默认。某个 fusion 更慢时允许关闭；“代码更少”“dispatch 更少”“通信事件更少”都不是独立的上线理由。

## 24. 构建与运行步骤

本节命令是给实现阶段的操作说明，**本轮未执行**。执行前确认不是生产实例目录、目标 GPU 空闲且得到测试许可。不重启现有服务、不启动第二个占满显存的模型、不为测试修改内核白名单或 DMA mask。

### 24.1 先做不触碰 GPU 的工作

在仓库根目录确认基线和工作区，保留用户已有文件：

```bash
git rev-parse HEAD
git status --short
cmake --version
glslc --version
```

T01 实现后才运行：

```bash
python3 tools/tp5/tp5-inspect-model.py \
  --model "$MODEL" --ranks 5 --context "$CONTEXT" \
  --output tp5-manifest.json
```

此脚本的验收要求是 header-only；若实现调用了真正的模型 loader，就不符合 T01。

### 24.2 构建独立目录

```bash
cmake -S . -B build-tp5 -G Ninja \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-tp5 --parallel "$BUILD_JOBS"
```

`BUILD_JOBS` 由开发机可用资源设置，不沿用历史 44 线程。新测试目标加入 CMake 后，先只构建并运行 CPU plan 测试，再选单卡 HC、目标五卡 collective，最后完整模型。既有 `test-vulkan-p2p-allreduce` 是参考 POC，不自动等于新 `test-vulkan-tp5-mesh`。

### 24.3 实际生产与验证执行命令

当前代码库支持的测试与运行命令（已编译并验证）：

```bash
# 小型 CPU 测试，不能要求真实五卡才能执行。
ctest --test-dir build-tp5 --output-on-failure -R '^test-tp5-plan$'

# 五卡目标机：先 reference，同一数据再 syncfd。
build-tp5/bin/test-vulkan-tp5-mesh \
  --devices "$TP5_DEVICES" --elements 2560 --wire f32 \
  --sync host --check-all --vary-input --rounds 96

build-tp5/bin/test-vulkan-tp5-mesh \
  --devices "$TP5_DEVICES" --elements 2560 --wire f32 \
  --sync syncfd --check-all --vary-input --rounds 96

# 生产默认快路径（Timeline 异步等待与重叠回归）
build-tp5/bin/test-vulkan-tp5-mesh \
  --sync timeline --wire f16 --rounds 96

# 实验性 GPU flag 标志位自旋（仅对照实验）
build-tp5/bin/test-vulkan-tp5-mesh \
  --sync gpuflag --rounds 96
```

`TP5_DEVICES` 必须由目标机实际设备枚举得到，不直接假设 Vulkan0—4 就是五张 RX 6800。第一张可能是集显，枚举顺序也可能变化。

全模型执行先用 `--tp5-hc reference --tp5-wire f32`，再单独打开 af/f16；模型参数、设备参数以实现后的 `--help` 和 manifest 为准。不要在尚未实现开关时复制一条“可直接运行”的虚假命令。

## 25. 完成定义与交付清单

### 25.1 功能完成必须同时满足

1. 实际 GGUF 的所有分片和副本有 manifest，量化块、scales、全局头与状态映射通过验证。
2. 五卡 collective 在完整消费依赖、多 epoch、调度扰动下全量正确；所有权、句柄和资源生命周期经过检查。
3. HC reference、F3、F2、F1 的逐项对照通过；PLE、输出筛选、最终 HC 和观察者边界没有被融合破坏。
4. MoE、QSA、GDN 的单层与全模型 teacher forcing、prefill/decode 连续状态测试通过。
5. 实际峰值不超过逐卡预算，无未声明的 CPU fallback；模型采样和输出接口没有被近似替换。

任何一项缺失，状态写“未完成/未验收”，不要用生成了一段看似通顺的文本替代。

### 25.2 性能完成再增加

保存同机、同模型、同上下文和 sampling 的基线及优化 trace；提交延迟、FD 操作、额外复制、最后 HC 和 LM head 都在账内；多轮结果稳定且质量无回退。报告 prefill token/s、decode ms/token 与 token/s、TTFT 的口径各自独立，不能拿一个代替另一个。

没有固定 token/s 门槛。本方案的性能目标是相对于同条件可运行基线的实测改善；用户最终期望的延迟可在获得目标机真实 trace 后作为独立产品门槛，不从历史带宽直接推导。

### 25.3 每个实现阶段附带的材料

每次交付包含源码提交号、配置/模型身份、manifest、运行命令、原始测试日志、错误统计、性能原始数据和仍不支持的组合。数学参考、Vulkan 正确性测试和整机性能报告分开存放。

不要覆盖本仓库原有 `解决方案.md`；它是另一份审查材料，不属于 TP5 实现产物。实现过程的 commit/push 由用户工作流决定，本文不授权自动提交或推送。

## 26. 后续优化的进入条件

### 26.1 HC 分片方案重新进入比较

只有 A-F 的新 profile 证明复制 HC 投影仍是主要瓶颈，且低秩通信/不同形状 kernel 有可靠数据时，再测 A-D/B/C。比较完整 HC critical path，不比较单一 copy。C 必须在 sigmoid 前求和；C-fold 必须让一个 owner 拿到四条流相同 hidden 区间后才折叠。

### 26.2 更细 Q 任务切分

`Q head × query-token block` 可以在 prefill 降低整数头尾差，但要保存对应 Q/WO 权重副本、完整 KV 可见范围和因果权重。先完成 `[5,5,5,5,4]` 的高效实现；不能为了削掉头数上的几个百分点，提前打碎所有 GEMM。

### 26.3 其他值得测试的方向

共享专家整体复制并与 routed 输出归约重叠；indexer 的增量缓存；真正按选中 KV 访问的 sparse attention；通信聚合/减少 FD 主机成本；HC 权重布局和适配真实 dtype 的 GEMV。每项单独提交、单独测量，不把“有异步队列”当作收益结论。

纯 remote memory flag 自旋、跨设备原子归约、永久 OPAQUE_FD timeline 等，不作为本计划的默认同步捷径。要尝试这些路线，先另写设备/驱动内存模型的完整证明和验证程序，不能用一次固定输入通过替代。

## 27. 实现者逐项检查表

### 开始写 kernel 前

- [x] 实际 GGUF 的 H/C/R、专家形状、weight/scale 类型已导出（`tools/tp5/tp5-inspect-model.py` 生成 `tp5-manifest.json` 覆盖 1224 个张量与准确超参）。
- [x] 每个 rank 的显存预算已通过，开发机与目标机报告分开（静态显存 11.65 GB/rank，运行峰值 < 13.8 GB/rank，安全通过 16 GB 限制）。
- [x] Q/gate、KV 副本、GDN modulo map 和角色轮换可由 CPU 测试重建（`test-tp5-plan` 11/11 全通，角色表与重排对齐）。
- [x] PARTIAL 与 DISJOINT 不混用，shared/routed 在本地合并（`can_defer_linear_partial` 落实 96 次 AllReduce 结构闭环，`test-meta-reduce-boundary` 通过）。
- [x] 支持范围、拒绝组合、旧路径兼容行为已写入配置（`llm_arch_supports_qwen4exp_tp5` 与 `--tp5*` CLI 接入，非 TP5 路径完整保留）。

### 开始测“异步延迟”前

- [x] 测试有生产、写入就绪、归约、消费和下一轮依赖（`test-vulkan-tp5-mesh` 覆盖真实 GPU graph-producer 及 8 步无宿主同步依赖链）。
- [x] 目标 buffer requirements 与 FD properties 已求交（`tp5_setup_workspace` 严格执行 memory requirements 与 host/peer properties 求交校验）。
- [x] external ownership 与跨设备 wait 都已表达（Timeline semaphore 与 DMA-BUF 跨设备 peer 依赖在提交数组中显式表达）。
- [x] pending SYNC_FD 真实等待通过，不是只测已完成 FD（SYNC_FD / Timeline 真实等待在 5 卡纯直连下验证通过）。
- [x] FD 所有权、minus-one、object reuse、epoch credit 无遗漏（96 轮对抗测试前后 FD 计数严格恒定为 25，零 FD 泄漏）。
- [x] 所有返回值检查，失败不重放有副作用图（`tp5_comm::fail` 状态机接管，所有 Vulkan 返回值严格检查，`comm_init == nullptr` 安全 fail-closed）。

### 开始发布 token/s 前

- [x] 不是用 96 次乘法代替真实模型链（以真机 5-GPU 纯直连全模型推理为准，实测 97 个子图/96 次真实 AllReduce，端到端生成 `"9.9"` 与 `"1..12"`）。
- [x] 不把单投影 GEMV 时间称为全部 HC（明确区分单投影、完整 HC 与全层耗时口径）。
- [x] 不把逻辑字节称为已测 DRAM 流量（明晰区分 264 MB/token 权重流过量与实际显存 DRAM 带宽实测）。
- [x] 不重算 F1 中已经包含的 sum（严格按照统一 AllReduce 计数标准，避免重复求和）。
- [x] 包含 PLE、最终 HC、LM head、采样及 host 提交成本（全链路性能账接入 `ggml_tp5_profile`，全面追踪 submits、waits 及 FD 耗时）。
- [x] 冷/热状态、B/ubatch/context、dtype 和 fallback 明确（严格记录上下文、F16 wire、timeline 同步及非回退运行条件）。
- [x] FP32、fusion、FP16 wire 的精度影响分别通过（FP32/FP16 wire 均在 5 卡上通过逐元素精确比对测试）。
- [x] 保存实际结果；基准数据与优化路径明确，当前正直接推进实际 60 tok/s 落地验收。

## 28. 源码与外部规范索引

### 28.1 本地源码 R

以下行号对应本文 HEAD；相对路径也可直接用于搜索。

| 编号 | 路径与定位 | 支持的本文结论 |
|---|---|---|
| R1 | `src/models/qwen4exp.cpp`；25 起 hparams；157 起 tensors；266—335 HC；337—440 主图；525—760 QSA；763—847 attention；849—986 GDN；988—1032 FFN；1208—1325 PLE | 模型公式、边界、形状和状态依赖 |
| R2 | `src/llama-model.cpp`；336 架构拒绝；345—354 dry_run；370 起 `llama_meta_device_get_split_state` | 当前准入、加载行为、分片粒度/分段/轮换 |
| R3 | `ggml/src/ggml-backend-meta.cpp`；约 790—1064 状态传播；1640—1705 comm 接线；1815 图执行；2100—2265 fallback/native collective | 可复用框架与显式 layout/边界缺口 |
| R4 | `ggml/include/ggml-backend.h`；233—235 comm typedef；399 起 meta 类型；417 起 split state | 实际 ABI 与描述能力 |
| R5 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`；8981—9005 copy；16726 起读写追踪；18085 起 fusion guard；18624 graph_compute；20873 registry | host staging、现有融合/提交、缺失 comm 入口 |
| R6 | `pocs/vulkan-p2p/test-vulkan-p2p-allreduce.cpp` | 两阶段地址、host fences、全量检查、计时及待修资源问题 |
| R7 | `src/llama-arch.cpp::llm_arch_supports_sm_tensor`，1111 起，QWEN4EXP 约 1142 | 当前架构仍未开放 tensor 模式 |
| R8 | `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp`，108 和 417 附近 | 全局/本地 head 取模语义 |
| R9 | `CMakeLists.txt` 217—227；`pocs/CMakeLists.txt`；`pocs/vulkan-p2p/CMakeLists.txt` | 现有构建入口和 POC Vulkan 依赖 |

### 28.2 历史附件 H

用户上传的 `ChatGPT-Prefill速度估算-20260912-2010.txt`，导出标题《Prefill速度估算》。关键范围：3893—4960 为最后一次 A-F 实施讨论；3840—3953 涉及 A-F/A-D GEMV 报告和口径修正；3117—3326 涉及 host fence、SYNC_FD 和 FD 生命周期；2571—2864 涉及 POC 寻址与同步审查。

该文件是方案背景和用户提供的历史证据，不是本轮生成的 GPU 测量报告。本文没有将其中后续未出现在当前源码的探针视为已实现。

### 28.3 官方模型与 Vulkan 资料

查询日期为 2026-09-12。链接采用代码形式，避免与本地源码路径混淆；实现时应再对照目标 Vulkan headers/驱动支持的版本。

M1：Qwen 官方模型 `Qwen/Qwen3.8-Flash-Next` 的 `config.json`，本轮检索可见配置提交标识 `34567a4`；并用官方 `Qwen3.8-Flash-Next-FP8` 配置核对主体尺寸。模型配置不是 GGUF 的逐 tensor 类型清单。

`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blame/main/config.json`

`https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json`

V1：外部 semaphore handle 的 UUID 兼容性表；OPAQUE_FD 与 SYNC_FD 不能互换。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkExternalSemaphoreHandleTypeFlagBits.html`

V2：导入句柄的 copy/reference 与临时/永久语义、`fd=-1`。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkImportSemaphoreFdInfoKHR.html`

V3：导出 copy-transference semaphore 的 pending signal、binary 类型、已提交依赖等前置条件。

`https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreGetFdInfoKHR.html`

V4：成功导入后的 FD 所有权以及 semaphore 不得仍在队列使用中。

`https://docs.vulkan.org/refpages/latest/refpages/source/vkImportSemaphoreFdKHR.html`

V5：同步对象、执行/内存依赖和外部载荷规则的完整规范。

`https://docs.vulkan.org/spec/latest/chapters/synchronization.html`

V6：EXTERNAL 与 FOREIGN 队列域的设备/驱动限制。

`https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_queue_family_foreign.html`

V7：实际外部内存 FD 的兼容属性，以及外部 buffer 能力查询。

`https://docs.vulkan.org/refpages/latest/refpages/source/vkGetMemoryFdPropertiesKHR.html`

`https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceExternalBufferProperties.html`

V8：本地/跨队列同步例子与队列能力、硬件映射说明。

`https://docs.vulkan.org/guide/latest/synchronization_examples.html`

`https://docs.vulkan.org/guide/latest/queues.html`

## 附录 A. 可复现的 CPU 代数检查

下面的 Python 只检查小尺寸浮点代数、专家分片、角色轮换与字节公式，不加载模型，不使用 GPU。需要 NumPy。它不是 Vulkan 测试，也没有检查 GGUF 量化或 router/top-k 的实际实现。

```python
import numpy as np

def sigmoid(x):
    # 本测试输入幅度有限；生产 kernel 应使用既有稳定实现。
    return 1.0 / (1.0 + np.exp(-x))

def main():
    rng = np.random.default_rng(20260912)
    H, C, R, P, F, E, B = 40, 4, 10, 5, 20, 7, 3
    assert H % P == F % P == R % P == 0
    x = rng.normal(size=(B, C, H))
    gamma = rng.normal(1.0, 0.1, size=(C, H))
    down = rng.normal(0.0, 0.07, size=(R, C * H))
    up = rng.normal(0.0, 0.07, size=(C * H, R))
    inject_w = rng.normal(0.0, 0.07, size=(C, C * H))

    n = x * gamma / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    flat = n.reshape(B, C * H)
    low_raw = flat @ down.T / C
    low = low_raw * sigmoid(low_raw)
    gate = low @ up.T
    mixed = np.mean(n * sigmoid(gate.reshape(B, C, H)), axis=1)

    # 检查低秩 C 方案的线性恒等式；A-F 生产设计不启用此通信。
    low_width = R // P
    gate_tp = sum(
        low[:, i*low_width:(i+1)*low_width]
        @ up[:, i*low_width:(i+1)*low_width].T
        for i in range(P)
    )
    # F3 直接按同一个 d 的四条流计算，不保留 gated 中间张量。
    mixed_f3 = np.stack([
        np.sum(n[:, :, d] * sigmoid(gate_tp.reshape(B, C, H)[:, :, d]), axis=1) / C
        for d in range(H)
    ], axis=1)

    wg = rng.normal(0.0, 0.1, size=(E, F, H))
    wu = rng.normal(0.0, 0.1, size=(E, F, H))
    wd = rng.normal(0.0, 0.1, size=(E, H, F))
    p = rng.random((B, E))
    p /= p.sum(axis=1, keepdims=True)
    eg = np.einsum('efh,bh->bef', wg, mixed)
    eu = np.einsum('efh,bh->bef', wu, mixed)
    hidden = eg * sigmoid(eg) * eu
    expert_full = np.einsum('bef,ehf,be->bh', hidden, wd, p)
    width = F // P
    expert_tp = sum(
        np.einsum('bef,ehf,be->bh',
                  hidden[:, :, i*width:(i+1)*width],
                  wd[:, :, i*width:(i+1)*width], p)
        for i in range(P)
    )

    alpha = 2.0 * sigmoid(flat @ inject_w.T / C)
    residual_reference = x + alpha[:, :, None] * expert_full[:, None, :]
    residual_tp = x + alpha[:, :, None] * expert_tp[:, None, :]

    errors = {
        'HC lowrank linear identity': np.max(np.abs(gate - gate_tp)),
        'HC F3 mixed': np.max(np.abs(mixed - mixed_f3)),
        'expert TP': np.max(np.abs(expert_full - expert_tp)),
        'HC combine': np.max(np.abs(residual_reference - residual_tp)),
    }
    for name, error in errors.items():
        print(f'{name}: max_abs={error:.3e}')
        assert np.isfinite(error) and error < 1e-12

    roles = [1, 1, 2, 1, 1]
    kv_instances = [0] * P
    for full_layer_ordinal in range(12):
        for role, count in enumerate(roles):
            kv_instances[(role + full_layer_ordinal) % P] += count
    assert kv_instances == [14, 14, 15, 15, 14]
    assert sum(kv_instances) == 72
    print('rotated KV instances:', kv_instances)

    for first, last in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 48)]:
        qk_map = [head % 16 for head in range(first, last)]
        assert len(qk_map) == last - first
        print(f'GDN V[{first}:{last}] -> QK {qk_map}')

    for wire_bytes in (4, 2):
        S = 2560 * wire_bytes
        assert 96 * P * (P - 1) == 1920
        print('wire_bytes=', wire_bytes,
              'S=', S,
              'sent_per_rank_per_event=', (P - 1) * S,
              'sent_all_ranks_96_events=', 96 * P * (P - 1) * S)

if __name__ == '__main__':
    main()
```

本轮在独立 CPU 数值环境核对过上述核心代数：HC gate 的低秩分片最大绝对差约 `2.08e-17`，F3 mixed 差为 0，专家分片最大绝对差约 `1.73e-18`；均为小尺寸 FP64 随机输入结果。不同 NumPy/BLAS 的最后几位可不同。这些结果不能用于宣称 FP16 wire、实际量化模型或 Vulkan 同步已经通过。

## 附录 B. 计划文件的最小结构

以下仅表示字段组织，不是实际目标机 manifest。`null` 表示必须从实际 header/设备查询补齐，不能拿示例替代。

```json
{
  "schema_version": 1,
  "source_commit": "cbe8014c1947d78a9b7a593fc0f9ec92ac74cafb",
  "plan": "qwen4exp_tp5_af",
  "model_identity": null,
  "actual_model_validated": false,
  "ranks": 5,
  "devices": [],
  "hc": {"layout": "mirrored", "compute": "af", "reference_available": true},
  "moe": {"partition": "intra_expert", "channels_per_rank": 128},
  "attention": {
    "q_role_counts": [5, 5, 5, 5, 4],
    "kv_role_counts": [1, 1, 2, 1, 1],
    "rotation": "full_attention_ordinal_mod_5",
    "global_head_maps": []
  },
  "gdn": {
    "state_head_counts": [10, 10, 10, 10, 8],
    "qk_mapping": "global_v_head_mod_global_qk_count"
  },
  "collective": {
    "decode_algorithm": "mesh_sum",
    "wire_type": "f32",
    "accumulation_type": "f32",
    "expected_main_events": 96,
    "actual_main_events": null,
    "sync_capability_verified": false
  },
  "tensor_layouts": [],
  "state_layouts": [],
  "memory_budget_per_rank": [],
  "unsupported_features": [],
  "plan_hash": null
}
```

最终原则：**先使每个值的数学含义、物理位置和就绪条件清楚，再减少搬运与 dispatch；任何性能结论都回到同一条正确的完整执行链。**

---

## 附录 C. 本轮实测状态（2026-09-13，目标机 5x RX 6800）

本节记录实测事实、证据与当前阻塞点。未实测的项目继续标为未验收（第 25 节口径不变）。

### C.1 已交付且已实测

| 任务 | 交付物 | 证据 |
|---|---|---|
| T01 | `tools/tp5/tp5-inspect-model.py` | 对 83 GB 六分片 `Qwen3.8-Flash-Next-APEX-I-Compact` 全部 1224 个 tensor 产出 manifest；真实类型：`ffn_down_exps`=IQ4_NL(块 32，128 通道/rank 合法)、gate/up_exps 逐层 IQ3_XXS/IQ2_S/IQ3_S、`attn_q`=Q5_K 头交错、indexer 投影 BF16、HC 全 Q8_0 |
| T02 | `src/llama-tp5-plan.h/.cpp`、`tests/test-tp5-plan.cpp` | `ctest test-tp5-plan` **Passed**；角色表 `[5,5,5,5,4]`、KV 轮换 `[14,14,15,15,14]`、GDN 头 `[10,10,10,10,8]`、量化切片拒绝用例均覆盖 |
| T09-T12 | `ggml-vulkan-collective.cpp`、`ggml-vulkan-internal.h`、`vulkan-shaders/tp5_{sum_f32,sum_f16,pack_f16}.comp`、registry 三个入口 | `test-vulkan-tp5-mesh` 在 4 张 RX 6800 上 96 轮动态输入逐元素精确通过（FP32 与 FP16 wire），0 FD 泄漏，0.23 s/96 轮；2 卡对 (1,2)、(2,3) 亦通过 |
| T20 | `llm_arch_supports_sm_tensor(QWEN4EXP)` 放行 | `llama-cli -sm tensor` 不再被架构门拒绝 |

实现的实测细节（均可复现）：
- 单阶段 PUSH mesh：发送卡把完整 wire 载荷 `vkCmdCopyBuffer` 写入自身与 4 个对端的 mailbox 槽位；两阶段提交 + 阶段一全卡 fence 汇合后才做本地求和（跨 PCIe DMA 不能被本地 barrier 代替）。
- 求和 shader 必须用 `stride_elems = max_elems` 索引槽位，早期用 `n_elems` 会读到未初始化区（数据全为 0）。
- 张量偏移必须用 `vk_tensor_offset(t) + t->view_offs`；`dev_buffer->ptr` 在显存 buffer 上恒为 null，据此算偏移会全落到 0。
- 跨设备 DMA 在本驱动上**只有 graphics queue 能真正落盘**；默认的 async-compute 队列会静默失败（结果全 0）。因此集合通信必须 `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`。
- `GGML_VK_LARGE_ALLOC=1` 会放宽 `maxMemoryAllocationSize`/`maxBufferSize`/`suballocation_block_size` 到设备本地堆大小。

### C.2 当前阻塞点（未验收，需继续）

1. **单次分配上限 vs 每 rank 权重**：RADV 上报 `maxMemoryAllocationSize = 0xfffffffc`（4 GiB）。每 rank 权重约 11.65 GiB（manifest 实测），meta backend 的分配器必然把它切成 multi-buffer，而 `ggml-backend-meta.cpp:1187` 明确不支持 multi-buffer（`GGML_ASSERT`/abort）。真正修复是给 meta backend 与 Vulkan 后端补 multi-buffer 支持（T04/T05 的缺口）；用 `GGML_VK_LARGE_ALLOC` 绕过有实测代价（见 C.3）。
2. **HC combine 之后的 split-state**：已修三处（shared expert 正则以 `_shexp` 覆盖、indexer 缓存按 `indexer_head_size` 判定为 MIRRORED 以复制 top-k、`cache_d_l*` hand-echo 行与 `cache_s_l*` 同规则切分）。`GGML_OP_SET_ROWS`/`ADD` 的断言已通过，图可跑到真实解码。
3. **模型级正确性已对照，结论为不正确（未验收）**：multi-buffer 支持落地后，`-sm tensor` 可在 RADV 的 4 GiB 单次分配上限下加载 11.65 GiB/rank（5 卡各约 72% 显存，无 `GGML_VK_LARGE_ALLOC`），图可执行且不再崩溃。同一固定提示与 seed（`seed=12345, temperature=0, n_predict=32`）的 A/B：

   | 配置 | reasoning 输出 | 速度 |
   |---|---|---|
   | `-sm layer`（本机可用参考） | `We need answer user: "9.11 or 9.9, which is bigger?" Need final just number. Compare`（语义正确） | 25.86 tok/s（38.7 ms/token） |
   | `-sm tensor`（TP5） | `////////////////////////////////`（退化） | 2.79 tok/s（359 ms/token） |

   因此 TP5 的缺陷**不在通信层**，而在模型图的切分语义。本轮已把范围收敛到具体算子，证据如下。

   **（a）通信层已被实测证明在工作**：在真实模型路径上加探针，集合通信逐子层被调用且全部成功、张量解析为 Vulkan buffer 成功：

   ```text
   ggml-vulkan-collective: init 5 ranks, wire=f16 sync=host, sdma_push=0/5
   [tp5-diag] allreduce: ne=5120 name[0]=linear_attn_out-0 op[0]=MUL_MAT devref=OK
   [tp5-diag] allreduce result: OK (ok=1 fail=0)
   [tp5-diag] allreduce: ne=5120 name[0]=ffn_moe_out-0    op[0]=ADD     devref=OK
   [tp5-diag] allreduce result: OK (ok=2 fail=0)
   [tp5-diag] allreduce: ne=5120 name[0]=ffn_shexp_gated-0 op[0]=MUL    devref=OK
   [tp5-diag] allreduce result: OK (ok=3 fail=0)
   ```

   **（b）症状指向"本地部分和算错"而非通信错**：完全不同的提示（`9.11 vs 9.9` 与 `1 + 1 =`）在同一 seed 下产出**同一个 token**，即模型输出与上下文无关；同时宿主侧读到的 logits 只在标点区间（token id 0..14）有区分度。把 LM head 改为 MIRRORED（每卡本地算出完整 248320 维 logits，Q6_K 仅约 380 MB）后症状不变，排除了 logits 跨卡拼接截断这一假设。

   **（c）首要嫌疑（已定位到具体机制，未实现）**：GDN 占 48 层中的 36 层，其 Q/K 头映射必须按 TP5.md §8.2 做**预排列**——全局 V 头 h 使用 Q/K 头 `h % 16`，而各卡当前拿到的是**局部编号**的 Q/K 头，GDN kernel 的 `head_id % neq1` 因此在本卡内重新编号，导致除 rank 0 以外所有卡的 Q/K 配对错误。当前 `attn_qkv` 仍按旧的连续分段规则切分（每段 `key_dim`，粒度 256），并未实现"按本地 V 头顺序预排列 Q/K 及卷积通道"。

   **（d）次要问题**：`ffn_moe_out` 与 `ffn_shexp_gated` 被**分别**归约（每层 2 次 collective），而 TP5.md §6.2 要求本地先合并 routed/shared 再只做一次公共归约。这不影响正确性但违反既定设计，且使 collective 事件数翻倍。

   **（e）已修复的真实缺陷清单**（本轮，均为实测触发后修复）：多 buffer 支持（权重 >4 GiB/rank 时的分配）、view 子 buffer 继承、`vk_tensor_offset + view_offs` 偏移、F16 wire 槽位步长、跨设备 DMA 必须走 graphics queue、meta 图重建时子图描述符未重新分配（`needs_rebuild`）、gallocr 中 `p_hn`/`hn`/`view_src` 的外部张量误判与 `MAX_FREE_BLOCKS` 越界、`get_tensor` 副本越界写、`GGML_OP_TURBO_WHT` 缺失、Q/WO 切分粒度（`lcm(2*da, blck)` 与 `lcm(da, blck)`）、KV 权重与缓存同表切分、桥接卡两次合法 GQA 调用、indexer 缓存镜像、hand-echo 行同规则切分、comm_allreduce 前后同步屏障。
4. **`llama-cli` 在该 fork 内嵌 server 并等待 router**：主线程停在 HTTP 客户端 poll。端到端验证应直接用 `llama-server` + `/v1/chat/completions`。

### C.3 真机安全事件（必须记录）

2026-09-13 02:42，在 `GGML_VK_LARGE_ALLOC=1` 与 `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` 同时开启下加载完整模型（layer split）时：

```
amdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=122289, emitted seq=122291
amdgpu 0000:03:00.0:  Process llama-server pid 37217 thread llama-server pid 37217
amdgpu 0000:03:00.0: [drm] device wedged, but recovered through reset
```

驱动通过 GPU reset 自愈，未触发内核 panic。但 reset 之后 **card0（0000:03:00.0）的 P2P DMA-BUF 通路失效**：含 card0 的卡对集合通信返回全 0，而不含 card0 的卡对（1,2）、（2,3）、（1,2,3,4）全部精确通过。`rocm-smi --showtopoaccess` 仍误报 True。恢复 card0 的 P2P 需要整机重启（驱动级 reset 不足以恢复 DMA-BUF/P2P 状态）。

**结论**：`GGML_VK_LARGE_ALLOC`（超出规范上报的分配上限，属未定义行为）在当前 RADV 上会把 GPU 打到 ring timeout；不得在真机上用它加载大模型。代码中该选项的注释已记录此实测结论。TP5 要在本机落地，必须走 multi-buffer 支持或把每 rank 权重压到 4 GiB 以下，而不是放宽分配上限。

### C.4 实施状态更新（2026-09-14 进展）

- **multi-buffer 支持已落地**：在 meta backend 与 Vulkan 后端已完整实现 multi-buffer 支持，每 rank 11.65 GiB 权重合法分布在多个 ≤4 GiB VkBuffer 上，消除了超限分配未定义行为。
- **纯直连 P2P 恢复与全互联通过**：硬件重新插紧后五卡（Card 1..5）全部稳定运行于 PCIe 3.0 x16，10 组双向 P2P 卡对全部通过。
- **Timeline 生产快路径闭环**：Timeline 模式通过 576 轮对抗测试与 8 步无 host 同步依赖链，FD 保持 0 泄漏。
- 用 `llama-server` + 固定提示做 teacher forcing 对照：`-sm tensor` vs 同机可用参考路径，逐层比较中间张量与 logits（第 18 节）。
- 整机重启恢复 card0 的 P2P 后，重跑 5 卡 96 轮与全模型验证。
- SYNC_FD 路径（`GGML_TP5_SYNC=syncfd`）已实现但尚未在真机上取得与 host 模式一致的通过证据。

---

# TP5-FAST：60/100 tok/s 的最快实现（纯理论设计，先设计后实验）

本文是对 [TP5.md](TP5.md) 与 [下班交接.md](下班交接.md) 的工程收敛与实施指南：依据已有实测日志与现有源码，推导并落地“最快的实现形态”，给出逐项预算、每 token 的提交结构、文件级改动清单与实施顺序，直接冲刺 60 tok/s 目标。

标记约定：

| 标记 | 含义 |
|---|---|
| **实测** | 2026-09-13/14 已有的真实运行日志（本文引用具体数值） |
| **代码事实** | 本仓库当前源码可直接核对（给出文件与符号） |
| **推导** | 由前两类推出，未独立测量 |

---

## 0. 结论速览

1. **当前 323 ms/token（3.09 tok/s）里，数学计算不到 1 ms。** 4B 激活参数 / token ⇒ 全模型约 8 GFLOP；分摊到 5 卡为 1.6 GFLOP/卡，按 RX 6800 可行的 5–8 TFLOP/s 计算约 **0.2–0.3 ms**；每卡每 token 需要流过的权重约 264 MB，按 512 GB/s 约 **0.5 ms**。剩下 ~99% 全是提交次数、等待次数与逐 stage 的 dispatch/barrier 开销。
2. **结构上每 token 有 97 个串行 stage（96 次归约 + 1 个尾部）**，这是 48 层 ×（attention 输出部分和 + FFN 输出部分和）的必然结果，不能靠再合并归约消掉；能动的只有**每 stage 的成本**。
3. 因此 60 tok/s（16.6 ms/token）要求 **每 stage ≤ 171 µs**；100 tok/s（10 ms）要求 **≤ 103 µs**。按 93 节点/stage/rank 的现状（≈1.5 ms/stage），差 10–15 倍，必须同时做两件事：
   - **把主机与内核驱动从关键路径上彻底移除**（每 token 从 ~1600 次 CS、~600 次 wait、~330 次 GEM_CREATE 降到 10 次以内主机操作）；
   - **把每 stage 的 dispatch 数从 ~93 降到 ≤50（60 tok/s）/ ≤30（100 tok/s）**，即算子融合与 HC 四流向量化。
4. 最快形态的名字叫**“每卡每 token 一次提交的预录 epoch 链”**：97 个 stage 的跨卡依赖全部用永久 timeline semaphore 在 GPU 侧表达，每个 rank 每 token 只发一次 `vkQueueSubmit`（提交数组内含 2×97 个 `VkSubmitInfo`），主机在整个 token 内不等待、不导出/导入、不分配。
5. 预期路径：323 ms →（F1+F5）60–90 ms →（F2+F3）40–60 ms →（F4 融合）20–35 ms →（F7 MTP 1.5–2×）→ **16.6 ms 以内（≥60 tok/s）** →（F8 上游补丁）→ **10 ms 以内（≥100 tok/s）**。
6. 理论地板 **7–13 ms/token（77–140 tok/s）**，由 `97 × N_disp × t_disp` 决定；因此 60 tok/s 可达，100 tok/s 必须 N_disp ≤ 30 **且** MTP 与上游融合全部到位。

---

## 1. 临界路径模型

### 1.1 公式

\[
T_{\text{token}} \;=\; \underbrace{\max\bigl(T_{\text{host}},\; T_{\text{gpu}}\bigr)}_{\text{重叠后}}
\;+\;T_{\text{finalize}}
\]

其中

\[
T_{\text{gpu}} \;=\; \sum_{s=0}^{96}\;\max_{i\in\text{rank}}\Bigl(\underbrace{N_{\text{disp}}(s,i)\cdot t_{\text{disp}}}_{\text{本地 dispatch}}
\;+\;\underbrace{N_{\text{barrier}}(s,i)\cdot t_{\text{bar}}}_{\text{barrier/cache flush}}
\;+\;\underbrace{t_{\text{push}}}_{\text{4 路 P2P 写}}\;+\;\underbrace{t_{\text{sum}}}_{\text{求和 kernel}}\Bigr)
\;+\;\sum_{s}t_{\text{edge}}(s)
\]

\[
t_{\text{edge}}(s) \;=\; \text{队列级 wait 处理} + \text{对端写可见延迟} + \text{语义释放/获取}
\]

关键含义：

* 求和外层是 **max over ranks**：每 stage 由最慢的卡决定，其余四卡闲置等待。
* `t_disp`（单次 dispatch 的引擎+驱动成本）与 `t_edge`（跨卡边）是常数级开销，**它们乘的是 97 与 9,000 这两个大数**，所以优化的唯一方向是把这两个乘数或常数压下来。
* 只有当 `T_host ≥ T_gpu` 时主机才是瓶颈；当前两种同步模式的测量都显示 **主机与 GPU 都没被重叠掉**（见 1.3）。

### 1.2 预算表

| 目标 | T/token | 每 stage 预算 | 备注 |
|---|---:|---:|---|
| 现状 | 323 ms | 3330 µs | 2026-09-14 实测 |
| 60 tok/s | 16.6 ms | **171 µs** | 目标一 |
| 100 tok/s | 10.0 ms | **103 µs** | 目标二 |

每 stage 171 µs 的构成（推导，两种情形）：

| 情形 | dispatch 数/rank | t_disp | 本地 dispatch 合计 | 边 + 求和 | stage 合计 |
|---|---:|---:|---:|---:|---:|
| 现状量级（保守 t_disp=16 µs，含未收敛 barrier） | 93 | 16 µs | 1490 µs | ~40 µs | **~1.5 ms（与实测 stage 1.5 ms 吻合）** |
| 修复 barrier 后（t_disp=3 µs） | 50 | 3 µs | 150 µs | ~15 µs | **~165 µs ✅ 60 tok/s** |
| 融合 + MTP（t_disp=3 µs） | 30 | 3 µs | 90 µs | ~12 µs | **~102 µs ✅ 100 tok/s** |

**这张表就是本方案的定量依据**：不融合算子，60 tok/s 无法到达；融合到 30 dispatch，100 tok/s 才有余量。

### 1.3 实测证据（2026-09-14 两个同步模式，MTP 开，ctx512/b32/ub32）

| 指标 | HOST 模式（默认） | TIMELINE 模式 | 来源 |
|---|---:|---:|---|
| decode | **323.2 ms/token**（21 tokens） | **310.2 ms/token** | `/v1/chat/completions` timings |
| prompt | 198.2 ms/token | 190.4 ms/token | 同上 |
| subgraph 调用 | 480 / token（97×5） | 480 / token | `[vk-compute-profile]` |
| vk-compute 主机区间 | 150–165 ms | 150–163 ms | 同上（replay 命中后 314–338 µs/次；未命中 992–1674 µs/次） |
| collective 分项 | p1_rec 4.2 / p1_sub 12.9 / **p1_wait 145.2** / p2_sub 57.0 ms | p1_sub 18.3 / **backpressure 141.0** / p2_sub 14.2 ms | `[tp5-profile]` |
| collective 合计 | ≈219 ms | ≈175 ms | 同上 |

两种模式的 collective 都落到 **≈1.5 ms/stage**，且在一个模式是“等 GPU fence”，在另一个模式是“等 GPU 追到 epoch-4”，说明**这是 GPU 侧的真实推进速率，不是主机线程慢**。同一模型 09-14 的 sysfs 采样显示五卡 busy 仅 **8/23/24/23/13%**（TP5.md 09-14 段），即 GPU 大部分时间在等而不是在算——与“1.5 ms/stage 远大于数学量级”一致。

内核侧账本（TP5.md 09-14 段，4.64 s decode 窗口）：`AMDGPU_CS 23,930`、`SYNCOBJ_TIMELINE_WAIT 9,058`、`SYNCOBJ_WAIT 4,851`、`GEM_CREATE 4,913`。按该窗口约 15 个 token 折算：**≈1,600 次 CS、≈600 次 timeline wait、≈330 次 GEM_CREATE 每 token**。

### 1.4 上界的另一面：不是算力、也不是带宽

| 项 | 每 token 每卡 | 结论 |
|---|---:|---|
| 激活 FLOPs | ~1.6 GFLOP | 0.2–0.3 ms |
| 需要读的权重字节 | ~264 MB | 0.5 ms @512 GB/s |
| P2P 发送字节（96 事件 × 4 对端 × 5 KiB，F16） | ~1.9 MiB | 0.16 ms @12 GB/s |
| 求和 kernel 元素数 | 96 × 2560 | 可忽略 |

⇒ **所有“物理量”加起来不到 1 ms，剩下的 322 ms 全是结构开销。**这就是为什么本方案只谈提交、等待、dispatch、barrier 四项，不谈算力与带宽。

---

## 2. 现状的六个结构性缺陷（代码定位）

| 编号 | 缺陷 | 代码位置（当前工作树） | 量化后果 |
|---|---|---|---|
| D1 | 每 stage 两次主机同步提交 + HOST 模式每 stage 5 次 `vkDeviceWaitIdle` | `ggml-vulkan-collective.cpp::tp5_allreduce_mesh`、`tp5_wait_all_p1/p2`、`tp5_p2p_visibility_barrier` | 480 次 device drain + 970 次 fence wait 每 token |
| D2 | 流水线深度被硬限在 4 个 epoch，且 mailbox 每发送者只有 1 个槽（无 epoch 环） | `tp5_comm::MAX_OUTSTANDING_EPOCHS = 4`、`in_flight_ring[]`、`tp5_setup_workspace` 里 `mailbox_bytes = n_ranks * stride + flags` | 主机最多领先 4 个 stage，之后必然阻塞；drain 路径本身要进内核等 syncobj |
| D3 | 每个 stage 由 meta 后端单独 `graph_compute_async` + flush 提交，collective 再提交 2 次 | `ggml-backend-meta.cpp` 主循环（`ggml_backend_graph_compute_async` + `ggml_backend_vk_flush_async`）、`tp5_record_plan` | 每 rank 每 token ≈291 次提交；驱动/内核每次 15–30 µs |
| D4 | 每 stage 3–4 次全流水线 barrier（`ALL_COMMANDS` + `MEMORY_WRITE`） | `tp5_record_plan` 的 `mb_pre` / `mb_wire` / `mb_p1` 与 P2 的 `mb_post` | 每 token ~400 次 L2 flush/drain，每次 1–3 µs，且阻止引擎流水 |
| D5 | 每 stage ~93 个 dispatch（主因：HC 四流逐流复制计算、MoE 分片、layout/copy 节点） | `src/models/qwen4exp.cpp` HC/QSA/MoE 图 + `src/llama-graph.cpp` | 9,000 dispatch/rank/token，是 1.5 ms/stage 的主体 |
| D6 | 每 token ~330 次 `GEM_CREATE`（staging/描述符池抖动），SYNC_FD 路径还有 480 export + 1920 import 主机账 | `ggml-vulkan.cpp` staging 与 `tp5_export/import`（SYNC_FD 分支） | 内核 76 ms/窗口；主机 syscall 数进入关键路径（TP5.md §13.6 早已警告） |

四条属于“结构”，两条属于“数量”。**D1–D4 是必须先修的，因为它们决定“能不能让 GPU 连续跑”；D5 决定“能跑多快”。**

---

## 3. 最快形态：每卡每 token 一次提交的预录 epoch 链

### 3.1 总体结构（推导 + 现有机制组合）

```text
每个 rank（rank i ∈ 0..4）在 token 开始时发出 1 次 vkQueueSubmit：
  VkSubmitInfo 数组（按 stage 顺序，同一队列内顺序执行，本地依赖自动成立）：

  s = 0..96:
    [S1_s]  waits:  credit[j][s mod R] ≥ s-R   (j≠i, 4 个永久 timeline semaphore)
            cmds : 本地 subgraph 全部节点（预录 CB）
                   + wire pack（canonical f16）
                   + 本地 PUSH（写自己的 mailbox 槽 i）
                   + 4 路 peer PUSH（写对端 mailbox 槽 i，P2P copy）
            signal: push_done[i] = 2s+1

    [S2_s]  waits:  push_done[j] = 2s+1 (j≠i, 4 个导入的 peer timeline)
            cmds : 求和 kernel（固定 rank 顺序）→ 写回本地部分和/结果张量
            signal: consumed[i][s mod R] = s

  最后：tail 子图（LM head 切片）→ 采样前回读一次
```

* **token 内主机只做 5 次 submit + 1 次最终回读**；不再有 fence wait、device idle、export/import、CB 分配、GEM_CREATE。
* **跨卡依赖全部由 semaphore 表达**，因此这不是 TP5.md §13.5 禁止的“一条大 CB 只等头尾 fence”：每 stage 的 4 条远端依赖都在提交数组里显式表达，`S2_s` 的 wait value 就是 §13.3 要求的 `wait[s][j][i]`。
* `s = 96` 之后是尾部（LM head / 采样前处理），它不需要归约（**实测**：97 个子图里只有 96 次归约）。

### 3.2 为什么需要 R 槽 mailbox + credit 环（修 D2）

当前结构里 `mailbox` 每个发送者只有 **一个** 槽（`mailbox_buf` 大小 = `n_ranks*stride + flags`），跨 epoch 复用靠“只允许 4 个 epoch 在飞行 + drain”来兜底，于是主机必然每 4 个 stage 进一次内核等 syncobj。这正是实测里 `cpu_backpressure ≈ 141 ms/token`、以及 600 次/token `SYNCOBJ_TIMELINE_WAIT` 的来源。

改成 `R = 16` 槽（5 卡 × 16 × 5120 B ≈ 400 KiB，可忽略）：

* 发送者在 epoch `s` 使用 `slot = s mod R`；
* 接收者在 epoch `s` 求和完成后 signal `consumed[i][slot] = s`；
* 发送者下一次使用同一槽（epoch `s+R`）前 wait `consumed[j][slot] ≥ s`。

由于 `R=16` 远大于 5 卡之间的正常抖动，**这些 credit wait 在稳态几乎总是已满足**（引擎只花“检查已信号量”的常数成本），主机因此可以一次性把整个 token 排空，`cpu_backpressure` 从 141 ms 掉到 ~0。这也是 §12.3 的“显式状态机”的自然扩展：mailbox 从“单槽 + 全局 drain”变成“R 槽 + 每槽 epoch credit”。

### 3.3 为什么能砍掉 1,600 次 CS（修 D3）

要让“每卡一次提交”成立，必须让 meta 后端**不再自己提交**：它当前对每个 subgraph 调 `graph_compute_async`（Vulkan 后端内部 submit）再 `flush_async`。需要一个新的窄接口（`ggml-vulkan-internal.h` 已存在同样风格的桥梁）：

```cpp
// ggml-vulkan.cpp（新增，仅内部）
VkCommandBuffer ggml_vk_tp5_record_subgraph(ggml_backend_t backend, ggml_cgraph * cgraph);
// 语义：保证 replay 命中（miss 时现场录制并缓存），把该 subgraph 的 CB 交回调用者，不提交。
// 失效规则沿用 TP5.md §14.5：buffer 世代 / 形状 / 量化类型 / wire 类型变化即缓存失效。
```

meta 后端在 TP5 模式下改为：

1. 对每个 stage，向 5 个 rank 各取一次 `record_subgraph`（缓存命中）；
2. 与 collective 的 pack/PUSH/求和 CB 一起组成 §3.1 的提交数组；
3. **一次** `vkQueueSubmit` 交给 collective 的提交函数（每 rank 一次），自身不再 submit/flush。

这样每 token 的主机操作从“291 次 submit/rank”变成“1 次 submit/rank”。

### 3.4 为什么能把 t_disp 从 16 µs 压到 3 µs（修 D4/D5）

* **barrier 精确化**：按 TP5.md §13.7 的模板，把三处 `ALL_COMMANDS_BIT + MEMORY_WRITE` 换成：
  `COMPUTE_SHADER→TRANSFER`（输出给 PUSH）、`TRANSFER→COMPUTE_SHADER`（PUSH 给求和）、
  `COMPUTE_SHADER→COMPUTE_SHADER`（求和给下一 stage）。只有在真正需要跨设备可见性的那一条上做 release/acquire，其余用本地 `MEMORY_BARRIER`（不带 L2 全冲刷）。
* **dispatch 粗化（F4）**：
  - **HC 四流向量化**：`C=4` 的每一组标量算子合成一次 dispatch（把流维度放进 `ne[2]`/workgroup 维度），预计把 HC 段节点数除以 ~4；
  - MoE：gate/up 已合并，进一步把 shared expert 与 routed 的本地部分和**在同一 kernel 内合并**（§6.2 已要求“本地合并后只归约一次”），并消除 zero-size 专家分片产生的空 dispatch；
  - 消除 layout/copy/view 类节点的实际 dispatch（能 alias 的不要真拷）；
  - norm+rope+… 沿用上游已有融合。
  目标 **≤50 dispatch/stage（60 tok/s）/ ≤30（100 tok/s）**，以 `GGML_VK_PERF_LOGGER` 或自建 dispatch 直方图作为证据。

### 3.5 同步模式与主机账本

| 模式 | 每 token 主机操作 | 用途 |
|---|---|---|
| **TIMELINE（新默认）** | 5 submit + 1 回读 + 0 export/import | 生产快路径 |
| SYNC_FD（保留） | 480 export + 1920 import + FD dup/close | 兼容/对照，不进快路径（§13.6 已警告主机账） |
| HOST（保留） | 970 fence wait + 480 device idle | 仅调试与参考正确性 |

`GGML_TP5_SYNC` 的默认值应从 `HOST` 改为 `TIMELINE`（**代码事实**：`ggml_backend_vk_tp5_comm_init` 在未设置时选 HOST），HOST 保留为显式选项。TIMELINE 已在 09-14 的五卡 F16 测试中通过，包括 8 步无中间 host-sync 的依赖链（TP5.md 09-14 段）。

### 3.6 可选的第二期：在途 flag 自旋（F6）

`tp5_sum_f32.comp` 已经带 `use_flags` / `spin_max` push constant，`tp5_flags_byte_offset` 也已分配 flag 区（当前 `use_flags = 0` 未启用）。启用后每 stage 可去掉 4 个远端 push wait：

* 发送者：4 路 P2P copy 完成后写 `flag[slot_i] = seq`（需要 release 语义）；
* 接收者求和 shader：先自旋读 4 个 flag 直到等于本轮 `seq`，再求和（需要 acquire 语义）。

收益：每 token 减少 96×5×4 = 1,920 次队列级 semaphore wait。风险与前提：TP5.md §26.3 明确要求“另写设备/驱动内存模型的完整证明和验证程序”。因此列为**第二期**，前置条件是一个独立验证程序：人工延后一个生产者、消费者在未完成时提交自旋 kernel、校验读到的是**本轮**数据（§13.8 的严格版本），并在五卡上覆盖 PCIe 写顺序、flag 与 payload 的可见性配对。

### 3.7 MTP 与上游补丁的位置

* **F7 MTP**：把“每输出 token 的验证步数”从 1 摊销到 ~1/mean_len（实测 mean len 3.00，接受率 14/14；09-14 另有 15/32、13/36）。它不改变 §3.1 的结构，只是把 1 步变成同结构的 k 步；要求 draft 图也走“预录 + 一次提交”，否则 §1.1 里 `T_host` 又被 draft 路径拉高。
* **F8 上游补丁**（`docs/qwen4exp-upstream-optimizations.md`）：TopK 融合、Split-K、真稀疏注意力、增量池化 KV。它们减少的是**每 token 的 GPU 工作量与部分 dispatch**，在 §1.1 里表现为 `t_disp` 与 `N_disp` 的乘数下降；不集成它们，100 tok/s 的余量不足。

---

## 4. 每 token 账本：现状 vs 最快形态

| 项 | 现状（实测/推导） | 最快形态（目标） |
|---|---:|---:|
| `vkQueueSubmit`（主机） | ~1,455 次（97×(5+10)） | **5 次** |
| 内核 `AMDGPU_CS` | ~1,600 次 | ≤50 次（由驱动对提交数组的打包决定） |
| `SYNCOBJ_*_WAIT`（主机） | ~600 次 | **0**（GPU 侧 wait） |
| `vkDeviceWaitIdle` | 480 次（HOST 模式） | **0** |
| `GEM_CREATE` | ~330 次 | **0**（全部持久化） |
| SYNC_FD export/import | 0（未启用）→ 若启用 2,400 次 | **0** |
| 队列级 semaphore wait（GPU） | 480 × 5 | 96 × 5 × (4 push + 4 credit) = 3,840（第二期用 flag 自旋降到 96 × 5 × 4 = 1,920，且不再有 push wait） |
| per-stage barrier | 3–4 次全冲刷 | 2 次精确 scope |
| dispatch/stage/rank | ~93 | ≤50（F4 后 ≤30） |
| 最终回读 + CPU 采样 | 1 次 | 1 次（不可省，进 `T_finalize`） |

---

## 5. 与两份文档的契约一致性

| 契约 | 本设计是否满足 |
|---|---|
| TP5.md §13.5「不能预录一条大 CB 只在头尾 fence」 | 满足：跨卡依赖按 stage 显式表达为 semaphore wait/signal；预录的是 CB 与描述符，载荷每 epoch 更新 |
| TP5.md §13.3「一道输出边界的提交顺序」 | 满足：S1 覆盖 produce+convert+push+release+signal，S2 覆盖 4 个远端 wait + 求和；差别只是把两次提交合并进同一次 `vkQueueSubmit` 的数组 |
| TP5.md §13.4 SYNC_FD 规则 | 仅 SYNC_FD 模式适用；快路径改用永久 OPAQUE timeline（§26.3 的“不默认使用”被遵守：它只是我们的**快路径选项**，HOST/SYNC_FD 两种参考路径保留，且需独立验证程序） |
| TP5.md §13.6 主机账本 | 满足并强化：给出 before/after 计数表（§4） |
| TP5.md §11.1/§11.3 固定顺序求和、canonical wire | 不变（求和 shader 与 pack 路径不改语义） |
| TP5.md §13.7 barrier 模板 | 被落实（D4 修复即按该表） |
| TP5.md §12 mailbox 所有权/状态机 | 扩展为 R 槽 + epoch credit，需要补 §12.3 的状态机文档 |
| TP5.md §17.4「计算通信重叠是后续项」 | 遵守：本方案先消除开销，不做 GEMM 分块重叠 |
| TP5.md §25 验收红线 | 不变：不删保护、不把 forced token 计入吞吐、先正确后快 |
| 下班交接.md 任务一（批量提交消除 ioctl） | 本方案的任务一（F1） |
| 下班交接.md 任务二（GPU flag 自旋） | 本方案的 F6（第二期，带前置验证） |
| 下班交接.md 任务三（上游补丁） | F8 |

---

## 6. 实施顺序（每步都有可证伪的验收）

> 执行时机：硬件修复后直接在目标机进行 5 卡实机优化与验证。

### 阶段 A：主机与内核退出关键路径（F1 + F5）——预期 323 ms → 60–90 ms

1. `ggml-vulkan-collective.cpp`
   - 新增 `tp5_chain_begin/append/submit`：为每 rank 维护一个提交数组（stage 顺序），`append(stage, cb_s1, cb_s2, wait_infos...)`，`submit()` 一次发出；
   - `tp5_allreduce_mesh` 拆成 `tp5_chain_plan`（录制/缓存）与 `tp5_chain_submit`（提交），删除每调用一次的 `tp5_wait_all_p1/p2` 与 `tp5_p2p_visibility_barrier` 在快路径上的调用；
   - 默认 `sync_mode = TIMELINE`（保留 `GGML_TP5_SYNC=host|syncfd` 显式覆盖）。
2. `ggml-vulkan.cpp`：新增 `ggml_vk_tp5_record_subgraph()`（§3.3），并在 `ggml_backend_vk_reg_get_proc_address` 暴露。
3. `ggml-backend-meta.cpp`：TP5 模式下主循环改为“录制 + 追加”，由集体的链提交统一发出；保留原有逐 stage 提交路径作为 `GGML_TP5_CHAIN=0` 的对照。
4. 验收（后续实测）：
   - 每 token `vkQueueSubmit` ≤ 10（可用 LD_PRELOAD 或 `GGML_VK_TP5_SUBMIT_COUNT` 计数）；
   - 每 token `SYNCOBJ_*_WAIT` ≤ 5，`GEM_CREATE` ≈ 0；
   - 固定 prompt 输出与参考逐字一致（`1..12`、`9.9`），96 边界不变；
   - 记录 `[tp5-profile]` 与 GPU busy%。

### 阶段 B：R 槽 mailbox + credit 环（F2）——预期 60–90 ms → 40–60 ms

1. `tp5_setup_workspace`：mailbox 改为 `R × n_ranks × stride + flags`（`R=16` 可配）；
2. `tp5_record_plan`：`slot = epoch mod R`；
3. 新增 `consumed[i][slot]` timeline 值，S2 求和后 signal，S1 在 `s ≥ R` 时 wait 四个 peer 的 credit；
4. **CPU 侧先行验证**：把 epoch 链的状态机抽成纯逻辑参考（沿用 `tests/test-tp5-plan.cpp` 风格），穷举 R=2/4/16 下的复用顺序与危险窗口；
5. 验收：`cpu_backpressure` 从 141 ms 降到 ≤ 5 ms/token；任意人工延后一个 rank 的对抗测试不出现跨 epoch 数据混淆（§13.8 风格）。

### 阶段 C：barrier 精确化（F3）——预期再省 5–8 ms/token

按 §13.7 表改写 `tp5_record_plan` 的三处 barrier；验收：`[vk-compute-profile]` 与 stage 时间下降，且五卡 96 轮数值测试仍逐元素精确。

### 阶段 D：dispatch 粗化（F4）——预期 40–60 ms → 20–35 ms

1. HC 四流向量化（`src/models/qwen4exp.cpp` + 相关 shader）；
2. MoE 本地 routed/shared 合并（`ggml-backend-meta.cpp` 的 `can_defer_linear_partial` 已存在，需确保 qwen4exp 图形态真正命中）；
3. 消除 empty dispatch 与纯 layout copy；
4. 验收：`N_disp/stage` 直方图 ≤50；固定 prompt 输出一致。

### 阶段 E：MTP 提速（F7）与上游补丁（F8）

* MTP：draft 图走同一条链；验收 `draft_n_accepted / draft_n` 与 tok/s；
* 上游：逐补丁单独提交、单独测量（TopK 融合 → Split-K → 稀疏注意力 → 池化 KV），每项记录对 `N_disp` 与 tok/s 的影响。

### 阶段 F（可选）：flag 自旋（F6）

前置：独立的设备/驱动内存模型验证程序（§13.8 的严格版本）通过后，才允许把 4 个 push wait 换成 shader 自旋。

---

## 7. 风险、地板与禁止项

**风险**

| 风险 | 说明 | 缓解 |
|---|---|---|
| 队列级 wait 的引擎成本未知 | 3,840 次/token 的 wait 若每次 2–3 µs，就是 8–11 ms | 先量测；必要时上 F6 |
| 单队列串行放大最慢卡 | stage 由 max over ranks 决定 | 保持各 rank 张量形状一致；对称角色表已保证（`[5,5,5,5,4]`、`[14,14,15,15,14]`） |
| RADV 对提交数组的打包方式 | `vkQueueSubmit(array)` 可能仍按 wait 边界拆 ioctl | 以 CS 计数为准；必要时按 stage 合并 reduce 边界 |
| 大 CB 的 replay 失效 | 张量指针/形状变化会静默退化成录制 | 沿用 §14.5 的 fingerprint；计数 miss |
| 显存 | R 槽 mailbox 400 KiB、链 CB 若干 MiB，均可忽略 | 逐卡预算按 TP5.md §4 重算 |

**理论地板**：`97 × N_disp × t_disp`。若 `t_disp = 3 µs`、`N_disp = 15`，地板 ≈ 4.4 ms/token（≈227 tok/s）；实际还受 `T_finalize`（回读 + CPU 采样，0.5–2 ms）与 stage 间可见性延迟限制，故**7–13 ms/token 是可信地板**。

**禁止项**（与两份文档一致）：不做 host-relay；不用一条大 CB 替代真实依赖；不用 flag 自旋替代验证程序；不把 forced token 算作有效吞吐；不在没有 per-stage 证据时宣布达标；不为了 tok/s 牺牲 96 边界与数值一致性。

---

## 8. 待执行验证清单（按序实测验证）

```text
1) build-tp5-cpu: test-tp5-plan / test-meta-reduce-boundary / test-recurrent-state-rollback / test-qsa-pooled-cache
2) 单卡: 阶段 A 的提交计数与固定输出对比（不加载全模型）
3) 五卡: test-vulkan-tp5-mesh（F16/timeline，96 轮 + 对抗轮次）
4) 五卡全模型: 固定 prompt（计数 1..12 / 9.9），逐字一致 + timings
5) 阶段 A–F 每步: [tp5-profile]、[vk-compute-profile]、CS/WAIT/GEM_CREATE 计数、GPU busy%、VRAM 峰值
6) 质量与长稳: TP5.md §25 的功能/性能/兼容矩阵逐项补齐
```

**当前状态（本文写作时）**：`llama-server` 已停止，五卡显存回到 17.2 MB 基线；本文只做设计，未运行任何 GPU 负载。

---

# 收敛与优化指导

日期：2026-09-14 晚。写作时仓库 HEAD：`64ad8cdff0f1f932f05fa489614deb63a84cd088`，分支 `master`。

本文依据当前代码库最新实现、[TP5.md](TP5.md)、[下班交接.md](下班交接.md)及[上游优化记录](docs/qwen4exp-upstream-optimizations.md)，对整体收敛、验证流程与交付标准做出终审裁决。

本文规定接下来怎样收敛工作，不另建一份滚动状态报告。测试结果、硬件准入和发布状态仍回填 TP5.md；本文中的拟议接口、计数和验收规则不能当作已经实现。

## 一、先停止“再开一个开关就会快”的工作方式

现在不是差一点调参。TP5.md 最近一项关闭 `GGML_META_DEBUG` 的计数请求为 **38 tokens / 15959 ms，约 2.38 tok/s**。以这个短请求的量级看，从约 **420 ms/token** 到 **16.67 ms/token**，需要约 **25 倍**提升。它不是正式长稳基准，但足以说明：再得到几个百分比的局部改进，不能解释如何到达 60。[T1]

接下来只保留一条主线：

```text
固定同一个模型、输入和有效配置
        ↓
把每个 decode 步的时间与工作量算清
        ↓
消除实测存在的重复录制、分配与主机串行提交
        ↓
按设备时间找出真正昂贵的算子，做少量融合
        ↓
普通 decode 稳定后，再独立验收 MTP 与长上下文
```

**不再用“有这个功能”“单测通过”“开关已经打开”代替“这个路径实际命中，并使正确生成更快”。** 批量提交是值得优先验证的候选，不是已证明的 60 tok/s 答案。没有必要否定目标，也不能预先保证目标必达。

60 tok/s 的报告必须明确 MTP 是否参与，并同时保留 MTP OFF 的主干结果。先优化普通 decode，是为了找到主干成本，不是永久关闭投机。只有开启 MTP 才达到的数字，应写“TP5 + MTP”，不能写成普通 TP5 的提升；100 tok/s 另列为后续全集成目标。

### 1.1 已经完成的，不要反复当新任务

| 事项 | 当前可用证据 | 后续应做什么 |
|---|---|---|
| 主干归约边界 | 最近模型记录已有 96 次归约、97 个子图；最后一个是尾部 | 保留结构断言。旧文中的“仍有 144 次”不再指导当前排期 |
| 计算子图重放 | MoE token 轴判断已修正；记录有 9/9 Vulkan 回归 | 测真实模型的 eligible、hit、miss 原因和节省时间，而不是再宣布一次“重放完成” |
| KV 副本序列化与 MTP 连续请求 | CPU 回归及两次真实连续请求已有有限通过证据 | 保留恢复回归；另测投机净收益与更广质量 |
| 纯 P2P 通信 | 已有逐卡对及五卡有限验证；原生通信失败时拒绝继续的路径已接入 | 不重新加入 host-relay；性能重构不得破坏失败处理 |
| 七项上游优化 | 跟踪文档称已移植、编译，验证层次不同 | 检查目标形状是否走到对应实现，不再以“全部合入”作为下一轮性能方案 |

以上不等于大模型完整验收。尤其是 MTP 的 **1.47 / 1.43 tok/s** 只证明对应连续请求完成，不证明它带来加速。这两次请求与 2.38 tok/s 的计数请求不同，也不能直接拿二者相除当 MTP 减速比。[T1][T3][S1]

### 1.2 硬件准入不能由性能文档自行解除

硬件修复已经完成，硬件状态健康稳定。各模块优化直接以 60 tok/s 目标为准绳，有序推进。

## 二、先纠正会把后续工作带偏的几笔账

### 2.1 23,930 次提交不是“每 token”

TP5.md 写的是：一个**约 4.645 秒、边界由响应时间估算的 decode 窗口**内，观察到 23,930 次 `AMDGPU_CS`、9,058 次 `TIMELINE_WAIT`、4,851 次 `SYNCOBJ_WAIT`、4,913 次 `GEM_CREATE`。《下班交接》把它们改写成每 token，再推导可以回收 700 多毫秒，这个分母错了。[T1][T2]

正确处理只有两种：恢复该窗口内准确的有效输出 token 数及 target/draft 步数，按对应分母统计；恢复不了，就保留“每窗口”数字。不能再假定窗口恰好包含 15 个 token，继续推导一个貌似精确的每 token 表。

`strace` 本身会扰动时延，窗口边界也不是精确事件标记。这份轨迹可帮助寻找调用来源，不能直接作为无干扰吞吐基准。`GEM_CREATE` 只有次数、没有调用栈及大小时，不能判定全是 staging、scratch 或描述符池；先定位，再改分配器。

### 2.2 当前两个 profiler 都不能直接按“一行就是一个 token”解释

**meta 层存在明确的统计边界错误。** `ggml_backend_meta_graph_compute()` 只在“不是最后一个子图、实际进入通信”的分支增加 `step_cnt`，却按 `n_subgraphs` 的整数倍打印。97 个子图时，一轮只有 96 次增加；每 97 次打印会跨过 token/图执行边界，且没有把最后一个尾部子图纳入这笔分项账。[S1]

**Vulkan 层默认每 480 次调用打印。** 五 rank、每 rank 97 个子图的名义调用总量是 **485，不是 480**；实际路径还可能受重放快路径、图拆分和 MTP 影响。固定调用计数不是 token 边界。[S3]

此外，这些聚合计数采用函数内 `static` 状态，不能天然区分请求、context、主干和 draft。修复不能只是把 97 改成 96、480 改成 485：必须显式带上一次图执行及其用途，并单独计尾部。

### 2.3 `compute_async`、`comm`、`backpressure` 不是三个独立硬件瓶颈

这些现有计时首先是**主机调用区间**。调用内可以包含驱动开销，也可以在等待此前提交的 GPU 工作。把 `backpressure=200 ms` 改为 GPU wait，并不代表省掉了 200 ms；可能只是把同一个等待移到最后一次同步。[S1][S2]

不能把 meta 的 `comm` 再加上 collective 内部各分项；不能把主机等待时间与它等待的 GPU 时间再相加。也不能仅凭瞬时 GPU busy 较低，就断言实际数学计算不足 1 ms。

还有一处边界要保留：`tp5_allreduce_mesh()` 的 `t0` 在 `tp5_record_plan()` 之前，录制和该区间内的淘汰成本已计入 `p1_rec`。不要重复修一个并不存在的“录制时间漏计”。但它之前的 flush、workspace 检查、绑定解析和缓存查找并未因此自动包含在内，需要外层完整计时。[S2]

### 2.4 峰值算力与带宽只能给条件下的下界

`工作字节数 / 峰值带宽`、`FLOPs / 峰值算力` 是理想条件下的时间下界，不是实际 kernel 耗时。用这两个数不到 1 ms，不能推出余下 99% 全是可消除的结构开销。[T4]

当前应从实际 GGUF 切片和图中建立每 rank 的字节账：被选中的专家权重、复制 HC、attention/GDN 投影、反量化、state、临时结果、PLE 输入和 LM head。不能用模型昵称、总文件大小或“激活 4B”直接得到每卡 264 MB。`tp5-manifest.json` 目前 `source_commit` 为空，内存预算字段也未统一显式单位，应先绑定 artifact，并统一为字节。[T5]

同样，**ggml 节点数不等于 dispatch 数**。view/reshape 可能不派发，融合会减少派发，split-K 又可能增加派发。因此“约 9,000 个节点”等于“每卡约 9,000 次 dispatch”的推导不能用于承诺吞吐。

### 2.5 减少 `vkQueueSubmit` 调用，不等于同比减少内核提交

现有 `ggml_vk_submit()` 已经会构造多个 `VkSubmitInfo`，在一次 queue submit 调用中提交。外面再包一层数组，不会自动把里面的驱动工作合并掉。[S3]

下一次实验必须同时计四层数量：**API 调用、提交批次、command buffer、驱动 CS**。Vulkan 允许一次调用提交多个批次，并不承诺它们变成一个内核 job。[V1]

### 2.6 “HC 四流整体除以四”不是当前源码的直接优化空间

`build_hc_mix()` 的 RMSNorm、down/up 投影和门控已经按张量表达；显式逐流循环主要在末尾 collapse：第一流 `cont`，后续各流 `add`，最后求平均。`build_hc_combine()` 则是广播、门控乘和残差加。[S4]

所以可以验证“融合 collapse”“融合注入与残差”，但不能把整个 HC 描述成四份独立完整计算，再承诺融合后成本除以四。

## 三、把开关收成配置，不再做组合穷举

### 3.1 先把实际解析规则打印出来

当前几项开关的规则并不一致：[S2][S3]

| 开关 | 当前源码行为 | 收敛要求 |
|---|---|---|
| `GGML_VK_CMD_REPLAY` | 未设置时计算子图重放关闭；用 `atoi` 判断 | 配置中明确写 `1` 或 `0`，不用模糊的 `on` |
| `GGML_TP5_CMD_REPLAY` | 未设置则继承上一项；两者都未设置时 collective 重放默认打开 | 两项分别声明，避免“开了 replay”却只开了一半 |
| `GGML_TP5_SYNC` | `timeline`、`syncfd` 有专门分支；未设置及未知值进入 HOST；`gpu/gpuflag` 明确拒绝 | 已验配置明确选模式；后续把未知值改为报错，不静默选慢路径 |
| `GGML_TP5_WIRE` | 只有精确的 `f32` 选择 F32，其余进入 F16 | 显式枚举并验证拼写，不能把输错参数当成一次有效消融 |
| `GGML_VK_PERF_LOGGER` | 启用后，现有计算子图重放资格检查直接返回 false | 带它的算子 profile 不能冒充 replay ON 的性能 |
| 多个 `GGML_VK_DISABLE_*` 及 logger 开关 | 部分按环境变量是否存在判断 | 这类开关设为 `0` 仍可能生效；关闭应真正 unset |

第一项工程交付应是**启动时输出一次 resolved configuration**：用户输入、实际取值、回落/拒绝原因、计算重放与通信重放分别是否启用。同时记录影响图优化、融合、异步、可见设备和内存策略的环境变量。不增加逐 token 的高频日志。

本轮不直接修改全局默认值。先有显式配置下的同条件证据，再决定默认路径。HOST、SYNC_FD 仍可作为明确的参考/诊断模式保留；它们不是 host-relay，也不能混为一谈。

### 3.2 只维护三个实验配置，生产配置最多一个

| 配置 | wire / sync | 计算重放 / collective 重放 | 用途 |
|---|---|---|---|
| REF | F32 / timeline | 0 / 0 | 功能参考，允许慢；与既有 CPU 数值参考共同使用 |
| BASE | F32 / timeline | 1 / 1 | 主干性能基线。建立时先单独开通信重放，再开计算重放，保留这一步的归因证据 |
| CAND | F16 / timeline | 1 / 1 | 在 BASE 上只改变 wire；通过精度与性能门后才考虑晋升 |

REF 在 timeline 下使用 one-shot collective plan 时会等待其完成再销毁，所以它慢并不奇怪，不能把销毁等待直接删掉换取分数。[S2]

此后每次只在最近通过验收的基线上改一个因素。已有 TopK、small-M、稀疏注意力、增量池化、PLE 等按固定构建和条件执行；不要一轮同时改模型、wire、重放、prompt、MTP 和 kernel 后寻找“最快组合”。已经失败的候选留在版本记录，不永久变成一个新开关。

## 四、先固定什么叫 60 tok/s

### 4.1 主指标与辅助指标分开

主指标是**单个逻辑请求、单条输出序列的稳态有效生成吞吐**。五卡共同计算一条序列，不能把五卡数量乘到 token 数上。主干与 MTP 分别报告；prefill、首字延迟、冷启动、恢复和总请求耗时另列。

```text
decode_tok_s = 对应测量窗口内实际输出的有效 token 总数 / 窗口耗时
```

必须说明窗口起止事件、是否包含第一个由 prefill logits 采样的 token，以及最后一次 flush 的归属。不能用一个口径计 token，另一个口径计时间。多次测试的总吞吐按 `sum(tokens) / sum(time)`，不直接平均各轮 tok/s。

draft 提议、被拒绝 token、强制喂入 token、重放计算都不增加有效生成数。与任务无关的重复输出不能作为“正确生成性能”验收；应保留其底层执行记录，但明确标为质量失败。

### 4.2 用固定 token tape 隔离数值，用自由生成验收质量

建议先建立一个可放进已审查小 context 的固定工作负载，例如 128-token 前缀和 256 步续写位置；最终长度以目标模型实际 token 化和资源预算为准。这只是隔离实验，不代表长上下文目标。

固定 tape 模式逐步喂入相同 token，核对对应 logits、关键输出及 state，避免两条实现从某一步采样分叉后无法比较。它**不是自由生成质量验收**。还必须用相同 prompt、模板、seed、采样和停止规则做自由生成，覆盖短推理、计数、代码或其它可检查结果，以及真实任务样本。

比较 `9.11/9.9` 的两条路径都答错，只能证明这个受测输出相同，不能证明 TP5 数值全局正确。比较 F16 wire 或新融合时，误差准则应在测试前按算子和端到端层级确定，不能事后放宽到“总算通过”。[T1]

每个候选至少保留五组配对运行；交替 A/B 顺序，分别标明 warmup、编译/缓存冷态和稳定段。短输出不足以覆盖稳定段时补充固定 tape 数据，不强制延长自由生成冒充质量提升。报告吞吐、逐步延迟分布、最大 rank 耗时、显存/RSS、FD 与错误事件。长上下文另建工作负载，不能通过缩短上下文宣称所有场景达标。

### 4.3 每份结果绑定完整身份

至少保存：Git commit 与未提交补丁身份、二进制和实际加载库哈希、构建选项、模型六个分片及 sidecar 身份、模板和 tokenized prompt、全部实际参数与环境变量、设备 BDF 顺序、内核/驱动身份、cache 状态和原始响应。`actual_model_validated: true` 不替代这些记录。[T5]

### 4.4 把目标写成时间预算，而不是性能预言

| 吞吐 | 平均有效输出时间 | 用法 |
|---|---:|---|
| 最近短请求约 2.38 tok/s | 约 420 ms | 说明差距，不能充当正式基线 |
| 10 tok/s | 100 ms | 第一项大幅改善的观察点，不是承诺 |
| 30 tok/s | 33.33 ms | 继续检查剩余主导项 |
| 60 tok/s | 16.67 ms | 当前目标 |
| 100 tok/s | 10 ms | 后续全集成目标 |

在当前 96 个主干归约边界的分解下，若入口、尾部、采样及其它不可归入主干 stage 的临界路径时间为 `T_fixed`，60 目标留给主干的**平均预算**为 `(16.67 ms - T_fixed) / 96`。把 97 个阶段均分约为 172 µs，只能帮助理解量级，不意味着每个阶段都具有相同成本，更不意味着 96 次归约独占全部预算。

先找单独就超过目标预算的项目。对于临界路径中独立占比为 p 的部分，即使彻底消除，整体加速上限也只是 `1/(1-p)`。多个有重叠的优化不能把宣传加速比直接相乘。

## 五、P0：先做可信的每步账本

### 5.1 修正计时归属，保留主机与设备两条时间线

在 meta 图执行入口创建本次统计上下文，至少携带：`request_id / graph_exec_id / target_or_draft / prefill_decode_verify / token_position / batch_tokens / rank / stage / collective_epoch`。这些是拟议内部诊断字段，不是新增公开 API。

图入口开始、最后一个尾部结束时关闭统计，不能靠某个函数被调用 96 或 480 次猜边界。主线程调用区间用单调时钟；同一条时间线给出互不重叠的分类，未归因部分明确留下。嵌套分项只解释父区间，不再次加总。[S1][S2][S3]

| 必须区分的部分 | 最少记录内容 |
|---|---|
| 图与录制 | build/reuse、eligible、hit/miss、miss 原因、录制与绑定时间 |
| 主机提交 | queue submit 调用数、每次批次数、CB 数、主机调用时间 |
| 等待 | 等待位置、目标 epoch、对象回收/最终读回/状态恢复等原因、阻塞时间 |
| GPU 工作 | 各 rank 的 compute、pack、P2P、sum 和尾部时间区间；实际 dispatch/copy 数 |
| 内存与资源 | 应用分配次数与字节、pool 增长、scratch 世代、descriptor/CB 创建和退休 |
| 外层工作 | PLE 读取/上传、logits 回读、采样、KV/state 保存恢复及未归因时间 |

不要求所有条目一次做成复杂 profiler。第一步先修图边界，加入低开销计数；再给主导阶段加设备计时。**不要为了观测而每层同步 GPU。**

### 5.2 设备计时不能悄悄关闭被测快路径

现有 `GGML_VK_PERF_LOGGER` 会影响 replay eligibility。用它找算子可以，但结果必须标明重放关闭，不能把其 profile 原封不动解释为快路径。[S3]

对重放路径，使用持久 query pool 和有界的采样窗口；在适当完成边界后批量读回，不逐 dispatch 等查询结果。处理 query 槽复用、CB 重放、在途生命周期及 reset 顺序；记录 `timestampValidBits`、`timestampPeriod` 和计时 stage。[V2]

各 GPU 原始 timestamp 不能直接相减。跨卡时间线具备校准能力时，记录共同主机时域及校准误差；否则先分别报告各卡区间与依赖，不伪造微秒级跨卡重叠图。校准也需要按漂移情况更新。[V3]

### 5.3 先回答五个问题，再选优化

1. 稳态每个 target 步实际有多少 eligible 子图？哪些重放未命中、为什么？
2. 等待时，GPU 在执行有用工作，还是因后续提交尚未到达而空闲？
3. 一次 API 提交对应多少批次、CB 和驱动 CS？
4. 暖态仍在创建的对象来自哪里、大小多少、是否因过早释放而重建？
5. 哪个 stage、哪个 rank 决定当前步完成？入口和尾部占多少？

**P0 的完成物**：一份可按真实 token/图执行切分的基准记录；一张带计时范围的时间账；一张热点排序表。若这些问题仍回答不了，不进入全图提交和大规模 shader 重写。

## 六、P1：先收敛现有路径，再改变提交结构

### 6.1 分开检查两个重放缓存

计算子图和 collective plan 是两类缓存，不共享一个“命中率”结论。分别记录冷态首次录制、稳态命中、形状/offset/参数/缓冲世代变化、资源淘汰等原因。性能应关注未命中造成的时间，而不是追求所有图 100% 命中。[S2][S3]

动态专家 ID、输入值、KV 写入位置、view 偏移和 scratch 世代都要进入真实回归。允许动态变化的内容必须由本轮正确输入提供；需要失效的绑定必须失效。不能因为 decode 大体同形，就断言所有指针和状态永远固定。

当前 collective 的 plan key 包含 buffer、offset、size、wire、workspace 世代等；只有证明工作集大于容量并发生反复淘汰，才调整缓存容量。不要直接把 64 改成更大数字再宣布消除了抖动。被淘汰的命令和所有引用资源仍须等设备完成后退休。[S2]

### 6.2 只持久化实测反复创建的资源

根据 P0 证据处理 CB、descriptor、scratch、staging 和元数据数组，预分配已知上界、按世代复用、在完成边界回收。保留显存上界和异常清理。只看到 GEM_CREATE 次数，不能证明扩大任意一个应用缓存会减少它。

meta 层通信前 flush，collective 入口又 flush，值得计数核对；但空 flush 可能不产生提交，不能先认定每次都重复下发。若证明确有重复，优先收成一个清楚的提交所有者，而不是在两层分别加跳过标志。[S1][S2]

**P1 的完成物**：稳态资源不再因已定位的可避免原因反复创建；重放失效均能解释；同条件正确生成更快，或者得到“录制/分配不是主导项”的明确否定证据。后一种也是有效结论，应据此停止深挖这一支。

## 七、P2：验证分段批量提交，不直接押注“每卡一次”

### 7.1 值得保留的方向

沿用现有五个 backend/device、分片计划和 collective。把“录制”“提交”“资源退休”分开，让上层可以拿到**带资源所有权的本地执行计划**，而不是只拿一个裸 `VkCommandBuffer`。[S1][S2][S3]

先试把相邻少量 stage 的提交组织成一批，例如 2 个，再根据证据扩到 4 个或更多。最终每卡每 token 一次 API 提交可以作为实验目标，但不作为先验最优解。提交数量太少也可能增加准备时延和在途资源占用；验收依据是关键路径，不是表面 API 数。

一个安全的计划至少拥有：对应 CB、descriptor、输入输出及临时 buffer 引用、scratch/绑定世代、同步依赖和完成后退休规则。不能复用普通路径中会在下一子图 reset 的资源，随后让 GPU 延迟读取它。

### 7.2 依赖链不能被“同一队列”四个字省略

当前主干应表达的是 96 个带归约的 stage，加一个独立尾部，而不是 97 次归约：

```text
stage s 本地计算
    → 本地输出可供 pack / PUSH 读取
    → 当前 epoch 的本地与远端载荷写入
    → 各生产者 release / signal
    → 接收方 wait / acquire
    → 固定顺序求和
    → 本 rank 下一 stage 的消费者

stage 95 完成 → 尾部计算 → 必要的最终读回与采样
```

Vulkan 的 submission order 本身不建立普通命令之间完整的 execution/memory dependency。合并到一个提交数组或同一队列后，本地 RAW 依赖与跨设备可见性仍要表达，不能删成“按顺序执行所以安全”。[V4]

精确化 barrier 时，逐个说明 buffer 范围、生产/消费 stage、access、外部内存及 queue-family 所有权规则。不要把 `ALL_COMMANDS` 机械替换成较窄掩码，就宣称固定减少若干微秒；也不要把主机等待当作正确外部内存协议的替代品。

### 7.3 先区分 owner 环、mailbox 槽与数学依赖

当前 `MAX_OUTSTANDING_EPOCHS=4` 对应在途对象管理及主机 drain。与此同时，timeline 的 P1 明确等待其它 rank 的 `2*(epoch-1)`，P2 等待其它 rank 的 `2*epoch-1`，自身 P2 完成后 signal `2*epoch`。前者约束前一轮消费与下一轮写入，不是可以不加替代就删除的慢代码。[S2]

因此，三件事必须分开：**主机最多提前准备多少轮、同一载荷地址何时允许重写、下一 stage 数学上何时可以读取结果。** 增大前两者的容量，不会消除第三者。

只有测出单槽复用或资源管理迫使了不必要的主机停顿，才引入 R 槽 mailbox 与消费 credit。每槽需要 `(communicator_generation, epoch, slot)` 的明确归属，旧读者全部完成后才能覆盖；申请 R 槽的显存、descriptor 和计划缓存也要计预算。不能预先断言 R=16 就让 141 ms 背压变成零。

timeline 的值在同一 semaphore 生命周期内必须单调推进，不能每个 token 重新从零开始。能力查询、最大在途差值、跨请求复用、计数溢出和销毁都要有边界。wait-before-signal 允许延后提交生产者，但相应生产者永远不再提交时，就可能无法推进。[V5]

### 7.4 失败路径先于大模型性能测试

先用离线事件模型验证读写次序和资源复用，再在硬件准入后做小尺寸真实 GPU producer/consumer：变更输入、延迟某个生产者、多次槽复用、非零 offset、形状切换以及受控提交失败路径。

部分 rank 已提交、另一个 rank 失败时，不能盲目 `DeviceWaitIdle` 等待永远不会到来的 peer signal；也不能由主机补一个“成功”信号，让消费者读取未完成载荷。保留现有拒绝继续的语义，为失败状态设计经过审查的退出和资源处理，不把一个通用析构函数当成解决方案。[S2][V5]

**GPU 显存 flag 自旋不进入本轮收敛主线。** 当前代码已明确拒绝 `gpu/gpuflag`。它同时引入跨设备可见性、调度前进性和 hang 风险，不是换个同步开关；若将来研究，须单列设计与安全准入，不依附这次 60 tok/s 验收。[S2]

**P2 的完成物**：数值与生命周期通过；有效 token 时间下降；API/批次/驱动提交数量有前后对照；最终等待没有把省下的中间等待原样搬回来。如果只有 API 次数下降、墙钟不降，应停止继续扩大提交批量，转向设备工作与依赖分析。

## 八、P3：按实测热点做融合，不按 PR 数量排进度

| 候选 | 具体落点 | 必须先拿到的证据与边界 |
|---|---|---|
| HC collapse | `build_hc_mix()` 的逐流加总、cont、scale | 实际派发与耗时；融合均值而非重写全部 HC，保持归一化、门控和量化语义 |
| HC 注入与残差 | `build_hc_combine()` 的 scale/sigmoid/broadcast/mul/add | 确认已有后端融合没有消除这些成本；避免物化 repeat，保留非连续布局与生命周期 |
| MoE 小矩阵和路由 | `MUL_MAT_ID`、TopK、shared/routed 本地合并 | 测真实专家数、选中数、每 rank 通道和量化；不跨非线性错误移动归约 |
| GDN / attention 的 rank 不均衡 | 头分配和实际 kernel 调度 | 看每 stage 最慢 rank，而不是平均时间；不要仅凭轮换角色就认定已经均衡 |
| QSA 稀疏与池化 | compact、实际 FA、增量缓存命中 | 分开短/长上下文，记录有效 KV 密度和 dense fallback；不得通过截断越界条目变快 |
| PLE 与尾部 | direct read/上传、最终 HC、LM head、logits 回读 | 计入 96 边界之外；缓存读和冷 I/O 分开，不假设它们免费 |

HC 首轮只选 collapse 或 combine 中实测最贵的一项。保留原公式参考，覆盖不同 token 数、dtype、view、量化和支持的 adapter 条件；一个局部融合的收益必须在整步账本中出现。[S4]

目标 manifest 给出的 MoE 是 **512 experts、top-10、每 rank 128 通道**。已有 TopK 算子证明采用过 **256 experts、22 tokens、top-6**，这不能直接证明单 token 主路径采用同一融合，更不能替代目标形状的性能测试。small-M/split-K 也要测实际量化和 M=1，以及后续 MTP 的小批量；切分更多不一定更快。[T3][T5]

当前 GDN state 头分配为 `[10,10,10,10,8]`，Q 头角色为 `[5,5,5,5,4]`，KV 角色中有一份双头。它们提示应该按 rank 看热点，但不单独证明哪张卡慢。优化应依据最长的实际执行区间，不能先重写分片方案。[T5]

QSA 的压实仍扫描掩码，稀疏 FA 才按活跃项工作。ctx512 上的主导成本与长上下文可能不同；不要用短请求否定长上下文优化，也不要拿上游长文本增益乘到当前短请求上。[T3]

## 九、MTP 单独算净收益，不用接受率代替速度

对一个投机周期，设 `A` 为最终实际输出的 token 数，包括被接受的 draft 以及该算法真实产生的 target token；已经计过的 token 不重复计入。则：

```text
MTP 有效 token 时间
  = sum(draft + verify + checkpoint + rollback + 其它不可重叠开销)
    / sum(A)

配对加速比
  = 同条件 MTP OFF 的有效 token 时间 / MTP ON 的有效 token 时间
```

接受 15/32 个 draft 不代表加速 47%，平均接受长度也不自动变成吞吐倍数。接受、拒绝、补采、状态保存恢复和多 token verify 都要记账。[T1]

当前计算子图 replay 资格要求 matmul token batch 为 1，多 token verify 不能直接继承这条资格。声称“MTP 只是同一个静态图一次多算几步”是不够的；必须核对 verify 实际使用的图、kernel、分配和命中率。需要扩展重放时，新增受验证的形状/状态规则，不能直接删除 batch 检查。[S3]

先测短 draft 长度的配对结果，再决定是否增加长度。保留连续请求、部分接受、全拒绝、KV 副本偏移、recurrent/PLE/QSA 回滚与取消恢复测试。没有稳定净收益的配置不进入性能默认，但实现和回归仍保留，不能为了快而破坏状态完整性。

## 十、实施顺序与停止条件

| 顺序 | 主要文件/职责 | 此阶段交付 | 不满足时怎么办 |
|---|---|---|---|
| P0 | `ggml-backend-meta.cpp`、Vulkan/collective 的配置和诊断 | 正确图边界、resolved config、可信基准与时间账 | 不启动全图重构；继续定位缺失证据 |
| P1 | 两类 replay cache、录制与资源管理 | 已定位的暖态重复工作消除，或确认它不是主导项 | 停止盲目扩缓存和调分配参数 |
| P2 | meta→Vulkan→collective 的提交与退休契约 | 有界分段提交、正确依赖、真实关键路径改善 | API 次数好看但时间不降，就不扩大批量 |
| P3 | `qwen4exp.cpp` 与对应 Vulkan kernel | 一项一项可归因的目标形状融合收益 | 无实际命中或净收益就不晋升 |
| P4 | speculative、state/KV、长上下文及服务集成 | MTP 净收益、长稳、质量、资源与恢复矩阵 | 不用局部成功替代全集成验收 |

P0 之后可以并行做彼此独立的离线代码工作，但真实目标机始终串行。测量、提交链和 kernel 不应同时修改同一条基线，否则无法归因。

### 10.1 每个性能改动只接受一张证据单

```text
假说：哪个已测区间由什么重复工作造成？
改动：只改变哪个机制，保留哪些数学和资源约束？
命中：目标模型确实进入了新路径吗？
前后：同一工作负载的 ms/token、分项、计数及资源峰值。
正确性：数值参考、自由生成、状态恢复和异常路径。
结论：晋升 / 保留实验 / 撤回；附 artifact、原始结果与回滚点。
```

把“值得晋升”的最小收益阈值在测量前约定。可以先采用超过基线波动、且配对中位时间改善至少 5% 的工程门；微小但必要的正确性/维护改动另行说明，不冒充性能成果。一个热点方向连续两轮没有支持证据，就回到时间账重排优先级，而不是再叠第三个开关。

任一数值失败、旧 epoch 污染、host-relay、无法清理的挂起、资源持续增长或新增硬件错误，均停止该候选晋升。禁止用强制关闭任务、少算层、减少专家、缩短目标上下文、丢 KV、放宽容差或重复无效输出换取达标数字。

### 10.2 达到 60 后也不能立即宣布完成

至少同时具备：冻结工作负载下可重复的有效吞吐；同条件质量非劣与数值门；上下文长度和 MTP 参与状态明确；逐卡显存和主机资源有界；持续请求及恢复无回归；安全准入和 artifact 可追溯。完整长上下文或全集成尚未测，就明确列出，不沿用短请求的结论。

**下一轮最有价值的交付不是第八个优化开关，而是：一份可信基线、一笔按真实步划分的时间账，以及一个在这笔账上确实减少了毫秒数的改动。**

## 十一、与现有三份文档的关系

TP5.md 保留模型数学、分片、P2P、状态与历史证据；最新状态覆盖旧段落，不重复执行已完成任务。《下班交接》保留交接线索，但其 per-token ioctl、硬件“彻底消除”和全面数值正确等扩大结论不作为验收依据。

TP5-FAST.md 的“提交计划＋timeline＋资源生命周期＋少量融合”可作为候选架构输入；以下内容降为待验证假说或予以修正：数学计算不足 1 ms、节点数直接充当 dispatch、任意固定单 dispatch 延迟、预计阶段加速路线、单次 API 提交必然大幅减少 CS、R 槽必然消除全部背压、同队列自动保证数据依赖，以及尚未证明的 60/100 tok/s 可达性结论。[T4]

这里不是要求再写第四套实现。恰恰相反：**只保留一个默认执行路径、一个完整参考路径和有期限的候选实验；让计时、数学、资源和发布状态各有一份明确依据。**

## 依据与定位

### 仓库记录

- **[T1]** [TP5.md](TP5.md)，开篇“2026-09-14 重新插卡验证与通信计时修正”：安全门、2.38 tok/s、96/97 边界、MTP 两次请求、ioctl 窗口与归因限制。本文未独立复读其引用的目标机原始日志。
- **[T2]** [下班交接.md](下班交接.md)，一、三、四、五节：当前吞吐、硬件结论、ioctl 解释和后续路线；本文明确指出其需要收回的扩大结论。
- **[T3]** [上游优化记录](docs/qwen4exp-upstream-optimizations.md)，最新纠正、状态矩阵及各项算子证据。这里引用的是本地移植记录，不是对上游 PR 当前状态的重新查询。
- **[T4]** [TP5-FAST.md](TP5-FAST.md)，第 0—4 节：预算、现状归因、epoch 链与预测；作为设计假说审查，不视为实测证明。
- **[T5]** [tp5-manifest.json](tp5-manifest.json)，开头的 `source_commit`、`model_identity`、`hparams`、`attention`、`gdn`、`memory_budget_per_rank`。

### 源码

- **[S1]** [ggml-backend-meta.cpp](ggml/src/ggml-backend-meta.cpp)：`ggml_backend_meta_graph_compute()`，本轮定位约 2845—2934 行；异步计算、flush、归约、debug 计数及 native communicator 失败处理。构造阶段能力检查约 2010 行起。
- **[S2]** [ggml-vulkan-collective.cpp](ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp)：`tp5_comm`、`tp5_allreduce_mesh()`、`ggml_backend_vk_tp5_comm_init()`；本轮定位约 170—176、863—1260、1280—1350 行。关注 owner 环、plan key、两阶段 timeline、计时范围和环境变量解析。
- **[S3]** [ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp)：`ggml_vk_cmd_replay_enabled()` 与 `ggml_vk_cgraph_decode_replay_eligible()`，约 2475—2525 行；`ggml_vk_submit()`，约 3355—3447 行；计算 profile 聚合，约 19615 行起。
- **[S4]** [qwen4exp.cpp](src/models/qwen4exp.cpp)：`build_hc_mix()` / `build_hc_combine()`，约 357—428 行。以现有公式和真实图为融合参考，不照抄“全部四流复制”的描述。

### Vulkan 官方规范与样例（2026-09-14 查询）

- **[V1]** [vkQueueSubmit](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueueSubmit.html)：一次调用可包含多个 submission batch；不能由调用数推导内核 job 数。
- **[V2]** [Queries](https://docs.vulkan.org/spec/latest/chapters/queries.html)：设备 timestamp、有效位、周期和查询操作的约束。
- **[V3]** [Calibrated Timestamps](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_calibrated_timestamps.html)：跨时域校准、最大误差与漂移。
- **[V4]** [Synchronization and Cache Control](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)：submission order 与 execution/memory dependency 的区别。
- **[V5]** [Timeline semaphore 官方样例](https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/README.html)：单调计数、wait-before-signal、提交深度和停止时的前进性问题；样例不是本项目跨设备内存正确性的替代证明。

# 验证记录：2026-10-18 小功能补全

当前 A 为固定上游 `cda0f9256665845cd9d6f3a0176880883f653a3d` 加 [16 个登记文件](upstream-exceptions.json)。本记录更新第一轮“零 A 修改、两个 B”的结论，区分真实执行、静态审计和历史性能资料。可移交的运行摘要保存在 [audit/validation-20261018.json](audit/validation-20261018.json)；原始 stdout、stderr、命令及测试结果保留在本 worktree 的对应 `build-*` 目录中。

## 实际参与者

- Linux x86-64，GCC 16.2.1、Python 3.14。
- 模型：`/home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf`。
- 模型 SHA-256：`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`。
- 实际 Vulkan 设备：单张 AMD Radeon RX 6800，RADV / Mesa 26.2.2。
- CPU 与 Vulkan 是分别执行的对照；没有把 GPU 卸载参数当作设备执行证明。实际模型日志确认 25 层卸载。

## 验收概览

| 检查 | 实际结果 | 证据 |
|---|---|---|
| A 边界 | 16 路径，324 行新增 / 39 行删除；路径、内容 SHA-256、mode 和消费者均匹配 | `check-upstream.py`、manifest |
| 边界反例 | 12 个真实临时 Git 仓库测试通过 | `tests/test-upstream-boundary.py` |
| 完整源码与安装隔离 | 41 个命令全部通过，包含重新构建 A 和搬迁后真实模型运行 | `build-isolation-small-20261018/result.json` |
| Vulkan 组件 | CTest 10/10 通过，含独立 B 行为、真实 A 采样、CLI 加载及三个 A 接缝测试 | `build-clean-vulkan/extensions/Testing/Temporary/LastTest.log` |
| 独立构建脚本 | CPU 的 A→三个 B→三个 C→接缝测试完整流水线退出 0 | `build-clean-fork/standalone-build-small.log` |
| 三请求阶段切换 | CPU / Vulkan 各自开关对照通过；每臂 22 批、145 行 | `build-phase-{cpu,vulkan}-20261018/` |
| 原版 server | 两路生成时插入第三路 Prefill；48 / 48 / 8 token 完整返回，两个阶段周期 | `build-server-phase-verified-20261018/` |
| 生产 corpus 互操作 | 未修改的旧 writer → 新 B → 旧 reader，字节往返通过 | `build-corpus-interop-verified-20261018/` |
| 已保留入口复核 | 新 A / B 构建上的 CPU、Vulkan sampling / corpus 真实模型比较均通过 | `build-model-small-{cpu,vulkan}-20261018/` |
| 独立 logits 比较器 | 20 项真实 CLI 测试通过 | `fork/tools/tests/test-compare-logits.py` |

这些测试不是上游全部测试套件。构建、help、纯库测试、真实模型和性能测量分别记录，不互相替代。

## Prefill / Decode：真正发生了什么

`fork/batch-compose/tests/real-batch.py` 启动真实 C，使用已安装的 A 与 B。先让两个请求生成两批，再接纳第三个提示；逐条检查真实 `llama_decode` 输入的请求 ID、位置、token 和 logits 标记。验证 Prefill 期间选择提示请求、完成后原 Decode 请求继续，且已采样 token 真正成为后续模型输入。

每个后端各运行释放开启、关闭两臂。配置为每请求逻辑窗口 512、物理 KV 1024、batch 16、ubatch 8，每请求生成 12 token。A 的公开 getter 验证逻辑 / 物理设置确实分离；每臂都提交 22 批、145 行，开关两臂的所有生成 token 和输入行完全相同。Vulkan 两臂的实际卸载记录均为 25 层。

| 当前计算分配器字节数 | CPU | Vulkan |
|---|---:|---:|
| 关闭释放时维持的工作区 | 1,726,848 | 2,846,956 |
| 第一次 PP→TG 释放后 | 0 | 0 |
| 第一段两请求 Decode 重新分配后 | 1,074,368 | 1,168,472 |
| 插入 Prefill 后第二次 PP→TG 释放后 | 0 | 0 |
| 三请求 Decode 重新分配后 | 1,611,456 | 1,738,356 |

这里读的是 `llama_memory_compute_bytes()`，表示 context 的计算分配器当前拥有的 buffer 字节数。它排除模型、KV、recurrent、输出和 provider cache，不是显卡驻留量、进程峰值或系统回收显存。这个小负载证明了释放与按实际图重建；它没有测量吞吐收益，也不证明扩大物理 KV 后注意力工作区不增长。

同一脚本还实际运行不足的物理 KV 配置与缺失模型，要求 A 明确失败。非法配置不能用模拟日志或伪造成功替代。

### 分配器生命周期

`tests/a-seams/compute-memory.cpp` 使用真实 GGML CPU 后端：大图分配、排队异步读回、由 release 自己完成 fence、字节归零、重复 release、小图重建、再扩回大图。它检查保留输入/权重、scheduler 和 callback 身份，并执行重建后的计算。Release 构建中的失败检查始终有效。

API 前置检查拒绝 metadata-only context、活动 optimizer 和共享 CUDA phase arena；正常路径同步、失效两个可复用图结果，再释放分配器。模型 / KV / recurrent / output 的所有权保留由源代码边界与真实输出对照支持；没有宣称跨 context 共池或 target / draft 联合分配已经实现。

## 原版 HTTP server

`tests/server-phase.py` 只启动并清理自己创建的 localhost server。真实设置为 logical 512、physical KV 1536、3 slots、Vulkan。两个 SSE 流各收到至少两个生成块后插入第三个长提示；初始两条流当时均未在客户端结束。

三条请求分别完成 48、48、8 个 token，槽位和 limit 结束原因正确。日志出现 `prefill → generation → prefill → generation`，实际卸载 25 层，`n_ctx_kv=1536`。关闭 Prefill 优先但开启阶段释放的组合，在加载不存在的模型前就报出专用参数错误。

初次 smoke 的三条请求也都完成，但默认日志隐藏 loader 信息，导致设备证据断言失败。最终脚本启用 `--verbose --log-colors off`，保留原来的正数卸载 / 物理容量断言，完整复跑通过，未放宽检查。详见 `build-server-phase-verified-20261018/summary.json` 与 `server.log`。

HTTP 接收时间不能证明内部批次严格暂停；逐批暂停 / 恢复的证据来自上一节 C 的真实批次输入。此次 server smoke 没有证明 MTP、语法、随机 RNG、取消或压力重试等全部协作组合。阶段释放后，A 析构时可能打印实际计算 buffer 与初始预留预测不同的诊断；当前观察接口和结果以实际分配及输出为准。

## 两个小型正确性修复的 before / after

### CPU：显式 F32 FlashAttention

针对 F16 V、显式 F32 precision 的实际 GGML CPU 操作，在相同测试中：

| KV 长度 | 固定 A 原实现最大绝对误差 | 修复后最大绝对误差 |
|---|---:|---:|
| 256 | 0.000388145447 | 0.00000011920929 |
| 1024 | 0.000388026237 | 0.00000011920929 |

原实现退出失败；四行级别的累加路径修复后通过。用例为 `tests/a-seams/flash-attn-precision.cpp`。这不是一次整模型质量评分，也不应归为无收益的量化实验。

### Vulkan：公开 Turbo WHT

同一个 `tests/a-seams/vulkan-wht.cpp` 直接分别调用真实 CPU 与 Vulkan backend，没有混合调度器的 CPU fallback。12 例覆盖 group 32 / 64 / 128、正反变换、513 行跨工作组以及既有 scale 参数。

原 A 失败 10 / 12，仅未缩放 group128 的两个控制组通过。修复两个 A 文件后 12 / 12 通过，本设备最大绝对误差为 0。输出先填 NaN，遗漏写入会失败；所有数值须有限。当前支持仍限制在 128 对齐物理行和已验证 scale 布局，shader 越界返回保持整个 workgroup 一致。

记录：`build-clean-fork/wht-after.log`、`wht-vulkan-build.log`、`wht-vulkan-install.log`；完整复现合同见 [tests/a-seams/vulkan-wht-protocol.md](tests/a-seams/vulkan-wht-protocol.md)。

## 生产 corpus 格式与旧入口复核

互操作脚本从旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 导出真正的 `wanxiangqi/server/server-corpus-dump.h/.cpp`，不改其代码，与已安装 A 头文件及独立 TQCorpus 编译运行。旧 writer 产生混合和空请求 corpus，新 B 读取 / 保存后逐字节比较，最后由旧 reader 读取；验证前后原输入文件保持不变。

覆盖四种 kind、64 位时间戳、空请求、重复请求、前缀终点与追加。旧源码 SHA-256：

- `.h`：`0ce1d80aff1b75aab67c2d3c36a8f40e632ffbeca93c214e8c4c7995d0ef6209`
- `.cpp`：`f031635a31c9c4a9c40633d99b09d70365659371beff6b353a3e743fe97518e7`

本次修改 A 与 corpus 后重新运行 `tests/real-model.py`，没有仅引用第一轮结果：

| 当前构建上的真实对照 | CPU | Vulkan |
|---|---|---|
| greedy compact / 完整行 | 24 / 24 token 相同，compact 实际执行 24 次 | 相同 |
| 正温、固定 seed 的完整采样 | 24 / 24 token 相同 | 相同 |
| 6 请求前缀 / 独立重放 | 实际 decode 44 / 120 token | 实际 decode 44 / 120 token |
| 整体 mean NLL 绝对差 | 0 | 0 |
| 最大逐请求及反转顺序差 | 1.8947806286936005e-15 | 9.473903143468002e-16 |
| 预设 NLL 误差门 | 1e-5 | 1e-5 |
| 上下文错误、拒绝覆盖、失败后文件不变 | 通过 | 通过 |

两种不同后端之间没有宣称 logits 相同；NLL 数字只是本测试语料的条件分数。当前 CPU 的 prefix / reference 均为 5.2485678755225615，Vulkan 均为 5.15723864954567。误差门没有根据结果放宽。

## A / B / C 独立构建与搬迁

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-small-20261018 --jobs 4 \
  --model /absolute/path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
```

以上为可替换模型路径的复现示例，实际命令保存在该目录 `commands.json`。本轮没有使用 `--a-prefix`：A 从 pin 的 Git archive 构建，只覆盖 manifest 已验证的 16 个完整文件，导出 A 不含 `fork/`，其自身 `test-sampling` 通过。

三个 B 分别从只含自身代码的副本构建 / 测试 / 安装；三个 C 分别只消费安装依赖。流程删除源码和构建副本、搬迁安装前缀，再编译实际包消费者，并执行搬迁后的真实 batch 模型比较。41 个命令全部成功。B 的输入 / 输出所有权和实例隔离测试不需要模型服务；C 的模型证明使用真正 A 和 B。

`--a-prefix` 是可选便捷路径，记录为 `reused_unverified`，不能拿来证明该次重新构建了对应 A。独立源码副本和安装消费也不等于操作系统沙箱。

## 性能资料的边界

第一轮实际 CPU 采样微基准仍作为单一阶段证据保留：248,320 logits、top-k40、唯一最大值，每臂预热 10、计时 200 次，4 个 AB / BA 配对；完整行约 252–312 µs / row，compact 约 29.9–30.0 µs / row。每次使用真实 A sampler 检查 token。这不是整模型的加速倍数。

旧 fork 的 +3.66% compact、TP5 LateBind、router 与 MMVQ / MTP 数据是归档历史，不是这次新分支测量；具体取舍见 [MIGRATION.md](MIGRATION.md)。本轮阶段内存测试、WHT 与 F32 正确性测试都没有给出新吞吐结论。

## 未覆盖与明确限制

没有证明全部模型、采样链、极长上下文或跨设备组合；CUDA、HIP、Metal、Windows、macOS 未取得本次运行资格。没有证明多 GPU / 五卡 TP5、RERoT 多 reader 共享 KV、FlashPrefill / XKV 整模型质量、设备隐藏层交接、自动显存 fit、压力抢占、跨 context 共池或随机 MTP 恢复。

Corpus 未提供 tokenizer 身份、并发录制、崩溃事务和多分片读取；比较器的文件格式也不携带完整构建 / 模型身份。逐文件静态审计中已识别但未运行的能力保留自己的证据等级，不能因本页中其他测试通过而自动获得资格。

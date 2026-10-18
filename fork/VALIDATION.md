# 验证记录：2026-10-18 发布候选补全

当前 A 固定为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。登记例外共18个文件：17个 A 代码文件合计 +328/-43，另有49行自有 CI 工作流。本轮新增的生产 A 修改只有 Meta 异步偏移一个文件的 +4/-4；其余新行为由 B、C 或独立测试承担。

本轮可移交摘要为 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)，记录关键源码摘要、模型身份、实际命令回执、反例和适用范围。原始 stdout、stderr、二进制 tape 与完整命令保留在对应 `build-*` 目录。本页后半保留同日上一轮记录，并明确标为历史验证，不能将其未复跑的部分自动当作本轮结果。

## 本轮实际参与者

- Linux x86-64，GCC 16.2.1；主测试 Python 3.14，CMake CLI 测试使用 Python 3.12.13。
- 模型：`/home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf`。
- 模型 SHA-256：`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`，本轮重新核对。
- Vulkan：单张 AMD Radeon RX 6800，RADV / Mesa 26.2.2。模型日志实际确认25层卸载；没有把请求的 GPU 参数当作执行证据。

## 本轮验收概览

| 检查 | 实际结果 | 主要证据 |
|---|---|---|
| A 边界与反例 | 18项内容 / mode / 消费者匹配，12个临时 Git 仓库测试通过 | manifest、`check-upstream.py`、`tests/test-upstream-boundary.py` |
| 有界预算 B | 2/2 CTest；双资源、失败保持、陈旧 / 外来 / 重复释放及整数边界 | `scheduling/tests/admission-test.cpp`、独立安装测试 |
| 队列与槽位复用 C | CPU、Vulkan各7个运行、60个请求成功；固定配置的阶段开关对照相同 | `build-release-admission-qualified-{cpu,vulkan}/` |
| Meta async 偏移 | 原 A 零偏移16控制通过、非零 get/set 分别断言；修复后48例通过 | `build-release-meta-before/summary.json`、Meta 协议 |
| recurrent 生命周期 | CPU、Vulkan各24邻居 / 8共享快照检查，恢复继续最大绝对差0 | `build-release-seams-{cpu,vulkan}/` |
| 完整 logits 采集 | 每后端15行×128000值，重复 tape 按位一致；标量形状行选择差0 | `build-release-logits-real-{cpu,vulkan}/` |
| logits 文件和 CLI | CPU、Vulkan链接构建均2/2，C++17严格警告；独立 UBSan writer通过 | `logits-compose/tests/` |
| 独立比较器 | 20项真实 CLI 测试通过 | `tools/tests/test-compare-logits.py` |
| 独立交付 | 47个命令成功，重新构建 A、三个 B、四个 C，搬迁后消费者及真实模型通过 | `build-isolation-release-20261018/result.json` |
| 构建入口 | CPU默认、Vulkan `--tools` 完整流水线均退出0 | `build-release-cpu.log`、`build-release-vulkan.log` |
| GitHub CI | 已配置无模型完整隔离流水线；本地对应流程通过 | 工作流尚未推送或在 GitHub 执行 |

构建、纯库测试、真实模型、数值对照和性能测试是不同证据。本轮没有吞吐或峰值显存测量，不是上游全部测试套件的通过声明。

## 准入、排队与真实状态释放

`admission_budget` 独立拥有 token 和驻留槽位预算。C 用实际 `llama_n_ctx_kv()` 初始化预算，每次预留已知上界 `prompt_tokens + predict - 1`。请求 ID 与 A 的可复用 seq ID 分开；reservation 带 owner domain 与 generation。队列可长于 batch 和驻留槽位数。

C 消费完当前批次所有输出并同步后，先成功调用 `llama_memory_seq_rm()`，再释放预算、增加可用槽位。B 的 reservation 析构不会代替 C 删除真实状态。容量不足留在 FIFO 队列；不能分配的单请求明确失败。本轮没有引入 HTTP 取消、抢占、LRU、失败重试或未知长度流式增长。

`batch-compose/tests/test_admission.py` 在每个后端执行以下七个运行。每请求135个提示 token，生成16个 token，实际 decode 行数150。

| 配置 | 请求数 | 驻留上限 / 物理 KV | 验证 |
|---|---:|---|---|
| three-wide，phase off/on | 每臂9 | 3 / 2048 | 相同配置两臂完整输入行与生成结果相同 |
| token-pressure，phase off/on | 每臂9 | 3 / 256，实际仅驻留1 | token预算等待、回收、固定配置结果相同 |
| slot-pressure，phase off/on | 每臂9 | 1 / 2048 | 槽位预算等待，和上述串行运行的结果相同 |
| repeat-wave | 6 | 3 / 2048 | 同三个提示在相同槽位复用后与第一波新槽位完全相同 |

合计每后端 `6×9+6=60` 个成功请求；两后端共120个。检查真实提交行、token反馈、位置、请求完成、generation、预算守恒，以及 phase release 前正数 / 后零的计算分配字节。等待计数按调度观察点累积，不能当作独立请求数或延迟样本。这里的120不是吞吐样本。

### 保留的跨批形状反例

最初要求并行与串行 greedy token 完全一致的测试失败。随后在本轮准入修改前的原 C 二进制上，关闭 phase release、保持提示与模型相同，独立复现三路与单路的结果差异；单路 KV2048 与 KV256 则完全相同。

原 C 来自本轮起点 `3746ebf82585f47c88fb48097f6ae45cb972b031`。其 CPU、Vulkan二进制摘要、实际命令、确切 UTF-8 stdin 与返回 token 保存在本轮摘要的 `shape_counterexample_before_admission`。CPU组选取原fixture的3/4/5，Vulkan选取0/1/2；已将全部提示token行逐条对上原运行记录。它们后来被完整安装覆盖，复现时应从该提交重建。最初失败记录仍在 `build-release-admission-cpu/run-cga4dqjq` 与 `build-release-admission-vulkan/run-cc0q5fad`，没有改写成成功。

本轮队列测试的 CPU 在请求4的第5个生成 token 出现差异，Vulkan在请求1、4、7出现差异。开关 phase release不改变各自结果。这个证据建立了修改前已存在的批形状敏感性，**没有定位数值原因**，也不能据此认定差异无害。验收明确比较同配置形状、串行容量对照和同槽位重复；跨形状差异继续单列。

## logits：全行采集与有限的独立对照

新 C `tq-logits-compose` 用真实 tokenizer 产生输入 token，以 teacher forcing 调用 A。默认逐 token 保存完整词表行；前缀模式只保存前缀最终行，再保存后续各行。每次 decode 后同步，行内非有限值立即失败。完整尺寸先检查，writer不保存历史行。

CPU、Vulkan各执行11条命令：完整采集、重复采集、标量形状suffix、标量形状prefix-only、chunked-prefix、旧比较器、新文件拒绝覆盖、上下文不足、输出尺寸不足、独立 batch C对照，以及 Linux `/dev/full` stdout故障。

| 验证内容 | CPU | Vulkan |
|---|---:|---:|
| 输入 token / 词表 / 完整行 | 15 / 128000 / 15 | 15 / 128000 / 15 |
| 重复 tape | 全部192万值及文件字节相同 | 相同 |
| 同标量形状suffix全行最大绝对差 | 0 | 0 |
| 同标量形状prefix-only全行最大绝对差 | 0 | 0 |
| 独立 batch C：相同提示行，最终greedy token / 最大logit，门限1e-5 | 通过 | 通过 |
| chunked vs scalar最终全行最大绝对差，仅诊断 | 2.6373519897460938 | 0.9928340911865234 |
| stdout写失败 | 非零退出；已完整发布的tape保留且字节正确 | 相同 |

独立 batch C只提供最终greedy token和最大logit，因此这个独立对照**没有比较第二个实现的全部词表值**。重复tape、suffix/prefix选择与chunked/scalar诊断则读取并比较实际完整行。chunked/scalar差异不能称作已证明无害的舍入误差；没有跨后端相等或模型质量声明。

Writer单测直接检查little-endian字节、负零、尺寸、token范围、NaN、缺行/多行、已存在文件、symlink、竞争发布者、临时文件样式目标名和真实文件大小限制故障。写或sync失败是终止状态；不能捕获后继续发布部分文件。完整文件先发布，stdout随后失败则明确报出“文件已完整发布”。独立UBSan检查通过；ASan环境未取得可靠资格，本记录不宣称ASan通过。

TP5LOG2头不含模型、tokenizer、构建和设备身份，应连同命令、日志及摘要保存。发布使用POSIX同目录hard link与调用者控制的目录；不承诺Windows、恶意目录写者防护、断电目录持久性或信号时临时文件清理。接口和复现命令见 [logits-compose/README.md](logits-compose/README.md)。

## Meta：公开异步传输的 before / after

新回归直接创建两个真实CPU child backend，覆盖split axes0/1/2与mirrored、F32/F16、3+7不均分片、full/middle/tail、独立get和set，以及前后guard与未触及字节。

原A的零偏移16个控制通过，非零get、set单独运行均在旧 `offset == 0` 断言处以SIGABRT(-6)失败。A修复只移除该过度限制，并把逻辑chunk起点换算为每个child自己的偏移。修复后48例通过。CPU安装的全部A接缝为4/4，Vulkan链接安装为5/5；**Meta用例在两次运行中都使用CPU child**，没有取得GPU异步、多GPU或异构Meta的运行资格。

连续、单segment、chunk对齐与repeat条件保持。没有新增API、后台队列、cache、pin或所有权协议。完整形状与检查方式见 [meta-transfers-protocol.md](tests/a-seams/meta-transfers-protocol.md)。

## recurrent：直接复用当前 A

源码复查表明：当前A在构造与 `clear(true)` 时整buffer清零；`seq_rm` / `seq_keep` 修改元数据，通过空闲zero row与图内初始化/聚集取得新状态。旧M33性能记录减少的是旧fork自己加的eager clearing；把它搬到当前A会增加写入和fence，本轮没有修改A的recurrent代码。

公开接口测试使用上述LFM2.5 hybrid convolution/KV模型，实际 `n_rs_seq=0`。CPU、Vulkan各执行24次decode、189行输入；24个邻居快照与8个共享快照检查通过，合法恢复后同形状teacher-forced继续的最大绝对差为0。覆盖三个槽位两次复用、COW存活者、dirty槽位恢复、`clear(false)` / `clear(true)`再填充，以及不支持的部分回滚失败后原快照不变。batch输入在显式同步前始终存活。

这不等于一般GDN/recurrent行数、MTP、PLE、HTTP取消或跨进程snapshot兼容性证明。源码另发现坏restore可能在完整验证前改变目的状态；**没有宣称损坏snapshot恢复具有事务性，也没有在本轮修复它**。操作边界见 [recurrent-lifecycle-protocol.md](tests/a-seams/recurrent-lifecycle-protocol.md)。

## 独立交付、构建入口和 CI

本轮实际运行：

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-release-20261018 --jobs 4 \
  --model /home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
python3 fork/build.py --backend cpu --build-dir build-clean-fork --jobs 4
python3 fork/build.py --backend vulkan --tools --build-dir build-clean-vulkan --jobs 4
```

47个隔离命令全部成功，A来源标记为 `built_from_pinned_archive_and_verified_overlay_bytes`，没有使用 `--a-prefix` 复用捷径。A从pin的Git archive导出，只覆盖18个经验证的完整文件；A源码没有 `fork/`。三个B、四个C从各自源码副本构建，B不调用A或兄弟B，C只查找安装包。

删除原源码/构建副本和原安装前缀后，搬迁后的真实包消费者仍能编译运行；新admission API参与外部消费者编译，安装后的batch模型与logits模型测试通过。logits的比较器和测试脚本在删除源码前拷出。这个流程证明安装消费与来源隔离，不等于操作系统沙箱。

两个真实构建入口退出0。Vulkan `--tools` 路径补建A自身tools安装要求的 `mtmd`，从C的构建脚本解决缺失安装依赖，没有改A的CMake。工具可执行文件位于对应 `upstream/bin/`。

新增 `.github/workflows/fork-extensions.yml` 配置只读权限、独立无模型CPU隔离检查和失败回执上传；没有改上游原工作流。本地完整流程已通过，GitHub工作流尚未推送或执行，不能称作远端CI绿灯。真实模型、Vulkan与可选recurrent资格仍由显式本地命令提供。

## 本轮范围与后续候选

没有新的整模型吞吐、reset加速、峰值VRAM或自动容量适配结论。CUDA、HIP、Metal、Windows、macOS、多GPU与异构Meta没有取得本轮资格。

auto-fit/slack、PP reserve、idle LRU、物理recurrent槽、HTTP取消/恢复、流式stop/usage、在线corpus、状态envelope、校准/imatrix和其他局部算子继续逐项列在 [MIGRATION.md](MIGRATION.md)。714路径的审计分母与546完整改动内容/168选定范围的阅读记录保持不变，不能用这些测试覆盖所有历史实现。

## 同日上一轮验证记录（保留原证据边界）

以下段落中的“本次”指本轮起点之前的小功能补全；当时为16个A代码例外、三个B和三个C，隔离流程41步。本轮重跑项目以上面的记录为准。其可移交摘要仍是 [audit/validation-20261018.json](audit/validation-20261018.json)。保留旧数值用于回查，不将未重跑的HTTP、corpus互操作或采样测量写成本轮新结果。

### Prefill / Decode：真正发生了什么

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

#### 分配器生命周期

`tests/a-seams/compute-memory.cpp` 使用真实 GGML CPU 后端：大图分配、排队异步读回、由 release 自己完成 fence、字节归零、重复 release、小图重建、再扩回大图。它检查保留输入/权重、scheduler 和 callback 身份，并执行重建后的计算。Release 构建中的失败检查始终有效。

API 前置检查拒绝 metadata-only context、活动 optimizer 和共享 CUDA phase arena；正常路径同步、失效两个可复用图结果，再释放分配器。模型 / KV / recurrent / output 的所有权保留由源代码边界与真实输出对照支持；没有宣称跨 context 共池或 target / draft 联合分配已经实现。

### 原版 HTTP server

`tests/server-phase.py` 只启动并清理自己创建的 localhost server。真实设置为 logical 512、physical KV 1536、3 slots、Vulkan。两个 SSE 流各收到至少两个生成块后插入第三个长提示；初始两条流当时均未在客户端结束。

三条请求分别完成 48、48、8 个 token，槽位和 limit 结束原因正确。日志出现 `prefill → generation → prefill → generation`，实际卸载 25 层，`n_ctx_kv=1536`。关闭 Prefill 优先但开启阶段释放的组合，在加载不存在的模型前就报出专用参数错误。

初次 smoke 的三条请求也都完成，但默认日志隐藏 loader 信息，导致设备证据断言失败。最终脚本启用 `--verbose --log-colors off`，保留原来的正数卸载 / 物理容量断言，完整复跑通过，未放宽检查。详见 `build-server-phase-verified-20261018/summary.json` 与 `server.log`。

HTTP 接收时间不能证明内部批次严格暂停；逐批暂停 / 恢复的证据来自上一节 C 的真实批次输入。此次 server smoke 没有证明 MTP、语法、随机 RNG、取消或压力重试等全部协作组合。阶段释放后，A 析构时可能打印实际计算 buffer 与初始预留预测不同的诊断；当前观察接口和结果以实际分配及输出为准。

### 两个小型正确性修复的 before / after

#### CPU：显式 F32 FlashAttention

针对 F16 V、显式 F32 precision 的实际 GGML CPU 操作，在相同测试中：

| KV 长度 | 固定 A 原实现最大绝对误差 | 修复后最大绝对误差 |
|---|---:|---:|
| 256 | 0.000388145447 | 0.00000011920929 |
| 1024 | 0.000388026237 | 0.00000011920929 |

原实现退出失败；四行级别的累加路径修复后通过。用例为 `tests/a-seams/flash-attn-precision.cpp`。这不是一次整模型质量评分，也不应归为无收益的量化实验。

#### Vulkan：公开 Turbo WHT

同一个 `tests/a-seams/vulkan-wht.cpp` 直接分别调用真实 CPU 与 Vulkan backend，没有混合调度器的 CPU fallback。12 例覆盖 group 32 / 64 / 128、正反变换、513 行跨工作组以及既有 scale 参数。

原 A 失败 10 / 12，仅未缩放 group128 的两个控制组通过。修复两个 A 文件后 12 / 12 通过，本设备最大绝对误差为 0。输出先填 NaN，遗漏写入会失败；所有数值须有限。当前支持仍限制在 128 对齐物理行和已验证 scale 布局，shader 越界返回保持整个 workgroup 一致。

记录：`build-clean-fork/wht-after.log`、`wht-vulkan-build.log`、`wht-vulkan-install.log`；完整复现合同见 [tests/a-seams/vulkan-wht-protocol.md](tests/a-seams/vulkan-wht-protocol.md)。

### 生产 corpus 格式与旧入口复核

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

### A / B / C 独立构建与搬迁

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-small-20261018 --jobs 4 \
  --model /absolute/path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
```

以上为可替换模型路径的复现示例，实际命令保存在该目录 `commands.json`。本轮没有使用 `--a-prefix`：A 从 pin 的 Git archive 构建，只覆盖 manifest 已验证的 16 个完整文件，导出 A 不含 `fork/`，其自身 `test-sampling` 通过。

三个 B 分别从只含自身代码的副本构建 / 测试 / 安装；三个 C 分别只消费安装依赖。流程删除源码和构建副本、搬迁安装前缀，再编译实际包消费者，并执行搬迁后的真实 batch 模型比较。41 个命令全部成功。B 的输入 / 输出所有权和实例隔离测试不需要模型服务；C 的模型证明使用真正 A 和 B。

`--a-prefix` 是可选便捷路径，记录为 `reused_unverified`，不能拿来证明该次重新构建了对应 A。独立源码副本和安装消费也不等于操作系统沙箱。

### 性能资料的边界

第一轮实际 CPU 采样微基准仍作为单一阶段证据保留：248,320 logits、top-k40、唯一最大值，每臂预热 10、计时 200 次，4 个 AB / BA 配对；完整行约 252–312 µs / row，compact 约 29.9–30.0 µs / row。每次使用真实 A sampler 检查 token。这不是整模型的加速倍数。

旧 fork 的 +3.66% compact、TP5 LateBind、router 与 MMVQ / MTP 数据是归档历史，不是这次新分支测量；具体取舍见 [MIGRATION.md](MIGRATION.md)。本轮阶段内存测试、WHT 与 F32 正确性测试都没有给出新吞吐结论。

### 未覆盖与明确限制

没有证明全部模型、采样链、极长上下文或跨设备组合；CUDA、HIP、Metal、Windows、macOS 未取得本次运行资格。没有证明多 GPU / 五卡 TP5、RERoT 多 reader 共享 KV、FlashPrefill / XKV 整模型质量、设备隐藏层交接、自动显存 fit、压力抢占、跨 context 共池或随机 MTP 恢复。

Corpus 未提供 tokenizer 身份、并发录制、崩溃事务和多分片读取；比较器的文件格式也不携带完整构建 / 模型身份。逐文件静态审计中已识别但未运行的能力保留自己的证据等级，不能因本页中其他测试通过而自动获得资格。

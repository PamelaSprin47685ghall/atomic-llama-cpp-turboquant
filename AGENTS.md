# AGENTS.md｜TP5 钢铁人基线与双边最优性研究规程

更新：2026-10-10。仓库：`~/atomic-llama-cpp-turboquant`。适用：本机 5 × RX 6800，Qwen3.8-Flash-Next-GSQ-RCO-Q2_0，单路自然文本生成，**RERoT 关闭**。

> **本文件是后续代理的执行指令，也是可照做的操作手册。** 先强固 TP5，再用它公平评价 RERoT。当前尚无通过新契约的 Q2_0 最强基线，也无覆盖全部允许算法的硬件必要下界。这里写的是执行顺序、待实现设计与验收条件；**不能把工单写成已完成结果**。更新本文件本身不表示执行过 GPU 测试。

## 0. 阅读顺序、判断优先级与十条红线

按顺序阅读：本文件 → [PERFORMANCE.md](PERFORMANCE.md) → [docs/performance-necessity.md](docs/performance-necessity.md) → 指定轮次的原始 `manifest.json`、`request-*.json`、`server.log`、`result.json` → 当前源码与测试。`TP5.md`、`MMVQ+MTP.md`、旧交接和历史 decision 只作证据索引，不作新的“已关闭”指令。

发生冲突时，以**用户最新明确裁决、当前实际源码/模型/二进制/DSO、可复核原始证据**为准；本文件的过时假设必须修正并注明。任何单次实验的“终局”不禁止在新模型、新机制下重新检验。

1. **不得损毁现场。** 先看 `git status`、现有 dirty diff、未跟踪源码、测试、模型和原始实验。禁止 `reset`、`clean`、`stash`、强删工作树或覆盖已存在的实验目录。用户未要求时不 `commit`、`push`。
2. **GPU 只容许一组实验。** 使用 `/var/tmp/tp5-gpu.lock`；已有 `scripts/tp5-gap-bench.py` **内部会取锁**，不得外层再对同一锁执行 `flock` 造成自锁。GPU 跑分期间禁止并行编译；遇到锁冲突或其他进程先停止自己的计划，不杀别人的进程。
3. **不得通过破坏保护装置过关。** 不更改驱动、GPU/CPU 时钟、功耗、BIOS、看门狗、恢复保护、强杀策略或超时阈值来取得速度/通过率。
4. **不得伪造资格。** timeout、GPU fault、非预期 fallback、HTTP 500、EOS/长度异常、冷暖正文差异、状态错误均保留原始失败；不能事后改容差、删失败臂、换问题或延长时限后沿用旧资格。
5. **只针对原定研究任务排名。** 原 Q2_0 与 APEX、HC-Q8 等衍生模型不混作同一数值路径；计数题峰值、插桩微基准及搜索最高单次值不是本题 SOTA。
6. **真实执行模型。** 禁止针对“如何减肥”硬编码、预存答案、查表返回、以已知参考答案选输出、将保护集泄漏到参数搜索。允许有根据的通用预处理、缓存、重排和推理算法创新。
7. **不得把现状当下界。** 当前的 96 stage、1510 dispatch、host relay 约 73 μs、镜像 HC/router、uncached mailbox、某次实测带宽、某个模型预测 Match 都不是所有合法实现必付的成本。
8. **资源与状态所有权不能减配。** 实际 `VkBuffer`、offset/size、epoch、extent、active/capacity rows、payload、双 bank 的真实读者完成关系必须正确。部分 native submit 后不可 fallback 重放；资源排空以实际提交和 fence 为准。
9. **区分三个“单”。** `-np 1` 是一个用户请求流；`-t 22` 是 CPU 主计算线程数；MTP 的有效验证行数另算。`-t 1` 也不代表进程只有一个 OS 线程。CPU 线程数本任务可调。
10. **独立证据才可晋级。** 先机制与数值资格，再无插桩完整请求，最后新的独立进程配对与未参与选型的保护集。理论证书和实测证据分别出具。

## 1. 当前事实：先看这一节再动手

### 1.1 当前主模型及其身份

原模型目录（示例取本机用户主目录，不写死用户名）：

`$HOME/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/`

两个原始分片的既有 SHA256：

| 分片 | SHA256 |
|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |
| `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

已有 MTP draft：`$HOME/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`，旧 SHA256 为 `8db8b4207bbe40286db910fae89928a8cc59b1f7c197aad0a1ebf1b12d5082ad`。**每轮须重新核实路径和哈希**，不能因为文件名相同就认为实际加载了同一模型。

五卡的既有 Vulkan rank→PCI BDF 映射：

| Vulkan rank | PCI BDF |
|---|---|
| Vulkan0 | `0000:03:00.0` |
| Vulkan1 | `0000:06:00.0` |
| Vulkan2 | `0000:09:00.0` |
| Vulkan3 | `0000:0c:00.0` |
| Vulkan4 | `0000:0f:00.0` |

这是 2026-10-10 旧测量的映射，**每轮重新确认**。硬件已记录为单路 Xeon E5-2699A v4（22 物理核），五个 PCIe 端点当时均为 8 GT/s ×16；五张 RX 6800 各有 16 GB 显存，标称峰值带宽 512 GB/s、Infinity Cache 128 MB。端点规格不是共享上游总带宽，也不是实际 GEMV 性能。

### 1.2 已知成绩与缺口：不同模型绝不拼账

| 时间、对象 | 观察 | 此刻的合法解释 |
|---|---|---|
| 2026-10-10 原 Q2_0 h0、LateBind off、MMVQ disabled | 暖请求 2 次、每次 140 token、50.806941 tok/s；生成均值 2755.529 ms、请求墙钟均值 5531.370 ms | **DIAGNOSTIC_ONLY**：低于预注册 160–240 token；同进程而非独立块；warmup/guard 有 fallback；没有全局最优证明 |
| 2026-10-08 原 Q2_0 h1 → HC-Q8/aggressive/h1 | 历史四块 57.17 → 62.28 tok/s，输出 156 → 167 token | 比较同时改变权重数值路径和 LateBind；不能称作原模型逐位等价提速；完整回答墙钟收益未获显著确认 |
| 2026-10-09 APEX 强无 MTP 路线 | 某轮约 67.57 tok/s；紧凑 host top-k 曾在四对比较有约 3.66% 收益 | **其他模型/量化路径**，不能充当本任务的 Q2_0 起点或终局 |
| 2026-10-09/10 router 输出行分片探索 | 在另一模型同冻结程序的自然文本 pilot 约 66.69 → 61.87 tok/s（-7.22%） | 淘汰的是该具体布局和协议；native AllGather 动态行数、BF16 router 固定轨迹对拍有独立窄资格，非整机提速 |
| 2026-10-10 HC 坐标分区 Python 原型 | 属性测试 6 项已记录为通过 | 只是 float64 单 site 代数验证；未有 Vulkan/真实模型状态/整机吞吐资格 |

上述原始入口：
- `artifacts/perf-gap/q2-baseline-20261010T0401/`：`PLAN.md`、`DECISION.md`、`A-q2-h0-run2/`、`model-identity.json`、`rank-identity.json`。
- `artifacts/perf-gap/run-20261008T1845/REPORT.md`：Q2_0、HC-Q8、MTP 与量化记录。
- `artifacts/perf-gap/frontier-20261009T154118Z/`：router、relay、h1 等近期原始材料。
- `artifacts/perf-gap/run-20261009T0145/`、`run-20261009T0730/`：APEX 及既有成本解释，不可移植为 Q2 极限。

原 Q2_0 的 HC down/up 主要为 BF16。当前 `ggml_vk_hc_segment()` 的相关 LateBind capture 检查要求 Q8_0 down/up，所以**仅打开 aggressive 开关不会保证真正命中 LateBind**。HC-Q8 衍生模型改变原 target 权重，必须单独标识。已有旧 router-bf16 冻结快照出现过库哈希不符，不能随意复用。2026-10-10 Q2_0 B–F 备选臂尚未取得该轮有效执行成绩。不得抹掉这些失败和未完成项。

### 1.3 理论结论已经正式撤回

已撤回将 `274 → 95–104 → 64.74–67.68 tok/s` 解释为全设计空间必要上界逐步收紧的推导。原始测量仍有效，但它们只量到了对应实现。冻结成本模型 66.64 与旧实测接近，也仅说明拟合/预测能力。

对于允许实现集合 A 与同一任务 Ω，只有

`T*(Ω,A) = inf { T(a) : a∈A }`，且 `L ≤ T* ≤ U`

才有资格讨论 `U/L`。测到一个程序用时 `T(a0)`，只能说明 `T* ≤ T(a0)`，不能说明 `T* ≥ T(a0)`。新的必要成本理论见 `docs/performance-necessity.md`。

## 2. 固定研究任务 Ω，但不冻结旧算法

| 项目 | 固定或准许变化 |
|---|---|
| 模型来源 | 上述原始 Q2_0 两分片和 tokenizer/模板；派生数值版本必须另记 target 身份 |
| 物理资源 | 本机 RX 6800 ×5，加现有 CPU/主机内存/PCIe；不改硬件保护状态 |
| 任务 | RERoT 关闭；一个用户流 `-np 1`；中文问句严格为“如何减肥” |
| 输出 | 约 200 **模型 token**、自然 EOS、切题、流畅、合理；新轮先预注册 160–240 completion tokens |
| 采样 | 初始 `temperature=0`、`seed=42`、`enable_thinking=false`、`cache_prompt=false`；如要改，须建立新契约 |
| 起始参照 | CPU 主线程 `-t 22`、ctx1024、batch/ubatch32、无MTP与h1分别建基线；这些是初始对照，不是硬件限制 |
| 允许 | 重写算子与推理图、TP分区、owner、layer/hybrid、通信、同步、缓存、预取、融合、CPU 分工、LateBind、MMVQ、MTP、Q8/INT8等合格近似 |
| 不允许 | 单题答案特化、评测集泄漏、跳过必需模型推理、隐藏失败、事后放宽质量或输出长度 |
| 保护集 | 不参与调参的中文/英文、不同长度、算术逻辑、代码、上下文、跨请求状态与异常恢复问题；先冻结再揭示 |
| 安全 | VK buffer/epoch/extent/bank/reader、KV/recurrent/carry/RNG、恢复和退出规则全部保留 |

**两个不同的资格集合，不能偷换：**
- `A_strict`：保持指定原 target 的数值/状态及 sampler 契约。数学上代数等价不自动等于 F32/BF16 位级等价；逐位对拍或已声明的严格误差契约须有证据。
- `A_quality`：允许权重/激活量化和浮点重排；必须通过预注册的未见问题能力门、合理性门和状态安全门。有限保护集不能证明对所有可能输入的语义恒等；其最优性范围只能按明确的形式化/经验资格说明。

**评测题是单题速度入口，不是训练信号。** 如果一个候选只在已知题上快速返回预制文本，即使每次满足 200 tokens，也不属于合格实现。

## 3. 计时、质量、统计和排名

固定三种不可混用的口径：

- `F（用户体验）`：收到自然请求至完整合格回答交付的墙钟时间；不包括独立记录的服务加载/初始化。**新契约主产品指标**为同一请求版本、同一质量门下的完整请求时间，另报 p50/p95。
- `S（解码核心）`：定义明确的暖态起止，公开 completion tokens / generation_ms，单位 tok/s；同时记录总 ms/公开 token 与 MTP 周期的实际 K。**核心理论/算子指标**须有同样定义的 Ω_S。
- `H（旧账兼容）`：`1000 × Σcompletion_tokens / Σpredicted_ms`，只用于与既有服务器 timings 对照，不自动等于客户端 F。

必须同时报告 F 与 S。答案长度可能因量化或 EOS 改变：即使 tok/s 变大，也可能完整回答更慢；更短的答案不能借此绕过 160–240 token 与语义质量门。固定 teacher-forced token 回放用于内核/状态因果对拍，**不替代自然 EOS 的排名**。给理论证书时说明约束 F、S 还是特定固定轨迹，不跨口径计算 Gap。

质量验收顺序：
1. 先读原始正文和真实 token 数，检查是否 160–240、自然 `finish_reason=stop`、流畅、切题、无明显重复或自相矛盾。
2. 检查回答具有合理健康减重建议，不出现明显危险内容或胡乱承诺；不能只靠长度或一个关键词判定。
3. 检查未见保护题的预先冻结标准；同一题型下不同模型可换措辞，但不能牺牲基本能力。
4. 针对结构改动另查 logits、expert 选择、状态、EOS、RNG/采样语义和跨请求不串扰。对量化方案明确标注“近似路径”。
5. 任何机械失败立刻停止该候选晋级；原始日志、失败过程及原因保持可追溯。

统计规则：
- 一进程的多次请求不等于多个独立样本；先按进程/配对块聚合，再推导效应和置信区间。
- 最终搜索胜者要在新冻结的二进制/DSO 和新进程上做预注册的 8 个独立配对块；两臂轮序交替 A→B、B→A，避免只取最有利顺序。
- 报告相同口径的绝对延迟、成对比值和 95% CI；平均与 p95 各自给证据。CI 跨越无差异不等于证明等价；若宣称非劣，须预先给非劣界。
- 工程初筛收益门可以取 2%，A/A 诊断漂移暂以 1% 为先查门；**均为待复核工程门，不是本机固定噪声定律**。
- 候选如仅改善启动或 prefill，明确写出改善对象，不说成 decode 提速；插桩成绩只进入诊断账。

## 4. 第一次接手：终端逐步核对（下列命令是手册，不是本次执行记录）

**步骤 4.1｜确认目录与现状。** 在本机 shell 中执行，出现异常先停，不继续跑 GPU。

~~~bash
set -euo pipefail
ROOT="$HOME/atomic-llama-cpp-turboquant"
cd "$ROOT"
pwd
git status --short
git branch --show-current
git diff --stat
# 查看已有成果，不运行 git reset/clean/stash。
~~~

**步骤 4.2｜填写真实模型路径并验证完整两分片。**

~~~bash
MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0"
MODEL="$MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
MODEL2="$MODEL_DIR/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
DRAFT="$HOME/models/Qwen3.8-Flash-Next-MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"
test -f "$MODEL"
test -f "$MODEL2"
test -f "$DRAFT"
sha256sum "$MODEL" "$MODEL2" "$DRAFT"
# 人工核对上节记录；若不符，停止本轮并登记新身份。
~~~

这些文件很大；哈希计算应在**GPU 跑分前**完成，不能与测量并行给 CPU/I/O 施压。

**步骤 4.3｜冻结可执行文件和所有加载库。**

已有旧诊断快照可作身份核对入口：

~~~bash
BIN="$ROOT/artifacts/perf-gap/q2-baseline-20261010T0401/measured-bin/llama-server"
test -x "$BIN"
sha256sum "$BIN"
# 只看可执行文件还不够，必须核对其目录全部实际加载的 DSO。
# runner 将把进程 /proc/<pid>/maps 的有效库身份写入 mapped-libraries.json。
~~~

若程序或 DSO 不符、冻结源身份不明，则在 GPU 空闲且没有实验时**另建干净构建与完整独立快照**；不得混用旧快照 binary 与会被改动的 build 中的库。旧 router-bf16-native-bin 曾有实际库哈希与声明不同的失败记录，不能改旧 `identity.json` 遮盖差异。

**步骤 4.4｜核查物理机器与 GPU 独占。**

~~~bash
# 如果命令存在，记录拓扑；不要据设备序号猜 BDF。
command -v lspci && lspci -tv || true
# 记录实际 rank 映射，核对本轮机器、驱动和模型加载日志。
# 有别人的 GPU 任务或编译作业就不启动自己的 benchmark。
~~~

CPU、PCIe、内存与现有五卡状态要记；不因“之前也是五张卡”而略过。runner 会检查五张 Navi21、空闲状态及部分编译进程，但**检查点无法发现所有短命竞争任务**。

**步骤 4.5｜在新的独立目录预注册问题版本。**

~~~bash
RUNROOT="$ROOT/artifacts/perf-gap/q2-frontier-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUNROOT/contract"
printf '如何减肥\n' > "$RUNROOT/contract/question.txt"
cat > "$RUNROOT/contract/system.txt" <<'EOF'
请用中文回答普通成年人如何健康减重。说明合理饮食、每周运动、睡眠和长期执行方法，语言自然，答案约200个模型token，避免冗长铺垫，正常结束。
EOF
sha256sum "$RUNROOT/contract/question.txt" "$RUNROOT/contract/system.txt"
~~~

这是**新版本候选系统提示**，不能保证模型一定生成 160–240 tokens。既有 `q2-baseline-20261010T0401/system.txt` 在旧轮给出 140-token 结果，不能回头把那次改成合格。每次新轮在跑性能候选**之前**冻结问题、提示、质量门、长度门、上下文、采样与版本号；若新版本本身仍达不到长度要求，标记该版本失败。修改提示意味着开始**新的完整 A/B 轮次**，所有候选必须用同一新版本重测，禁止挑某个方案生成更短的特例。

保护集另登记哈希与只读身份，正式揭示前不作优化输入；runner 默认的一道“哈希表与数组” guard 仅是基础回归，**不足以证明通用能力**。

## 5. 用仓库现有 runner 跑参考臂：不要猜选项

`scripts/tp5-gap-bench.py` 现有选项已核查：`--bin --model --draft --out --horizon --latebind --mmvq --threads --ctx --warmup --repeats --prompt-file --system-prompt-file --max-tokens --placement --profile --env`。当前 `--placement` 只有 `tp5` 与 `layer`；HC 坐标分区、owner、hybrid、peer relay **还不是可直接切换的 CLI**。不要把待实现的方案写成已存在的开关。

下列 shell 与第 4 节使用**同一个终端**和变量。最初配置刻意固定 attention 分片/headmap 与其余开关；这是公平对照起点，不是对允许算法的约束。

~~~bash
run_arm() {
  local name="$1" binary="$2" target="$3"
  local horizon="$4" latebind="$5" mmvq="$6"
  python3 scripts/tp5-gap-bench.py \
    --bin "$binary" --model "$target" --draft "$DRAFT" \
    --out "$RUNROOT/$name" \
    --placement tp5 --horizon "$horizon" \
    --latebind "$latebind" --mmvq "$mmvq" \
    --threads 22 --ctx 1024 --warmup 2 --repeats 3 \
    --prompt-file "$RUNROOT/contract/question.txt" \
    --system-prompt-file "$RUNROOT/contract/system.txt" \
    --max-tokens 400 --profile none \
    --env GGML_TP5_REPLICATE_ATTN=0 \
    --env GGML_TP5_QSA_HEADMAP=1 \
    --env GGML_TP5_GDN_HEADMAP=1 \
    --env GGML_VK_SMALL_Q8_COLUMNS=1 \
    --env GGML_TP5_GREEDY_HOST=1 \
    --env GGML_TP5_PARALLEL_SUBMIT=1 \
    --env GGML_TP5_Q_WORKGROUP=256 \
    --env GGML_TP5_NORM_WORKGROUP=256 \
    --env GGML_TP5_NORM_PREFETCH=0 \
    --env GGML_TP5_PACK_SINGLE_WG=0 \
    --env GGML_TP5_SHARD_ROUTER=0
}
# 同一参考臂在两个全新进程中复验漂移。
run_arm aa-01 "$BIN" "$MODEL" 0 off disabled
run_arm aa-02 "$BIN" "$MODEL" 0 off disabled
# 上一步身份、输出、质量和漂移合格后，再做单一因素的诊断。
run_arm b0-q2-h0 "$BIN" "$MODEL" 0 off disabled
run_arm b1-q2-h1 "$BIN" "$MODEL" 1 off disabled
~~~

**执行解释：**
- 每条 `run_arm` 为独立进程，目录必须事先不存在；runner 自动获得 GPU 锁、检测空闲与编译、采集模型/程序/DSO/响应，并尝试正常退出。不得外加 `flock` 套住此命令。
- `--horizon 0` 仍要求 `--draft` 指向存在的文件，但不启用 MTP。`--horizon 1` 需确实检查 draft 加载/接受/回退。
- `--max-tokens 400` 仅作防失控上限；`finish_reason=length` 不合格，不能硬切 200 冒充自然结束。
- `--profile none` 才用于正式速度排名；`cycle/full/kernels` 是后续诊断，需另开目录，不能混入正式样本。
- `result.json` 即使写 `MEASURED_NOT_QUALIFIED` 也不表示已通过人工质量和状态验证；必须逐条读 `request-*.json` 与日志。
- 冷 warmup 或 guard 异常不能因两条 measured 恰好没有 fallback 而被忽略；标为诊断且保留。
- 如果本次 160–240 token 长度门未通过，停止该契约下的候选排名；不得去优化器里选择能生成更多/更少 token 的数值路径来规避长度资格。
- 每个候选若需改变层布局、线程、Q8权重等，可以在明确控制变量后另建有名字的臂；禁止偷偷顺带更改其他因素。
- `--placement layer` 是已有的五卡 layer 对照，但其数学划分不同；需显式记下与 TP5 的实际参数差异，不能直接交给上面的固定 `tp5` 函数冒充同臂。

**每一份结果的最低字段：**

~~~text
contract_version, workload_hash, quality_gate_version, timing_boundary
all_model_shard_SHA256, draft_SHA256, source_dirty_identity
binary_SHA256, actual_loaded_DSO_SHA256, rank_to_BDF
resolved_env_and_CLI, process_id, paired_block, phase
public_completion_tokens, finish_reason, complete_text_hash_and_raw_body
generation_ms, prompt_ms, first_token_ms, request_wall_ms, startup_ms
mtp_horizon, accepted_prefix_distribution, cycle_D_V_S_H_K
actual_pipeline_hits, active_rows, capacity_rows, bank_epoch
fallback, timeout, GPU_fault, HTTP_error, teardown, quality_verdict
~~~

旧 runner 尚未自动收齐这里全部字段（例如完整周期账/能力保护集），**缺失要补受维护采集器或标 `NOT_COLLECTED`**，不能在报告里凭推测填数。

## 6. 总控算法：每轮必须能解释为什么做下一步

~~~text
算法 T｜TP5_FRONTIER(任务 Ω，允许实现 A，最优性容差 ε)

T1. 若源身份、环境、质量门或 A/A 未合格：
        执行 §4–5；尚不能比较速度。
T2. 取得合格 B0 与 B1；再加入必要的 layer、HC-Q8 等强对照。
    不把弱的默认 attention 布局故意保留作“钢铁人”对手。
T3. 收集完整请求、周期、stage/rank 三张账（§7）。
T4. 找到最大且可以因果解释的暴露成本。
    若只是被其他等待覆盖的小核：
        不先改这个小核。
T5. 根据实际证据选一项：
        BF16 LateBind / HC 分区 / relay / router / MTP-MMVQ / kernel。
    编写“五句工单”：瓶颈、最多可省、机制、对照、失败出口。
T6. 先单 site/单算子数值与状态对拍。
    失败：保留证据、撤回候选、修机制；不跑大规模排名。
T7. 通过后才跑完整 target 自然 EOS pilot。
    若只有微基准快：不晋级，记录被什么掩盖。
T8. 冻结胜出候选并独立确认 (§14)。
    合格者改善 U；失败者保留原强参考。
T9. 并行进行不改硬件的理论必要性论证 (§15)。
    有效证明收紧 L；预测 Match 只进诊断。
T10. 若 U/L 已在所声明 A、Ω 上达到 1+ε，
     且质量、覆盖、可靠性、统计资格全部成立：
         标记 CERTIFIED_WITH_SCOPE。
     否则：
         标记 OPEN 或 UNRESOLVED；
         选择尚可能改变结论的最大合法分支，回 T4。
结束。
~~~

该流程**不预先保证有限步可对所有可能算法给出数学精确最优值**；如果合法分支尚未覆盖，只能报告“最快已知的合格实现”，不能宣布全局 SOTA 证明完成。

## 7. 先拆关键路径：三张账同源、不可重复收费

### 7.1 请求账

每个请求记录 prompt n/ms、首 token、公开 completion n、generation_ms、完整 wall_ms、EOS、长度、质量、fallback、程序退出与环境身份。**冷首请求、暖请求、启动/模型加载三者分账**。旧历史 176 公开 token 可能只对应 175 次单行 decode，因为第一个来自 prefill logits，不得将所有 token 当作一次独立 target 调用。

### 7.2 周期账（MTP）

对每个真实周期 j，记录无重叠的 `D_j` draft、`V_j` verify、`S_j` rollback/catch-up/state、`H_j` 其他 host，公开提交 `K_j`。若有异步重叠，则按**同域时间线**拆互斥区间，并另报重叠和未归因残差，不可先简单相加再假装恰好等于墙钟。

~~~text
总周期 ms = Σ_j 真实不重复经过的时间
每公开 token 成本 = 总周期 ms / Σ_j K_j
MTP 盈利门 = (总周期 ms / Σ_j K_j) < 同数值路径无 MTP 的 ms/token
~~~

`accepted/drafted` 不等于 `K`；EOS、全拒绝、尾段、bonus 和 catch-up 均会改变分母。动态 h 必须只用运行时可获得的信息，不许事后看真实答案选 h。

### 7.3 rank/stage 账

记录每个 `role × phase × rank × stage × active_rows × M/N/K × weight_type` 的：
- 已选 MoE experts、权重与中间量占用、实际访问/驻留、Q8_0/Q8_1/BF16/其他类型、所选 native pipeline；
- producer 开始/完成、publish、host 可见、归约完成、consumer 可读、consumer 实际开始与 reader done；
- CPU/DRAM/Infinity Cache/PCIe/设备执行单元的工作与峰值；这些计数器如果未采到就写缺失；
- 前后 stage 的实际 ownership 和通信对象，区别慢 rank 到齐与到齐后的 relay。

跨 GPU timestamp **未校准前不能直接相减**；插桩本身会扰动端到端表现，稀疏诊断与无插桩正式排名分开。`P2 wait` 不全是 PCIe 传输；`compute` 不全是 ALU。只有不可重叠的成本才能累加。

## 8. 工单 A：原 Q2_0 的 BF16 LateBind，优先补算法缺口

**源码入口：**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_hc_segment`；
- `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp`；
- `ggml/src/ggml-vulkan/vulkan-shaders/tp5_hc_latebind.comp`；
- `src/models/qwen4exp.cpp::build_hc_mix/build_hc_combine`。

现状：相关 capture gate 检查 `down_node->src[0]->type == GGML_TYPE_Q8_0` 与 up 同条件；原 Q2_0 的 BF16 HC 无法由 aggressive 开关名称自动获得这条路径。需要设计而非简单删检查。

~~~text
算法 A｜BF16_LATEBIND

A1. 在原 Q2_0、h0 下冻结 reference logits/HC state 与权重身份。
A2. 逐 site 记录为何未 capture；确认 BF16 权重类型与真实 shape。
A3. 比较两种实现草案：
    A-native：原生 BF16 权重参与提前 Q/LO 计算。
    A-convert：经过合格的权重或 activation 表示转换再执行融合。
A4. 对每个算子严格区分：
    提前可线性分解的部分；等待完整 Y 或 rho 的非线性部分；
    scratch 生命周期；Q/ρ/UP 的发布与读取契约。
A5. 先做单 HC site，比较 X、RMS、gamma、down、SiLU、
    up、gate、inject、fold、下一次 residual 和状态。
A6. 再扩大真实模型层数与 h0/h1 行数。
A7. 逐 rank 检查 late site 命中、实际 pipeline 与 ISA。
    若仅环境变量改变而 site=0：记 NO_HIT，不宣称 LateBind 提速。
A8. 先对拍，再无插桩测完整请求。
    严格数值路径不过关，不得降级成“逐位等价”；可另立近似资格。
结束。
~~~

必要对照：原 Q2_0/off、原 Q2_0/BF16 LateBind、HC-Q8/off、HC-Q8/aggressive（后两项是**不同 target**，需通过近似质量门）。仓库 `scripts/tp5-hc-q8-variant.py` 已有衍生转换工具，源模型禁止覆写。2026-10-08 变换 290 个 HC 张量并命中约 94 site 的记录只是历史路径资格，不是新 BF16 路线的结果。

## 9. 工单 B：HC 坐标分片，计算与数据布局一起改

**源码入口：** `src/llama-tp5-plan.cpp::plan_tensor`（当前 HC mirror）、`src/llama-tp5-plan.h`、`src/models/qwen4exp.cpp`、`ggml/src/ggml-backend-meta.cpp`、collective 与 HC shader。代数参照：`scripts/tp5-hc-coordinate-prototype.py`；CPU 测试：`tests/test-tp5-hc-coordinate.py`。

令五 rank 的坐标集合 `J_r` 不交且覆盖全部宽度；每条 stream b 维护 residual `R_b`、scale `s_b`，各 producer 部分和为 `z_r`。

~~~text
算法 B｜COORDINATE_HC(site, active_rows, capacity_rows)

B1. 校验 0 ≤ active_rows ≤ capacity_rows，
    各 J_r 不重叠且覆盖全部坐标；记录真实 VkBuffer 和所有权。
B2. 按 J_r reduce-scatter：
        Y[J_r] ← Σ_p z_p[J_r]。
    注意：取平方以前必须得到完整 Y[J_r]。
B3. 对每条 HC stream b、每个 j∈J_r：
        X_b[j] ← R_b[j] + s_b·Y[j]
        G_b[j] ← gamma_b[j]·X_b[j]
    本地累加：
        v[r,b] ← Σ_j X_b[j]^2
        d[r,b,k] ← Σ_j W_down[k,b,j]·G_b[j]
        i[r,b,o] ← Σ_j W_inject[o,b,j]·G_b[j]
B4. 跨 rank 归约小统计量：
        rho[b] ← 1/sqrt(Σ_r v[r,b]/width + eps)
        u[k] ← (1/streams)·Σ_b rho[b]·Σ_r d[r,b,k]
        h[k] ← SiLU(u[k])
        inject[o] ← Σ_b rho[b]·Σ_r i[r,b,o]
B5. 在本 rank 拥有的输出坐标上：
        gate[b,j] ← sigmoid(Σ_k W_up[b,j,k]·h[k])
        mixed[j] ← (1/streams)·Σ_b (rho[b]·G_b[j]·gate[b,j])
    完整保留原先的 stream 平均、inject 与下一次
    2·sigmoid(inject/streams) residual 更新。
B6. 将 J_r 的输出/状态所有权传播到下一真实消费者。
    若消费者需要完整向量，实际执行并计时必要 gather；
    不把布局重排、通信或尾段操作默认为零。
B7. 先单 site，后单层，后真实 target；先 active=1，
    后 active>1、active<capacity、零行、尾段和 MTP 路径。
B8. 与物化 reference 比较中间值、完整 logits、
    KV/recurrent state、下一 token 的 carry/RNG。
B9. 若通信+变换消耗覆盖了所有计算收益：
        淘汰该具体布局，保留数学原型与失败证据。
结束。
~~~

**不能把 `||Σ z_r||²` 错替换成 `Σ||z_r||²`**；这会丢跨 rank 交叉项。float64 数学原型成立不意味着 BF16/F32 bit-exact；更不意味着设备布局、bank 和消费者无需改动。

对照至少包括：
- 镜像 HC + 现有 LateBind；
- coordinate ownership + 完整下游通信账；
- owner 算 HC 再分发；
- layer 与混合布局（若有合格实现）。

改变分区时同时维护**权重、activation、residual/recurrent state、下游消费者**四张所有权表。不得只分权重，后面再把所有数据无条件广播五次。

## 10. 工单 C：重新设计跨卡交接，而不是只调小核

**源码入口：** `ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp::tp5_relay_submit_epoch_chain`、`ggml-vulkan-relay.h`、`tp5_hc_publish.comp`、`tp5_hc_resume.comp`。

先按真实时间线拆开 `rank_last_arrival` 和 `after_arrival_transport`。前者要查负载均衡、权重分区和执行顺序；后者才查通信协议。不要拿旧 73 μs 循环次数作为物理下界。

~~~text
算法 C｜SAFE_HANDOFF(stage s, epoch e, bank b, active a, capacity c)

C1. 校验 a≤c、s/e/extent/源 VkBuffer 与定义相符。
    若上一次 b 的真实读者未完成：等待合法完成信用，不能复用。
C2. producer 写本次 payload；完成该内存域要求的发布。
C3. 发布 (s,e,b,a,extent,ready)。
C4. relay 等全部必要 rank 的同代 payload 就绪；
    按声明的数值顺序归约，并将结果写到约定空间。
C5. 先发布 payload 可见性，再发布 output_ready。
C6. consumer 验证同 epoch/extent/bank，执行合法 acquire。
C7. consumer 真正完成读取后才发布 done。
C8. 若已有任一 rank 部分 native submit：
        记录故障，按实际提交/fence 排空；
        禁止 fallback 再重放同一轮计算。
C9. 候选只有在完整请求和状态不退化时才晋级。
结束。
~~~

候选路线：
1. 现有 host relay 的批处理与合理 tile（先 1/2/4 tile），核算新的门、CPU 线程调度与载荷开销；
2. 控制头与大 payload 分开，让后者在合法可见性的 device-local 副本/共享局部中被消费；
3. 如果**实际**设备组、heap 的 Vulkan peer COPY/GENERIC 功能与同步要求通过资格，再尝试设备侧通信；仅 COPY 可用不代表 shader 任意跨卡 load/store；
4. 改 owner、坐标分区及实际消费者需求，从根上减少需要过总线的信息，而非加快无谓的全量广播。

不得简单去掉 `DEVICE_UNCACHED`、`coherent`、barrier 或 payload 校验让微基准变快。实际 memory type flags、发布/获取作用域和 fence 要可审计。保留旧 CPU relay tile pilot 无收益等负证据；新设计须说明为何不同。

## 11. 工单 D：router / MoE 分工与中间量

代码入口：`src/llama-tp5-plan.cpp`、`src/models/qwen4exp.cpp`、`ggml/src/ggml-vulkan/vulkan-shaders/qwen4_router.comp`、collective、router 原始 pilot。

~~~text
算法 D｜ROUTER_OWNERSHIP

D1. 验证各 rank router 输入是否相同。
D2. 输入相同时才有资格比较：
    mirrored 五份计算；
    score 输出行分片 + gather + 原后处理；
    owner 完整 router + 分发下游必需结果。
D3. 比较全部 score、expert ID、weights、mask、
    分组归一化与下一计算，不只核 top-1。
D4. 对每种分区算上权重读取、owner 等待、消息、
    consumer 转换、真实五卡负载。
D5. 若该具体方案变慢，保留结果并分析原因。
    旧行分片 pilot 曾约慢 7.22%；不能据此封杀
    其他合法分区或复用策略。
结束。
~~~

BF16 router 曾取得固定 token 全词表 logits/serialized-state/trace 的窄范围逐字节对拍，但还缺独立自然文本性能、长期 sampler/RNG 与整机可靠性。不要把“对拍能力通过”当“router 加速成功”。

## 12. 工单 E：新 target 上重新选 MTP × MMVQ × LateBind

**源码入口：** `common/common.cpp::common_tp5_apply_env`、`ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_should_use_mmvq`、`ggml-vulkan-collective.cpp`、MTP workspace/量化 scratch 与 target/draft state 代码。

- MTP 默认单卡 draft；隐式 horizon 仍为 6。**每次显式填写** `--horizon 0/1/2/3`；深度变化要验证实际 row/capacity 和 draft 设备布局。
- MMVQ 在本工程指 **Q8_1 activation 转换 + 量化权重收缩**，不同于将 BF16 HC 权重派生为 Q8_0，也不同于 aggressive LateBind 的 Q8_0 Q 路线。
- 当前 selector 支持 Q2_0，但受设备整数 dot、类型、actual M/N/K、small-column lowering 和 pipeline 可用性约束；AMD auto 对 Q6_K 有选择限制。**开 forced 不等于每个算子命中，也不应拆掉正确的融合区**。
- 只有相同 target 数值路径、相同 attention/headmap/CPU 条件下，才能归因 MTP 或 MMVQ 的单因素收益；如果精度改变，MTP 的 K 分布也要重测。

~~~text
算法 E｜JOINT_MTP_MMVQ(target)

E1. 用已合格 target 作 2×2 因果起点：
        h0/disabled, h0/auto,
        h1/disabled, h1/auto。
    必须让无 MTP 对照也使用合适、相同的 attention 布局。
E2. 对实际 verify 行数 N=1,2,3... 分桶；
    记录矩阵 M/N/K、rank、权重类型、专家并集、
    量化输出缓存 key、native pipeline 与复用次数。
E3. 比较 h1、h2、h3 的周期总成本与真实 K。
E4. 在资格通过时尝试 forced MMVQ、形状专用融合、
    activation Q8_1 复用、量化与整数点积合并。
E5. 对每个新 target 重新测接受前缀分布和 EOS；
    对数值近似检查守护问题与状态门。
E6. 若扩大 h 的 ΔK≤0，或 ΔC/ΔK≥当前 C/K：
        暂停扩大该具体候选；不是证明所有深 MTP 均无效。
E7. 验证全接受、首/中拒绝、EOS、预算尾段、零草稿、
    cached prefix、连续请求、rollback、KV/recurrent、
    carry 与 sampler/RNG 的所有权。
E8. 只按 Σ真实周期成本/Σ公开 token 和完整请求排名，
    不按单独 accepted/drafted 或 MMVQ 命中数排名。
结束。
~~~

## 13. 工单 F：资源局部性、融合、首轮与采样收尾

仅对第 7 节确认仍暴露在关键路径上的部分立项。

- **权重与算子。** 检查 Q2_0 原生收缩、bf16/Q8 转换、专家局部性、prefetch、LDS、寄存器、occupancy 和编译后 ISA。更少字节通常伴随解码/占用开销；不许假设解包免费。预取已知不可变权重可以早于 activation；专家预取须计错误预测损失。
- **小 kernel 融合。** 先给出该段最多可节省的绝对毫秒；无法缩短消费者实际开始时间的小核不作主线。只修改 GLSL、ISA 完全相同不算新硬件收益。
- **词表与采样。** 现有紧凑 host top-k 已有历史收益；再做局部候选/device 合并时严格保持 ties、NaN/Inf、grammar、过滤顺序与 RNG。禁止“先 argmax 再多抽一次随机数”造成隐性状态改变。
- **首次 decode。** 可尝试合法 replay/定义预热，但计入服务启动、冷请求和显存驻留成本，不能仅把耗时从 F 转移到 startup 后宣称成本消失。
- **CPU 线程。** 22 为起点，若时间线显示 CPU contention 再隔离比较 4/8/22/44；不要以 `taskset` 限制整个服务为一个核冒充只设 `-t 1`。
- **可靠性。** 严格保留真正的 VkBuffer 身份、cache key 包含数据世代/真实 owner，双 bank reader done、退出 signal 与实际 pending submission。新增缓存/采样不得导致跨请求状态泄漏。

## 14. 从 pilot 晋级到公平“钢铁人”基线

~~~text
算法 V｜QUALIFY_AND_PROMOTE(A, B)

V1. 在没有 GPU 测量时编译 B；冻结 server 及全部动态库。
V2. 运行对应 CPU 和协议测试；保留既有失败基线。
V3. 在真实模型做固定 token 对拍、完整状态和故障边界。
V4. 先用小型自然 EOS pilot 判断是否值得投入。
    若机械失败：B=FAILED，不增加超时或删除失败请求。
V5. 冻结 B 的 hash/配置/质量契约，预注册主指标、样本数
    与 A/B 轮序，关闭正式测量中的性能插桩。
V6. 在新的 8 个独立配对块中交替 A→B、B→A。
    每次 run_arm 启动独立服务；一时刻只允许一个 GPU 组。
V7. 按块聚合、对数比与恰当的置信区间评估；
    同时比较完整请求 wall、公开 tok/s、长度与质量。
V8. 用未参与选型的独立保护集与跨请求状态长测确认。
V9. 只有 B 正确、质量不退化、无未解释故障且
    独立确认支持事先声明的收益时：
        PROMOTE(B)；保存前一最强臂及完整证据。
    否则：
        KEEP(A)；保留 B 的失败、阴性和统计不确定性。
结束。
~~~

**可直接接在 §5 shell 后的确认轮序示例：**

~~~bash
# 仅示例：同一原 Q2_0、同一 bin，把 h0 与 h1 做正式对照。
# 开始之前必须完成 pilot、身份核对、质量和状态资格。
for block in 1 2 3 4 5 6 7 8; do
  if (( block % 2 == 1 )); then
    run_arm "pair-$block-A" "$BIN" "$MODEL" 0 off disabled
    run_arm "pair-$block-B" "$BIN" "$MODEL" 1 off disabled
  else
    run_arm "pair-$block-B" "$BIN" "$MODEL" 1 off disabled
    run_arm "pair-$block-A" "$BIN" "$MODEL" 0 off disabled
  fi
done
~~~

如果对比的是新代码，需**另冻 B 的二进制和所有 DSO**，只改其工单要求的因素；如果对比的是 HC-Q8，则它属于另一数值 target，必须走近似质量门且明确输出差异。允许更快的全新 owner/layer/hybrid 实现进入“钢铁人”竞争，但每个强候选均需自身资格，不把历史最强片段拼在一起假想一套从未运行的配置。

## 15. 理论侧：从必要工作证明 L，而非拟合旧时间

### 15.1 先界定可证明的命题

分别定义严格数值类 `A_strict` 与在明确保护集/质量规则下的 `A_quality`。记录硬件状态、初始模型/缓存、请求、输出语义及计时边界。对平均耗时、单次确定轨迹和 p95 必须**分别**定义目标变量。有限人工能力保护集不会自动赋予“所有可能自然语言输入都保持能力”的数学保证。

对于同一实现集合 A：

~~~text
T* = inf { T(a) | a 属于 A 且满足 Ω }
L ≤ T* ≤ U
Gap = U/L - 1            (仅当 0 < L 且证书有效)
~~~

`U` 来自正确实现的可达耗时及独立统计资格，测量随机性要有保守估计和适用范围；绝不能把一个样本均值写成任何运行条件下确定的上界。`L` 来自数学/硬件必要性。二者必须是**同一 Ω、同一 A、同一 F/S 时钟边界**。

### 15.2 合法的资源引理

给定一种明确算法族与存储边界，若证明任何允许实现至少要做 `W_min` 必要工作，且该资源速率至多 `C_max`，才可写 `T ≥ W_min/C_max`。

若多个必要资源可重叠，只能先共同取 `max`，不能无证明相加。只有被证明不可重叠的必经依赖边才能加入同一关键路径。实测 390–403 GB/s、某个 shader 28.6 μs、旧 relay 73 μs 等只是条件观测，不是资源容量上限或全局固定税。

若声称“U 字节必须穿过某快存储边界，初始最多保留 C 字节等价信息”，至少需要跨边界 `max(0,U-C)` 字节；前提是独立必要信息、表示压缩/重构、代码常量、缓存、预取和流间复用都已纳入证明。**4.614 GB 按张量名去重清单 ≠ 所有算法每 token 至少从显存读取 4.614 GB。**

### 15.3 合法策略分支与可核验证书

~~~text
算法 P｜PROVE_FRONTIER(Ω, A, ε)

P1. 声明 A 的全部可替代策略族：
    mirrored / owner / coordinate / layer / hybrid；
    host / peer / 不同合法通信；
    原格式 / 合格数值表示与缓存；
    不同 MTP 验证、draft、融合与调度。
P2. 对每个分支 Aj 写必要性前提：
    权重/状态必需信息、物理 cut-set、存储容量、真实算术、
    非线性依赖、可重叠资源以及允许的压缩/预取。
P3. 构建对实际实现“有利”的放松优化问题。
    要证明 Aj 中任一合法实现能映射到放松问题可行解；
    否则这个问题不是 Aj 的有效下界。
P4. 得到 Aj 的可核验耗时下界 Lj。
    LP 可以保存对偶可行证书；MILP 保存有效全局 lower bound。
    求解器仅给 incumbent/一份可行排程时，不得冒称 Lj。
P5. 用全部合法分支合并：
        L ← min_j Lj。
    若有尚未覆盖或无界的合法分支：
        global_L = UNRESOLVED。
P6. 用合格实现的保守时间统计建立 U；核对 Ω 与计时边界。
P7. 若 global_L 有效，且 U/L ≤ 1+ε，
    同时质量、可靠性和统计门均通过：
        输出 CERTIFIED_WITH_SCOPE(Ω,A,ε,L,U,证书)。
    否则：
        找到仍可能最快或证据最弱的分支；
        一面改进 U，一面补必要耦合约束以收紧 L；
        保留未解决分支与失败，不制造终点。
结束。
~~~

不同替代分区的条件下界**取最小**，不能以旧 relay 分支最多 90 tok/s 排除另一种可能达到 160 的合法分区。跨 token 权重预取、不同阶段缓存、MTP 多行共享权重都必须留在允许集合中。若只证明有限算法族的最佳解，只能称**该明确算法族内的条件最优**；不能宣称证明了开放的所有可设计算法。

阶段里程碑可先登记 `U/L≤2`、`≤1.5`、`≤1.2`，有严格覆盖证书后才讨论 `≤1.02` 等更紧门。阈值是研究目标，不是通过撤回不利分支或放宽质量门就能获得的成绩。**若尚无有效 L，Gap 必须输出 UNPROVEN，不计算一个好看的伪比例。**

## 16. 具体测试入口与故障处理

### 16.1 先跑 CPU/逻辑层检查，再跑 GPU

可在无 GPU 测量时执行；这是未来操作步骤，本次修改文件不代表已经重跑。

~~~bash
cd "$HOME/atomic-llama-cpp-turboquant"
python3 tests/test-tp5-gap-tools.py
python3 tests/test-tp5-hc-coordinate.py
python3 tests/test-tp5-necessity-audit.py
python3 tests/test-tp5-perf-model.py
# 如果 build-perf-gap 的构建存在，先列出实际测试名：
ctest --test-dir build-perf-gap -N
# 只对已存在的目标跑对应 CPU 定向测试，不能假定全部已构建。
~~~

生产定向入口包括 `test-predefined-capacity`、`test-predefined-coverage`、`test-target-capacity`、`test-tp5-row-program`、`test-mtp-workspace`、`test-server-signal`、`test-greedy-unique-host`；Vulkan replay/relay、实际五卡和真实模型独立验证。旧 `test-mtp-workspace` 失败曾未修，必须检查本轮真实结果并保留，不将“部分 CPU 测试通过”写成“五卡生产全绿”。

仅运行文档/自检时不调用 GPU。涉及五卡的 Vulkan 测试与服务器 benchmark 分别申请 GPU 独占，防止两组交叠；构建与 benchmark 分阶段，不随意调整恢复/保护机制。

### 16.2 错误处置表

| 观察 | 必须处理 | 禁止的捷径 |
|---|---|---|
| 哈希/动态库不一致 | 停止比较，冻结真实身份并追查来源 | 改旧清单配合现状 |
| GPU 锁被占或编译并行 | 本轮停止并保存阻塞事实，待资源正常后新轮 | 杀其他任务、绕过锁 |
| 140 token、预算截断或坏答案 | 保留原样，长度/EOS门失败；修改提示必须新契约全臂重测 | 事后降低长度门、删文本 |
| h1 加载失败/RELAY 故障 | 保留原始出错臂，诊断行数、提交与状态 | 加长超时、失败后重放 |
| 某开关开了却没命中 | 查实际 site、M/N/K、pipeline 与权重类型 | 从 env 名推断优化已执行 |
| 与参考 logits/state 不同 | 判断严格或近似路径；后者重新质量资格 | 只看一个 scalar 或 top-1 |
| router 分片 kernel 变快但整机变慢 | 计通信、owner 等待、分区传播与最慢 rank | 用孤立 kernel 分数“证明”收益 |
| 成本模型吻合实测 | 只作解释/预测能力证据 | 声称硬件理论最优 |
| 某分支没法证明 L | 标 UNRESOLVED、继续补证明 | 假设分支不存在 |
| SIGTERM 后强杀 | 记失败，检查真实 pending submission/信号路径 | 写“正常退出” |

## 17. 每轮最小证据包和下一轮选择

不得覆盖已有 `artifacts/perf-gap/` 轮次。建议每轮一个唯一时间戳目录，包含：

~~~text
contract/
    question.txt, system.txt, contract-id, guard-set-identity, preregistration
identity/
    git-status, source/dirty identity, target/draft SHA256
    frozen binary/DSO hashes, rank→BDF, build/toolchain, config
arms/
    pilot-A/, pilot-B/, pair-1-A/, pair-1-B/, ...
diagnostics/
    rank-stage-timeline, cycle-ledger, shader/ISA, ownership/inventory
review/
    raw-audit, quality-review, paired-statistics, reliability-check
theory/
    declared-class, branch-coverage, necessary-cost-lemmas
    verified-solver-certificates, unresolved-branches
DECISION.md
~~~

这里只规定**未来**证据结构，不要求无意义地复制旧大日志或创建多份同义交接。清理只按 `artifacts/README.md`，保留当前有效原始数据、失败、fixture 与必要历史索引。

每个工单的 `DECISION.md` 必须用同一格式填写：

~~~text
问题与契约版本：
本轮原始/衍生 target 与全部身份：
目标瓶颈（同请求、同进程/块）：
最多可节省的绝对时间及计算依据：
具体改变的数值/布局/通信/调度机制：
实现文件与单因素对照：
测试已执行/未执行及状态资格：
无插桩 pilot 与独立确认（原始统计、CI）：
失败、回退、EOS、质量、长测原样记录：
理论 Lj 的已证前提、未解分支；整体 L/U/L 是否有资格：
晋级 / 淘汰该具体设计 / 暂停；下一项可证伪问题：
~~~

没有核实的字段写 `NOT_RUN`、`NOT_COLLECTED` 或 `UNRESOLVED`，不要填“预期通过”。

## 18. 第一轮排程与 TP5 → RERoT 的交接门

第一轮优先交付（按真正关键路径可调整先后，GPU 仍串行）：

~~~text
算法 FIRST_WAVE

F1. 冻结新 Q2_0 自然问句契约和独立保护集。
F2. 复验原始两分片/实际 DSO/rank；合格 A/A。
F3. 取得原 Q2_0 h0 与 h1 的同身份强对照；
    若新契约不合格：停止排名，解决资格，不跑配置海选。
F4. 做三张账和一份 HC/router/attention 所有权清单。
F5. 选择最可能实质缩短暴露时间的一个：
    原 BF16 LateBind 单 site；
    HC coordinate 单 site + 下游布局；
    host/peer 合法交接原型。
F6. 保留原结构作严格对照，先数值/状态，后真机 pilot。
F7. 如果 target 路线合格变快，再比较 h0/h1/h2/h3 × MMVQ。
F8. 将正式候选冻结，执行 8 个独立配对块和保护集。
F9. 对允许分支构造必要工作/容量证书，标记未解决项。
F10. 输出最强合格 TP5 实现、速度/质量/可靠性边界、
     理论证明范围和残余缺口。
结束。
~~~

**只有合格且已独立确认的 TP5，才能作为 RERoT 的公平强基线。** 研究中可以先冻结“当前最快已知合格 TP5”开展探索，但必须写 `BEST_KNOWN_NOT_PROVEN_OPTIMAL`；不能把尚未得到 L 的程序叫“已证极限”，也不能为了比较 RERoT 而故意保留较弱 TP5。

最终验收有三种诚实的状态：
- **`QUALIFIED_BEST_KNOWN`**：有真实质量、可靠性和独立速度确认，但最优性证书仍不完整。
- **`CERTIFIED_WITH_SCOPE`**：在清楚声明的 Ω/A/ε 内，必要下界与可达上界及分支覆盖均有效。
- **`OPEN/UNRESOLVED`**：模型、机器、质量、身份、可靠性或理论必要性存在未闭合问题；保留进展与下一步。

不得把“做了很多实验”“拟合到实测”“某轮没有新收益”作为关闭条件。

## 19. 主要证据与规范入口

- 本仓 `PERFORMANCE.md`：TP5、LateBind、MMVQ、MTP 的受维护源码与历史成绩。
- `docs/performance-necessity.md`：撤回伪上界后的必要性重建、反例与证明门。
- `artifacts/perf-gap/q2-baseline-20261010T0401/`：严格匹配当前 Q2_0 问句的最近诊断；`DECISION.md` 明确无合格终局。
- `artifacts/perf-gap/run-20261008T1845/REPORT.md`：原Q2、HC-Q8、MTP/MMVQ、激活量化的历史真实观察及局限。
- `artifacts/perf-gap/frontier-20261009T154118Z/`：router 动态行数、BF16 对拍与 pilot 的原始证据。
- `scripts/tp5-gap-bench.py`、`scripts/tp5-hc-coordinate-prototype.py`、`scripts/tp5-hc-q8-variant.py`、`scripts/tp5-necessity-audit.py`：现有可维护入口。
- [AMD RX 6800 官方规格](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800.html)：硬件标称容量/带宽，不替代本机必要性论证。
- [Vulkan Memory Model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)：跨内存域的 availability/visibility、release/acquire 语义。
- [Vulkan peer memory flags](https://docs.vulkan.org/refpages/latest/refpages/source/VkPeerMemoryFeatureFlagBits.html)：COPY 与 GENERIC 区分；须核实实际设备组/heap。
- [Vulkan memory property flags](https://docs.vulkan.org/refpages/latest/refpages/source/VkMemoryPropertyFlagBits.html)：实际内存类型属性，不能靠去掉同步换速度。
- [Speculative Decoding（Leviathan 等）](https://arxiv.org/abs/2211.17192)：固定 target 下的投机验证原理；不能替近似 target 恢复数值语义。

**最终规则：先正确、再加速、再证明；模型能力与状态不能用性能数字抵偿。**

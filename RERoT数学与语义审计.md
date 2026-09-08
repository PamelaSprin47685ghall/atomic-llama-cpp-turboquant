# RERoT 数学与语义审计

日期：2026-09-07。项目：`atomic-llama-cpp-turboquant`，分支 `master`，接班 HEAD `0aede756d`。

本班按“数学定义 → 实现 → 语义测试 → 最后才性能”推进。以下是局部修复记录，不是 RERoT 完成交付报告。接班时工作树已经有未提交改动；本班在其上修改，没有清理、提交、推送或部署。

**最新状态见 §10（2026-09-08）。前文保留历史证据，不覆盖当前 child native recurrence、STRONG barrier-after 和随机 ID completion 定义。真实回答质量仍未通过。**

## 1. 数学上先纠正两项规定

### 单笔退化不能依赖正则化 block 公式

原生 GDN 在固定本 token 的参数后是仿射算子：

```text
T_i(S) = A_i S + C_i
A_i = exp(g_i)(I-beta_i k_i k_i^T)
C_i = beta_i k_i v_i^T
```

§14.1.2 原式带 `eps I`，N=1 的实际写入系数是：

```text
beta / (beta*||k||² + 1-beta + eps*beta)
```

它并不普遍等于 beta。保留既有的原生单笔分支，把规范改为明确的 N=0/1/>1 分段定义；N=1 不加 eps，也不假定 key norm 恰好为 1。数学上的相等不代表 FP16 hand 写回后的 bitwise 相等。

多笔采用 `sqrt(beta)` 对称缩放，解与原 ridge 方程等价的正定系统：

```text
Bbar = exp(mean g_i) B
C = diag(sqrt(beta_i))
M = C K K^T C + diag(1-beta_i) + eps C²
M z = C(V-K Bbar)
B' = Bbar + K^T C z
```

合法 beta 范围含端点 `[0,1]`。beta=0 的写入严格为零，不需要除以 beta 或给 beta 加下限。该 token 的 decay 仍计入其公开组的时钟。N>1 使用现有 `eps=1e-4`，没有另调超参数。

这证明的是给定 q/k/v/g/beta 下算子的定义与边界，不是未重训模型使用共享状态的质量保证。Parallel Delta 不是标准串行推理的等价变换。

外部原生实现核对对象为 Hugging Face/Qwen 的 `modeling_qwen3_5.py` 中 `torch_recurrent_gated_delta_rule` 与 `l2norm`。本项目的 block 共享和 hand 是自己的推理期契约，不能把原生实现当成这套共享方案已经有效的证据。

### fork 快照不能清除私有指令

§21.4 旧规定要求在 final fence 后恢复 fork hand seed，声称可清除 child/planner 指令。该规定已撤销：旧快照会同时覆盖当前 conv、R0-R2 和 shared-layer hand，丢掉 fork 后的局部演化，并把旧 hand 接到当前 brain/KV；它不是按指令语义做的逆变换。

`rerot_begin_serial_tail()` 已移除这次覆盖。survivor 和 N=1 root 都保留当前局部状态。fork seed 仍用于 admission，未删除其正常用途。这里只修了错误回退，尚未证明最终生成恢复正常。

## 2. 实现中的确定错误与修复

### A. 混合组批偷换更新算法

`uses_parallel_delta()` 原来只有单组、全部 rows 同属该组时才成立。加入 PRIVATE/PENDING row 或另一 episode 后，`build_recurrent_attn()` 转而采用 candidate mean，且 output 读各自 native candidate。这样，同一公开写入集合会因为无关 batch rows 改变 brain 和 readout。

现在 `n_rs_seq=0`、每 seq 一 token 的 mixed path 按 brain group gather，分别执行相同的 block 算子，再 scatter output/hand。不合并不同 episode，也不把 PRIVATE/PENDING child 加入公开组。root 的专用 private shadow group 仍只提交到 shadow，不广播到 PUBLIC brain。

### B. 混合 batch 删除私有 transition

固定旧输入 brain B，公开和私有 row 的 hand 不能用同一个减法：

```text
PUBLIC:         H' = T(B+H)-T(B) = A H
PRIVATE/PENDING H' = T(B+H)-B
```

旧 mixed path 因为存在公开 sibling，给私有 row 也减去 `T(B)`，错误抵消了本次私有 transition。修复后非提交 rows 先保存完整的 `T(B+H)-B`，只有参与某个 brain 提交的 rows 才替换为其 `A H`。

### C. beta 下限偷偷开启 write gate

Vulkan RBB 原来把 beta clamp 到 `[1e-6,1]`。beta=0 和 1e-8 的明确反例均失败，N=1 也不是原生更新。

已删除该下限，改用上文对称缩放求解；CPU 新增独立 double Cholesky 正确性实现。GPU 仍用原有 matrix-free CG，不做性能优化。删除“NaN/Inf solution 替换为 0”的掩盖，并在数值测试中显式检查有限值；没有声称任意非法输入都已被服务层优雅拒绝。

### D. 通过组数猜 ordinary root，误丢真实 hand

Qwen3.5 MoE / Qwen3Next 原来把 `groups.size()==2 && first.brain_row==0` 当成普通 root，直接使用 B、忽略 H。两个真实 episode 也能满足这个条件。

这两个模型的 state prep 已统一为 `B+H`。普通请求的 hand 应由内存生命周期保持为零，而不是从 batch shape 猜身份。旧 multi-token candidate 路径也移除了同样的身份猜测；这不代表其 Parallel Delta 语义已修齐，见 §4。

### E. 非 grouped 路径使用未初始化的 brain_copy

新增 mixed 图测试改变堆布局后，直接运行 recurrent 测试出现 SIGSEGV；GDB 定位到 `llm_graph_input_rs::set_input_recurrent()`。`brain_copy` 未初始化，非 grouped builder 不设置它，setter 却检查并解引用这个不确定值。这可能影响 RERoT OFF。

已在 `llama-graph.h` 初始化相关 tensor pointers 与标量。增加用 `0x45` 填充回收存储后 placement-new 构造输入、调用真实 setter 的回归，不依赖 malloc 恰好返回零内存。

## 3. 测试证据与复现

先改测试预期、未改实现时，实际得到：recurrent 2 个断言失败，attention 6 个断言失败。不是先写实现再让测试照抄实现。

| 项目 | 修复前 | 修复后的已测结果 |
|---|---:|---:|
| mixed 纯 decay 的 brain | 错误 0.75 | `2*sqrt(0.5*0.25)` |
| mixed PRIVATE hand | 错误 3 | 2 |
| GPU N=1、beta=0，brain 最大绝对误差 | 0.00102399988 | 1.86264515e-9 |
| GPU N=1、beta=1e-8，brain 最大绝对误差 | 0.00101375987 | 1.86264515e-9 |
| GPU N=64、相关 keys，brain 最大绝对误差 | — | 5.7220459e-6 |
| 两个 brain、两 head、混合 visibility、倒序 rows | 未覆盖 | CPU 与 Vulkan 对独立标量 oracle 均通过 |

新 mixed fixture 使用非零 k/v/H、指向 peer key 的 q，以及较大的私有 v；同时核对 brain、output、FP16 hand 和未触及的其他 brain rows。CPU brain 误差为 0，Vulkan 为 2.38419e-7；两者 output 误差约 1.10891e-6，hand 最大误差约 0.0156088（该 fixture 的私有 hand 约为 60，按 FP16 精度单独设门槛）。自然顺序与倒序结果都符合 oracle。

block op 在 CPU/GPU 各跑 9 组：N=1/3/12/32/64，含相关 keys、非零 H、beta=0 和极小 beta。测试 reference 用 double 带选主元消元，CPU 用 double Cholesky，Vulkan 用 FP32 CG，不是三份同一个求解循环。

**不能只看 CTest 摘要。** 本班曾出现 CTest 8/8 但直接运行 recurrent 崩溃，随后用 GDB 找到上面的未初始化指针。修复后已直接跑通 CPU/Vulkan；每次交付仍需保留直接运行的设备信息与数值结果。

复现命令（在仓库根目录）：

```bash
cmake --build build-vulkan-localhost --target \
  llama-server test-rerot-recurrent test-rerot-attn test-rerot-runtime \
  test-rerot-parser test-rerot-view test-rerot-ddvr \
  test-kv-cells test-triattention-score -j6
ctest --test-dir build-vulkan-localhost \
  -R 'rerot|kv-cells|triattention-score' --output-on-failure
build-vulkan-localhost/bin/test-rerot-recurrent
build-vulkan-localhost/bin/test-rerot-attn
git diff --check
```

验证设备为现有 AMD Radeon RX 6800 / Vulkan RADV。没有启动真实模型服务，也没有改动桌面 GPU 服务。最终一次全量相关构建及上述命令的结果见下方收尾记录；不得把中间构建结果冒充最后版本。

## 4. 仍然阻断正确性验收

1. **完整 final acquire fence 没有实现。** `refresh_final_fence()` 只安装 stable view；`llama_rerot_refresh_barrier()` 只 synchronize。没有重新 forward 完整 `</ID>`。修复不能简单把闭合 tokens 再喂一次；必须恢复正确的局部 checkpoint、处理对应 KV 替换，同时保留其他 lane 已提交的 PUBLIC brain/KV。源注释已纠正，未虚构回放完成。
2. **snapshot/MTP 与 active multi-token 路径未统一。** `n_rs_seq>0` 仍走旧 candidate/snapshot 提交；active episode 的 multi-token mixed path 也未按每个时步实现本文 block/hand 契约。本班修复范围是无 snapshot 的单步 frontier；普通 prefill 的 native 对照通过不等于 active 并发 prefill 正确。需要真实 snapshot 恢复后的 brain/hand/conv/output 数值对照，不能只看 shape 或保存成功。
3. **真实模型质量未重新验收。** 原题的重复、跨洲污染、空正文和自然结束问题不能凭本次 kernel/graph 回归宣布根治。后续先测 root private/public shadow 切换、child admission、final fence 的实际状态连续性，再做带完整 response 与 trie 的原题运行。CG 在更接近奇异的参数、长序列 FP16 hand 漂移也仍需扩展覆盖。

性能阶段保持未开始。本班没有新的 tok/s 成绩，没有修改原题、seed、预算、prompt 或 stop 条件，没有部署和推送。

## 5. 收尾记录

最后一次核验包含去掉 ordinary-root 组数猜测之后的代码：`llama-server` 与上述全部测试 target 构建成功；CTest 8/8；从仓库根目录直接运行 `test-rerot-recurrent`、`test-rerot-attn`，均为 0 failures，日志明确包含 CPU 与 Vulkan0 测试。`git diff --check` 通过，组合命令退出 0。

没有真实模型端到端新结果；§4 的未完成项保持未完成。构建通过不是部署许可。

## 6. 第二轮精修：局部 checkpoint、完整 fence 与 root 基底切换

### 6.1 hand restore 必须恢复一个完整局部时点

`apply_hand_seed()` 原来只覆盖字节，不更新已有 cell 的位置；共享 tail 没有先 COW；尺寸错误可能在若干层已经写入、甚至新 cell 已分配后才返回 false。二进制 loader 还接受末尾垃圾和不匹配的层数。

先加回归、保留旧代码，实际得到 **8 个失败**。修复后先验证所有 conv/local S/hand 行，再选择独占 cell，全部写完后提交引用、source、position 和 rollback selector。共享兄弟不被覆盖，坏输入不分配 cell、不半写状态；旧字节流若不是完整 SEE2 快照会明确拒绝。

`capture_hand_seed()` 同时修正为读取真实 source row 和当前 rollback snapshot plane；共享层将所选旧 brain 与旧 hand 折算到当前 PUBLIC brain，而不是把旧 hand 直接接到新 brain：`H_seed = B_selected + H_selected - B_public_now`。新增回归用 `4+2 = 10+(-4)` 检查此语义，并核对 conv 与 R0–R2 的 snapshot 字节。这是 capture/restore 的验证，不是 MTP snapshot 生成算法已经正确的证明。

### 6.2 完整 final acquire fence 进入正常 batch 路径

新增 `server_rerot_fence_checkpoint`，跟随 logical node 保存闭合候选的原始 token ID、storage position、run ID，以及首行 token 之前的 hand checkpoint。假候选被否定时丢弃 checkpoint；它不是重新 tokenize 一个 `</ID>`。

最终 survivor 确定后：

```text
安装稳定 reader view
恢复首个闭合 token 之前的局部状态（不回退当前 PUBLIC brain）
移除本 Lane 的原 PRIVATE suffix KV（不动公开 archive 或其他 Lane）
用原 token/position/run 逐行 PRIVATE 回放，经过正常 batch/decode
确认所有行已成功提交
才允许 complete_serial_tail / 恢复原用户 grammar / serial resume
```

回放不再次扩展 logical run、不写 prompt tape、不重复计入 model/sample/public/private token，也不重复发 stream。没有在 `post_decode()` 递归调用 `llama_decode()`，避免覆盖其他 slot 尚未采样的 logits。非最终退出者及时释放 fence checkpoint；最终回放完成后释放其 hand 字节。

原来的 end-of-batch fence poll 会与新回放路径重入。第一次原题实测因此返回 HTTP 500：已经排入 8 行回放，却再次 prepare。已修成幂等进入：保留 prepared 状态和 cursor，不再次恢复 checkpoint、不从头重放。

Episode state **version 2 → 3**，保存 fence 的 token tape、prepared 状态和 cursor；旧 RERoT episode blob 明确拒绝，不能静默丢失新字段后恢复。这不是 TriAttention calibration version 的变化。已覆盖中途 episode blob roundtrip；完整 server RAM demotion/resume 矩阵仍未验收。

### 6.3 root 的 PRIVATE/PUBLIC 切换是坐标换算，不是新递推

root 使用独立的 PRIVATE brain。旧 `rerot_set_write_tag()` 只切换基底，不换算 hand，因而没有输入新 token 也会改变有效状态。新反例在旧实现触发 **2 个失败**：原状态 `B_private=4,H=2`，切成 `B_public=1` 后必须变为 `H=5`，不能仍为 2。

现在在同 episode、同 root 的基底切换时保持：

```text
B_old + H_old = B_new + H_new
H_new = H_old + B_old - B_new
```

反复安装相同 tag 不重复换算；两个 brain 本身不修改。仍有 FP16 hand 的正常舍入误差，不能称 bitwise 等价或“清除私有指令”。这也修正了 N=1 root 从 private serial-resume 转入 public 串行 token 时的状态断裂。

### 6.4 数值与真实模型证据

两步闭合回放的 production recurrent graph 对独立标量参考：output 最大误差 `2.74169e-9`，FP16 hand 重构后的 state 最大误差 `0`，当前 PUBLIC brain 保持不变。runtime 回归检查跨 token 闭合、拒绝只换视图就进入 serial、逐行确认、中途保存、重复 prepare 不重置，以及所有逻辑计数/epochs 不重复变化。related CPU/Vulkan 测试仍需与真实回答质量分开看。

真实模型使用原题、temperature 0、seed 424242、max_tokens 8192、MTP 未启用、trie 开启；开发 context 为 **131072**，不是生产全配置验收。实例为本次前台 harness 的子进程，结束后已停止，生产 unit 未启动。

| 运行 | 证据目录 | 结果 |
|---|---|---|
| 首次接入回放 | `/tmp/rerot-semantic-20260907-210636` | HTTP 500；重复 prepare；4.653 s；该版本不通过 |
| 修正重入后 | `/tmp/rerot-semantic-icb_fqv1` | HTTP 200；`finish_reason=length`；102.613 s；正文 10,250 字符；reasoning 421 字符；completion 8426；仍不通过 |

第二次日志确认 `final fence replay ... rows=8`，随后进入 `final child fence`。公开文档只有 803 bytes，包含北美洲段写“非洲国家”、残缺 HTML 标题等明显错误。不能把 HTTP 200 或 fence 成功当成回答正确。请求预算 8192 与 completion 8426 也不一致，预算核算尚待修复。

新增 `scripts/rerot-semantic-smoke.py`，每次保存 request、原始 response、前后 metrics、完整日志、独立 trie 目录及 binary/library hashes；只检查传输/生命周期，不给国家清单打“质量通过”分数。复现：

```bash
python3 scripts/rerot-semantic-smoke.py --model /opt/llama/data/Ornith-1.5-35B-Uncensored-YMQ-S-MTP.gguf
```

脚本遇到 HTTP 错误、非 stop 或空正文均返回非零；本轮第二次返回 1，开发 server 正常退出 0。没有修改原始洲题脚本的错误八洲评分，也没有沿用其 PASS。

### 6.5 当前剩余语义门

- 并行正文仍串线、碎片化，串行正文耗尽预算。具体模型级根因未确认，不能归结为本轮任何一个已修局部 bug。
- `server_rerot_child_grammar()` 仍在首个换行之后只允许精确 close marker。它确实把 child 限制为一行，不是任意长度任务的自然退出证明。本轮未同时改变该协议变量，也未把它当作已证明的唯一质量根因。
- 总 completion 预算、active multi-token/snapshot 的 block 数学、一般 topology barrier 的 logits refresh、真实 RAM/MTP/context-shift 恢复仍需核验。

性能未优化，也没有新的 500 tok/s 验收成绩。本轮不提交、不推送、不部署；未触碰 `lkjdfk/` 或桌面会话。

### 6.6 第二轮收尾验证

最后一次构建包含 fence poll 重入修正、root 换基底、snapshot capture 回归及独立的非法 serial-transition 测试：`llama-server` 与全部 8 个相关 test target 构建成功，CTest **8/8**；直接运行 recurrent 与 attention 均 **0 failures**；smoke 脚本语法检查和 `git diff --check` 通过。构建日志 `/tmp/rerot-refine-final-build.log`，直接数值测试日志 `/tmp/rerot-refine-recurrent-final.log` 与 `/tmp/rerot-refine-attn-final.log`。

最后实查 production unit 为 `inactive`，没有 `llama-server` 开发进程残留。真实模型复现仍以 §6.4 的失败结果为准，不能用这组单元测试覆盖它。

## 7. 第三轮：撤销一行限制，区分数值错误与算法语义

### 7.1 多段 child 正文恢复

`server_rerot_child_grammar()` 不再将首个换行当成退出条件。换行、空行、代码块、嵌套 `<ol>` 都可继续正文，只有 owner 的精确 `</ID>` 结束工作块。原测试中“第二行必须被拒绝”的断言已反转，增加 256 行正文等用例；只改测试、保留旧实现时有 **4 个断言失败**，改 grammar 后通过。指南 §4.2.1 和 admission 中残留的 one-line 要求一并撤销。

默认路径没有增加 per-child 长度预算、重复字符串截断或题目专用 stop。长时间不退出的模型仍会暴露为失败，不能把它伪装成已完成。

### 7.2 真实张量反例：CG 的 N 步结论误用于浮点数

新增显式 `--audit` 诊断，记录 raw logits、采样结果、非有限数、逐层小 activation 统计，并可捕获 RBB 的原始输入及输出。默认关闭。它会增加同步和读回，所以不计作性能测试。

从真实 Ornith 的 layer-4 执行捕获 6 个 block，用 `scripts/rerot-audit-replay.py` 和 `test-rerot-attn --rbb-replay` 对同一组输入独立求解，避免比较两条已经分叉的自回归轨迹。输入为实际的 `D=128, H=32`，writer 数分别为 3/6/6/7/7/4；观测到的最大条件数约 468。

原 Vulkan 实现对 N≤32 只做 N 次 CG。**N 步在精确算术中成立，不是 FP32 的收敛保证。** 真实四笔例子触发了数值断言：

| 同一 captured block | 修正前 | 修正后 |
|---|---:|---:|
| shared brain 最大绝对误差 | 0.00918055 | 4.76837e-7 |
| output 最大绝对误差 | 6.5567e-5 | 7.45058e-9 |
| hand 最大绝对误差 | 1.19209e-7 | 1.19209e-7 |

已改为 4N 次有界求解，每 N 步重新计算真实残差 `b-Ax` 并重启方向，避免只累积递推残差。所有 workgroup 使用相同循环边界，保留 barrier 的一致性。新增高 beta、相关 key 回归；CPU double Cholesky 仍是独立参考，不是照抄 GPU CG。

证据：`/tmp/rerot-real-block-replay.json`、`/tmp/rerot-cg-before.log`、`/tmp/rerot-cg-after.log`、`/tmp/rerot-cg-regression.log`。捕获目录为 `/tmp/rerot-semantic-r753heg6/tensor-audit/block-{0..5}`。4N 修正通过了这些输入，不等于给所有病态输入证明了误差上界；残差契约仍需继续加强。

### 7.3 数学上确实存在并发宽度引起的写入增强

令 N 个 PUBLIC writer 的单位 key、value、beta 和 decay 都相同。当前 block 规则沿该 key 的有效写入门为：

```text
beta_effective(N) = N*beta / (1 + (N-1)*beta + eps*beta)
```

例如 beta=0.3，8 个相同 writer 会产生约 0.774 的有效门，而不是 0.3。log-decay 只走一拍，**不意味着写入力度也只走一拍**。正交写入不稀释，和重复证据不放大，是两个不同条件。这个反例是确定的代数事实；它说明 block 规则会改变训练时的记忆动力学，不自动证明该规则非法，更不证明它是此次模式坍缩的唯一原因。

已加入重复 writer 的 CPU/Vulkan 断言，并保留 `coherence-write` 显式研究对照：按 key 相关度调整正则项，减小重复证据的宽度放大。该模式不是默认算法，也没有通过质量验收。正则项和 N=1 的特殊处理仍需区分，不能把“约等于单笔”写成浮点严格相等。

### 7.4 消融没有给出“根治”证据

`scripts/rerot-semantic-smoke.py` 现在明确记录研究参数，所有模式均另建证据目录；无实验开关时仍为默认共享算法、strong frontier、Turbo4/Turbo2。研究模式必须与发布验收分开：

| 对照 | 证据目录 | 观察与边界 |
|---|---|---|
| 撤销一行限制，默认算法 | `/tmp/rerot-semantic-dct2mw4y` | 240 s 客户端超时；并行 Lane 早期出现重复 `>`，不是只有 serial tail 出错 |
| raw-logit 审计 | `/tmp/rerot-semantic-jv0apmkp` | 重复字符可成为 raw top；各 row hash 不同，观测窗口无 NaN/Inf；排除的是“全部读同一行”，不是所有可能的行排列错误 |
| 同题 RERoT OFF，单请求 | `/tmp/rerot-semantic-27f1erl3` | HTTP 200、stop、2075 completion tokens、19.568 s；未对各国名单作全面评分 |
| native-read | `/tmp/rerot-semantic-i7rc3j10` | 只改变共享层当前 readout，仍未自然完成；不是有效修复 |
| local-state + lag1 | `/tmp/rerot-semantic-b5lc4945` | 研究窗口超时；不能据此将共享 block 定为唯一根因 |
| coherence-write | `/tmp/rerot-semantic-ms8zox6x` | 70 s 窗口超时，没有质量通过证据 |
| 显式通用 child 指令 | `/tmp/rerot-semantic-pht89r0d` | 70 s 窗口超时；没有把该实验升级成默认 prompt |
| local-state + ancestors-only + child 指令 | `/tmp/rerot-semantic-idoviim0` | 45 s 窗口超时；不再是全 Lane 同字符，但仍续写其他章节，没有得到完整正确退出证据 |
| RERoT OFF，8 个独立并发请求 | `/tmp/rerot-semantic-onp0bc13` | 一个 stop，七个在 120 s 超时；取消之前仍在推进。超时本身不能证明七个答案都错误或永不结束 |
| F16 KV，context 131072 | `/tmp/rerot-semantic-peqvlyvo` | auto-fit 在 ready 前失败，没有推理结果 |
| F16 KV，开发 context 8192 | `/tmp/rerot-semantic-z1gjw2ra` | 启动成功，50 s 超时；日志反复生成 `</h>`。增加 KV 精度没有消除观测到的异常 |

`native-read`、`local-state`、`coherence-write`、`ancestors-only` 和通用 child 指令都只是因果定位开关，不是自动 fallback。它们既不能用来交付“真实共享”，也不能和默认配置混报速度。短窗口的超时保留为未完成观察，不写成自然 length/stop；已保存能取得的 metrics 和完整服务日志。

### 7.5 固定 token 的全模型单路/多路对照

新增 `test-rerot-model-batch`（显式运行，需要本地模型，不注册到无模型 CTest）。从同一 prefix checkpoint 出发，先记录单路逐层/逐步输出，再用相同 token 重放到多路。`distinct` 模式为各 Lane 使用不同的固定 token 序列，并分别与它自己的单路参考比较，避免“全相同输入看不出互换”。这不是原题质量评测。

F16 对照中的单路 restore 重放与原单路日志一致；不同 token 的 4 路、4 步已执行成功，各路 raw top 与自己的参考一致。最后一步四路 logits relative L2 分别为约 `5.31e-4 / 1.22e-4 / 1.96e-4 / 2.43e-4`。证据 `/tmp/rerot-model-batch-distinct-f16.log`。

先前 2 路相同输入的对照显示：前三个 recurrent 层先保持一致，差异从 Full Attention 开始，Turbo 比 F16 的后续差异大。它是量化/执行形状敏感性的证据，不足以认定 KV 损坏。第一次 4 路尝试因给 shared-COW 源行没有留够物理槽位而失败；诊断程序已显式预留一行，并完成上述 4 路测试。早期失败日志不能当成 4 路数值结果。

```bash
cmake --build build-vulkan-localhost --target test-rerot-model-batch -j6
build-vulkan-localhost/bin/test-rerot-model-batch "$MODEL" 4 4 f16 distinct
```

该程序 exit 0 只表示调用成功、logits 有限；不把诊断差异报告伪装成任务质量或完整 batch 等价门通过。

同样的 `4 4 turbo distinct` 对照也完成了，日志 `/tmp/rerot-model-batch-distinct-turbo.log`。两种精度各有 28 个记录（1/2/4 路各 4 步），都没有 raw top 与本路单路参考不同的记录；但最大 logits relative L2 为 F16 `0.000560437`、Turbo `0.0621743`。在第 4 步的 4 路输出中，Turbo relative L2 为 `0.0525697 / 0.0259824 / 0.054966 / 0.00885057`。这进一步说明“argmax 相同”和“数值等价”是两回事，不能因为四步 argmax 都相同就宣布长期 greedy 完全一致。

### 7.6 仍未证实的主根因

已证实并修复的是多行协议错误和真实张量触发的 CG 精度错误。已证实的算法性质是重复写入放大。**默认 RERoT 仍没有完整、正确、自然结束的原题结果**；不能把任意一个局部修复宣称为科研问题已经解决。

后续应优先做固定分支 token tape 的层级反事实，核对 fork 时 KV/recurrent 上下文是否一致、DDVR 的实际 reader 内容与 query/logits 绑定，而不是继续调停止条件。现有数值对照仅覆盖普通模型路径和局部算子，不是完整分支/递归/队列/恢复的等价证明。全局 budget、MTP、RAM/context-shift 仍是前两轮记录的未完成项。

### 7.7 本轮最后核验

已重新构建 `llama-server`、8 个相关无模型测试及 `test-rerot-model-batch`。CTest **8/8**；从项目根目录直接运行 attention、recurrent 均 **0 failures**。新 model-batch target 经过显式 CMake reconfigure 后构建成功，Turbo distinct 实际运行 exit 0，不能把首次未刷新 CMake rules 的 `No rule to make target` 当成最终构建失败或测试通过。

日志：`/tmp/rerot-science-configure.log`、`/tmp/rerot-science-final-build.log`、`/tmp/rerot-science-final-attn.log`、`/tmp/rerot-science-final-recurrent.log`。最后实查生产服务为 `inactive`，没有 `llama-server` 进程残留，`git diff --check` 通过。所有实验开关均为显式研究控制；默认原题未过质量门，不部署、不提交、不推送，不报新性能成绩。

## 8. 第四轮：阶段分离与 recurrent sharing 降级

### 8.1 同一 frontier 的 peer 不得同拍穿透

此前 `STRONG` 把同一联合 forward 内、`meta.frontier == reader.frontier` 的 peer PUBLIC K/V 直接放进当前 query 的 attention。物理上 KV copy 发生在 attention 前，不等于逻辑上 peer write 已完成。同步 frontier 应分成：

```text
read stage:  自己的 causal history + 自己 current K/V + barrier 前已提交的 peer memory
write stage: 本 frontier 各 Lane 写入自己的 K/V / local recurrent transition
barrier:     frontier 完成
next read:   才能读到上一 frontier 的 peer PUBLIC writes
```

已直接修改默认实现：STRONG 只读 `peer.frontier < reader.frontier`；自己的 causal history 不受 peer policy 延迟。LAG1 在此基础上再多延迟一个已提交 frontier。`test-rerot-view` 与 attention reference 同步改写，避免“同拍 peer 可见”继续作为正确性断言。focused view/attention 测试均 0 failures。

### 8.2 shared RBB 的当前 token read 也必须分阶段

RBB 原先在同一 recurrent layer 先合并本 frontier 所有 writer 得到 `B'`，再令每个 `q_i` 立即读取 `B'+H_i'`。这使联合 batch 具备训练时不存在的瞬时多人通信。已改为：当前 output 严格使用 `native_GDN(T_i(B+H_i))`；`B'` 只提交给下一 frontier。CPU/Vulkan、mixed-person/mixed-visibility oracle 通过。

同时保留前一轮已经直接升为默认的两项数学修正：重复 writer 的 evidence-density normalization（8 个相同 writer、`beta=0.3` 从旧 `beta_eff≈0.774` 收回 `≈0.300`），以及零均值 self-echo `C_i-mean(C)`，其 writer 和严格为 0，不改变共享 brain，N=1 为 0。

### 8.3 更根本的问题：DDVR-conditioned hidden 再写 global recurrent 会重复广播

经过正交消融，低层因果关系变得清楚：

- lane-local recurrent + peer DDVR：35 秒窗口可连续生成不同洲的正常国家文本，没有字符级坍缩；
- 修正后的 shared RBB + ancestors-only（不读 peer KV）：同样不出现字符级坍缩；
- 两条共享通道同时开启：长跑仍会重新同步到 planner 标题 / HTML 结构片段；
- 这不是 batch row 绑定错误：两 Lane 物理行完全反转的固定-token 模型对照，Turbo 12 步最大 logits relative L2 约 `6.3e-7`，各自 top token 不变。

原因不是一个可由 block algebra 单独修正的小系数。Full Attention 已将 peer PUBLIC 内容写进 `h_i`；后续 GDN 参数 `q_i,k_i,v_i,g_i,beta_i` 是 `h_i` 的非线性投影。只观察这些量，不能唯一恢复“这部分来自 peer 公共记忆、这部分是 Lane 新创新”。因此把所有 `T_i` 再 merge 到 global brain 会形成第二条反馈通道，却没有可识别的去重项。

据此，**child Lane 的正式默认已改回模型训练过的 lane-local native recurrence；跨 Lane 只由 DDVR 共享。** Shared RBB 不删除，改为 `LLAMA_REROT_RBB_ABLATION=shared-rbb` 显式科研模式；`raw-redundant` 保留错误对照。这是数学可识别性决定，不是按输出质量动态 fallback。

### 8.4 “不同笔必须看不同地方”已直接测量

新增 A3 `rerot_indexed_attn` 的只读 tensor audit，离线重建每个 reader、每个 head 在实际 physical K 上的 softmax distribution。早期默认运行的 reader attention 已不同；在阶段分离 + lane-local recurrent 后，用 F16 KV、context 8192 的开发审计继续抓到 observation 128/192/256：

| observation | pair attention cosine mean | top-key overlap mean |
|---:|---:|---:|
| 128 | 0.04088 | 0 |
| 192 | 0.03542 | 0 |
| 256 | 0.03469 | 0 |

因此当前实现**没有把不同 Lane 的 Q/attention 强制成同一个**；越到后期，实际 attention distribution 反而高度 reader-specific。证据目录 `/tmp/rerot-semantic-sxog2mev/tensor-audit`，分析脚本 `scripts/rerot-attention-audit.py`。Turbo 快照没有被脚本冒充 F16 解码；Turbo offline dequant 支持尚未补，故这些定量值来自明确的 F16 数值控制。

### 8.5 当前问题分层判断

到本轮为止，可以把问题分层改判：

1. **实现/数值层：曾经确有硬 bug，且已修。** 包括 CG N-step 浮点误用、beta=0 泄漏、checkpoint/COW、root 基底切换、mixed batch 偷换算法等。
2. **数学/动力学层：曾经是字符级坍缩的主要来源，且已直接改默认。** 包括重复证据随 Lane 数放大、同层即时读 peer recurrent write、同 frontier peer attention 穿透，以及无法识别 innovation 时的 DDVR→global-RBB 二次广播。
3. **语义/协议层：现在才真正开始成为主问题。** 在低层修正后，不再观察到所有 Lane 共用同一 attention；输出也从单字符/`</p>` 死循环上移为“各 Lane 看不同位置，但有的重复章节结构、补首都/语言等未要求内容、迟迟不闭合自己的 task”。这才符合“句子/任务层的偏航”定义。

默认原题 90 秒开发运行 `/tmp/rerot-semantic-797cuej1` 仍超时，因此不能宣布质量通过；但其失败形态已不同于前面的字符级模式坍缩。下一步应检查 child task identity / close contract / PAC-DFS reader 中“自己的任务描述”是否持续可辨，而不是回头增加长度截断或把 attention 强行做成一样。

## 9. 第五轮：语义 phase 收束与 OpenAI reasoning_effort

### 9.1 child 身份与 planner phase 显式化

默认 child 现在会收到 owner-only PRIVATE task contract，内容绑定自己的 exact title；兄弟章节只能作为参考。递归 planner 不再靠“正文中出现 `<ol>`”暗中触发，而是显式进入 PRIVATE planner phase、安装 planner grammar，N=1 后再显式切回 worker phase。正文阶段的普通 `<ol>/<li>` 永远只是正文格式，不能改变拓扑。

这是协议消歧，不是内容 stop hack。相关 runtime 测试覆盖：child 正文列表不 fork、显式 planner 才能 recursive fork、N=1 planner 后正文列表仍保持 terminal worker。

### 9.2 随机 ID 仍是唯一 child completion delimiter

随机 8 位 base62 `</ID>` 结束结构保持不变。期间曾实验“模型 EOG 触发内部 PRIVATE close”，但用户明确否决该方向；该实验代码已完整撤回。当前正式协议仍为：worker 在 owner-specific close grammar 下生成，只有自己的精确随机 `</ID>` 完成 child；EOG 在 delimiter 关闭前仍是协议错误。几千或几万 token 的 child 思考本身不作为异常，也不加长度/重复截断。

### 9.3 OpenAI reasoning effort 已落到请求链路

OpenAI Chat Completions 顶层 `reasoning_effort` 现在严格接受：

```text
none / low / medium / high / xhigh / max
```

Responses API 的 `reasoning.effort` 继续转换到同一 `reasoning_effort` 路径。顶层字段覆盖同名 `chat_template_kwargs`，并原样传给支持该 kwarg 的 chat template；`none` 关闭 thinking，其余档位显式开启。

对于具备 thinking start/end tag、且没有显式 `reasoning_budget_tokens` / `thinking_budget_tokens` 的普通非-RERoT请求，增加确定性的本地 budget fallback。若请求有有限 output budget `M`：

重新查证 Anthropic 官方与开源实现后，撤销最初的 `1/8,2/8,4/8,6/8,7/8` 本地比例。Anthropic 当前 `effort` 是软行为信号，不是严格 thinking-token quota；新 Claude 的正确路径是直接传 effort/adaptive thinking，而不是把 effort 硬换成 budget。只有本地/旧后端**仅支持 token budget**时才做兼容映射。

当前 fallback 对齐 OpenRouter 的 Anthropic budget-only 公式（Cline 也使用同一主梯度，`max` 的抽象 ratio 唯一略有不同）：

```text
low    = 20% * M
medium = 50% * M
high   = 80% * M
xhigh  = 95% * M
max    = 95% * M
```

常规 clamp 为 `[1024, 128000]`，同时本地实现始终至少给最终输出保留 1 token；显式 token budget 永远优先。若没有有限 output budget，则不再凭空发明 2K/8K/32K/64K 这类绝对预算，只保留原生 template effort。RERoT 继续拥有自己的随机-ID生命周期，不能拿 reasoning budget 强制替代 child close。

`task_params` 现在保存并在 generation settings 中暴露 `reasoning_effort` 与最终 `reasoning_budget_tokens`；Web UI API type 也加入完整枚举。README 已从“除 none 外无效”更新为真实行为。

开源实现并非完全一致。OpenRouter 对 budget-only 模型公开的比例是 `low=.20 / medium=.50 / high=.80 / xhigh=.95 / max=.95`；Cline 的完整 effort ladder 是 `none=0 / minimal=.10 / low=.20 / medium=.50 / high=.80 / xhigh=.95 / max=1.00`。因为本项目 API 同时暴露 `xhigh` 和 `max`，fallback 采用 Cline 的 distinct-max 语义：`max=100%`，再由 `budget < max_tokens` 的硬约束落成至多 `max_tokens-1`。这仍然只是 budget-only 兼容层，不冒充 Anthropic adaptive effort 的模型原生分配。

测试：`test-chat` 覆盖五档 template kwarg、五档 8192 output budget 映射（1638/4096/6553/7782/8191）、无有限 output cap 时不制造 synthetic budget、显式 budget override、非法 effort 拒绝，以及 Responses `reasoning.effort -> reasoning_effort`。最后相关 CTest 8/8、`test-chat` all passed、child parser 0 failures，`git diff --check` 通过；生产服务保持 inactive。

## 10. 2026-09-08：recurrent rollback 修复与真实决策分叉定位

### 10.1 本轮基线和已有 F32 修复

基线仍为 `master...origin/master`，HEAD `205913a2af1846521687fc8d47bf91df459ae8bd`。本节记录未提交工作树，不能只用这个 HEAD 或 build number 复现；必须同时使用各 run 的 `source.patch`、`source.json`、库哈希和完整参数。

恢复工具时，工作树已有 full-F32 scalar/indexed attention、F32 persistent hand 及相关测试修改。本轮保留并验证，不把这些已有改动冒充新的修复：

- ordinary scalar 的非 MMQ F32 路径不再在 Q staging、softmax/PV 中偷偷回到 F16；indexed 路径的 Turbo dequant 也使用 F32 模块，shared-memory 预算与实际模块一致。
- persistent hand 从 F16 改为 F32；SEE3 hand seed 明确拒绝旧 SEE2 格式。原生 recurrent state 不是可任意有损压缩的 KV cache。
- 已有 128 步确定性 recurrent 反例：F16 hand 的 output/state 误差约 `5.76e-4 / 1.72e-3`；F32 降到 `3.58e-7 / 7.15e-7`。证据在 `/tmp/rerot-p0-fix-20260908-_gintnjm`。

### 10.2 新修复：rollback 开关改变了 child 方程

`build_recurrent_attn` 的 `n_rs_seq > 0` 分支原来只提交 shared brain，没有保存 hand snapshots。默认 child 不提交 shared brain，因此即使没有真正执行 rollback，仅启用两个 rollback slots 就会丢失持续 recurrent transition。单步 output 测试看不到这件事。

新增生产 graph/memory 路径反例，128 步结果：

| rollback slots | 修复前 output/state 最大误差 | 修复后 output/state 最大误差 |
|---:|---:|---:|
| 0 | `3.58e-7 / 7.15e-7` | `3.58e-7 / 7.15e-7` |
| 2 | `2.17231 / 1.14956` | `2.68e-7 / 7.15e-7` |

现在每份完整 GDN state snapshot 都保存对应的 hand：默认 child 以不变 input brain 为参照，真正 shared writer 以本 snapshot 的实际 committed brain 为参照。先写齐 hand，再提交 brain。

继续执行“三份 snapshot → 丢弃 suffix → 续跑”又发现第二个错误：child hand 的 rollback index 被错误地用于选择 root 的历史 brain 槽。默认 child 从未写过这些槽。已在 `brain_copy` 和 `capture_hand_seed` 中区分 native child 与实际 shared writer，保留 PUBLIC root 的 snapshot rebasing。回归同时覆盖三份快照、`seq_rm`、checkpoint capture/apply 和 resume。

核心数学、CG、重复证据 normalization、默认 child 策略和结束协议均未回退。

### 10.3 固定旧 tape：rollback 0/2 已经数值一致，但不能宣告总体等价

使用修复前保存的 `/tmp/rerot-p0-fix-20260908-_gintnjm/teacher.tape`，prompt 为原始文本 `世界上每个大洲有哪些国家`，无 eval callback，128 步：

| KV | argmax mismatch | logits 相对 L2 最大值 |
|---|---:|---:|
| Turbo4/Turbo2 | 0/128 | 0.0742375 |
| F16/F16 | 0/128 | 0.0348572 |

每种 KV 下，rollback 0 与 2 的逐步 logits 报告完全相同。prefill logits 完全相同。这证明本反例中的 rollback 开关不再改变持续推理；几个百分点的误差仍然存在，不能说所有 shape sensitivity 已解决。

诊断现在默认严格检查，分叉返回失败；`--report-only` 才明确允许只报告。prefill 非有限值/不相等也失败。显式 `--precision-only` 没有 GPU 时不再 skip-success。I32 expert IDs 单独报告排序变化与集合变化，不再把编号差当作浮点 activation 误差。

### 10.4 512 步反例及 counterfactual：第 29 层 expert membership 改变最终 token

另生成 512 步 native Turbo teacher tape，并让 F16 使用同一份 tape。这不是上面的旧 tape，不能把旧实验的 step=33 与下面的 step=509 当成同一输入的前后延迟指标。

- Turbo：1/512 argmax mismatch，首次 step=509（零起算），原生 token `97852`，RERoT token `134536`；严格进程 exit 1。
- F16：0/512 mismatch，但 logits relative L2 最高仍到 0.132615。
- 两种模式 prefill 都完全相等。所有 run 保存源码 diff、实际加载库路径/哈希及参数；运行期间源码和库哈希不变。

在 step=509 完整逐层 trace 中再次复现同一分叉，logits 相对 L2 `0.0953225`，与无插桩的 `0.0953226` 接近：

1. 第一处非零浮点差异是 A3 attention output，relative L2 `2.43e-7`；A19 上升到 `0.00133105`，A27 到 `0.0379533`。
2. 十层 full attention 的 visible key 集合、顺序均相同（516 keys，单 Q group）；A3 实际 K/V 字节也相同。A19 已有历史 Turbo K 的离散 code 差异，后续层差异增多，不能只看当前 Q/K/V 投影。
3. 第 26/28 层仅 selected expert 顺序变化。第 29 层第一次改变选中的 expert 集合：native 的 `203` 被 RERoT 的 `5` 替代。
4. **仅在诊断中**将第 29 层的 8 个 selected expert IDs 换回 native 值，其他历史不变，最终 token 恢复为 `97852`。这是本反例一个可直接改变 sampled decision 的位置。

counterfactual 不是 production 修复，也不证明 router 算错了。上游细微误差、量化阈值和历史状态仍须继续定位；不得把 native expert ID 硬塞进生产来掩盖差异。这一因果实验也尚未证明就是 9.11 数字漂移的根因。

### 10.5 短题 smoke：完成改善，数字漂移仍在

先发现 harness 配置错误：旧默认 `context=131072` 与显式 `total-kv=8192` 不相容，server 在模型初始化阶段拒绝，未产生回答。未放宽 server 的检查；脚本改为：manual KV 且未显式给 context 时选择 `min(131072,total-kv)`，显式冲突报错，auto 仍保留原默认。请求内容、采样和结束协议不变，解析后的容量写入 diagnostics。

沿用短题、seed=424242、temperature=0、max_tokens=512、KV8192/parallel2，解析后的 context8192：

```text
evidence = /tmp/rerot-semantic-5k83b_1v
HTTP = 200
wall = 17.991691 s
finish_reason = stop
root children = 3
```

三个 child 都通过 `injection=0` 的采样自然生成自己的 random-ID close。最终回答正确比较 `9.9 > 9.11`。但是第一 child 的公开正文仍出现 `3.9 和 3.11`、字面 `</think>`，另有重复标题；对应 trace 也都是 `injection=0`。这是模型生成偏航，不是 PRIVATE transport 直接泄漏。不能据此宣布语义质量通过，不能用去字串或更换结束条件遮住它。

本次 context 明确为8192，不冒充之前其他 capacity/context 配置的严格质量 A/B。

### 10.6 验证与后续入口

主证据目录：`/tmp/rerot-resume-fix-20260908-tdq0ptmb`。重点看：

```text
rollback-before.log                 # 修复前失败
final-build.log / final-ctest.log    # 最终 build / 9 of 9 PASS
frozen128-*-rollback*/              # 同一旧 tape，rollback 0/2
extended512-turbo/                  # 无插桩、step509 真实失败
trace509-turbo/ / step509-tensors/   # 逐层原始张量
counterfactual509-moe29/            # 仅诊断注入，不能作 production pass
counterfactual-summary.json
step509-kv-differences.json
```

最终提供的 build targets 构建成功，CTest `rerot|kv-cells|triattention-score` 为 **9/9 PASS**（原8项加新 harness 测试）。原有 child mixed-batch CPU/Vulkan regression 保留。生产服务保持 inactive，没有部署、重启、提交或推送，两个原有未跟踪项未动。

下一条数值路线已明确：使用保存的512 tape，回溯 A19 首个不同 Turbo K code 对应的历史 token（本快照最早物理 key397），固定量化前 tensor 和持久 state，区分舍入起点与离散阈值放大；不要改 prompt 或强制 router 决策。

## 11. 2026-09-08 后续核对：CPU F32 indexed attention 与误差门禁

本次实际恢复了命令执行和写入，在已有未提交修改上继续；§10 的 Vulkan full-F32、F32 hand、rollback 修复不是本次新增代码。没有改默认 child 定义、RBB 数学、STRONG frontier 或 completion 协议。

### 11.1 先修数值测试的假通过

`test-rerot-attn.cpp::max_abs_diff` 使用较短向量长度比较，且 `std::max(m, NaN)` 会保留旧的 `m`。因此空输出、长度不一致、NaN 可能被报告为零误差。recurrent 的同类函数已有长度 CHECK，但同样会忽略 NaN。

已先加入失败反例，两个 focused 进程均返回 1；再修改两个比较器：空输出、长度不一致、任一侧非有限数统一返回正无穷误差。反例覆盖 NaN、正负无穷在左侧、右侧和两侧同时出现，不再依赖正常张量恰好不产生 NaN。

### 11.2 独立 double 参考暴露了 CPU 的 Q 降精度

精度测试现在对 F16 和 Turbo 都从实际相同的量化 K/V 字节解码，独立使用 double 累加 QK、softmax、PV；不调用任一 attention 算子生成参考答案。CPU indexed 和 Vulkan ordinary/indexed 都分别与它比较。

这直接发现 CPU `ggml_compute_forward_flash_attn_ext_rerot` 无视显式 F32 选择，按 K 的 vec-dot 类型转换 Q。F16 K 会使原始 F32 Q 先舍入成 F16。修改仅限 RERoT indexed 的 `GGML_PREC_F32`：保留 Q，按需把 K 解码到 F32 后点积；普通 attention 和 indexed DEFAULT 分支不变。CPU scratch planner 同步按 F32 K 临时行预算，避免扩大缓冲使用却不扩大分配。

| 独立 double 参考，F16 KV | CPU 修复前最大绝对误差 | CPU 修复后 |
|---|---:|---:|
| 33 keys | 5.55948e-5 | 1.19209e-7 |
| 257 keys | 2.067e-5 | 8.9407e-8 |

Turbo CPU 的同两项最终为 `1.19209e-7 / 8.9407e-8`。Vulkan 两条路径在这些 F16/Turbo 合成用例中均不超过 `1.19209e-7`。CPU 参考门即使没有 GPU 也会执行；显式 GPU gate 仍不得 skip-success。这些结果不是完整模型或所有 tensor shape 的误差上界。

### 11.3 当前真实模型问题仍未解决

本次旧 `/tmp` 证据目录受工作区读取权限限制，没有重放旧 step33 tape。另生成并保存 512 步 native Turbo tape，随后所有模式都读取这同一份外部 tape，均不安装 eval callback。

| KV / rollback slots | argmax mismatch | 首次分叉（零起算） | logits 相对 L2 最大值 | 真实测试 exit |
|---|---:|---:|---:|---:|
| Turbo / 0 | 1/512 | 509 | 0.118488 | 1 |
| Turbo / 2 | 1/512 | 509 | 0.118488 | 1 |
| F16 / 0 | 0/512 | 无 | 0.132615 | 0 |

三种 prefill 完全相同。CPU 修复前后的 Turbo 本反例都在 step509 分叉，不能把 CPU 修复说成 Vulkan 根因修复，也不能把历史 step33 与本次 step509 当成同一输入的前后改善。本次未重新执行 9.11 chat smoke，语义质量状态仍按 §10.5 保留为未通过。

### 11.4 验证与证据

证据使用仓库内被构建目录忽略的相对路径，便于后续工作区读取：

```text
build-vulkan-localhost/rerot-evidence/20260908-AuxjZd/
  guard-red-attn.log / guard-red-recurrent.log  # 误差门禁修复前失败
  guard-green-attn.log                         # 独立参考抓到 CPU Q 精度错误，exit 1
  final-precision.log / final-trajectory.log
  final-build.log / final-ctest.log             # build 成功，9/9 PASS
  model512-inherited/teacher.tape              # 本次新生成的固定输入
  final512-{turbo,f16}-rollback*/               # 源码 diff、参数、log、exit、库哈希
```

每次最终真实模型运行的前后 artifact 哈希相同。128 步 recurrent focused 测试：rollback0 output/state `3.57628e-7 / 7.15256e-7`，rollback2 `2.68221e-7 / 7.15256e-7`；三快照、suffix discard、capture/apply/resume 回归通过。未部署、未提交、未推送；下一步仍应查 Vulkan 长轨迹舍入/量化阈值放大，不调 prompt 或伪造结束。

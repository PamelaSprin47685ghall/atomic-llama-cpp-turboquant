# DECISION.md｜q2-contract-20261012 轮裁决

生成：2026-10-12。本文档由本轮文档工程师依据 `artifacts/perf-gap/q2-contract-20261012/` 轮次目录实际数据逐项核对后，按 AGENTS.md §17 模板成文。与 Manager 交接事实清单不符之处，一律以轮次目录实际数据为准，并在文中以【出入N】显式注明。未核实字段写 `NOT_RUN` / `NOT_COLLECTED` / `UNRESOLVED`。

**更新（2026-10-12，Manager 复核后裁决补录）：** 本更新只补录 Manager 对 V9 记录的复核结论（第七节【出入4】）与 h1 晋级裁决（第十一节），并相应更新下一轮可证伪问题清单；不改动其他事实陈述。

**第二轮补录（2026-10-12 晚）：** 补录第二轮 session 的 F 口径正式结论、h2/h3 深度闭环与乱码归因收口（新增第十二节），并同步更新第十一节可证伪问题清单。第一轮全部事实陈述与 h1 的 PROMOTE 裁决不变；理论 L=UNRESOLVED 不变。

**第三轮补录（2026-10-12/13，第 2 个 Manager 任期）：** 补录独立评审结论与 GAP 推进——profile 诊断轮周期账（GAP 4 部分闭环）、乱码根因钉死与修复的阴性一轮（GAP 1 定位闭环、修复未闭环），新增第十三节并更新问题清单。h1 的 PROMOTE 状态与理论 L=UNRESOLVED 不变。

**第四轮补录（2026-10-12 深夜，第 3 个 Manager 任期）：** 补录第 3 任 Manager 独立评审结论与 GAP 1 修复链的四轮推进（r3→r7）——静态深挖证伪 r3 根因叙述（r3 修复为死代码）、r5 运行时判别、r6 守卫阴性、r7 同一 run 决定性判别与 r5/r6 伪影结论修正（前两轮「身份分叉」结论错了，如实写明），根因收窄到写入可见性层。新增第十四节并更新问题清单。h1 状态与理论 L=UNRESOLVED 不变。

**第五轮补录（2026-10-12/13 深夜）：** 补录 r8/r9 两轮终局判别与归因链翻转——r8 sourcetrace 判「覆盖」坐实后，r9 phasestrace 三时点判别以 host/GPU 读取路径分裂的决定性洞察将其推翻，根因最终收窄为 host-GPU 可见性断裂。新增第十五节并更新问题清单。off 臂连续六轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第六轮补录（2026-10-13，第 4 个 Manager 任期）：** 补录三 Engineer 交付（守卫处置、rank/stage 采集器、三套 CPU 测试全绿）与 GAP 1 的 r10/r11/r12 三轮判别——fork 被四元组铁证排除、staging 写入被 fence 读回证实到达 device buffer，最终裁决情形 A「写后覆盖」并锁定窗口 T2→T3（首次图执行内部），r9 的可见性断裂表述被修正。新增第十六节并更新问题清单。off 臂连续九轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第七轮补录（2026-10-13，第 5 个 Manager 任期）：** 补录 r13 覆盖者定位轮——fork 被双侧四元组直接排除（r11 双指针观察确系打印语义伪差异）、覆盖被证明是选择性的（同一 VkBuffer 内两区间命运不同，r8 的全局性判断被修正为外推）、候选 (2) 正式排除、候选 (1)/(3) 因 watch 限流盲区保持未决。新增第十七节并更新问题清单。off 臂连续十轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第八轮补录（2026-10-13，第 6 个 Manager 任期）：** 补录 r14/r15 伪影判别终局——r14 三修正插桩落在 AMBIG2（无翻转、盲区闭合），r15 SYNCCMP 判别证明「W 全零」自 r2 以来是 dump 插桩自身的 deferred_memcpy 伪影：权重在设备上从未为零，覆盖者与 fork 均不存在，r2–r14 的「覆盖/fork」叙事整体证伪为连锁误读；乱码根因重定向为 BF16 kernel 数值缺陷（未定位）。新增第十八节（含 r2→r15 归因链修正表）并重写问题清单第 2/9 项。off 臂连续十二轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第九轮补录（2026-10-13）：** 补录 r16 可信读数对拍轮——dump-fixer 修复两版 dump（deferred→同步读）后首次取得可信读数：W/lo/mixed/normalized 全部非零、量级合理，GAP 1 因果链从设备侧真值上彻底关闭；两臂同崩于 dump 修复的 ensure 副作用（对拍未完成）；乱码根因新增提交粒度/barrier 缺失候选（数值缺陷候选降权）。新增第十九节并更新问题清单。off 臂连续十三轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第十轮补录（2026-10-13）：** 补录 r17 双证伪轮——ensure 副作用假设证伪（补回 ensure 崩溃不变）、barrier/提交粒度假设被既有数据证伪（r13–r15 有 mid-graph sync 仍乱码，r16「无 500」系请求未发出的误读，第九轮 19.5 推理错误修正），BF16 kernel 数值缺陷成为唯一存留候选；崩溃线（r16/r17 树四次全崩 vs r13–r15 树六次零崩）独立于乱码线，最强嫌疑为交替编译的 SPIRV 新鲜度伪影，待全新构建复验；dump 工具存在「读数可信但破坏图执行」的结构性矛盾。新增第二十节并更新问题清单。off 臂连续十四轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第十一轮补录（2026-10-13）：** 补录 r18 全新构建复验与崩溃分界档案审查——SPIRV 新鲜度伪影被哈希一致性证伪（全新构建 = 增量构建 = 9b8890af）；决定性发现：**崩溃分界不存在**——r13 server.log 最后一行就是 r16+ 的同一断言（本轮直接核验），崩溃从 r12/r13 起一直存在，「r13–r15 不崩」是 r13 verdict 的 teardown 归档惯例与 r16 verdict 的「请求未发出」两份误读叠加制造的假象（r16 dump 修复经静态审查无罪释放）；真实机理链为首个 HTTP 请求图触发 moe_route prealloc realloc → 越界断言，最强嫌疑 s_gap1_weight_simple_registry（r10/r11 引入）。新增第二十一节并更新问题清单。off 臂连续十五轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第十二轮补录（2026-10-13）：** 补录 r19/r20 崩溃线因果闭合轮——r19 纯读判别定位崩溃引入点 = r10 精确分界（r2–r9 九轮零断言正常退出、r10 首次断言即末行、与 registry 引入精确同步；r13–r15「进程活着出乱码」系第三次误读修正：500 与崩溃共存）；r20 门控因果验证三方闭合（r9 树不崩 / registry 活跃崩 / registry 旁路精确回到 r9 形态）——**s_gap1_weight_simple_registry 确认为崩溃根因**（r10 引入的诊断插桩非产品代码，存在理由已被 r15 伪影裁决抽空）；**崩溃线关闭、乱码线解锁**（数值对拍障碍解除；「dump 新版破坏图执行」待重审——dump 可能无辜）。新增第二十二节并更新问题清单。off 臂连续十六轮被前置阻断（如实）；h1 状态与理论 L=UNRESOLVED 不变。

**第十三轮补录（2026-10-13）：** 补录 r21 registry 移除验证轮——registry 四触点三块 51 行干净删除（grep 零命中）+ 测试配套两缺陷修复（__main__ 入口根因 + RegistryRegressionTests 4 项回归守卫，49 项全绿）；r21 默认配置验证：断言 0 次、正常退出、请求完成，与 r20 门控臂逐字节等价——**崩溃线终局关闭（四臂因果链：r9 不崩 / r10–r18 崩 / r20 旁路不崩 / r21 移除不崩）**；乱码线对拍通路打开（dump 无辜性待对拍实践确认、参照臂应可正常运行）。新增第二十三节并更新问题清单。off 臂连续十七轮被前置阻断（崩溃维度已通过）；h1 状态与理论 L=UNRESOLVED 不变。

**第十四轮补录（2026-10-13）：** 补录 r22 fused vs 入口还原参照数值对拍轮——乱码分歧首次定位：两臂同树同配置顺序运行（max_tokens=32、GGML_VK_CMD_REPLAY=0、GGML_VK_HC_DUMP_SITES=1），fused 臂输出乱码、参照臂（q8_projection 的 BF16 条件还原为纯 Q8_0 的 14 行临时改动）输出 c2d7bdd3 已知正常开头，**前 2 个 token「健康」逐字一致、第 3 个 token 分歧（「做」vs「减」）**——分歧起点推断在 prefill 尾部至第 2 个 decode 步之间的 HC 段计算，fused 段 BF16 变体的 lo/mixed 具体数值错误（量级对值错）为最强解释（间接对拍，边界如实标注——参照臂 dumps=0，逐张量对拍结构性不可行）；**dump 无辜性正式确认**（registry 移除后 dump 活跃下进程完全健康、teardown returncode 0——sitedump 系列首次干净退出、读数与 r16 逐字节一致——第二十节 20.5「dump 破坏图执行」叙述正式修正）；逐值验证通路打开（CPU 参照计算 vs dump 真值逐元素对拍可行）。新增第二十四节并更新问题清单。off 臂连续十八轮（r3/r5–r22）被前置阻断——阻断只剩乱码数值缺陷本身；h1 状态与理论 L=UNRESOLVED 不变。

**第十五轮补录（2026-10-13）：** 补录 r23 CPU 参照对拍尝试 + S1/S2 静态深查轮——GPU 被外部进程 strata（PID 1450245，5 卡 71.4GB VRAM，非我方进程）占用，按 AGENTS.md 红线 2 停止（不杀别人进程），CPU 参照对拍与 FORCE_ITER 判别未执行（如实记录）；dump 全量增强已落地（copy_bytes 64→全量 subbuffer，编译 9a69607d）待 GPU 释放。**S1 静态排除**（三层证据：BF16 变体宏字典不含量化参数、QUANT_K=1/QUANT_R=1 via DATA_A_BF16 注释实锤、push constants 全字段元素单位一致——32 倍/2 倍错位假说不成立）；**S2 静态排除**（五层证据：iqs=0 硬编码且 B/A 同一变量、dequantize4_2aligned 体内 iqs 未用、A/B 同 col 配对、packed32 无奇偶截断、两解码路径逐位等价——与第二轮 V-A 运行时观察互洽）；**r15a 方案盲点发现**（normalized 只被当输入锚点从未验证自身——对拍前先验证 normalized，否则上游错会误判 down kernel）；形态再权衡（状态累积型上游错天然解释「前 2 token 对、第 3 token 乱」）；**剩余嫌疑重排为 H1–H4**（H1 host 值组装最强——连续三轮静态缺口、唯一未审 host 环节）。新增第二十五节并更新问题清单。off 臂连续十八轮（r3/r5–r22）被前置阻断 + 本轮 GPU 外部阻塞；h1 状态与理论 L=UNRESOLVED 不变。

**第十六轮补录（2026-10-13）：** 补录 r24 CPU 参照对拍数值级实锤轮——H1 静态排除（ggml_vk_segment_project 函数体逐项核对：push constant 13 字段全匹配、k/m 无填反、descriptor 绑定正确、spec constants 正确——连续三轮静态缺口清偿）+ 根因时间线矛盾发现（R2「W 全零」叙述 vs r22 lo「量级合理」——交 r24 裁决）；GPU 释放后 fused 臂一次运行成功（不崩 + 全量 dump 6553600 字节/张量）；**W 权重正确性终局裁决：W 与 GGUF 逐字节一致**（V 写入可见性嫌疑排除、时间线矛盾消解——r6/r7「W 全零」确认为 dump deferred 伪影）；**lo 对拍分歧实锤**（dump lo[0]=+0.132684 vs CPU ref=-0.014946，符号反、前 32 worst 3.61；6 种公式变体全不匹配——排除公式歧义）+ **mixed 双重判别**（用 CPU 正确 lo_ref 重算仍 worst 4.80——up/gate 段自身也错）；综合裁决：**fused HC segment 两个 BF16 kernel（qwen4_hc_down_silu_bf16 + qwen4_hc_project_fold_bf16）数值计算错误实锤（down + up 双段都错，指向共享解码/累加路径）**——从「最强解释间接推断」升级为「CPU 对拍直接证据」；静态排除（S1/S2/H1）与运行时实锤的表面矛盾在 26.6 写清（不矛盾——静态排除的是参数语义不匹配机理、实锤的是输出数值错事实）。新增第二十六节并更新问题清单。off 臂连续十九轮（r3/r5–r24）被前置阻断——阻断只剩乱码数值缺陷本身的修复；h1 状态与理论 L=UNRESOLVED 不变。

**第十七轮补录（2026-10-13）：** 补录 kernel-defect-locator 静态深查 + r25 FORCE_ITER 判别轮——判读边界 (a) 清偿（qwen4exp.cpp build_hc_mix/build_hc_combine 363–431 逐行核对：lo/gate/mixed/normalized 公式与 r24 参照逐项一致——r24 判读成立无需重估）；共享层静态全排除（Q8_0 与 BF16 变体编译字典 B_TYPE/B_TYPEV4/D_TYPE/USE_SUBGROUP_ADD_NO_SHMEM 完全相同、唯一差异 DATA_A——B 读取骨架/get_offsets/reduce/D 写回有 Q8_0 生产正确性作运行时证人）；Manager 归因收敛：V/C 排除（r24 W 逐字节 + 对拍输入控制）、**H3（驱动/编译器对 aligned 路径编译产物行为）收敛为唯一剩余嫌疑**；r25 FORCE_ITER 三轮 A/B 判别**实验无效**（B/B2 乱码与 dump 与 A/r24 逐字节一致——两独立数值实现不可能逐字节同值 ⇒ 开关运行时效果未达 kernel；与 r13 capture gate「静态正确≠运行时活跃」同型教训）——H3 裁决悬置、开关链断裂点待插打印定位。新增第二十七节并更新问题清单。off 臂连续二十轮（r3/r5–r25）被前置阻断；h1 状态与理论 L=UNRESOLVED 不变。

**第十八轮补录（2026-10-13）：** 补录 r26 插打印运行时裁决 + 归因重估轮——switch-chain-tracer 插打印（GGML_VK_HC_TRACE_FLAGS=1 门控：pipeline 身份布尔×7 + force_iter_env/match + pc.fusion_flags 十六进制 + grid）+ 候选 (2) shader 侧静态排除（is_aligned_nonquant 是运行时 const bool——正确形态；附带勘误第十七轮行号标注张冠李戴）+ 五环链静态全自洽；r26 两跑直接证据：跑 1 证明 fused 段两算子都真实走 BF16 变体（grid/k/m/n_rows 合理、fold grid-y=4 upfold 形态在位），跑 2 证明 **FORCE_ITER 开关链 host+shader 双侧全通**（force_iter_env=1 + force_iter_match=1 + fusion_flags=0x00000200 注入全部 32070 行 dispatch）——**r25「开关无效」推断被直接证据推翻**：路径确实切换了，而 **iter() 标量路径与 aligned 路径同错同值**（正文与 dump 逐字节一致）。归因重估：H3（aligned 编译产物缺陷）排除（标量路径也错）；错误在两路径共享的层——行映射/reduce/B 语义被 Q8 证人排除，剩余 **BF16 特有的 A 寻址契约层**（QUANT_K=1 下 ib→data_a packed32 寻址换算——三个 dequantize 函数共享）为最强嫌疑（hypothesis）；静态审计「位展开正确」与运行时错的矛盾消解（审计覆盖位运算、寻址契约层未被覆盖）。新增第二十八节并更新问题清单。off 臂连续二十一轮（r3/r5–r26）被前置阻断；h1 状态与理论 L=UNRESOLVED 不变。

**核对方式说明：** 本文档编写时直接读出并核对了以下原始文件：`identity/model-and-binary-identity.json`、`identity/source-identity-latebind.json`、`identity/binary-identity-latebind.json`、`identity/clean-head-bin-identity.json`、`identity/base9be-bin-identity.json`、`identity/quarantine-{bf16pipe,gateRevert,fusedRevert,phaseRevert}/` 全部 identity/diff 登记文件、`contract/paired-blocks-preregistration.md`、`review/paired-statistics.json`（含 primary/secondary 统计与 pair-01/02/08 明细）、`review/guardset-v1-h1-responses.json`（01/02/03/08/09 条目、h0_parity_check 与 notes 全文）、`review/v9-cross-request-longrun.json`（全文）、`review/guard-set-freeze-confirmation.json`、`review/guard-set-v2-freeze-confirmation.json`、`arms/{pilot-v2-01,aa-01,aa-02,b1-q2-h1}/result.json`（池化统计与请求明细）、`diagnostics/commit-range.txt`。`diagnostics/commit-2379e1253.patch` 确认存在（内容未读）。其余交接清单事实（CPU 测试计数、shader audit 细节、dispatch 差异数值、FP-TMISS 计数）在轮次目录内无独立文件，按交接清单记录并标注来源。

---

## 一、问题与契约版本

- **契约：** q2-contract-20261012 **v2**。`contract/question.txt` 为中文问句「如何减肥」，`contract/system.txt` 为新版本候选系统提示（约 200 模型 token、自然结束导向）。预注册 **160–240 completion tokens**、`temperature=0`、`seed=42`、`enable_thinking=false`、`cache_prompt=false`。
- **配对块预注册：** `contract/paired-blocks-preregistration.md`，其 SHA256 由 `review/paired-statistics.json` 登记为 `cd906aeb014ebd2dd8df43e2bf9b1b179e1ca6d6807c04ac6ff2feef3ee295fd`（与交接清单一致；该哈希本轮未独立重算，以 paired-statistics.json 登记值为准）。
- **预注册判定规则（跑分后不得更改，已从 preregistration 原文读出）：** 8 配对块 × 2 臂（h0/h1）× 3 measured 请求，奇块 h0→h1、偶块 h1→h0 轮序交替；每臂取池化 tok/s 与 generation_ms 均值；每块计算 h1/h0 池化比值；8 比值取**几何均值**与 **95% t-CI**（t 分布 df=7）；CI 完全高于 1 视为 h1 显著更快，CI 跨 1 视为未达显著（不宣称等价）。
- **预注册质量门（任何一臂失败则该块作废并保留原始记录，不重跑不删除）：** `length_gate.passed == true`；measured 相位 3 请求全部 n=160 且 `finish_reason=stop`；三次 measured 正文 SHA256 全部为 `c2d7bdd3f18f0d9a496087157c8773f26296076c0f979c60bce3ef509effe62c`（与 q2-baseline h0 资格轮逐字一致）；measured 相位 fallback 为 0 行；runner 退出码 0、server 正常退出（returncode 0、非强杀）。

## 二、本轮原始/衍生 target 与全部身份

**排名臂（全部使用同一冻结二进制与原始 target，无衍生数值路径）：**

- **模型：** 原始 Q2_0 两分片 + MTP draft，`identity/model-and-binary-identity.json`（frozen 2026-10-12T11:15:32Z，`computed_before_gpu_benchmark: true`）：
  - 分片 1 `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`：SHA256 `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`，match=true；
  - 分片 2 `…00002-of-00002.gguf`：`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`，match=true；
  - draft `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`：`8db8b4207bbe40286db910fae89928a8cc59b1f7c197aad0a1ebf1b12d5082ad`，match=true。
  三者均与 AGENTS.md §1.1 记录一致。
- **二进制：** `artifacts/perf-gap/q2-baseline-20261010T0401/measured-bin/llama-server`，SHA256 `e9aca36509cbd2d6a4d448dc08d49f21907b9d5f37084d4c94929328ae019c6c`；`identity/model-and-binary-identity.json` 对该目录重算 `all_files_match: true`、`all_symlinks_match: true`。**【出入1】** 交接清单称「15 DSO 哈希全 MATCH」；实际该文件登记 **10 个文件哈希**（9 个库 + llama-server，含 `libtp5-hc-quant-helper.so`）另加 symlink 匹配。以实际登记为准。
- **rank→BDF：** `identity/` 下**无**独立 `rank-identity.json`（**【出入2】** 交接清单暗示身份文件在 identity/ 下）。实际复验以各臂 `result.json` 的 `gpu_before` 呈现：五卡 BDF `0000:03/06/09/0c/0f:00.0`，与 AGENTS.md §1.1 的 Vulkan rank→BDF 映射一致；跑分前五卡 gpu_busy 0%、8.0 GT/s PCIe ×16。
- **源码 dirty 身份：** 排名臂不涉及 dirty 源码（用 2026-10-10 冻结二进制）。工单 A 编译时的 dirty 状态登记于 `identity/source-identity-latebind.json`（git HEAD `3e28b121a38bc425dada175e54a1cef6d5ad1f8d`，M 文件含 `ggml-vulkan-collective.cpp`、`ggml-vulkan.cpp`、`multi_add.comp`、`vulkan-shaders-gen.cpp`、`scripts/tp5-gap-bench.py`、`scripts/tp5-necessity-certificate.py`、两个 tests 文件）。

**归因/隔离用衍生构建（非排名臂，全部登记于 identity/）：** `measured-bin-latebind`（llama-server `a14d5c69…`，工单 A 编译快照）、clean-HEAD 3e28b121a 独立构建（`d047c4a6…`）、base9be 9be06d5ab 独立构建（`d1600f14…`）、四个 quarantine 实验构建（各自 experiment-bin-identity.json 登记 DSO 哈希与 diff）。

## 三、目标瓶颈（同请求、同进程/块）

- h0 与 h1 同契约、同冻结二进制、同原始 Q2_0 权重、同五卡 TP5 布局、同 `-t 22` / ctx1024 / ubatch32 / warmup 2 / repeats 3 / max-tokens 400 / profile none；**唯一差异为 `--horizon 0`（无 MTP）vs `--horizon 1`（MTP draft 加载 + verify + rollback/catch-up）**。
- S 口径（decode 核心）下 h0 池化 48.999 tok/s → h1 51.670 tok/s；generation_ms 均值 3265.359 → 3096.650 ms/160 token。瓶颈对象是 MTP 对 decode 周期成本的净效应，不是任何算子/通信子项（本轮未做 §7.3 rank/stage 账）。

## 四、最多可节省的绝对时间及计算依据

- **已实现节省（h1 相对 h0）：** 每 160-token 请求 generation_ms 均值差 `3265.358708333333 − 3096.64975 = 168.709 ms`，折合约 **1.054 ms/token**；p95 口径 `3291.50865 − 3143.512 = 148.0 ms`。依据 `review/paired-statistics.json` `secondary`（8 块 × 3 请求池化）。
- **机制上限解释：** MTP h1 每请求 accepted_draft=67（draft 92 → 接受率 67/92 = 0.72826）。理论上限受「被接受 token 免去独立 target 前向」与 draft/verify/rollback 开销联合约束；本轮实测净收益 +5.449%（几何均值比），**不构成对 h2/h3 的外推依据**。
- **工单 A（BF16 LateBind）节省：** 0——该路线被乱码事件阻断，未取得任何合格成绩（NOT_RUN）。工单 B/C/D 未立项（NOT_RUN）。
- **F 口径：** 未做正式结论（prompt 约 3.1 s 会稀释 S 口径收益，估算约 +2.9%；NOT_RUN 正式统计）。

## 五、具体改变的数值/布局/通信/调度机制

- **排名臂不改变数值路径：** h0/h1 同一二进制、同一权重、同一采样契约；唯一机制差异是 MTP horizon（draft Q4_K_M 加载、verify 行数、rollback/catch-up、每请求 accepted_draft=67）。h1 正文与 h0 逐字一致（哈希同 `c2d7bdd3…`），MTP 验证未改变输出语义。
- **工单 A 的机制改动**（capture gate BF16 接受条款放宽、fused 段 hc_bf16_weights pipeline、ubatch_execution_phase 判据）**只存在于乱码归因的隔离实验**：第一轮在 `build-perf-gap`（dirty 构建树）上做 bf16pipe 隔离，后续三轮在 `/var/tmp/tp5-clean-head-src`（git archive 干净导出树）上做单因素还原。**【出入3】** 交接清单称「四轮单因素还原实验均在 /var/tmp/tp5-clean-head-src 隔离树」；实际 `identity/quarantine-bf16pipe/experiment-bin-identity.json` 登记 `bin_dir = build-perf-gap/bin`（dirty 树），仅 gateRevert/fusedRevert/phaseRevert 三轮在 clean-head 树（各自登记 `repo worktree untouched`）。以实际登记为准。
- 仓库工作树在全部归因实验期间零接触（各 quarantine identity 文件均含备份 + diff + DSO 哈希核对记录；`identity/quarantine-bf16pipe/restore-identity.json` 记录实验后源码按备份哈希还原并重建）。

## 六、实现文件与单因素对照

- **排名工具：** `scripts/tp5-gap-bench.py`——`--horizon` 单因素；机械门 `--length-min/--length-max`、`evaluate_length_gate`、`extract_cycle_ledger`、`audit_minimum_fields`（交接清单记录 `tests/test-tp5-gap-tools.py` 32 项全绿，本轮未重跑）。
- **工单 A 源码：** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`、`ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp`、`ggml/src/ggml-vulkan/vulkan-shaders/multi_add.comp`、`vulkan-shaders-gen.cpp`（M 清单见 `identity/source-identity-latebind.json`）；快照 `identity/binary-identity-latebind.json`（first-compile success, zero fixes）。
- **归因对照臂：** `arms/latebind-parity-off`（dirty 构建首现乱码）、`arms/latebind-parity-off-cleanHead`（干净 3e28b121a 复现乱码）、`arms/latebind-parity-off-base9be`（9be06d5ab 连模型都加载不了）、`arms/latebind-parity-off-{noBf16Pipe,gateRevert,fusedRevert,phaseRevert}`（四轮单因素还原）。全部保留，未删除未重跑。

## 七、测试已执行/未执行及状态资格

**已执行、本轮直接核验（读出原始数据）：**

- **A/A 资格三进程：** `pilot-v2-01` 池化 49.108、`aa-01` 49.172、`aa-02` 49.523 tok/s；跨进程漂移 `(49.52299553870031−49.17202001424144)/49.17202001424144 = 0.7138%` < 1% 先查门。三进程 measured 共 9×160 token 全部自然 stop（另各含 warmup 1×160 亦 stop），全部请求正文哈希 `c2d7bdd3…` 一致，零 fallback，runner 退出 0。
- **b1-q2-h1 单臂：** 池化 52.397 tok/s，MTP draft 真实加载（每请求 accepted_draft=67），3×160 stop，正文与 h0 逐字一致。
- **8 配对块正式统计：** 见第八节。
- **V8 保护集施测：** `review/guardset-v1-h1-responses.json`——h1 施测 10 题；本轮直接读出 01/02/03/08/09 条目 `mechanical_gate: "PASS"`（01：235 token stop 瑞利散射答案；02：452 token stop（maxtok500 重跑）；03：21 token stop「长平之战…赵括」命中确定性目标；08：双请求 request_a「已记下」/request_b「未接收到任何口令」，leak_check `琥珀-8839/8839` 均 false，server returncode 0 非强杀；09：首 token 后中断，follow-up 21 token stop，server 正常退出）。05 的 105 km 命中由文件 notes 记录（`final answer 105 km matching deterministic_target`，ctx2048/max_tokens1000 重跑 908 token stop）。06/07 的 V8 哈希存在且被 V9 长测用作逐字节可比基线。04/10 的逐题字段值本轮未逐条重读，以原始文件为权威记录。**h0 抽查 3 题（01/05/09）**：同文件 `h0_parity_check` 段，三题 `h1_h0_identical: true`（01/05 附 h0 哈希 `87bd0599…`/`fa2d76f9…`，09 为 follow-up 内容比对）；原始请求体在 `arms/guardset-v1-h0/{guard-01,guard-05-ctx2048,guard-09}/`。Manager 语义门（切题/流畅/事实合理）判定通过（交接清单记录）。
- **V9 跨请求长测：** `review/v9-cross-request-longrun.json`——**【出入4，本轮最重要出入，已由 Manager 复核解决】交接清单称「V9 未做」；实际该文件存在且完整**。单 h1 进程 20 个顺序请求（main×6 → guard-01×3 → guard-03×3 → guard-06×3 → guard-07×2 → main×3 尾段重复探测晚序列状态漂移），每 guard 题与 V8 独立进程施测使用相同 prompt/system/max_tokens 以保证逐字节可比。结果：**20/20 `matches_baseline: true`、全部自然 stop、server returncode 0 非强杀，文件内 verdict `V9_PASS: true`**；main 尾段 3 次与前 6 次哈希一致（无晚序列漂移）。以实际数据为准。**复核结论（Manager，2026-10-12，裁决补录）：V9 记录有效**——该长测在本 session 内由 Manager 授权 DevOps 执行（与本文档写作并行）：GPU 锁与 runner 同一把（flock 串行）、server argv/env 与 runner 臂同构、单进程 20 请求逐项哈希对照独立运行基准、verdict `V9_PASS=true`。晋级裁决见第十一节。
- **保护集冻结：** v1（`artifacts/perf-guardset-v1`，guardset-v1-20261010）13 文件哈希经 DevOps 干净主机重算全一致（`review/guard-set-freeze-confirmation.json`，2026-10-12T15:31:18Z）；v2（本轮 `guard-set/` 23 文件）DevOps 冻结完成，manifest 更新 questions[].sha256、`FROZEN_BY_DEVOPS`、guard-03 char_count 修正留痕（declared 139 → actual 140，UTF-16 code units 口径）（`review/guard-set-v2-freeze-confirmation.json`，2026-10-12T15:58:26Z）。

**已执行、来源为交接清单（轮次目录内无独立运行记录，本轮未重跑）：**

- 工单 A 编译后 CPU 测试 72 Python + 7 ctest 全绿；
- `tests/test-tp5-gap-tools.py` 32 项全绿；`tests/test-tp5-necessity-certificate.py` 40 项全绿；
- 乱码归因的 dispatch 差异观察（prefill 4080→4063、decode 3391→2248/2175，≈96 层×12）与 FP-TMISS 20/20 非判别性观察（原始日志在 latebind-parity 各臂）。

**未执行：** F 口径正式统计（NOT_RUN）；h2/h3 配对设计（NOT_RUN）；工单 A 数值对拍——被乱码阻断（NOT_RUN）；乱码 hunk 级定位（UNRESOLVED）；§7.2 完整周期账 D_j/V_j/S_j/H_j 分解（NOT_COLLECTED——本轮 MTP 证据为 accepted_draft=67/请求与池化 generation_ms，未读出逐周期分解）；§7.3 rank/stage 账（NOT_COLLECTED）。

## 八、无插桩 pilot 与独立确认（原始统计、CI）

- **口径分层：** pilot 级 b1 单臂 52.397 相对三 h0 资格进程池化均值 49.268 为 **+6.34%**（单进程估计，仅诊断）；正式结论以 8 配对块为准。
- **正式统计（`review/paired-statistics.json`，预注册规则执行）：** 16 臂 48 请求全部 160 token 自然 stop、正文哈希 `c2d7bdd3…` 一致、零 fallback、零作废块（`invalidated_blocks: []`，`quality_gates_all_pass: true`）。
  - **几何均值 h1/h0 = 1.0544932935131766**；95% t-CI **[1.048390017707107, 1.060632099966177]**（ln-ratio mean，t 分布 df=7，t(0.975,7)=2.3646）；`significant_faster: true`，CI 完全高于 1。
  - h0 池化 48.99946183849799 ± 0.11850337560993876 tok/s；h1 池化 51.6702016922948 ± 0.29392767236979167 tok/s。
  - generation_ms：h0 均值 3265.358708333333（p95 3291.50865）；h1 均值 3096.64975（p95 3143.512）。
  - 8 块逐块比值 1.0460 / 1.0517 / 1.0500 / 1.0662 / 1.0546 / 1.0586 / 1.0467 / 1.0623，全部 >1。
  - h1 每请求 accepted_draft = 67（92 drafted，接受率 0.72826）。
- **S 口径为 decode 核心（generation_ms / 池化 tok/s）；F 口径（完整请求墙钟）未做正式结论**，prompt 约 3.1 s 会稀释收益（估算约 +2.9%），不跨口径计算 Gap。

## 九、失败、回退、EOS、质量、长测原样记录

**乱码事件（工单 A，本轮最大失败，全部原始记录保留）：**

- `arms/latebind-parity-off`（dirty 构建快照 `measured-bin-latebind`）自首请求起数值乱码 + HTTP 500，错误体 `{"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}}`，400 token 无 EOS。
- **归因链（commit 级锁定，hunk 级 OPEN）：**
  1. 干净 HEAD `3e28b121a`（git archive 独立构建，零 dirty，DSO 无混合，`identity/clean-head-bin-identity.json`）复现同样乱码 → **dirty 全部洗清**。注：该文件记录「task brief said 79e69b7e0 but that object is a BLOB, not a commit」——乱码二进制实际构建于 HEAD 3e28b121a。
  2. `9be06d5ab` 干净版（`identity/base9be-bin-identity.json`）**连模型都加载不了**（GGUF 加载器在 2379e1253 才进 tracked；构建时 LLAMA_BUILD_TESTS=OFF 绕过该 commit 缺失的 tracked 测试源）→ **不存在更早的干净正常态**。
  3. 实验 1 gateRevert（capture gate BF16 放宽条款还原，clean 树）：乱码仍在且 off 臂不活跃 → **排除**。
  4. 实验 1' fusedRevert（fused 段 hc_bf16_weights 选择还原，clean 树，gate 还原保留）：乱码仍在，dispatch [4063,2248,2175] 不变 → **排除**。
  5. 实验 2a phaseRevert（ubatch_execution_phase 三重判据还原至 pre-2379e1253 形态，clean 树）：三重判据全部不变 → **排除**。
  6. 更早的 bf16pipe 隔离实验（dirty 构建树）：BF16 fused pipeline 注释掉后 HTTP 500 仍在 → **BF16 fused pipeline 归因被拒**，嫌疑当时转向 undeclared worktree changes（后续被干净 HEAD 复现取代）。
  7. 元发现：FP-TMISS 在正常/乱码轮全部出现（交接清单记 20 次）——**非判别性标志**。
  8. **剩余嫌疑收敛到 commit `2379e1253` 的 `ggml-vulkan-collective.cpp` +1543 行（mailbox 寻址 / predefined plan 重排）及配套 shader——OPEN。** 根因为 commit 级锁定，非 hunk 级。`diagnostics/commit-range.txt` 登记 3e28b121a 范围内的 6 个提交（3e28b121a / 0707ea187 / 7686fbf8f / 2379e1253 / 25ea60642 / 646a41d36）；`diagnostics/commit-2379e1253.patch` 存在（+2372/−423 vk 行，交接清单记录，本轮未读内容）。
- **BF16 up_fold shader 已知缺陷（待修，工单 A 恢复验证的前置）：** `mul_mat_vec.comp` 宏分支的 BF16 up_fold 变体存在 `iter_aligned_nonquant` 缺 HC_UP_FOLD 折叠的索引缺陷 + `dispatch_rows` 缺 bf16 分支。**【出入5】** 该发现交接清单归于「bf16-shader-audit」；轮次目录内未见独立 audit 文件（diagnostics/ 仅 patch 与 commit-range.txt），此条按交接清单记录，来源标注为清单而非本轮目录文件。

**V8 施测适配（如实记录，非候选缺陷）：**

- guard-02：原约束 max_tokens=300 截断（finish=length），500 重跑 → 自然 stop 452 token；原始失败目录保留。
- guard-05：300/500/800 均截断（ctx=1024 装不下表格式完整答案），ctx=2048/max_tokens=1000 重跑 → 自然 stop 908 token，最终答案 105 km 命中确定性目标。
- guard-07：多轮对话压平为单轮 prompt（adaptation recorded）；「灵隐寺多轮引用命中」为交接清单记录，正文未逐字核读。
- guard-08/09：runner 单 prompt 模式无法表达协议，用自定义脚本（相同 server argv/env、GPU lock via flock）执行。
- runner 字段 `runner_status: MEASURED_NOT_QUALIFIED` / `qualification: NOT_REVIEWED` 为待人工评审标记，与机械门 PASS 并存，符合 AGENTS.md「result.json 写 MEASURED_NOT_QUALIFIED 不代表已通过人工质量验证」的约定；人工质量结论为 Manager 语义门通过（交接清单记录）。

**V9 长测：** 见第七节——20/20 匹配基线、自然 stop、进程干净退出，文件内 verdict `V9_PASS: true`。原始逐请求记录（elapsed_ms、哈希、基线来源）全部在该文件内。

## 十、理论 Lj 的已证前提、未解分支；整体 L/U/Gap 是否有资格

- **算法 P 已落地：** `scripts/tp5-necessity-certificate.py`（120 分支枚举、四槽前提检查、LP 对偶验证、UNRESOLVED 传播）；`tests/test-tp5-necessity-certificate.py` 40 项全绿（交接清单记录，本轮未重跑）。
- **冒烟运行结果：** `global_L = UNRESOLVED` → **`Gap = UNPROVEN`**（无有效 L 时不计算伪比例）。
- **已证 Lj：无。** 全部 120 分支的必要性前提未闭合，无任何分支取得可核验下界证书。
- **U 候选：** h1 池化 **51.670 tok/s（S 口径）**，由 v2 契约资格 + 8 独立配对块统计 + V8 保护集（+ V9 长测记录）支撑。
- **结论：** 整体 `L = UNRESOLVED`，`Gap = UNPROVEN`；本轮不具备计算 U/L 的资格。U/L 里程碑（≤2 / ≤1.5 / ≤1.2）均未登记。

## 十一、晋级 / 淘汰该具体设计 / 暂停；下一项可证伪问题

**本轮整体状态：`QUALIFIED_BEST_KNOWN`**（AGENTS.md §18 三态）。TP5 v2 契约下最强臂 h1 具备真实的质量、可靠性与独立速度确认，但最优性证书不完整、乱码根因未闭合。

**裁决（Manager，2026-10-12，复核后补录）：`PROMOTE(h1)`。** Manager 正式裁定 h1（MTP horizon=1，原 Q2_0、v2 契约、冻结 measured-bin）晋级为 **v2 契约下当前最强已知 TP5 合格臂，状态 `BEST_KNOWN_NOT_PROVEN_OPTIMAL`**（AGENTS.md §18：有真实质量、可靠性与独立速度确认，但最优性证书不完整；不得称「已证极限」）。晋级依据逐条：

- **(a) 正确性：** 48（8 配对块）+ 20（V9 长测）+ 10（V8 保护集）请求正文与 h0 逐字节一致（greedy 分布保持）。
- **(b) 质量不退化：** V8 保护集 10/10（机械门 PASS + Manager 语义门通过）+ 主契约 160 token 自然 stop 门。
- **(c) 无未解释故障：** 全程零 fallback、进程干净退出（returncode 0、非强杀）。
- **(d) 独立确认支持事先声明的收益：** 预注册 8 配对块 +5.45%（几何均值 1.05449，95% t-CI [1.048390, 1.060632] 完全高于 1）。

前一最强臂 h0（同数值路径无 MTP）及其完整证据保留（`artifacts/perf-gap/q2-baseline-20261010T0401/` 与本轮全部臂目录，不覆盖不删除）。§14 算法 V 的 **V5–V9 全部完成**（V9 复核结论见第七节【出入4】）。原资格分解（V5 冻结/预注册、V6 8 配对块轮序、V7 按块聚合统计、V8 独立保护集）如前文所录，全部成立：

- V5（冻结二进制/DSO/预注册）✅；V6（8 独立配对块轮序交替）✅；V7（按块聚合、几何均值 + t-CI）✅；V8（未参与选型的独立保护集 + 跨请求隔离与中断恢复协议）✅。
- **V9（已复核，2026-10-12 Manager 裁决补录）：** 交接清单称未执行；实际 `review/v9-cross-request-longrun.json` 有完整执行记录且 verdict 为 `V9_PASS: true`。Manager 已复核其执行身份与时间（本 session 内 Manager 授权 DevOps 执行，与本文档写作并行；GPU 锁与 runner 同一把 flock 串行、server argv/env 与 runner 臂同构、单进程 20 请求逐项哈希对照独立运行基准），复核结论：**V9 记录有效**。据此作出本节 `PROMOTE(h1)` 裁决。
- 乱码最终修复 / 工单 A 对拍 / 工单 B / rank-stage 采集器真机验证：**OPEN**（第十三至十八轮补录：**崩溃线终局关闭**——registry 物理移除 + 四臂因果链（r9 / r10–r18 / r20 / r21）+ 回归守卫 4 项，见 23.1–23.4；**乱码根因数值级实锤**——r24 CPU 参照对拍直接证据：fused HC segment 两个 BF16 kernel（qwen4_hc_down_silu_bf16 + qwen4_hc_project_fold_bf16）数值计算错误（down lo 符号反/前 32 worst 3.61；up mixed 正确输入下仍错 worst 4.80——双段都错），W 与 GGUF 逐字节一致（V 排除）、H2 排除（对照臂逻辑）、H1 静态排除——见 26.1–26.5；**静态域全排除 + H3 曾收敛为唯一嫌疑**（公式一致清偿 + Q8_0 生产证人 + BF16 特有层逐位正确，见 27.1–27.3）；**r26 插打印推翻 r25「开关无效」推断——开关链双侧全通、两解码路径同错同值，H3 排除，嫌疑收敛到 BF16 特有的 A 寻址契约层（三个 dequantize 函数共享，hypothesis）**，见 28.1–28.4；off 臂连续二十一轮（r3/r5–r26）被前置阻断——崩溃维度已自动通过、阻断只剩乱码数值缺陷本身的修复；rankstage 采集器 CPU 验证已过、真机待排；h2/h3 与 F 口径已闭环——见 12.1/12.2）。
- 理论 L：**UNRESOLVED**（Gap = UNPROVEN）——晋级不改变最优性证书缺失的事实，故状态为 `BEST_KNOWN_NOT_PROVEN_OPTIMAL` 而非 `CERTIFIED_WITH_SCOPE`。

**可证伪问题清单（最终状态；第 1 项第一轮完成，第 2–4 项第二轮更新，第 5/6 项第二轮完成，第 7 项第三轮定位闭环，第 8 项第三轮新增，第 2/9 项逐轮更新至第十三轮，第 9 项为第十四轮主线）：**

1. ~~**Manager 复核 `review/v9-cross-request-longrun.json` 并完成 h1 晋级裁决**~~ **已完成（2026-10-12，Manager 裁决补录）**：复核确认 V9 记录有效（第七节【出入4】），`PROMOTE(h1)` 裁决记录于第十一节。剩余 5 项顺延如下。
2. **乱码根因——第十八轮更新：开关链双侧全通（r26 直接证据）+ H3 排除 + A 寻址契约层为最强嫌疑——fused HC segment 两个 BF16 kernel（qwen4_hc_down_silu_bf16 + qwen4_hc_project_fold_bf16）数值计算错误（r24 CPU 对拍直接证据，down lo 与 up mixed 双段都错）；r25「开关无效」推断被 r26 插打印推翻（0x200 注入全部 dispatch + is_aligned_nonquant 运行时切换——**iter() 标量路径与 aligned 路径同错同值**）；H3（aligned 编译产物缺陷）排除（标量路径也错）；错误在两路径共享层——行映射/reduce/B 语义被 Q8 证人排除，剩余 **BF16 特有的 A 寻址契约层**（QUANT_K=1 下 ib→data_a packed32 寻址换算——dequantize1/dequantize4/dequantize4_2aligned 三函数共享）为最强嫌疑（**hypothesis 状态——待第十九轮深查定位实锤**；历史演进：第十六轮矛头收回 kernel 内部 → 第十七轮静态层清空、H3 唯一嫌疑 → 第十八轮开关全通推翻 r25、H3 出局、嫌疑收敛 A 寻址契约层）**（详见 22.1–22.4、23.1–23.5、24.1–24.5、25.1–25.6、26.1–26.6、27.1–27.5 与 28.1–28.5）：(崩溃线) registry 四触点三块 51 行干净删除（grep 零命中）+ RegistryRegressionTests 4 项回归守卫（旧机制复活时测试变红）+ r21 验证：默认配置（无任何 env 开关）下断言 0 次、进程正常退出、请求完成——与 r20 门控臂**逐字节等价**（乱码正文 hash 20c0ddaa0a45a3f5 一致、fallback 10 相同）。**四臂因果链终局：r9（无 registry）不崩 → r10–r18（registry 活跃）崩 → r20（门控旁路）不崩 → r21（物理移除、默认路径）不崩。**(乱码线) r22 对拍：fused 臂输出乱码（「健康做，，，，，，，，且，，，在rata，让我，rata，FP，，rata，，rata，且，，」）vs 参照臂（入口还原）输出正常中文（c2d7bdd3 已知开头形态），**前 2 个 token「健康」逐字一致、第 3 个 token 分歧（「做」vs「减」）**——分歧起点推断在 prefill 尾部至第 2 个 decode 步之间的 HC 段计算，fused 段 BF16 变体（qwen4_hc_down_silu_bf16 / mul_mat_vec 的 BF16 路径）的 lo/mixed 具体数值错误（量级对值错）为最强解释（推断而非逐值证明——参照臂 dumps=0，逐张量对拍结构性不可行）；dump 无辜性在对拍实践中正式确认（dump 活跃下进程健康、读数与 r16 逐字节一致）。第十五轮静态深查：S1（host↔shader 单位契约 32 倍/2 倍错位）三层证据排除；S2（iter_aligned_nonquant packed32/iqs 索引错位）五层证据排除（含两解码路径逐位等价证明——与第二轮 V-A 运行时观察互洽）；静态排除边界如实保留（不能裁决驱动/编译器实际行为与 host 填入的具体数值——即 H3 与 H1）；r15a 方案盲点修正（对拍前先验证 normalized 自身正确）。**第十六轮 r24 CPU 参照对拍数值级实锤**：lo 对拍（lo_ref=SiLU(0.25·W_down@norm) vs dump lo——lo[0] dump=+0.132684 vs ref=-0.014946，符号反；前 32 输出 worst rel err 3.61；6 种公式变体（0.25 位置/无 SiLU/4 倍缩放）全部不匹配（maxerr 0.44–26）——排除公式歧义；dump 值与 r16/r22 跨轮逐字节一致——确定性错误）；mixed 双重判别（用 dump lo 输入 worst rel 2.59；**用 lo_ref（CPU 正确值）重算仍 worst rel 4.80——up/gate 段自身数值也错**，不只是 down 错误的传播）。综合裁决：W 正确（GGUF 逐字节）+ normalized 假设成立（H2 靠入口还原对照臂逻辑排除）+ lo/mixed 双段数值错——**两个 BF16 kernel 数值计算错误实锤，从「最强解释间接推断」升级为「CPU 对拍直接证据」**。判读边界（如实标注）：(a) 参照公式取自方案文档 + AGENTS.md 工单 B 数学交叉印证、qwen4exp.cpp:382–400 的精确实现未由本轮直接读取核对；(b) normalized 未做独立数值验证（H2 靠对照臂逻辑排除而非逐值验证）。**第十七轮结果**：静态深查完成——判读边界 (a) 清偿（qwen4exp.cpp build_hc_mix/build_hc_combine 363–431 逐行核对：lo 的 scale 在 SiLU 前、gate 无缩放、mixed 0.25 按流求和、normalized RMSNorm gamma 折叠——与 r24 参照逐项一致，r24 判读成立）；共享层静态全排除（Q8_0 与 BF16 变体编译字典 B_TYPE/B_TYPEV4/D_TYPE/USE_SUBGROUP_ADD_NO_SHMEM 完全相同、唯一差异 DATA_A——B 读取骨架/get_offsets/reduce/D 写回有 Q8_0 生产正确性作运行时证人）+ BF16 特有层逐位正确（B vec4 col 恒 4 倍数、A packed32 对齐 1049041280 mod 4/8/16 全零 + ib 恒 4 倍数、位展开逐位正确、subgroup 单 wave、迭代尾段闭合）；Manager 归因收敛：V/C 排除（r24 W 逐字节 + 对拍输入控制）、**H3 收敛为唯一剩余嫌疑**。r25 FORCE_ITER 三轮 A/B 判别**实验无效**（B/B2 乱码与 dump 与 A/r24 逐字节一致——两独立数值实现不可能逐字节同值 ⇒ 开关运行时效果未达 kernel；与 r13 capture gate「静态正确≠运行时活跃」同型）——H3 裁决悬置；开关链断裂点三候选（pipeline 指针匹配/shader 0x200 分支/fusion_flags host→shader 传递）待第十八轮插打印定位。**第十八轮结果**：r26 插打印两跑直接证据——跑 1（32565 行）：dispatch_is_down_bf16=1 + dispatch_is_fold_bf16=1 交替（**fused 段两算子都真实走 BF16 变体**）、grid/k/m/n_rows 合理（fold grid-y=4 证明 upfold 修复形态在位）；跑 2（32070 行）：force_iter_env=1 + force_iter_match=1 + **fusion_flags=0x00000200 注入全部 dispatch**——**开关链 host+shader 双侧全通**（shader 侧 is_aligned_nonquant 是运行时 const bool 读 p.fusion_flags & 0x200——注入后真实切换到 iter() 标量路径）。**r25「开关无效」推断被推翻**（r25 判读边界当时已标注「机理推断非直接观察」——直接观察一到即被替换，推断未固化为事实）：路径确实切换了，而 iter() 与 aligned 两独立数值实现产出**逐字节相同的错值**（正文与 dump 不变）。归因重估：H3 排除（标量路径也错）；错误在两路径共享层——行映射/reduce/B 语义被 Q8 证人排除，剩余 **BF16 特有的 A 寻址契约层**（QUANT_K=1 下 ib=(weight_row·ncols+col) 元素索引 → data_a packed32 寻址换算（(a_offset+ib)/2 与元素内偏移）——dequantize1/dequantize4/dequantize4_2aligned 三函数共享）为最强嫌疑；静态审计「位展开正确」与运行时错的矛盾消解（审计覆盖位运算、**寻址契约层未被覆盖**）；与 r24 CPU 对拍自洽（W 字节正确 + A 寻址错 → kernel 读错 A 位置 → 值错但量级合理——读到别处的真实权重）；乱码形态跨 5 轮（r22–r26）逐字节稳定。**乱码根因修复未闭环（A 寻址契约层为最强嫌疑、待深查定位实锤——保持 hypothesis 状态，见第 9 项）。**
3. **BF16 up_fold shader 缺陷——已修且验证真实，但非唯一根因**（第二轮补录，详见 12.3）：`mul_mat_vec.comp` iter_aligned_nonquant 补 HC_UP_FOLD 折叠 + `ggml-vulkan.cpp` dispatch_rows 加 bf16 分支（4*n_rows）；`docs/TP5-BF16-UPFOLD-FIX.md` + `tests/test-tp5-upfold-indexing.py`（交接清单记 16 项全绿）；`identity/quarantine-upfoldFix/` 验证：乱码仍在但是第三种形态——修复真实生效。
4. **工单 A 对拍重执行——挂起，依赖乱码最终修复**（第二轮补录）：最终修复落地并复现 baseline 前对拍无意义；V-B 之后恢复。
5. **h2/h3 深度扩展——已完成，结论：最优深度=1，h2 淘汰**（第二轮补录，详见 12.2）：几何均值 h2/h1 = 0.8765（95% CI 上界 0.8867 远低于 1），显著慢 12.4%；E6 判读 PAUSE_DEEPENING、h3 不测；h1 最强臂地位经受深度方向钢铁人检验。
6. **F 口径正式结论——已完成**（第二轮补录，详见 12.1）：几何均值 h1/h0 = 0.9820（墙钟缩短 1.8%），95% CI [0.9769, 0.9872] 完全低于 1；与 S 口径 +5.45% 互洽（decode 提速被固定 prompt 段稀释）。h1 双口径证据链完整。
7. **单 site 定位——第三轮已由 sitedump 直接证据完成**（详见 13.2）：site-dumper 插桩（GGML_VK_HC_DUMP_SITES opt-in + rank 区分 + A binding dump）比 V-B 设计更直接地钉死了根因；V-B 设计（teacher-forced 固定 token 回放逐层对拍）保留为备用方案与修复后的验证工具。
8. **profile 诊断轮（GAP 4）——周期账第三轮闭环；采集器第六轮交付（CPU 验证过、真机验证待排）**（详见 13.1 与 16.1）：周期账已完成（184 周期，V 91.6% / D 7.0% / S 1.4%，每公开 token 19.912 ms，接受率 72.8% 与正式轮一致）；rankstage-collector 已交付（RANK_STAGE_LINE_SPECS 十种 tag + extract_rank_stage_ledger + RankStageLedgerTests 14 用例，三套 CPU 测试全绿）——**GGML_TP5_PROFILE=1 真机诊断轮未跑**；五字段中 active_rows/M/N_K/weight_type 源码无打印点（如实标注）、bank_epoch 保持 NOT_COLLECTED；中段 2 个异常周期（疑似图重编译/缓存失效）待 --profile full 定位——该嫌疑与 GAP 1 覆盖者候选 (2)（cache invalidate/buffer 池）存在潜在交集，待真机轮一并观察。
9. **（第十八轮重写，第十九轮主线）Engineer 深查三个 dequantize 函数的 BF16 A 寻址契约 → 依定位最小修复 → off 臂三重判定 → 工单 A（+4 项历史哨兵失配的文本级同步顺带）**：第十八轮已完成开关链双侧全通的运行时裁决（r26 插打印——28.2/28.3）与归因重估（H3 排除、A 寻址契约层最强嫌疑——28.4；r25 推断被直接证据推翻的修正记录见 28.3）。第十九轮推进：(a) Engineer 深查 base glsl 三个 dequantize 函数（dequantize1/dequantize4/dequantize4_2aligned）的 **BF16 A 寻址契约**（ib/iqs/a_offset → data_a 偏移的换算：QUANT_K=1 语义下 packed32 对齐、a_offset 单位、元素内偏移的契约；**与 Q8 路径的寻址差异对比**——三函数共享的寻址层是唯一能解释「两条解码路径同错同值」的层）→ 定位实锤；(b) 依定位最小修复（guess-based-fix 纪律：named invariant + recurrence regression——修复需解释原失败、解释保留改动的阻止作用、排除替代假设、附复发回归；**r24 CPU 对拍脚本可直接作回归门/contract test**：dump → CPU 复算 → 逐元素对拍）；(c) off 臂三重判定（乱码消失 + 哈希 c2d7bdd3 逐字一致 + dispatch 恢复 [4080,3391,…]——**崩溃维度已自动通过（r21 默认路径零断言）**）→ 工单 A 对拍恢复（NO_HIT 判定 / 单 site 对拍 / aggressive 臂）；(d) 4 项历史哨兵失配的文本级同步（r4 mmap 删除 / loadtrace 插桩 / r16 dump 改造让旧断言过期——非本轮缺陷，交 Engineer 同步——顺带完成）。**off 臂已连续二十一轮（r3/r5–r26）被前置阻断。**

## 十二、第二轮补录（2026-10-12 晚）：F 口径、h2/h3 深度闭环与乱码归因收口

本节补录第二轮 session 的全部新事实。核对方式：`review/f-caliber-paired-statistics.json` 与 `review/h2-paired-statistics.json` 的统计字段、`identity/quarantine-{mailboxRevert,upfoldFix,entryRevert,forceIter}/` 的实验登记文件、`arms/{pilot-v2-h2-01, pairh2-01..08-h1/h2, latebind-parity-off-{mailboxRevert,upfoldFix,entryRevert,forceIter,hcq8}}/` 的存在性、`docs/TP5-BF16-UPFOLD-FIX.md` 与 `tests/test-tp5-upfold-indexing.py` 的存在性，均直接核验；运行观察细节（乱码第三形态文本、HC-Q8 臂 tok/s 与 dispatch、entryRevert 臂池化值）来自 Manager 交接清单，逐条标注来源。

### 12.1 F 口径正式结论（原问题清单第 6 项：已完成）

- **数据源：** `review/f-caliber-paired-statistics.json`。F 口径 = runner 原生 `request_wall_ms` 字段（每臂 measured 3 请求、warmup 剔除；含 prompt 处理 + generation、不含 startup/loading，符合 AGENTS.md §3 F 定义）。**复用 S 口径同批 8 配对块的同一批请求**（per_arm 的 generation_ms 与 paired-statistics.json 逐臂一致），无新 GPU 跑分。
- **结果：** 几何均值 h1/h0 = **0.9820141406040861**（完整请求墙钟缩短 1.80%），95% t-CI **[0.9769008856316421, 0.9871541591682084]**，`ci_entirely_below_1: true`；8 块比值 0.97361–0.99181 全部低于 1。
- **secondary：** h0 墙钟 6410.757 ± 31.719 ms（p95 6460.268）；h1 墙钟 6295.420 ± 21.826 ms（p95 6348.114）；prompt 段约 3.10–3.22 s。
- **与 S 口径互洽：** decode 核心 +5.45%（几何均值 1.05449）被固定 prompt 段（~3.1 s，两臂几乎相同）稀释为墙钟 −1.8%，方向与量级一致。**h1 的双口径（S + F）证据链完整。**

### 12.2 h2/h3 深度闭环（原问题清单第 5 项：已完成，结论最优深度 = 1，h2 淘汰）

- **pilot 资格：** `arms/pilot-v2-h2-01/`——h2 池化 45.439 tok/s；draft acceptance 83/154 = 0.539、mean len 2.08、cycles 154（`review/h2-paired-statistics.json` pilot_reference）；160 token 自然 stop、哈希 `c2d7bdd3…`、零 fallback。
- **预注册：** `preregistration_sha256 = 825962ae8ed65553ceb90ae4f6a70babee84469b6af347af20053d4c79536c02`（`review/h2-paired-statistics.json` 登记）；设计 8 配对块 × 2 臂（h1/h2）× 3 measured，奇块 h1→h2、偶块 h2→h1，baseline = h1（已晋级最强臂）。
- **正式统计：** 16/16 臂全过质量门（48 measured 请求全部 160 token 自然 stop、哈希 `c2d7bdd3…` 与 h0/h1 逐字一致、零 fallback、`invalidated_blocks: []`）；**几何均值 h2/h1 = 0.8765112758560594，95% t-CI [0.866448606501328, 0.8866908099778212]，CI 上界远低于 1——h2 显著慢 12.4%**。h1 池化 52.010 ± 0.733 / h2 池化 45.585 ± 0.364 tok/s；gen_ms h1 3076.837（p95 3151.210）/ h2 3510.118（p95 3575.034）。（交接清单称「96 请求」，统计文件直接记录 48 measured；96 应为含 warmup 的全请求口径，以统计文件为准。）
- **E6 判读：** `e6_verdict: PAUSE_DEEPENING`——暂停加深，h3 不测。机理（交接清单记录；统计文件直接支撑接受率与周期数）：接受率 0.728→0.539、verify 行 2→3、每 token verify 成本 +50% 而提交收益仅 +2.3%，深度 2 净亏损。
- **数值等价性无虞，纯性能淘汰：** h2 全部请求正文与 h0/h1 逐字一致（哈希同）。**h1 的最强臂地位经受深度方向的钢铁人检验。**

### 12.3 乱码归因链收口（原问题清单第 2/3 项的推进：从 commit 级到组合根因）

第一轮将根因锁定在 commit 2379e1253（commit 级）；第二轮经五步实验收口到「fused 段 × BF16 权重的组合」。全部实验在 `/var/tmp/tp5-*` 隔离树执行，仓库工作树零接触，每次备份 + diff + DSO 哈希核对（identity/quarantine-*/ 各自登记）。

- **实验 2b（mailbox 寻址还原，`identity/quarantine-mailboxRevert/`）：排除。** 静态数学推导证明新旧寻址在 TP5_MAILBOX_BANKS=2 下逐 s 等价（`tp5_mailbox_bank(e)=(e-1)&1`，调用方传 `tp5_mailbox_bank(ac+1)`）；实验还原后三项指标不变正是该等价性的运行时印证。第一轮「剩余嫌疑 collective.cpp +1543」中的 mailbox 寻址部分洗清。
- **upfold 修复（Engineer 源码 + 验证，`identity/quarantine-upfoldFix/`）：真实缺陷已修，但非唯一根因。** `mul_mat_vec.comp` 的 iter_aligned_nonquant 补 HC_UP_FOLD 折叠 + `ggml-vulkan.cpp` dispatch_rows 加 bf16 分支（4*n_rows）；`docs/TP5-BF16-UPFOLD-FIX.md` 与回归测试 `tests/test-tp5-upfold-indexing.py`（交接清单记 16 项全绿，本轮未重跑）均已入库。验证构建（clean 3e28b121a + 仅两个 upfold hunk，libggml-vulkan `9452b39a…`，`/var/tmp/tp5-verify-upfold-src`）：乱码仍在但是**第三种形态**（交接清单记录「健康做…且…在rata」）——数值路径改变证明修复真实生效，但乱码另有根因。
- **实验 3c（判别力最强，`arms/latebind-parity-off-hcq8/`）：公共框架无罪。** HC-Q8 衍生模型（`models/perf-gap-q2-hcq8/`，65.8 GB，2026-10-08 转换产物）在 upfoldFix 二进制 off 臂完全正常（自然中文、59.11 tok/s、dispatch [4063,2248,2175]——交接清单记录）——fused 段公共框架（norm/SUM/segment_project）对 Q8_0 无罪；dispatch 差异（4080→4063、3391→2248/2175）是 fused 段激活的**无害结构标志**（Q8_0 进段也出现）。
- **实验 3a（入口还原，`identity/quarantine-entryRevert/`）：单点即恢复。** `ggml-vulkan.cpp` ~24076（patch 4405-4406 行）的 q8_projection BF16 放宽单点还原（14 行 diff，树 = verify-upfold-src 含 upfold 修复，libggml-vulkan `eaa94f05…`）→ 一切恢复 baseline：乱码消失、哈希 `c2d7bdd3…` 逐字一致、dispatch 恢复 [4080,3391,…]、pooled 49.003 / gen 3265 与 h0 逐项吻合（交接清单记录）。
- **3c/3a 夹逼结论：** 乱码根因 = **fused 段 × BF16 权重的组合**（入口放行 + 段内 BF16 特有缺陷）。Q8 twin 正常（3c）+ 入口单点还原即恢复（3a）+ 纯 BF16 臂乱码（第一轮全部 BF16 臂）三者交集唯一。
- **V-A 判别实验（`identity/quarantine-forceIter/`）：两解码路径数值等价，无法用路径切换判别。** 判别开关 `GGML_VK_HC_BF16_FORCE_ITER=1`（三文件、默认零行为变化、16 项 CPU 测试全绿——交接清单记录）将 HC BF16 收缩路由到 iter() 标量解码；开关开后乱码与开关关**逐字相同**。switch-chain-audit 生效链审计六环全闭合（env 解析 → pipeline 匹配 → flags 组装 → push constant 字段映射（output_flags = p.fusion_flags 同一成员，偏移 28 字节）→ shader 宏与运行时分支 → 执行流到达），开关确实生效；V-A 原判据（「路径切换必有 ulp 差异」）前提错误——两条 mul_mat_vec 解码路径数值完全同构（相同元素映射、bitwise 相同 B 值、同一 dot 结构、共用归约），逐字相同是预期结果。
- **最终归因（诚实边界）：** 缺陷在 **fused 段对 BF16 的处理链上、两条 mul_mat_vec 解码路径的共享层**。这是排除法交集，**不是直接证据**——最终缺陷未钉死。剩余候选（Q8 twin 正常 + BF16 乱 + 两 BF16 路径等价的交集）：BF16 特有且两路径共享的**寻址单位语义**（QUANT_K=1 下 ib/iqs/a_offset 的解释与 host 侧 binding 布局的一致性）、HC_UP_FOLD 折叠 arm、dispatch 参数。静态审计（bf16-segment-fixer 穷尽）无缺陷——「静态无错 ≠ 运行正确」：iter_aligned_nonquant + DATA_A_BF16 组合全仓库只有 HC 两个变体，从诞生起无独立执行见证。

### 12.4 修复资产清单（全部保留）

1. **入口还原 diff**（`identity/quarantine-entryRevert/entry-revert.diff`，14 行）：立即可用的规避修复——恢复 Q8_0-only 入口即恢复 baseline；代价是放弃 fused 增益。
2. **upfold 修复**（工作树 + `identity/quarantine-upfoldFix/`）：真实缺陷已修（第三种乱码形态证明数值路径改变）。
3. **判别开关**（工作树三文件 + `identity/quarantine-forceIter/`）：`GGML_VK_HC_BF16_FORCE_ITER`，默认关闭零行为变化。
4. **V-B 设计**（bf16-segment-fixer 收尾报告有完整方案）：单 site teacher-forced 固定 token 回放，逐层比较 normalized/lo/mixed 与 CPU F32 参考，把偏差首次出现的环节钉死——最终定位的直接证据路径。

### 12.5 裁决影响

- **h1 的 PROMOTE 状态不变**（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）：F 口径（12.1）补全双口径证据链，h2 淘汰（12.2）加固最强臂地位。
- **理论 L = UNRESOLVED、Gap = UNPROVEN 不变**；乱码最终缺陷未钉死（12.3），工单 A 对拍继续挂起。
- 本轮整体状态仍为 `QUALIFIED_BEST_KNOWN`。

## 十三、第三轮补录（2026-10-12/13，第 2 个 Manager 任期）：独立评审与 GAP 推进

本节补录第三轮 session 的评审与执行事实。核对方式：`review/profile-cycle-h1-01-summary.md`、`identity/quarantine-{rebuild,sitedump,uploadfix}/` 全部登记与分析文件、`arms/{profile-cycle-h1-01, sitedump-fused, sitedump-ref, sitedump-fused-r2, sitedump-fused-r3, latebind-parity-off-rebuild}/` 的存在性与状态字段，均直接核验；评审结论与 down-wire-fixer 静态审计细节按 Manager 交接清单记录并标注来源。

### 13.0 第 2 个 Manager 的独立评审

评审确认前任交付与宣称一致：h1/F 口径/h2 统计的 JSON 关键字段、工作树五处修复落盘、四个 quarantine diff、臂数据真实性、v9/guardset 证据全部核验（其中 glob 对 arms/ 返回 0 是 gitignore 语义，非数据缺失）。同时确认五个 GAP：**GAP 1 乱码根因未钉死、GAP 2 工单 A 被阻断、GAP 3 工单 B 未启动、GAP 4 profile 诊断轮 NOT_RUN、GAP 5 理论 L=UNRESOLVED。** 本轮随后推进了 GAP 4（部分闭环）与 GAP 1（定位闭环、修复未闭环）。

### 13.1 GAP 4：profile 诊断轮（部分闭环）

- **数据源：** `arms/profile-cycle-h1-01/`（status=DIAGNOSTIC_NOT_RANKED，--profile cycle、warmup 1 + repeats 2、冻结 measured-bin、h1 配置；result.json 状态字段直接核验）与 `review/profile-cycle-h1-01-summary.md`。诊断口径 pooled_tps 50.387（插桩扰动下仍接近正式轮 51.67；不进排名）。
- **周期账（§7.2 口径）首次取得逐周期原始数据：** 184 周期（2 请求 × 160 token；K 合计 318 = 2×160 − 2 首 token 来自 prefill）；总周期 6332.0 ms = **V（verify）5801.2（91.6%）+ D（draft）445.3（7.0%）+ S（rollback/catch-up）85.5（1.4%）+ H 0.0**；每公开 token **19.912 ms**；MTP 盈利门成立（h1 19.912 vs h0 ~20.41，约 2.4%）；K=1 × 50、K=2 × 134 → 接受率 72.8% 与正式轮 67/92 精确一致；V_ms p50 25.9 / p95 28.8 / min 24.2 / max 285.8，D_ms p50 2.41，S_ms p50 0.48；accepted_prefix 新增逐周期 K 序列与直方图（比 timings 的 per-request 汇总更细）。
- **rank/stage 账（§7.3）：实测确认采集器不存在**——五字段（rank_stage_ledger、actual_pipeline_hits、active_rows、capacity_rows、bank_epoch）全部 NOT_COLLECTED；契约 README 已声明未来入口为 GGML_TP5_PROFILE=1 的 server.log rank/stage 时间线解析。**缺口从评审推断变为实测确认；闭合需 Engineer 实现该解析采集器。**
- **新发现：** 异常周期 4 个（V_ms > 50ms，~10 倍于 p50 25.9）：cycle[0] 272.2、cycle[89] 264.8、cycle[92] 265.4/285.8——每请求首周期（图缓存冷启动）+ **中段 2 个**（疑似图重编译/缓存失效；summary 记单周期影响 ~0.5% 总时长，Manager 交接清单估潜在 1–2% 收益）；--profile full 可定位。

### 13.2 GAP 1：乱码根因钉死（定位闭环，直接证据）

三轮递进，全部实验在 /var/tmp/tp5-* 隔离树执行、仓库工作树零接触（identity/quarantine-*/ 各自登记 diff 与 DSO 哈希）：

1. **裁决实验（`identity/quarantine-rebuild/`，20:21Z）：** 全新构建（git archive 3e28b121a + 存档 diff + 从头 cmake configure，libggml-vulkan `db1f28ef…`）的乱码与增量构建（`9452b39a…`/`9e6f0a54…`）**逐字节相同**——unit-semantics-audit 的 SPIRV 新鲜度假说被否定；V-A 共享层判读保持有效；三构建乱码一致排除构建伪影。
2. **单 site dump 对拍（`identity/quarantine-sitedump/`，20:51–20:57Z）：** site-dumper 插桩（`GGML_VK_HC_DUMP_SITES` opt-in 开关 + mid-graph synchronize 读回，文档 `docs/TP5-HC-SITE-DUMP.md`）。**关键执行细节：TP5 下 GGML_VK_CMD_REPLAY 默认开启（common.cpp 置 1）会跳过 dump，冷路径（=0）为必需**；warmup 0、max_tokens=32（两臂 runner exit 1 截断为诊断轮预期，非异常）。fused 臂三个 site 一致呈现 **normalized 全非零（n=10240，真实激活值 0.23/−0.42/1.20…）+ lo 全零（n=320）+ mixed 全零（n=2560）**；ref 臂（3a 入口还原，vk `be4946ab…`）无 dump——HC 段不激活，正是 3a 的预测。
3. **rank 区分 + A binding dump（r2，`arms/sitedump-fused-r2/` 50 个 dump，21:24–21:25Z）：** **W_down 与 W_up 的 A binding 内容全零（10 组 rank×site 一致；verdict 表 ALL 10：normalized 10240/10240 非零、lo 0/320、mixed 0/2560、W_down 0/64 u16、W_up 0/64 u16）而 normalized 真实。** subbuffer 结构存在（r0 W_down {buf=0x6419d207ea30, off=1049041280, size=6553600} = 320×10240×2 BF16 全量、W_up {同 buf, off=1055594880}；r1 句柄不同——逐 rank 分片结构在）但**字节从未写入**。rank 区分用 ctx->device.get() 指针序数（限制已记录：device 指针序数，非 Vulkan rank 号）。

**根因钉死：BF16 HC 权重在 dispatch 时刻不在收缩绑定的 subbuffer（零页）——TP5 权重上传链的 BF16 特有缺陷。** SPIR-V 执行层无罪（kernel 忠实执行零×激活）；该结论完整解释此前全部观察：lo=0（W_down 零）、mixed=0（W_up 亦零，r2 证实 down-wire-fixer 的 fold 数学——mixed=fuse0×sigmoid(0)×0.25≠0 需要 W_up 也零）、V-A 两路径同零（读同一零权重）、3c Q8 正常（Q8_0 走不同上传路径）、upfold 修复是零权重之上的二阶问题、ref 臂无 dump。down-wire-fixer 静态排除四个接线嫌疑域（descriptor 模板家族无关 / grid 无零产出 / dispatch 无条件提交 / 无覆盖）并发现 dump 覆盖写限制（五 rank 同文件名最后写者获胜——r2 用 rank 区分修正）。

### 13.3 GAP 1：修复的阴性一轮（修复未闭环）

- **修复尝试（upload-chain-fixer）：** 定位 mmap 分支（`llama_model_loader::load_all_data` 的 `cur->data==nullptr` 时 tensor_alloc 只绑定不 fan-out）并修复（meta buffer 显式 tensor_set 补齐，`identity/quarantine-uploadfix/upload-fix.diff` 30 行）+ UploadChainAuditTests 断言 + 文档根因章节。DevOps 应用到 rebuild 树编译：原 diff 引用 internal 头符号（`ggml_backend_buffer_is_meta`）编译失败，两次非架构级自修（先试 `ggml_backend_meta_buffer_type()` 比较——符号不在公共头；最终改 `ggml_backend_buffer_name` 前缀匹配 "Meta("）后通过（`identity/quarantine-uploadfix/uploadfix-bin-identity.json`，21:39Z；交接清单称「三次」，以 execution-log.md 记录的两次自修为准）。
- **阴性结果（`arms/sitedump-fused-r3/`）：** r3 的 50 个 dump 与 r2 逐文件一致——**W 仍全零（全部 10 组 rank×site）**。修复编译进二进制但未改变现象，排除「mmap 分支 tensor_alloc 缺 fan-out」为实际解释。
- **三个候选原因待判别（判别方法已写入文档：加载期打印 W {buffer, data}，本轮未执行）：** (a) 修复点未命中（BF16 HC 权重的 cur->data 非 null 或 buffer 非 "Meta(" 名）；(b) 修复点命中但 meta set_tensor 不做 MIRRORED fan-out；(c) TP5 tensor-split 权重经 clone/replicate 链上传而非 loader mmap 分支。
- **off 臂三重判定（乱码消失 + 哈希 `c2d7bdd3…` 逐字一致 + dispatch 恢复 [4080,3391,…]）未跑**——前置 dump 复查（W 非零）未通过，按步骤判定停止。**GAP 1 修复未闭环。**

### 13.4 测试与插桩资产

- `tests/test-tp5-upfold-indexing.py` 经多轮扩充 22 → 28 项（新增 HcSiteDumpSentinelTests、DownWiringAuditTests、UploadChainAuditTests）；其中 DevOps 修正一处断言公式笔误（4 流 fold 期望值 0.115 = 0.25×4×0.23×0.5，对齐实现的正确公式；修正过程两轮，先误写 0.1325——为断言对齐实现，非弱化门禁）。
- 插桩资产保留在 rebuild 树与工作树：site-dump（opt-in）+ rank 区分 + A binding dump。

### 13.5 裁决影响（诚实边界）

- **GAP 1：定位闭环（直接证据），修复未闭环（阴性结果如实保留）。** 乱码最终修复仍 OPEN；工单 A 对拍继续挂起。
- **GAP 4：部分闭环**——周期账完成；rank/stage 采集器确认缺失（需 Engineer 实现）；中段异常周期待 --profile full 定位。
- **GAP 3（工单 B 未启动）、GAP 5（理论 L=UNRESOLVED、Gap=UNPROVEN）不变。**
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；本轮全部工作为诊断与归因，不改变任何排名资格。

## 十四、第四轮补录（2026-10-12 深夜，第 3 个 Manager 任期）：评审与 GAP 1 深挖

本节补录第四轮 session 的评审与执行事实。核对方式：`identity/quarantine-{loadtrace,routefix,dispatchtrace}/` 的全部 identity/verdict/diff 文件（r5 verdict、r6-verdict、r7 dispatchtrace verdict 的逐 rank 对照表）、`arms/sitedump-fused-{r4,r5,r6,r7}/` 存在性，均直接核验；upload-path-tracer 静态深挖与 container-router-fixer 算术反推细节按 Manager 交接清单记录并标注来源。

### 14.0 第 3 个 Manager 的独立评审

评审确认第 2 任交付与宣称一致：十三节内容、r2-verdict 的设备侧证据链、execution-log 的诚实边界、r3 dump 真实存在于 dumps/ 子目录、profile 数据真实。确认四个 GAP：**GAP 1 修复未闭环、GAP 2 工单 A 挂起、GAP 3 工单 B 未启动、GAP 5 理论 L=UNRESOLVED**；另 **GAP 4 残余**（rank/stage 采集器与异常周期定位，本轮未动）。

### 14.1 upload-path-tracer 静态深挖：证伪第三轮根因叙述（r3 阴性的机理级解释）

静态读码（交接清单记录）证明：**meta.cpp:2637 给所有 meta 张量设假指针 0x2000000000000000（非 null）**，loader.cpp:1670 的 `buf_mmap && cur->data==nullptr` 分支对 meta 权重**恒为假**——**第三轮的 r3 修复（mmap 分支 tensor_set 补齐）是死代码**，这正是 r3 阴性结果的机理级解释。真实上传链：`load_all_data` 恒走 else 分支（loader.cpp:1699-1701）→ meta set_tensor → MIRRORED fan-out（meta.cpp:2330-2338）→ per-rank write。第三轮三候选的静态判别结论：(a) 成立（结构性——修复点未命中）；(b)(c) 排除（真实链即 meta set_tensor → MIRRORED fan-out，非 clone/replicate 链）。**重要负结果：上传链上无 BF16/Q8_0 类型分叉**——类型差异的分叉点在别处（见 14.6 的 Q8_0 相容性问题）。

### 14.2 r5 运行时判别（identity/quarantine-loadtrace/，2688 行 loadtrace）

- **假设 (i) 排除：** 384 次 MIRRORED set_tensor 全覆盖（48 层 × 8 HC 张量，含 blk.0.hc_ffn_down.weight），写入尺寸 6553600 = 完整 BF16 权重字节数（320×10240×2）。
- **假设 (ii) 排除：** 零 SIMPLE_TENSOR_NULL——5 rank 的 simple tensor 全部非空有效。
- **当时判「假设 (iii) 坐实」**（加载期写入 buf 与 dispatch 绑定 buf 不匹配）——**但这是跨 run 比较，后被 r7 修正为伪影（见 14.5）**。
- **r4 插桩笔误（如实记录）：** r4 的 loadtrace 零行是插桩过滤条件笔误（`strncmp(name, "hc_", 3)`，真实张量名为 `blk.N.hc_...`），非加载链缺陷；r5 修正为 `strstr(name, ".hc_")` 后 2688 行正常输出。

### 14.3 container-router-fixer 修复 + r6 阴性（identity/quarantine-routefix/）

- **定位（交接清单记录，算术级反推）：**「alloc_ctx_tensors_from_buft 无条件重绑导致身份分叉」——由 dispatch off=0x3E801000 反推 data=0x3E801000+base=第二 meta buffer 的 tallocr 分配。
- **修复：** `route-fix.diff` 48 行、两道守卫（重绑循环 continue + lazy-init 路径 GGML_ABORT fail-closed）+ ContainerRoutingAuditTests 3 项（测试文件补 `import os` 笔误修正，非架构级、仅测试文件）。
- **r6 阴性（r6-verdict.json，直接核验）：** CPU 前置 34 项全过；GGML_ABORT **零触发**；绑定仍「不匹配」（跨 run 比较）；W/lo/mixed dump 仍全零——ROUTE_FIX_INSUFFICIENT。诚实细节：guard 1 的 continue 路径未加 print，「无 print 不是未触发的证据」（r6-verdict 原文注明）。
- **off 臂 NOT_RUN：** 步骤 4 被步骤 3（dump 复查）门阻断。

### 14.4 r7 决定性判别（identity/quarantine-dispatchtrace/）：身份完全匹配

**同一 run 内**逐 rank 精确对照（r7 verdict.md 的对照表，直接核验）：

- 加载期 MIRRORED fan-out 写入目标（meta.cpp:2330）与 dispatch 侧 W_down 张量（ggml_vk_hc_segment 内）的 **buffer 与 data 指针在全部 5 rank 完全一致**（如 rank0：双侧 buf=0x6490fe8fb9f0、data=0x3e872980；rank1–4 逐项同样匹配）。
- dispatch 张量 standalone **无 view_src**（非 VIEW、非 clone——VIEW/clone 假设排除）。
- **但 W dump 仍全零。** W dump 的 subbuffer 显示 buf=0x6490fe8fb730、off=1049041280——这是 `ggml_vk_tensor_subbuffer` 经 Vulkan 后端解析后的**底层 VkBuffer**，与 `tensor->buffer` 的 `ggml_backend_buffer_t` 是**不同间接层**。

### 14.5 r5/r6 结论修正（本轮最重要：前两轮结论错了，如实写明）

**r5/r6 的「身份分叉」结论是跨 run 地址比较的伪影。** r5 比较的 dispatch buf（0x5705644297d0）与加载期 buf（0x570564428160）来自**不同 run**（ASLR/分配布局不同）；r7 在同一 run 内比较，buffer 与 data 完全匹配。r6 的守卫未生效不是因为「路由分叉」，而是**根本没有分叉——守卫保护的问题不存在**。这是 r5/r6 两轮阴性的机理级解释。

### 14.6 剩余嫌疑收窄到最后一环：写入可见性

身份匹配 + 写入调用执行 + W 仍全零 → **写入未到达 dispatch 可见的内存**。三嫌疑（r7 verdict 剩余嫌疑段）：

1. **ggml_backend_tensor_set 静默失败：** simple tensor 的 per-rank Vulkan buffer 可能不支持 set_tensor（返回错误被吞），或写入到 host staging 而非 device memory。
2. **host mapping 非 HOST_COHERENT：** simple tensor 的 data 指针是 host mapping，但底层 VkMemory 的 memory type 非 HOST_COHERENT——host 写入对 GPU 不可见，需显式 flush（vkFlushMappedMemoryRanges）。
3. **subbuffer 解析链分叉：** `ggml_vk_tensor_subbuffer` 解析到的底层 VkBuffer（0x…b730）与 host 写入经 tensor->buffer 的对象（0x…b9f0）是不同间接层——解析链或位置换算不自洽（dispatchtrace 实测 data 与 subbuffer off 的反推矛盾）。

另注意：**Q8_0 正常（3c）与 coherence 假设的相容性待解释**（Q8_0 写入路径可能不同；14.1 已确认上传链无类型分叉，分叉可能在 set_tensor 实现或 buffer 属性层）。

### 14.7 测试与插桩资产

- `tests/test-tp5-upfold-indexing.py` 扩充至 **34 项**（含 ContainerRoutingAuditTests 3 项；import os 笔误修正）。
- 插桩资产保留在 rebuild 树与工作树：loadtrace（三点）+ dispatchtrace（W 张量身份/view_src 链）+ sitedump 全套。

### 14.8 诚实边界与裁决影响

- **GAP 1：根因收窄到「写入可见性」层（身份匹配 + 写入执行 + 数据在 dispatch 时不可见），修复仍未闭环。** 第三轮的「权重零页」结论在数据层面保持（dispatch 时数据确实为零），但「不在绑定的 subbuffer」的机理表述被 r7 精细化为「写入不可见」——以 14.4–14.6 为准。
- **off 臂三重判定连续四轮（r3/r5/r6/r7）被前置 dump 复查阻断**——如实记录，不跳门。
- **GAP 2/3/5 与 GAP 4 残余不变（本轮未动）。**
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；本轮全部工作为诊断与归因，不改变任何排名资格。

## 十五、第五轮补录（2026-10-12/13 深夜）：r8/r9 终局判别与归因链翻转

本节补录第五轮 session 的 r8/r9 两轮判别。核对方式：`identity/quarantine-{sourcetrace,phasestrace}/` 的 verdict.md 与 bin-identity.json（含 r8 的 src/readback hex 证据与 r9 的三时点对照表）、`arms/sitedump-fused-{r8,r9}/` 存在性，均直接核验；visibility-adjudicator 的静态结论与下一轮判别方向按 Manager 交接清单记录并标注来源。

### 15.1 r8 终局判别（identity/quarantine-sourcetrace/，23:06Z）

三点插桩：meta.cpp MIRRORED fan-out 内源 data 前 16 字节 hex 打印 + write-then-readback（`ggml_backend_tensor_get` 读回同位置）+ loader.cpp else 分支 offs/mapping 打印（**笔误如实记录**：打印位于 `if(use_mmap)` 的 else 分支内，runner 传 `--no-mmap` 实际走非 mmap 路径未触发——meta.cpp 的 MIRRORED 打印不受影响，非加载链缺陷）。

裁决（当时判定）：

- **假设 A 排除：** 源数据真实非零——src_hex 如 blk.0.hc_attn_down.weight `2d3bd83b1cbb9abb63ba40bdb2bcbcbb`（BF16 真实权重字节）、hc_attn_norm `00c0913f…`（F32）。
- **1920 对 src/readback 逐字节一致、零失败**（384 hc 张量 × 5 rank）——当时判「写入真实到达 per-rank simple tensor 内存」。
- W dump 仍全零（与 r2–r7 完全一致）→ **当时判「假设 B 坐实：加载完成到 dispatch 之间被全局性覆盖」**。覆盖者候选（当时）：ggml-alloc 图构建 zero-init / Vulkan 缓存失效重分配（`ggml_vk_cache_invalidate_all` / buffer 池）/ stc_compute 初始化。

### 15.2 r9 相位定位（identity/quarantine-phasestrace/，23:28Z）——推翻 r8 覆盖假设

三时点 readback 插桩（编译修复一处：`extern "C"` 声明从函数体内移至文件头——函数内声明 C 链接函数在 C++ 中不合法，非架构级）：

| 时点 | 读取路径 | 结果 |
|---|---|---|
| T1_post_all_load（全部加载完成） | `ggml_backend_tensor_get`（**host 路径**） | `2d3bd83b…` **非零** |
| T2_first_graph_compute（首次 graph_compute 入口） | `ggml_backend_tensor_get`（**host 路径**） | `2d3bd83b…` **非零，与 T1 逐字节相同** |
| T3（dispatch W dump） | vk staging copy（**GPU 路径**） | **全零**（5 rank × 0/64） |

### 15.3 决定性洞察：host/GPU 读取路径分裂

**T1/T2 用 `ggml_backend_tensor_get`（经 meta buffer → simple tensor → 底层 buffer 的 host 接口），T3 用 `ggml_vk_tensor_subbuffer` + `vkCmdCopyBuffer`（Vulkan staging）——两条路径看到不同的世界：host 路径看到写入的字节（T1=T2=非零且从加载到首次计算稳定），GPU 路径看到零。**

由此 **r8 的「覆盖/清零」假设被推翻**：数据可能从未被覆盖，而是 **MIRRORED fan-out 的 host 写入从未到达 GPU 可见的内存**。r8 的 1920 对 readback 也全部走 host 路径——其「写入成功」结论只对 host 路径成立。第四轮 14.6 三嫌疑经 r8/r9 重新裁决：嫌疑 1（tensor_set 静默失败）被 r8 的 write-then-readback 排除（写入在 host 路径真实生效）；嫌疑 2（非 HOST_COHERENT）与嫌疑 3（subbuffer 解析链分叉）获 r9 直接支撑，合并为「host-GPU 可见性断裂」框架。

### 15.4 最终三候选（GAP 1 归因链最后一环）

1. **per-rank VkBuffer 实际非 HOST_COHERENT：** host mapping 写入后 device memory 不可见，需 `vkFlushMappedMemoryRanges`；T1/T2 的 host readback 看到的是 host mapping，GPU 读的是 device memory——两者不同步。**矛盾点待实测：** visibility-adjudicator 的静态分析曾排除此项（rebar 分支全带 COHERENT 且 GGML_ASSERT 强制——交接清单记录），但运行时实际 flags 未实测、fallback 分支可能不同——静态结论与 r9 运行时证据的冲突正是下一轮第 (2) 步的判别对象。
2. **host write 与 GPU read 指向不同 VkBuffer/VkDeviceMemory：** `ggml_backend_tensor_set` 走的 buffer 链与 `ggml_vk_tensor_subbuffer` 解析的 VkBuffer 不同（r7 已见 subbuffer buf 与 tensor->buffer 是不同间接层）。
3. **graph_compute 内 buffer 重映射：** 入口时 host 路径可见、执行中被重映射/重绑定。

另注意 **Q8_0 正常（3c）与三候选的相容性仍待解释**（Q8_0 写入路径可能不同）。

### 15.5 GAP 1 归因链全程（六轮，r3→r9）

| 轮次 | 假设 | 结局 | 证据级别 |
|---|---|---|---|
| r3 | mmap 分支缺 fan-out | 阴性——修复是死代码（meta 假指针使分支恒假） | 行号级静态（14.1） |
| r5 | 三候选运行时判别 | (i)(ii) 排除；(iii)「身份分叉」当时坐实 | 2688 行 loadtrace（14.2） |
| r6 | 容器路由重绑 | 阴性——守卫保护的问题不存在 | 34 项测试 + r6 verdict（14.3） |
| r7 | 身份分叉 | 证伪——跨 run 比较伪影，同 run 内身份完全匹配 | 同 run 逐 rank 对照（14.4） |
| r8 | 覆盖/清零 | 当时判 B 坐实——被 r9 推翻 | 1920 对 src/readback（15.1） |
| r9 | host-GPU 可见性断裂 | **当前最终形态**（三候选待判别） | 三时点 host/GPU 路径对照（15.2） |

每一环的排除/翻转都有行号级静态证据或同 run 运行时证据；归因链从未回退到无证据的猜测。

### 15.6 诚实边界与裁决影响

- **GAP 1：根因最终收窄为 host-GPU 可见性断裂（身份匹配 + host 写入生效 + GPU 读取为零），修复仍未闭环。** r8 的「覆盖」结论被 r9 推翻，如实写明；第四轮 14.6 的三嫌疑经 r8/r9 重新裁决（嫌疑 1 排除，嫌疑 2/3 合并入可见性断裂框架）。
- **off 臂三重判定连续六轮（r3/r5/r6/r7/r8/r9）被前置 dump 复查阻断**——如实记录，不跳门。
- **GAP 2/3/5 与 GAP 4 残余不变（本轮未动）。**
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；本轮全部工作为诊断与归因，不改变任何排名资格。
- **插桩资产：** rebuild 树上现有 loadtrace（三点）+ dispatchtrace（张量身份/view_src）+ sourcetrace（源 hex + write-then-readback）+ phasestrace（T1/T2 相位 readback）+ sitedump 全套；`identity/quarantine-{loadtrace,routefix,dispatchtrace,sourcetrace,phasestrace}/` 五个目录 + `arms/sitedump-fused-{r4..r9}/` 六个臂全部保留。

## 十六、第六轮补录（2026-10-13，第 4 个 Manager 任期）：三 Engineer 交付与 r10/r11/r12 判别

本节补录第六轮 session 的三 Engineer 交付与 GAP 1 三轮判别。核对方式：`identity/quarantine-{forkfix,wrtrace,quadtrace}/` 的 verdict.md / instrumentation.md / bin-identity.json（含 r12 三实验的对照表与综合裁决）、`arms/sitedump-fused-{r10,r11,r12}/` 存在性、`identity/cpu-tests-round4/summary.md`（三套 CPU 测试计数），均直接核验；vk-settensor-buf-audit 静态结论、guard-disposition 与 rankstage-collector 的交付细节、r11 的语义未对齐与 r12 请求状态按 Manager 交接清单记录并标注来源。

**网络故障恢复的诚实记录：** r12 途中发生网络故障，Chronicle 末条记载「插桩已植入并完成运行」与当时盘面不符（`arms/sitedump-fused-r12/` 与 `quarantine-quadtrace/` 均不存在）——DevOps 从头执行了完整 r12，现存文件为重执行的结果。

### 16.1 三 Engineer 交付（CPU 测试全绿验证）

- **guard-disposition（守卫处置）：** 可达性评估——守卫一（重绑 continue）静态不可达、守卫二（GGML_ABORT）条件不可满足。处置决策：**保留守卫 + 改写注释**（fail-closed 不变量正确、移除收益与代价不成比例）；meta.cpp 两处注释改为「historical defensive guard」，删除伪影地址与「never-written second allocation」叙述（如实引用 r6/r7 证据——该叙述源于已被 r7 证伪的跨 run 伪影）；tests docstring 同步；docs 更正框。**守卫代码零改动。**
- **rankstage-collector（rank/stage 采集器，GAP 4 残余的 Engineer 部分）：** `RANK_STAGE_LINE_SPECS` 十种 tag 规格 + `parse_rank_stage_line` + `extract_rank_stage_ledger`（per-request 归属、relay_stage 聚合、slowest_rank 定位、malformed 容错）+ RankStageLedgerTests 14 用例。诚实边界：active_rows/M/N_K/weight_type 源码无打印点（如实标注、不虚构）；bank_epoch 保持 NOT_COLLECTED（读者完成关系无打印点）。
- **CPU 测试验证（`identity/cpu-tests-round4/summary.md`，直接核验）：** test-tp5-upfold-indexing **34 项 OK** + test-tp5-gap-tools **46 项 OK**（修正 2 处断言笔误：lines_total 3→4、lines_parsed 2→3——rankstage fixture 4 行但原断言少算 1 行，为断言对齐事实、非弱化门禁）+ test-tp5-necessity-audit **40 项 OK**——三套全绿。

### 16.2 r10（forkfix，identity/quarantine-forkfix/）：meta registry 零效果

meta 层 registry 修复（权重→simple tensor 全局注册表，fan-out 时填充、解析恒返回注册项）实施后 **W dump 仍全零**（与 r2–r9 一致）。结合 r7（dispatch 张量的 buffer/data 与写入目标完全一致），**meta 路由层彻底排除**——dispatch 已持有正确对象，registry 零效果。r10 verdict 同时给出关键代码发现：`ggml_vk_buffer_write_2d` 的 host 路径（HostVisible+HostCoherent）直接 memcpy 到 `dst->ptr + offset`——r8 的 readback 走同一路径看到非零，dispatch 的 GPU 读看到零。

### 16.3 r11（wrtrace，identity/quarantine-wrtrace/）：非 HostVisible + 双 vk_buffer 指针

W/R 双端打印（GAP1-W 写入链 / GAP1-R 读取链，BF16 only；**字段语义未对齐**——GAP1-W 打印的 dev_buffer 是 vk_buffer 对象、GAP1-R 是 context 对象；DevOps 上下文耗尽、最终对照表与原始行的矛盾按交接清单记录）。确立：**ggml 层身份完全匹配**（ggml_buf/vk_ptr_base/offset/size 逐项一致，如 blk.0.hc_attn_down rank0 双侧 off=100597760、size=6553600）+ **buffer 非 HostVisible**（双侧 ptr=nil——写入走 staging + command submission 路径，非直接 memcpy）。两个不同 vk_buffer 指针（写侧 vk_buffer 对象 0x62d2f8fb5a30 vs dump 读 0x62d2febe9f70）成为关键线索。

### 16.4 vk-settensor-buf-audit（Engineer 静态）：fork 收窄到三候选

静态审计（交接清单记录）：vk 解析层无 fork（dev_buffer 构造后无替换、三重约束）。两个不同 vk_buffer 指针逻辑上只能是两个不同的 `ggml_backend_buffer_t`——「数据写进 buffer A、dump 读 buffer B」假设成立，fork 收窄到三候选：meta 容器影子 / view_src 分叉 / 跨运行伪差异。

### 16.5 r12（quadtrace，identity/quarantine-quadtrace/）：三实验，fork 排除的铁证

三实验同一宏（`ggml_vk_gap1_quad`）、字段语义逐字节一致（写侧与 dump 侧走同一函数——修正 r11 的语义未对齐）：

- **实验 1（四元组）：** dump 四元组（ggml_backend_buffer_t / context / vk_buffer / VkBuffer）与写侧 set_tensor 序列在 5 rank 逐一完全匹配（output_hc 5 rank 全部一致，如 rank0 双侧 0x64cbdca613c0 / 0x64cbdca61370 / 0x64cbd6e2e4f0 / 0x64cbd73cfff0；offset 亦一致：output_hc_down 双侧 off=40960）；GAP1-R 显示读链 offset 无错位（blk.0.hc_attn_down 读 off=100597760 与写侧一致）。**fork 三候选（meta 影子 / view_src / 跨运行伪差异）全部排除、offset 错位排除。**
- **实验 2（fence 读回，GGML_VK_GAP1_READBACK=1）：** write_2d 非 HostVisible 分支的写 fence 退休后立即从同一 (buffer, offset) 读回——前 200 次写**无一全零**（output_hc_down off=40960 读回 `39b63b42 3d003c3b…` 真实 BF16 权重位模式）。**staging 写入真实到达 device buffer，写入失效假设排除。**
- **实验 3（dump 自检）：** dump 机制有效——同一机制读出 normalized 非零（10240/10240，5 rank），W 读零不是 dump 伪影。
- r9 phasestrace 本轮复现：T1/T2 readback=`2d3bd83b…` 仍非零。

### 16.6 最终裁决：情形 A——写后覆盖（GAP-1 缺陷形态彻底改写）

同一 (buffer, offset) 的完整时间线：**写入（fence 读回非零）→ T2 首次 graph_compute 入口（非零）→ 首个 HC 段 dispatch（全零）。**

**覆盖窗口锁定：T2（首次图计算入口）→ T3（首个 HC 段 dispatch）之间——首次图执行过程中某操作清零了权重 buffer。**

**缺陷形态改写：** 不是 r9 表述的「host 写入从未到达 GPU 可见内存」——数据到达且在位（fence + T2 双证），**被首次图执行中的未知操作抹掉**。r8 的「覆盖」方向被 r9 推翻后，r12 以更精确的证据重新确立覆盖（但窗口从「加载完成→dispatch」收窄到「图计算入口→HC dispatch」）。归因链的这次再翻转与前几次同样有同 run 铁证支撑，如实记录。

### 16.7 覆盖者候选（未定位）与下一轮方向

**静态已排除：** vk 层 init_tensor（no-op，仅 view_src 断言）、COUNT_EQUAL memset（dst 是小计数 buffer）、prealloc_add_rms_partials memset（独立 prealloc buffer）、bst 小 memset（4×int32 计数器）、读链 subbuffer 解析错位（GAP1-R offset 一致）。

**剩余候选（按嫌疑排序）：** (1) 图执行中某算子把激活 dst 写进权重区间（GAP1-R 只追踪 BF16、激活 F32 的 subbuffer 解析未追踪——**盲区**；写侧 fork 而非读侧 fork，r5–r10 的守卫只保护读侧所有权）；(2) `ggml_vk_cache_invalidate_all` / buffer 池释放重分配（首次图执行的 cache 初始化路径）；(3) stc_compute 容器 lazy 初始化触碰权重区。

**下一轮定位建议：** 权重区间写入 watchpoint（off 落在 HC 权重区间时打印调用栈）或 graph_compute 内分段 readback 二分。

### 16.8 诚实边界与裁决影响

- **r12 请求状态：** fused 版请求未完成（teardown SIGABRT -6、FP-TMISS=0、request-*.json 未产出）——诊断轮，dump 采集完成即达目的，不进排名。
- **off 臂三重判定连续九轮（r3/r5/r6/r7/r8/r9/r10/r11/r12）被前置 dump 复查阻断**——如实记录，不跳门。
- **GAP 4：** 采集器 CPU 验证已过（16.1）、**GGML_TP5_PROFILE=1 真机诊断轮未跑**（真机验证待排）；中段异常周期的「图重编译/缓存失效」嫌疑与 GAP 1 覆盖者候选 (2) 存在潜在交集，待真机轮一并观察。
- **GAP 3（工单 B 未启动）、GAP 5（理论 L=UNRESOLVED、Gap=UNPROVEN）不变。**
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；本轮全部工作为诊断、归因与工具交付，不改变任何排名资格。

## 十七、第七轮补录（2026-10-13，第 5 个 Manager 任期）：r13 覆盖者定位轮

本节补录 r13（quarantine-watchtrace）。核对方式：`identity/quarantine-watchtrace/verdict.md` 全文（三重证据、fork 排除对照、选择性覆盖证据链、候选收窄表与 r14 建议）与 `r13-bin-identity.txt`（libggml-vulkan `7e45f022…`）、`arms/sitedump-fused-r13/` 存在性，均直接核验；与 Manager 交接清单一致。

### 17.1 插桩与基线复现

- **插桩（全部 env 门控 GGML_VK_GAP1_WATCH / GAP1_BISECT）：** watch 区间表（`g_gap1_watch_regions`，写链 23014 与 subbuffer 解析链 10315 双登记）+ 四类 HIT 检查（SUBBUFFER-HIT / MEMSET-HIT / COPY-HIT-W2DA/WNC/BC / DISPATCH-HIT）+ vk 层批次块 bisect 同步读回探针（blk.0.hc_attn_down off=100597760，600 次批次边界）。新库哈希冻结于 identity/quarantine-watchtrace/。
- **基线形态复现确认：** dump weightsA 全 5 rank 全零、lo/mixed 全零、normalized 非零——与 r5–r12 完全一致，**无观察者效应**（bisect 探针未改变 T3 结果）。teardown SIGABRT -6 与 r12 同形态（插桩树已知 teardown，非新异常）。

### 17.2 fork 终局排除（写读两侧位置完全一致的直接证明）

output_hc_down rank0（唯一两侧都有四元组打印的张量）：写侧 [GAP1-QUAD tag=W] 与 dump 读侧 [tag=DUMP] 的 ggml_buf/ctx/dev_buffer/vkhandle 四元组与 offset（40960）**完全一致**——写与 dump 读指向同一 VkBuffer 同一位置。**vk 层 buffer/offset fork 直接排除**（r12 裁决的「写后覆盖」方向维持；r11 的双 vk_buffer 指针观察确系打印语义不对齐的伪差异）。

### 17.3 决定性新发现——覆盖是选择性的，不是全局的

三重证据拼出同一块 VkBuffer 内两个区间的不同命运：

- r12 FENCE：off=40960（output_hc_down）写入时刻 5 rank 全部读回非零（`39b63b42 3d003c3b…` 真实 BF16 位模式）→ 写入成功到达设备；
- r13 dump（T3）：同一位置读**零**；
- r13 bisect：**同一 rank0 buffer 的另一区间** off=100597760（blk.0.hc_attn_down）在全部 600 次批次边界（贯穿 T2→T3→执行结束）**保持非零**（`3bd83b2d bb9abb1c`）。

**被清区间：output_hc（40960，第一组 buffer 开头）与 blk.47.hc_ffn（61012480，第二组 buffer）；幸存区间：blk.0.hc_attn（100597760）。** r8 的「384 张量全局清零」判断被修正——r8 只观察了 dump 的 2 个 site，「全局性」是外推不是观察。选择性模式未定（buffer 开头 vs ffn/output 家族——现有数据无法区分）。覆盖窗口维持：加载完成 → 首个 HC site dispatch（T3）之间，只波及部分权重区间。

### 17.4 候选收窄与盲区披露（诚实边界）

- **候选 (2)（cache invalidate / buffer 池）正式排除**：全程零次 invalidate。
- **三大家族零命中**：MEMSET-HIT=0、SUBBUFFER-HIT=0、COPY-HIT-BC=0——这三个写入家族排除。
- **候选 (1)（激活 dst 错位写入权重区间）与候选 (3)（stc_compute lazy init 产物被 dispatch 写）未被排除也未被证实**——它们的覆盖动作走 dispatch 写 dst 路径（DISPATCH-HIT 家族），而 watch 插桩存在两个限流盲区：W2DA 家族 400 条限流被加载期 1920 个 fan-out 写入打满（T2 之后的 write_2d_async 行为未监控）；DISPATCH 家族 200 条限流被 rms_norm_mul_rope 超大 binding 噪音打满（binding_range≈512MB 与一切区间相交）。collective 侧（peer copy/relay）同样不能排除。
- **本轮不实施修复**：覆盖者指纹未捕获（恰在盲区），不满足「指纹明确指向具体错位机理」的自修条件。

### 17.5 r14 建议（verdict 已写，交 Manager 裁定）

1. **修正 watch 限流门控：** W2DA 家族跳过加载期 self 写（时间门控：首个图计算开始后才计数，或写入区间=权重自身区间时跳过）；DISPATCH 家族过滤 rms_norm 噪音（binding_range > 1MB 跳过或单独计数）。
2. **bisect 探针扩展到三个位置：** output_hc(40960) + blk.47.hc_ffn(61012480) + blk.0.hc_attn(100597760)（含两个已知被清区间与幸存对照）——直接抓 1→0 翻转节点 + 确认选择性模式。
3. 翻转节点定位成功后，对照该节点区间的 dispatch 算子定位覆盖者（候选 1/3 走 dispatch 写 dst 路径、collective 亦不能排除）；或在 T2→T3 窗口对 off=40960 做逐图节点 readback 二分收敛。

### 17.6 执行偏差与诚实边界

- **r13 执行偏差如实记录：** 漏设 GGML_VK_GAP1_READBACK=1（fence 实验未开）与 GGML_VK_GAP1_TRACE=1（GAP1-R 未开）——属 env 配置差异、非代码缺失；证据链完整（r12 已有同型 fence 数据，r13 的 bisect 600 次读提供更强时间序列）。
- **off 臂三重判定连续第十轮（r3/r5/r6/r7/r8/r9/r10/r11/r12/r13）被前置 dump 复查阻断**——如实记录，不跳门。
- **GAP 2/3/5 与 GAP 4 残余（真机诊断轮待排）不变。**
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；本轮为诊断与定位，不改变任何排名资格。

## 十八、第八轮补录（2026-10-13，第 6 个 Manager 任期）：r14/r15 伪影判别终局

本节补录 r14/r15 两轮。核对方式：`identity/quarantine-watchtrace2/verdict2.md` 全文（r14 AMBIG2 结果、r15 SYNCCMP 对照、伪影机理与连锁重估）与 `bin-identity.txt`（r14 libggml-vulkan `3261fc45…`、r15 `ba34cba1…`）、`arms/sitedump-fused-{r14,r15}/` 存在性，均直接核验；与 Manager 交接清单一致。**本轮是 GAP 1 归因链的终局翻转——r2 以来的「W 全零」被证明是观测工具自身的伪影，此前多轮的覆盖/fork 叙事整体重估。**

### 18.1 r14 覆盖者定位轮（三修正插桩，AMBIG2）

r14-instrumenter 三项修正落地（隔离树 ggml-vulkan.cpp，+136 行）：(1) W2DA 家族盲区修正（精确区间 self 写跳过 + 图启动时间门）；(2) DISPATCH 噪音过滤（binding range > 64MiB 跳过、单独计数、前 8 条 SKIP 行）；(3) bisect 三探针扩展（P1=output_hc 40960 被清区 / P2=blk.47.hc_ffn 61012480 被清区 / P3=blk.0.hc_attn 100597760 幸存对照，每批读 64 字节，MISS/WARN 机制）。

**r14 结果落在 AMBIG2（源码预设形态之一）：** 三探针 600 次批次边界全部非零（p1=1 p2=1 p3=1）、数据恒定、无 1→0 翻转、无 MISS/WARN；**W2DA 修正后零条命中**（加载期 fan-out 全被 self-skip、T2 后零 host 写入——r13 盲区终局闭合，该家族正式排除）；MEMSET/SUBBUFFER/BC 零条；cache-invalidate 零次（候选 (2) 维持排除）；修正后显形的 200 条 DISPATCH-HIT 全部是 rms_norm_mul_rope 约 5.9MB binding 与 blk.15 权重区间（q_proj/k_proj/ffn_gate_inp）的真实相交——**不覆盖探针区间，与 bisect 无翻转自洽**。基线形态（dump weightsA 全零）复现；teardown SIGABRT 已知形态。

### 18.2 r15 伪影判别轮（SYNCCMP，DevOps 自修权限内的最小判别插桩）

AMBIG2 的两个子假设（批次间隙覆盖 vs dump 读取路径本身）由 SYNCCMP 判别插桩裁定：在 weights dump 函数内、同一 (src.buffer, src.offset) 上追加 bisect 同款同步读（`ggml_vk_buffer_read`：submit + fence + 立即 memcpy）。

**决定性结果（同一函数、同一位置、同轮运行）：** 同步读（SYNCCMP）全部 20 行（2 site × 5 rank × down/up）非零——site1 down（blk.47.hc_ffn）`first2=3ba53d76 bcaabc41` 与 bisect P2 逐字节一致；site2 down（output_hc）`39b63b42 3d003c3b` 与 bisect P1 逐字节一致；up 两 site亦非零——而 **deferred 路径（原 dump 机制）读零（raw16 全 0）**。

**伪影机理：** weights 版 dump 的 `deferred_memcpy` 挂 `subctx->out_memcpys`，而 `ggml_vk_synchronize` 只冲刷 `in_memcpys`（实现只迭代 in_memcpys）；out_memcpys 的落地依赖 submit 后 fence 路径，dump 随即 `ggml_vk_ctx_begin` 重开录制——**host vector 保持零初始化**。

### 18.3 GAP 1 叙事的终局重估（本轮最重要）

- **「W 全零」自 r2 轮以来就是 dump 插桩自身的 deferred_memcpy 伪影。W 在设备上从未为零**——多重独立证据全部非零且数据恒定：r8 readback（1920 对一致）、r12 fence（写入时刻）、r13/r14 bisect（两轮各 600 次批次边界、r14 含三探针）、r15 SYNCCMP（dump 时刻同位置同步读）。
- **覆盖者不存在**（bisect 无翻转 + W2DA/DISPATCH T2 后零非法写入 + invalidate 0 + memset/subbuffer/BC 零 + DISPATCH-HIT 不覆盖探针区间）；**fork 不存在**（r13 四元组 + r15 同位置同步读）。
- r8「384 张量全局清零」、r9–r13「T1/T2 非零 → T3 全零的覆盖窗口」、r12「写后覆盖」、r13「选择性覆盖」——**全部是 dump 伪影的连锁误读**：T1/T2 的 readback 走同步路径读到真实非零，T3 dump 走 deferred 路径读到伪影零，「覆盖窗口」从未存在。
- **lo/mixed 的 dump 读数同样不可信**（tensor 版 dump 用同一 deferred 机制）；normalized 非零的机理存疑（sync_staging 残留候选，未验证）。

### 18.4 r2→r15 归因链修正表（哪些结论被证伪、哪些证据仍有效）

| 轮次 | 当时结论 | 第八轮终局重估 | 证据命运 |
|---|---|---|---|
| r2–r7 | W dump 全零（乱码根因线索） | **伪影**——deferred 读零非设备状态 | dump 读数不可信；normalized 非零读数机理存疑 |
| r8 | 写入成功（readback 1920 对一致）→「加载后全局覆盖」 | 写入成功**成立**；「覆盖」证伪 | 同步 readback 证据**有效**；dump 侧结论证伪 |
| r9 | host/GPU 路径分裂→「写入未到达 GPU 可见内存」 | **证伪**——数据到达且恒在位 | T1/T2 同步读数据**有效**；T3「零」是伪影 |
| r10 | meta registry 零效果（阴性） | **维持**——registry 与根因无关 | 阴性结果有效 |
| r11 | 非 HostVisible + staging 写入 + 双 vk_buffer 指针 | 非 HostVisible 与 staging 路径**成立**；双指针系打印语义伪差异（r13 已证） | staging 发现**有效** |
| r12 | fence 非零→「写后覆盖、窗口 T2→T3」 | fence 非零**有效**（写入到达）；「覆盖窗口」**证伪**（T2 同步读真实、T3 dump 伪影——窗口不存在） | 四元组/fence/dump 自检证据**有效**；情形 A 裁决证伪 |
| r13 | fork 排除 +「选择性覆盖」 | fork 排除**维持**；「选择性覆盖」**证伪**（三个位置设备上全部非零） | 四元组对照与 bisect 数据**有效**；选择性结论证伪 |
| r14 | AMBIG2（三探针无翻转、盲区闭合） | **正确**——无覆盖者的直接证据 | **全部有效** |
| r15 | SYNCCMP 判别 | **终局**：dump 伪影机理钉死 | **全部有效** |

**归因链的教训（供后续轮次引用）：** 观测工具与被观测对象共享执行路径时，工具缺陷会伪装成被观测系统的缺陷；r2–r13 的每一轮「阴性/翻转」都在真实证据与伪影读数的混合上推理。本轮之后，凡 dump 读数参与推理的结论必须先通过同步读交叉验证。

### 18.5 乱码根因重定向与下一轮方向

- **乱码（BF16 变体路径 HTTP 500）的真实根因回到 BF16 kernel 数值缺陷本身（未定位）**——r10 全新构建（含 upfold 修复）后仍乱而权重一直在位，上传链/覆盖/fork 假说域整体证伪。
- **下一轮方向：** 用同步读（SYNCCMP 同款，一行改动）修复 dump 插桩（deferred→同步读，weights/tensor 两版），重做单 site 中间值对拍（fused vs 3a 入口还原参照），拿到可信的 normalized/lo/mixed 数值后定位真正的数值分歧环节。
- **本轮不实施产品修复（无覆盖者可修——正确决定）**；dump 插桩修复留给下一轮对拍 charge。

### 18.6 诚实边界与裁决影响

- **off 臂三重判定连续第十二轮（r3/r5–r15）被前置 dump 复查阻断**——且 r2–r13 的阻断依据（dump 读零）本身是伪影；如实记录，不跳门。
- **GAP 2/4 残余/3/5 状态不变**（本轮未动）。
- **h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**；h1 的全部排名与资格证据（V5–V9、F/S 口径、保护集）从未依赖 dump 插桩读数，不受本轮伪影发现影响。
- 两轮 teardown 均 SIGABRT（插桩树已知形态）；插桩资产（watchtrace2 三修正 + SYNCCMP）保留在 rebuild 树。

## 十九、第九轮补录（2026-10-13）：r16 可信读数对拍轮

本节补录 r16（quarantine-sitedump-r16）。核对方式：`identity/quarantine-sitedump-r16/verdict.md` 全文（真值表、两臂崩溃记录、根因重指向与遗留）与 `ggml-vulkan.cpp.pre-refarm.bak`、`arms/sitedump-{fused,ref}-r16/`（manifest/mapped-libraries/result.json）存在性，均直接核验；与 Manager 交接清单一致。

### 19.1 dump 修复与 CPU 测试

dump-fixer 修复两版 dump：读取从 deferred_memcpy 改为 `ggml_vk_buffer_read` 同步读（移除 ensure_sync_staging_buffer / copyBuffer 录制 / has_pending_mem_work / deferred_memcpy，**保留 sync_buffers→synchronize→ctx_begin 完整配对**——正确性必需：dispatch 提交时序）。`tests/test-tp5-upfold-indexing.py` 新增 DumpReadbackTests（三个坏样本红路径：纯 deferred、混合残留、次序颠倒——回归能力：dump 再退回 deferred 形态时测试会红）；隔离树 `docs/TP5-HC-SITE-DUMP.md` r16 修订版。CPU 测试 34+46 全绿。编译两版：fused `6c9ee061…`、入口还原参照 `e3ebb7a0…`；对拍后树已恢复 fused 状态（备份回拷 + 重编验证哈希一致）。

### 19.2 首次可信读数（核心成果：GAP 1 叙事从设备侧真值上彻底关闭）

修复版 dump 在崩溃前完成 site 1/2 采集，SYNCCMP 交叉核对全部一致（raw16 ↔ SYNCCMP first2 逐字节对应）：

| 张量 | 真值（site1 全 5 rank 一致） | 判读 |
|---|---|---|
| normalized | 非零（0.232/-0.422/1.197…合理 RMS 量级） | 旧读数是真的——**staging 残留候选裁决：非残留** |
| **lo** | **非零（0.133/-0.163/0.442…合理收缩量级）** | **r2–r15 的「lo 全零」是 dump 伪影——down dispatch 实际一直在写** |
| **mixed** | **非零（0.204/-0.324/-0.476…）** | **「mixed 全零」同样是伪影——up/gate 段在写** |
| **W down/up** | **非零（与 r8 readback / r12 fence / r13–r15 bisect 全部一致的 BF16 位模式）** | **W 从未为零** |

**结论：「W 全零 → lo/mixed 零 → kernel 不写 → 乱码」的整个 GAP 1 因果链，至此从设备侧真值上彻底关闭**——与第八轮的 dump 伪影裁决互为印证（第八轮证明读数不可信，第九轮用可信读数证明真值非零）。

### 19.3 两臂崩溃与对拍未完成（新执行故障，原始失败保留）

- 两臂同崩于 `ggml-backend.cpp:349: GGML_ASSERT(offset + size <= ggml_nbytes(tensor)) "tensor write out of bounds"`——预热阶段（Prewarming TP5 decode/verify graph 之后的 router 路径），**请求从未发出**（result 全 NOT_COLLECTED、teardown -6）。DevOps 初读曾误判「fused 臂零 500 = 乱码消失」——实际是请求未发出，已在 verdict 修正（**误读过程如实记录**）。
- 参照臂（入口还原版，`arms/sitedump-ref-r16/`）崩溃更早且伴随 meta-chain-miss（stage=0 nodes=165）与 480 子图 CRITICAL-FALLBACK——与 3a 轮（upfold 树、正常运行）行为不同（rebuild 树多 r13–r16 插桩，插桩组合交互未定位）。
- **对拍（fused vs 参照数值分歧定位）未完成**——按「失败不重放」保留原始失败。

### 19.4 崩溃机理高嫌疑候选：dump 修复的 ensure 副作用

dump 修复移除了 `ggml_vk_ensure_sync_staging_buffer` 调用——旧版 dump（r13–r15）是 sync_staging 初始化的一个触发点；r16 新版不调用 ensure，预热/router 路径的其他用户在 staging 未初始化状态下使用（copy 到空 buffer → 越界断言）。**r13–r15 不崩（dump 活跃时 ensure 被调用）、r10 无插桩树不崩（乱码但不崩）——分界与该假设自洽。** 属 dump-fixer 修复的副作用缺陷（插桩级），需补回 ensure 或另立初始化保证。

### 19.5 乱码根因新候选（第十轮方向）：提交粒度 / barrier 缺失

fused 段数值在 dump 触发的 **mid-graph 分段提交**下非零且量级合理，而 r10（无任何插桩、**单大批量提交**）乱码——**乱码可能与提交粒度相关：fused 段 dispatch 序列缺少 barrier/依赖声明、单大批量提交下 GPU 乱序执行。同步类缺陷候选，而非纯数值缺陷候选**（数值缺陷候选降权：真值表显示数值合理——但未排除，对拍未完成）。

**验证实验（已写入 verdict，交 Manager 裁定）：** 在无插桩树上加「每 HC 段后 mid-graph synchronize」最小改动跑 off 臂——乱码消失则坐实同步根因。**注意：若坐实，修复方向是补 barrier/依赖声明（而非永久加 synchronize——性能代价）。**

### 19.6 r2→r16 归因链的第九轮增量

第八轮证伪读数（dump deferred 伪影）、第九轮真值确认（同步读下 W/lo/mixed/normalized 全部非零）——两轮合起来把 r2 以来的一切「设备侧权重缺失/覆盖」叙事从两个方向关闭。归因链的当前终点：**乱码根因的两个候选——(A) 提交粒度/barrier 缺失（同步缺陷，第十轮首选验证）、(B) BF16 kernel 数值缺陷（降权但未排除，对拍未完成）。**

### 19.7 诚实边界与裁决影响

- **off 臂三重判定连续第十三轮（r3/r5–r16）被前置阻断**——本轮的阻断原因是两臂崩溃（ensure 副作用），如实记录，不跳门。
- **本轮不实施产品修复：** 数值分歧定位未完成（对拍臂崩溃）、barrier 候选需独立实验验证。
- **GAP 2/4 残余/3/5 状态不变（本轮未动）；h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**——h1 的排名与资格证据不依赖 dump 读数，不受影响。

## 二十、第十轮补录（2026-10-13）：r17 双证伪轮

本节补录 r17（identity/quarantine-r17/）。核对方式：verdict.md 全文（ensure 确证实验、barrier 判别实验与推理修正、崩溃线与下一步），直接核验；与 Manager 交接清单一致。臂目录 `arms/sitedump-fused-r17/` 与 `arms/segmentsync-off-r17/` 由 verdict 引用。

### 20.1 r17 执行（segment-sync-fixer 两项改动）

- 两版 dump 函数的同步读路径前补回 `ggml_vk_ensure_sync_staging_buffer(ctx, bytes)`（幂等初始化、数据源不变仍同步读、注释注明 initialization only）。
- `ggml_vk_hc_segment` 函数尾新增 `GGML_VK_HC_SEGMENT_SYNC` 门控 mid-graph synchronize 实验开关（默认关零行为变化、replay/program 录制模式按设计跳过、三元配对）。
- `tests/test-tp5-upfold-indexing.py` 新增 SegmentSyncSwitchTests（4 测试含两红样本）+ ensure 正向 sentinel。CPU 测试初跑 3 失败——r17 注释里 `ctx->sync_staging` 字面量触发文本级判别，DevOps 在自修权限内改写注释措辞（**二进制零变化**）后 **39+46 全绿**。编译 `9b8890af…`（fused/segmentsync 共用）。

### 20.2 ensure 副作用假设证伪（十九节 19.4 候选关闭）

r17 fused 确证臂（r16 同配置重跑）：**崩溃仍在**——`ggml-backend.cpp:349 GGML_ASSERT "tensor write out of bounds"`（行 6779），崩溃链与 r16 完全一致（meta-chain-miss stage=0 nodes=165 → 480 子图 CRITICAL-FALLBACK ×7 → router 越界），请求从未发出（result 全 NOT_COLLECTED）。dump 读数可信性保持（SYNCCMP 20 行非零且与 bisect 一致、lo 与 r16 逐字节一致 0.132684/-0.163211/0.441517…）。**ensure 补回未消除崩溃——十九节 19.4 的 ensure 副作用假设证伪。**

### 20.3 barrier 假设证伪 + 第九轮推理错误修正（候选 A 关闭）

- segmentsync off 臂（默认 replay 模式、无 dump/watch/bisect）：**同崩溃且 [hc-seg-sync] 零触发**（开关在 replay 模式下按设计跳过）——实验本身无效；但本轮无 dump 无 sync 仍崩——**崩溃与 dump 调用和开关触发都无关、与树状态相关**。
- **DevOps 修正自己上轮（第九轮 19.5）的推理错误：** r13–r15（dump 旧版 deferred、含完整 mid-graph synchronize 三元配对）**乱码 HTTP 500 仍在**——「mid-graph sync 存在时执行正确」的观察从未成立；r16 的「无 500」是请求根本未发出的误读（该误读在 r16 verdict 已修正）。**有 mid-graph sync（r13–r15）与没有（r10）都乱——提交粒度/barrier 假设（候选 A）被既有数据证伪。**
- 第三次判别运行未跑——假设已被既有数据证伪，无需再验（预算节约的诚实说明）。

### 20.4 崩溃线新事实（独立于乱码线）与第十一轮方向

r16/r17 树四次运行（fused/参照/segmentsync）全部崩溃于同一链（meta-chain-miss → 480 子图 fallback ×7 → router tensor write 越界）；r13–r15 树六次运行零崩溃。**分界 = r16 dump 修复的树状态 + r16 轮同一 build 目录的「入口还原→恢复」交替编译。最强嫌疑：交替编译留下的 SPIRV/shader 新鲜度伪影**（与 r10 轮曾怀疑的 shaders-gen 增量重生成同类）。

**第十一轮方向：** 全新构建（删 build 目录全量重编、消除 SPIRV 新鲜度）复验——不崩则坐实编译伪影（崩溃线关闭）；仍崩则 diff 审查 r16 dump 修复代码的静态影响。

### 20.5 dump 工具的结构性矛盾（第十一轮需解）

dump 新版**读数可信但破坏图执行**（同步读在录制中间破坏命令缓冲状态 → meta-chain-miss → fallback → 崩溃）。乱码数值线的对拍需要「读数不破坏执行」的 dump 设计——例：请求完成后离线读、或把读移出录制路径。

### 20.6 r2→r17 归因链的第十轮增量

第九轮双候选 → 第十轮单候选：**数值缺陷（BF16 变体 kernel）成为唯一存留候选**；候选 A（barrier/提交粒度）证伪。第九轮 19.5 的推理错误修正记录在案——**误读链：r16「无 500」≠ 乱码消失（请求未发出）；r13–r15 的 mid-graph sync 存在时乱码仍在，「sync 存在时执行正确」从未被观察过。**

### 20.7 诚实边界与裁决影响

- **off 臂三重判定连续第十四轮（r3/r5–r17）被前置阻断**（本轮为两臂崩溃），如实记录，不跳门。
- **本轮不实施产品修复：** 唯一存留候选（数值缺陷）的对拍被崩溃线阻断，需先全新构建复验与「不破坏执行的 dump」设计。
- **GAP 2/4 残余/3/5 状态不变；h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**——h1 的排名与资格证据不依赖本轮任何诊断路径。

## 二十一、第十一轮补录（2026-10-13）：r18 复验与崩溃分界档案审查——分界不存在

本节补录 r18 全新构建复验与崩溃分界审查。核对方式：本轮文档工程师直接核验了 `arms/sitedump-fused-r13/server.log` 尾部（约 735k 字符的最后一行即断言）、`arms/sitedump-fused-r16/server.log` 的 warmup/listening/launch_slot 存在性、`arms/sitedump-fused-r13/result.json` 的 RemoteDisconnected、`arms/freshbuild-off-r18/` 存在性——**r13 日志以同一断言结尾这一核心证据由本轮直接读出**；其余审查结论（哈希一致性、静态审查细节、registry 嫌疑）按 Manager 交接清单记录并标注来源。

### 21.1 r18 全新构建复验（SPIRV 伪影裁决）

- 第一次全新配置漏 GGML_VULKAN=ON（**无效运行保留在 `arms/freshbuild-off-r18/` 作为配置错误记录**）；第二次补齐全参数真删 build 目录全量重编（RelWithDebInfo + GGML_VULKAN=ON + SPIRV-Headers）。
- **决定性哈希证据：全新构建 libggml-vulkan = `9b8890af` 与 r17 增量构建逐字节相同**——SPIRV/shader 新鲜度伪影假说被哈希一致性直接证伪（增量=全新，不存在旧 SPIRV 残留）；同理 r16 轮交替编译的恢复版哈希与原版一致——交替编译无伪影。第二十节 20.4 的「最强嫌疑」被证伪。
- off 臂复验：**仍崩（裁决 (c)）**——同链断言（meta-chain-miss → 480 fallback → router 越界）、请求全 NOT_COLLECTED、teardown -6。
- 对照修正：**fallback 非崩溃判别特征**——r10（乱码不崩）20 次 CRITICAL-FALLBACK、r13（崩）7 次、r18b（崩）5 次——所有运行都有 fallback，崩溃分界在 fallback 之后。

### 21.2 决定性发现：崩溃分界不存在

**r13 的 server.log 最后一行（约第 7959 行）就是 r16+ 的同一断言**（`cmd_child_to_router:error:…/ggml-backend.cpp:349: GGML_ASSERT "tensor write out of bounds"`——本轮直接核验）。崩溃链与 r16 fused 逐行同构：prealloc-realloc trigger=moe_route（moe=662572，五 rank 各一条）→ meta-chain-miss stage=0 nodes=165 → CRITICAL-FALLBACK cnt=5/6/7 → allreduce p1_wait 603–724ms 超长等待 → 同一断言 → SIGABRT -6。r13 的日志同时含 warmup completed 与 launch_slot_（请求到达）。**崩溃从 r12/r13 起一直存在**（r12 有 SIGABRT -6 与 request 未产出记录；r13 由本轮直接核验；r10/r11 待判别实验 1 确认）。

### 21.3 两份误读的揭露（假分界的成因）

- **(a) r13 verdict 的 teardown 归档惯例：** 把 -6 SIGABRT 归档为「与 r12 相同的插桩树 teardown 形态，非新异常」——这条归档惯例从 r12 起从未对过日志，把**请求处理中的崩溃**吞进背景（r13 result.json 的 error=RemoteDisconnected 即请求处理中连接断开，本轮直接核验其存在）。
- **(b) r16 verdict 的「请求未发出」误读：** 日志实际显示 warmup completed → model loaded → listening → 请求到达（launch_slot_）→ 崩溃在请求图计算中——NOT_COLLECTED 是请求未完成（连接断开）非未发出（本轮对 r16 日志的 warmup/listening/launch_slot 存在性直接核验成立）。r17 verdict 转述的「r16 无 500 是请求未发出的误读」这条修正本身也建立在同一误读上。
- 两份误读叠加制造了第二十节 20.4 的「r16/r17 树四崩 vs r13–r15 树六不崩」假分界——**该分界不存在，本轮推翻**。

### 21.4 r16 dump 修复无罪释放（静态审查）

静态审查确认：r16 改动全部在两个 dump 函数体内（deferred→同步读、移除 ensure；r17 补回 ensure + segment-sync 开关默认关）；out_memcpys/deferred_memcpy/ggml_vk_buffer_read 是既有基础设施（早期树 bf16-bak 就有）；push constants/vk_context 布局 r13–r16 间无变化（hc_dump_site_count 是 r13 加的字段、两边都有）；r14 的 g_gap1_graph_started 全局静态变量 r14/r15 树同样存在（非分界）。**「静态存在、不被调用的代码不产生行为差异」的直觉从头到尾正确——需要解释的不是代码，是两份 verdict 的误读。**

### 21.5 崩溃真实机理链（日志直接支撑）

每个图 tp5-meta-check valid=0（meta 链校验从未通过）→ meta-chain-miss → 480 子图 fallback；warmup 图（1 token、moe=0/0）不触发 moe_route prealloc realloc、能完成；**第一个 HTTP 请求图（n_sub=48、moe=662572）第一次触发 prealloc-realloc trigger=moe_route** → fallback 重算 → ggml_backend_tensor_set 写越界断言 → SIGABRT。「router 路径」= moe_route prealloc 重分配后的张量写（cmd_child_to_router 只是错误转发通道前缀，非 server router 进程）。崩溃与 dump 活跃与否无关（r13 fused 有 dump 也崩、r16 ref/r17 off/r18b 无 dump 也崩）。

### 21.6 崩溃引入点最强嫌疑与判别实验设计

**最强嫌疑：s_gap1_weight_simple_registry（meta.cpp，r10/r11 引入）**——无条件注册+查询、fallback 路径的 ggml_backend_meta_buffer_simple_tensor 优先返回注册指针；moe_route realloc 后的新张量若经 registry 解析到错误指针 → 越界。与 r10 rebuild（registry 引入前）「乱码不崩、请求能完成」相容。候选 3（弱）：r13/r14 的函数体外改动（g_gap1_graph_started、hc_dump_site_count 字段）。

**判别实验（DevOps 可执行，按优先级）：** (1) 崩溃起点定位（零改动纯读）——读 `arms/sitedump-fused-r10/r11/r12/server.log` 尾部，确认断言首次出现于哪轮；(2) 断言现场增强（一行插桩）——349 断言前打印 tensor 名/offset/size/nbytes，重跑默认 off 臂拿越界张量身份；(3) registry 门控实验——registry 注册与查询加 env 门控（默认关）重跑，不崩则坐实 registry 根因；(4) r14/r15 日志尾部确认（纯读——本轮 r14/r15 同形态是基于 -6 记录 + 同配置的强推断，未逐行读）。

### 21.7 审查边界与诚实边界

- **本轮只读未改未跑**；r14/r15/r17-off/r18b 的崩溃时点未逐行复核（采信 verdict 记录 + 断言相同）；meta.cpp fallback 与 registry 交互细节未逐行审（只确认 registry 存在于 .orig→当前约 110 行 diff）。
- **off 臂三重判定连续第十五轮（r3/r5–r18）被前置阻断**——新理解：r12 起的阻断原因是崩溃（此前未被识别，被 teardown 归档惯例吞没），r2–r11 的阻断原因是 dump 读零（伪影）；如实记录，不跳门。
- **乱码线与崩溃线两线独立：** 乱码线唯一候选（BF16 kernel 数值缺陷）不变，其对拍在崩溃线修复后恢复（需「不破坏执行的 dump」设计，见 20.5）。
- **GAP 2/4 残余/3/5 状态不变；h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**——h1 的排名与资格证据不依赖 rebuild 树上任何诊断运行。

## 二十二、第十二轮补录（2026-10-13）：r19/r20 崩溃线因果闭合——registry 坐实

本节补录 r19（纯读判别）与 r20（门控因果验证）。核对方式：`identity/quarantine-crash-origin-r19/verdict.md`（17 个 sitedump 系臂 + 基准的断言判读表）与 `identity/quarantine-registrygate-r20/verdict.md`（门控实验对照表与裁决）全文直接核验；与 Manager 交接清单一致。

### 22.1 r19 纯读判别：崩溃引入点 = r10 精确分界

- **判读表（完整表在 verdict）：** 基准臂（latebind-parity-off-rebuild，registry 引入前）零断言正常退出；sitedump-fused/ref 首轮与 r2–r9 九轮全部零断言、「cleaning up before exit」正常结尾；**r10 首次出现 ggml-backend.cpp:349 断言且断言即末行，r10–r18 九轮同崩同点**（r10 崩溃上下文与 r16+ 逐行同构——meta-chain-miss nodes 差异为树状态差异，断言同点）。
- **registry 嫌疑强化到最强：** r10 = forkfix 轮 = s_gap1_weight_simple_registry 引入轮——**崩溃引入与 registry 引入精确同步**；对照基准（registry 引入前）请求能完成。
- **DevOps 第三次自我修正：** r13–r15 的「进程活着出乱码」判断错误——它们的 server.log 同样以断言结尾（乱码 500 与崩溃共存：请求发出得到 500 后进程崩）。**dump 新旧只影响崩溃时点与 500 有无，不影响崩溃本身**——这解释了此前在 r13–r15 看到 500 时误判进程存活。
- **完整时间线修正：** r2–r9（dump 旧版伪影时代）乱码 500 不崩 → r10–r15（registry 活跃 + dump 旧版）乱码 500 + 崩溃 → r16–r18（registry 活跃 + dump 新版）崩溃无 500（崩溃时点提前）→ r20（registry 旁路）乱码 500 不崩。r18 哈希一致性结论维持（崩溃是 r10 起的确定性二进制行为，与构建方式无关）。

### 22.2 r20 门控因果验证（判别实验 2——裁决 (a) 坐实）

registry-gate-fixer 落地门控：隔离树 meta.cpp 的 `ggml_backend_meta_registry_disabled()`（`GGML_VK_META_REGISTRY_DISABLE=1` 激活、静态缓存一次读取、[meta-registry-bypass] stderr）+ 查询端守卫（simple_tensor 的 registry 优先返回处）+ 注册端守卫（MIRRORED fan-out 登记处）——双端旁路 = 等价 r9 树该部分行为；RegistryGateTests 6 项（含两红样本）；编译载体 libggml-base `246012b1…`。

**门控实验（`arms/registrygate-off-r20/`，与 r18b 完全同配置、唯一新增门控 env）：** bypass 触发 1 次确认；**断言从 1 次降为 0——崩溃消失**；进程末态从断言即末行变为正常「cleaning up before exit」；请求从未发出变为发出并完成（乱码 500、正文 hash `20c0ddaa…` ≠ baseline `c2d7bdd3`）；CRITICAL-FALLBACK 5→10。

### 22.3 三方因果闭合（决定性）

- r9 树（无 registry）：乱码 500、不崩、正常退出。
- r10–r18 树（registry 活跃）：崩溃（r10–r15 500+崩溃共存；r16+ 崩溃且请求未发出）。
- r20 树（registry 门控旁路）：乱码 500、不崩、正常退出——**精确回到 r9 形态**。

**裁决：s_gap1_weight_simple_registry（r10 引入的 fallback 路径注册指针优先返回）确认为崩溃根因**——fallback（480 子图）路径的 simple_tensor 解析经 registry 返回注册指针后，moe_route prealloc realloc 后的 router 路径对其执行越界 tensor write 触发断言；旁路后回到 r9 行为。

### 22.4 崩溃线关闭 + 乱码线解锁

- **registry 的存在理由已被抽空：** 当初为诊断「W 全零」引入，r15 已证 W 全零是 deferred 伪影——修复方向大概率直接移除（隔离树插桩，仓库工作树无此代码；连同 r10/r11 forkfix 相关插桩一起清理）。
- **乱码线解锁：** 唯一候选（BF16 kernel 数值缺陷）的对拍最大障碍（进程崩溃）已解除——r20 旁路树下乱码 500 维持 r9 形态，可在 registry 旁路/移除树上重做 fused vs 参照的单 site 数值对拍。
- **「dump 新版破坏图执行」需重审：** 崩溃根因既归 registry，r16 dump 新版可能无辜——第二十节 20.5 的 dump 结构性矛盾叙述待对拍实践修正（若对拍顺利，dump 设计无需重造）。

### 22.5 测试配套缺陷与推断边界

- **测试两处缺陷记录在案（交 Engineer）：** (1) RegistryGateTests 未挂进 __main__ 自定义 runner（39 项不含它们，仅 unittest 发现可跑）；(2) 红样本 test_red_sample_env_name_removed 失败——判别函数 _registry_gate_property_holds 对 env 名重命名 mutation 不敏感（mutation 后仍返回 True）——判别函数缺陷，不影响门控行为本身（正样本全绿、门控代码在树并编译进库）。
- **diff 复核边界：** r9→r10 无树存档直接 diff——r10 改动清单基于 quarantine-forkfix/verdict.md 推断（registry 是 verdict 记录的唯一代码改动）——推断边界如实标注；但 r20 门控实验的因果验证已独立闭合（无需 diff 兜底）。

### 22.6 r2→r20 归因链第十二轮增量

- **第二十节 20.4 的「树状态/交替编译」嫌疑链被 r19/r20 关闭**（崩溃 = r10 起的 registry 因果，与构建方式无关——r18 哈希裁决维持并获新解读）。
- **第二十节 20.5「dump 新版破坏图执行」待重审**（dump 可能无辜，见 22.4）。
- **二十一节 21.2 的「崩溃从 r12/r13 起」修正为「从 r10 起」**（r19 判读表：r10 首次断言、r10–r18 九轮同崩；r13 日志的直接核验维持有效）。
- 完整时间线（r2–r9 / r10–r15 / r16–r18 / r20 四段）见 22.1。

### 22.7 诚实边界与裁决影响

- **off 臂三重判定连续第十六轮（r3/r5–r20）被前置阻断**——崩溃障碍已解除，下一轮阻断只剩乱码本身；如实记录，不跳门。
- **本轮不实施 registry 移除**（隔离树插桩清理交 Engineer，含测试配套两缺陷）。
- **乱码线与崩溃线关系终结：崩溃线关闭后，乱码线（BF16 kernel 数值缺陷）成为唯一主线。**
- **GAP 2/4 残余/3/5 状态不变；h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关。

## 二十三、第十三轮补录（2026-10-13）：r21 registry 移除验证——崩溃线终局关闭

本节补录 r21（quarantine-registryremoved-r21/）。核对方式：`identity/quarantine-registryremoved-r21/verdict.md` 全文（移除复核、四臂对照表、等价性与裁决）直接核验；registry-remover 的移除清单与测试修复细节按 Manager 交接清单（源自其交付报告）记录并标注来源。与交接清单一致。

### 23.1 registry 移除（registry-remover）

隔离树 meta.cpp 的 registry 四触点三块 **51 行干净删除**：s_gap1_weight_simple_registry 定义 + ggml_backend_meta_registry_disabled 门控 helper + 查询端 fast-path + 注册端登记块——**grep 复核四标识符零命中**。保留项在位：stc 容器查找、r5 container routing guard、phasestrace、dump/watch 插桩（r20 门控实验已证明 dump 插桩与崩溃无关）。编译一次通过（libggml-base `dab9d159…`——registry 移除载体；libggml-vulkan 9b8890af 不变）。

### 23.2 测试配套两缺陷修复（registry-remover）

- **(a) __main__ 入口根因修复：** 原入口在文件中段 L942，其后定义的测试类（UploadChainAuditTests / ContainerRoutingAuditTests / RegistryGateTests）永远不会被 unittest 发现——入口移至文件末尾 L1153，全部纳入自动发现。
- **(b) RegistryGateTests（6 项）改造为 RegistryRegressionTests（4 项回归守卫）：** 断言隔离树 meta.cpp 中 registry 四标识符不存在——**旧机制复活时测试变红报警**；判别函数 _registry_absent_property_holds 检查完整机制标识符集合（map + helper + env 名 + 日志标记）而非单一 env 名；两个红样本：(1) 注入 registry map 定义、(2) registry 以全新 env 名复活（GGML_VK_META_REG_OFF——**原缺陷的回归编码**：r20 版判别函数对 env 重命名 mutation 失明，全标识符判别下无论 env 名如何改必然变红）。
- **CPU：49 项全绿**（原 39 + UploadChain 3 + ContainerRouting 3 + RegistryRegression 4——预期数精确达成）+ gap-tools 46 项回归无破坏。

### 23.3 r21 验证与等价性（arms/registryremoved-off-r21/）

**off 臂复验（默认配置、无任何 env 开关）：**

| 观察 | r18b（registry 在） | r20（门控旁路） | **r21（物理移除）** |
|---|---|---|---|
| ggml-backend.cpp:349 断言 | 1（崩溃） | 0 | **0** |
| 进程末态 | 断言即末行 | 正常退出 | **正常 "cleaning up before exit"** |
| 请求 | 未发出 | 完成（乱码 500） | **完成（乱码 500：peg 2 次）** |
| CRITICAL-FALLBACK | 5 | 10 | **10** |
| 正文 hash | — | 20c0ddaa0a45a3f5… | **20c0ddaa0a45a3f5…（逐字节一致）** |

**等价性运行时确认：** 门控旁路（r20）与物理移除（r21）产出完全相同行为——乱码正文逐字节相同、fallback 计数相同、退出形态相同。

### 23.4 崩溃线终局关闭（四臂因果链）

**r9 树（无 registry）不崩 → r10–r18（registry 活跃）崩 → r20（门控旁路）不崩 → r21（物理移除、默认路径）不崩。** 移除干净双重保障：grep 零命中 + 4 项回归守卫绿。乱码与崩溃确认为两条独立故障线（r10–r15 曾共存；r21 崩溃消除后乱码 500 仍在——预期形态，数值缺陷线未修）。

### 23.5 乱码线解锁（对拍通路打开）

- 崩溃消除后 fused vs 入口还原参照对拍的最大障碍解除。
- **dump 无辜性可在对拍实践中确认：** 崩溃根因既归 registry（r10 起就存在），r16 fused 臂的崩溃是 registry 而非 dump——dump 新版很可能无辜，对拍可在 dump 活跃下进行（r16 fused 臂的 dump 读数在崩溃前采集且经 SYNCCMP 交叉核对有效）。
- **参照臂（入口还原）在 r16 的崩溃同属 registry 根因，registry 移除后应可正常运行——对拍两臂通路均已打开。**

### 23.6 诚实边界与裁决影响

- **off 臂三重判定连续第十七轮（r3/r5–r21）被前置阻断**——**崩溃维度已自动通过（r21 默认路径零断言），阻断只剩乱码本身**；如实记录，不跳门。
- **GAP 2/4 残余/3/5 状态不变；h1 的 PROMOTE 状态（`BEST_KNOWN_NOT_PROVEN_OPTIMAL`）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关。
- r2→r21 归因链第十三轮增量：崩溃线完整闭环（引入 r10 → 定位 r19 → 因果坐实 r20 → 移除验证 r21）；修正史见 22.6 与本节。

## 二十四、第十四轮补录（2026-10-13）：r22 fused vs 参照数值对拍——乱码分歧首次定位

本节补录 r22（identity/quarantine-parity-r22/ 与 arms/parity-fused-r22/、arms/parity-refentry-r22/）。核对方式：identity/quarantine-parity-r22/verdict.md 全文、两臂 request-0.json（输出正文与 timings）、两臂 result.json（运行身份与 GPU 状态）直接核验；树恢复、dump 文件计数与 SYNCCMP 行数按 verdict 记录。与 Manager 交接清单一致，仅一处过程性细节的记载位置差异（见 24.3 末注）。

### 24.1 两臂运行状态与树恢复

两臂顺序运行、同树同配置（--warmup 0 --repeats 1 --max-tokens 32、GGML_VK_CMD_REPLAY=0、GGML_VK_HC_DUMP_SITES=1）：

- **fused 臂**（arms/parity-fused-r22/，bin 9b8890af——树默认状态）：不崩（断言 0）、**正常退出（teardown returncode 0——sitedump 系列首次干净退出、非 SIGABRT）**、dump 采集成功（25 个文件）、SYNCCMP 交叉核对 10 行非零。
- **参照臂**（arms/parity-refentry-r22/，bin 2a525504——入口还原：q8_projection 的 BF16 条件还原为纯 Q8_0 的 14 行临时改动）：不崩、正常退出、请求完成、**dumps=0（预期——入口还原后 fused 段不激活，dump 挂 fused 路径采不到）**。
- **树状态恢复**：对拍后入口还原从备份恢复，重编哈希 9b8890af = fused 开放状态（与对拍前一致，diff 验证）。

### 24.2 dump 无辜性正式确认（第二十节 20.5 修正）

registry 移除后 dump 活跃下进程完全健康（不崩 + returncode 0），且 **dump 读数与 r16 逐字节一致**（normalized 0.232353… / lo 0.132684… / mixed 0.203644…——数值跨轮稳定可复现，SYNCCMP 交叉核对有效）。r16 轮「dump 新版破坏图执行」的归因确系 registry 误判——第二十节 20.5 的「dump 结构性矛盾」叙述正式修正：dump 同步读设计无需重造，对拍可在 dump 活跃下进行。

### 24.3 对拍逐字分歧首次定位

| 臂 | finish | 32 token 输出 |
|---|---|---|
| fused（fused 段激活） | length | **「健康做，，，，，，，，且，，，在rata，让我，rata，FP，，rata，，rata，且，，」**（乱码） |
| 参照（入口还原，fused 段不激活） | length | **「健康减重并非靠极端节食，而是建立可持续的生活习惯。在饮食方面，优先选择高纤维的蔬菜、优质蛋白和全谷物，」**（正常中文——c2d7bdd3 的已知开头形态） |

**首个分歧 token：前 2 个 token「健康」逐字一致，第 3 个 token 分歧（fused=「做」 vs 参照=「减」）。**

**peg 校验细节的核对说明（出入如实记录）：** Manager 交接清单称「fused 臂 peg=0 仅因 32 token 太短未触发格式校验，DevOps 初判『乱码消失』已在判读中修正」——本轮直接核对的文件中未见该过程细节的独立记载（verdict.md 全文无 peg 字样；两臂 request-0.json 的 finish_reason 均为 length、completion_tokens 均为 32；两臂 result.json 的 quality_verdict 实测为 NOT_COLLECTED——与 32 token 诊断臂质量门未参与判定一致）。乱码判定本身不依赖该细节（输出正文直接可见乱码）；该过程性细节按交接清单记录（源自 DevOps 判读修正过程）并标注来源，不入直接证据。

### 24.4 间接对拍推断链与边界（如实标注）

参照臂采不到 dump（fused 段不激活）——**逐张量对拍结构性不可行，分歧定位是间接的**（正文分歧位置 + 两臂唯一差异 = fused HC 段激活）。推断链：(1) 第 3 个 token 依赖前 2 个 token 的 KV 与 HC recurrent 状态——分歧起点在 prefill 尾部至第 2 个 decode 步之间的 HC 段计算；(2) fused 臂 dump 真值显示 lo/mixed 非零且量级合理——**量级合理 ≠ 数值正确**（浮点错误不改变量级）；(3) 两臂唯一激活差异是 fused HC 段的 BF16 变体（qwen4_hc_down_silu_bf16 / mul_mat_vec 的 BF16 路径）——**lo/mixed 的具体数值错误（量级对值错）是分歧的最强解释**。与 r10 起的乱码形态、3a 轮「入口还原=数值正确」的历史证据全部自洽。**此为推断而非逐值证明**——逐值证明需要「参照臂 fused 段等价张量」的采集设计（受 dump 挂 fused 路径的结构性限制）。

### 24.5 逐值验证通路与下一步（kernel 级定位）

dump 真值可信且跨轮稳定（r22 与 r16 逐字节一致）——**CPU 参照计算 vs dump 真值的逐元素对拍可行**。kernel 级定位交 Engineer 静态审查：分歧环节收窄到 fused 段的 down（lo 产出）或 up/gate（mixed 产出）收缩计算——qwen4_hc_down_silu_bf16 / mul_mat_vec.comp 的 BF16 路径（iter_aligned_nonquant 位展开、vec4 B 读取、HC_UP_FOLD 折叠——upfold 修复后仍乱，嫌疑重心在 down 端或折叠之外的 BF16 解码细节）。

### 24.6 诚实边界与裁决影响

- **off 臂三重判定连续第十八轮（r3/r5–r22）被前置阻断**——崩溃维度已通过（r21），阻断只剩乱码数值缺陷本身；如实记录，不跳门。
- 间接对拍的推断边界如实保留（24.4）；「最强解释」不写成「已证根因」。
- GAP 3 / GAP 4 残余 / GAP 5 状态不变；**h1 的 PROMOTE 状态（BEST_KNOWN_NOT_PROVEN_OPTIMAL）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关；理论 L = UNRESOLVED 不变。
- **r2→r22 归因链第十四轮增量**：dump 无辜性修正（r16「dump 破坏图执行」归因确系 registry 误判——第二十节 20.5 叙述正式修正）；乱码线从「唯一候选 BF16 kernel 数值缺陷（对拍通路打开）」推进到「**分歧首次定位（第 3 token / HC 段 / BF16 变体 lo-mixed 值错——最强解释，间接对拍）**」——kernel 级逐值定位为下一主线（见第 9 项）。

## 二十五、第十五轮补录（2026-10-13）：S1/S2 静态排除 + CPU 对拍被 GPU 阻塞

本节补录 r23（identity/quarantine-cpuparity-r23/）与 S2 静态深查（artifacts/perf-gap/hc-bf16-s2-verdict-r15c-20261013.md；r15a 原方案 hc-bf16-audit-r15a-20261013.md 同目录在案）。核对方式：r23 verdict 全文与 r15c 裁决文档全文直接核验（r23 含 GPU 阻塞事实与 S1 三层证据；r15c 含 S2 五层证据、盲点修正与嫌疑重排）。与 Manager 交接清单一致，无出入。时间线说明：r23 verdict 写作时 S2 尚未排除（列为最强剩余嫌疑），r15c 随后将其排除——本节按 r15c 之后的最终格局落笔。

### 25.1 GPU 被外部进程占用（本轮执行边界）

- fused 臂运行失败于 runner 前置检查（GPUs not idle）；占用者 **strata（PID 1450245，5 卡 71.4GB VRAM，非我方进程）**。按 AGENTS.md 红线 2 停止（不杀别人进程、遇冲突停止自己的计划）。
- **CPU 参照对拍（numpy 复算）与 FORCE_ITER 判别均未执行**——对拍需全量 dump 采集（GPU 运行）；如实记录，待 GPU 释放。
- dump 全量增强已落地：weights dump 的 copy_bytes 从 64 元素改为全量 subbuffer（r23 插桩级修改——满足方案文档「dispatch 实际绑定权重」的全量需求），编译成功（libggml-vulkan 9a69607d）——待采集。

### 25.2 S1 静态排除（决定性三层证据）

S1 假说（host 侧 push constant 单位契约随 QUANT_K 变化——BF16 的 A 索引错位 32 倍或 2 倍）：

1. **变体注册**（vulkan-shaders-gen.cpp:1133-1147）：qwen4_hc_down_silu_bf16 / qwen4_hc_project_fold_bf16 宏字典只含 DATA_A_BF16/B_TYPE/D_TYPE/HC_DOWN_SILU/HC_UP_FOLD——**不含 QUANT_K**（无量化参数注入）。
2. **QUANT_K 语义实锤**（vulkan-shaders-gen.cpp:1130 注释原文，types.glsl:43-45 同证）：「QUANT_K=1/QUANT_R=1 via DATA_A_BF16」——BF16 变体为元素语义，A 索引 ib=(weight_row*ncols+col)/QUANT_K = 元素偏移（正确）。
3. **push constants 单位一致**（ggml_vk_segment_project:16970）：pc{k, k, k, m, nelements, k, m, …}——k=weights->ne[0]（元素单位）、stride_a/b=k、batch_stride_a=nelements，与 shader 侧 B 的 blocks_per_row（B_QUANT_K=32 仅 HC_DOWN_Q8_DOT 变体定义，BF16 变体不涉及）及 A 的元素索引语义全部一致；K_PER_ITER 非量化族（F32/F16/BF16）=4——元素单位自洽。

**S1 排除：不存在 32 倍/2 倍错位。**（边界注意：r23 排除的是单位契约，不是 host 填入的具体数值——值的正确性归 H1。）

### 25.3 S2 静态排除（五层证据链）

S2 假说（iter_aligned_nonquant 的 dequantize4_2aligned packed32 索引错位——(a_offset+ib)/2 整除截断 + iqs 元素内偏移语义 + B/A 列偏移分歧）：

1. **iqs=0 硬编码**（mul_mat_vec.comp:403-406）：iqs 不是从 col 推导而是常量 0；B 读取（line 408）与 A 传参（line 430）引用**同一个 iqs 变量（同作用域、同值 0）**——「B 用列相关值而 A 用 0」的分歧不存在。
2. **实现体内 iqs 未用**（dequant_funcs.glsl:54-61）：BF16 位展开只用 (a_offset+ib)，iqs 仅存于签名（为与量化类型共享接口保留）；BF16 的 QUANT_K=1 无块内偏移概念——调用侧传 0 与实现的忽略互相印证、无错位源。
3. **A/B 同 col 配对**（两分支核对）：HC_DOWN_SILU 的 ib=(ibi+col)/QUANT_K 与 HC_UP_FOLD 的 ib=weight_row*ncols+col——A 读权重 col..col+3 列、B 读激活 col..col+3，dot(v,b) 配对正确；B 的 vec4 整除由 is_aligned_nonquant 门槛（batch_stride_b/b_offset/ncols 全 4 倍数）保证无跨 vec4 错位。
4. **packed32 无奇偶截断**：col 是 4 的倍数、ncols=hc_dim=4*n_embd（偶）、row*ncols 偶、a_offset 偶（batch_stride_a=行数×ncols，偶）——**a_offset+ib 恒偶，(a_offset+ib)/2 无整除截断错位**。
5. **两路径逐位等价（FORCE_ITER 判据的理论基础）**：bf16_to_fp32（types.glsl:1966）= uintBitsToFloat(u<<16) 与 dequantize4_2aligned 的位展开（packed32[i] 低 16 位=元素 2i、高 16 位=元素 2i+1）同构；lane→K 切片两路径同公式、HC_UP_FOLD/HC_DOWN_SILU 行映射两路径同码、dot 累加同序、reduce 同函数——**同一 buffer、同一位展开、同一元素序，两解码路径逐位等价**——与第二轮 V-A 运行时观察（FORCE_ITER 开/关乱码逐字相同）互洽：静态证明解释了当时的运行时现象。

**S2 排除（源码语义层）。** 诚实边界（r15c 原文）：静态排除的是**源码语义层的错位机理**（参数不匹配、iqs 分歧、packed32 截断、A/B 列错配）；静态分析不能裁决：(a) 驱动/编译器对 aligned 路径实际生成的机器行为——理论逐位等价 ≠ 实测等价，且 HC BF16 是全树仅有的 mul_mat_vec+DATA_A_BF16 实例（无独立执行证人）——即 H3；(b) host 填入 push constant 的具体数值与 descriptor 绑定（r23 排除的是单位契约，不是值的正确性）——即 H1。

### 25.4 r15a 方案盲点发现（重要修正）

r15a 的 CPU 参照方案把 normalized 只当输入锚点、**从未验证自身正确**——若上游（residual/rms_norm）错，用它复算 lo 会误判 down kernel（冤案风险）。修正（r15c）：对拍前先验证 normalized——用前一层的 residual dump + GGUF hc_attn_norm 权重在 CPU 复算 normalized 对拍 dump normalized；**若 normalized 已错直接定罪上游（H2），HC 段无罪释放**；若 normalized 对，再按 r15a 流程对拍 lo/mixed。

### 25.5 形态再权衡与剩余嫌疑重排

- **形态分析**：「前 2 token 对、第 3 token 乱」对状态累积型上游错（H2）的兼容性优于已排除的 S1/S2——确定性 kernel 错位从 token 0 就错（「前 2 token 对」需要错误恰好不改 argmax 的巧合）；**状态逐 token 漂移的上游错误天然解释「前面对、第 3 个乱」**。
- **剩余嫌疑重排（S1/S2 排除后）**：
  1. **H1（最高）**：host 侧 ggml_vk_segment_project 的 push constant 值组装与 descriptor 绑定——stride_d/batch_stride_b/base_work_group_y/ne02/ne12/broadcast2/broadcast3 的具体值、A/B/D/fuse0 绑定的实际 buffer 与 offset。**静态缺口跨三轮遗留（r15a 列为第一优先、r23 排除的是单位契约而非值、r15c 预算仍未读到该函数体）——唯一未被静态审过的 host 环节。**
  2. **H2（高）**：normalized 输入锚点本身的正确性（上游 residual/rms_norm 的状态累积型错误）——形态兼容性更优 + r15a 盲点佐证。
  3. **H3（中）**：驱动/编译器对 aligned 路径的实际行为（vec4 B 读取 + packed32 位操作在 RADV 上的生成代码）——FORCE_ITER（GGML_VK_HC_BF16_FORCE_ITER=1）一发 A/B 裁决。
  4. **H4（中）**：up 段 host 绑定（data_fuse0 布局、up 权重 rank 切片）——与 H1 同属 host 侧，合并审查。

### 25.6 运行时判据优先级（GPU 释放后）与诚实边界

- **判据优先级**（r15c）：(1) **CPU 参照对拍（r15a + 本轮盲点修正版）**——前置验证 normalized（25.4），再对拍 lo/mixed（判读规则：相对误差 <3e-3 → 正确；>5e-2 或结构性模式 → 定位）；(2) FORCE_ITER 对照臂（H3 判别——注意 S2 已排除，FORCE_ITER 的角色是裁决编译器 aligned 行为而非 S2）；lo 对拍失败时的归因顺序：**先查 normalized（H2）→ host 参数（H1）→ kernel（FORCE_ITER 区分 H3）**。
- **off 臂三重判定连续第十八轮（r3/r5–r22）被前置阻断 + 本轮 GPU 外部阻塞**（strata 非我方进程、按红线 2 停止——如实记录，不杀别人进程）。
- **静态排除 ≠ 运行正确**：S1/S2 排除后乱码根因仍未钉死（H1/H2/H3 均未排除也未证实）；「最强解释」的重心从 kernel 数值缺陷转移至 host 值组装（H1）与上游状态累积（H2）——均为候选，非定论。
- GAP 3 / GAP 4 残余 / GAP 5 状态不变；**h1 的 PROMOTE 状态（BEST_KNOWN_NOT_PROVEN_OPTIMAL）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关；理论 L = UNRESOLVED 不变。
- **r2→r23 归因链第十五轮增量**：乱码线的 kernel 侧静态假说域（S1 单位错位、S2 索引错位）经两轮静态深查清空；嫌疑重心从 kernel 转移到 host 侧（H1——唯一未审 host 环节）与上游（H2——形态兼容性更优 + normalized 盲点）；CPU 参照对拍因 GPU 外部阻塞未执行（判据已备：全量 dump 增强编译就绪 + normalized 前置验证方案）。

---

## 二十六、第十六轮补录（2026-10-13）：H1 静态排除 + r24 CPU 参照对拍——数值级实锤

数据源：`identity/quarantine-cpuparity-r24/verdict.md` 与 `artifacts/perf-gap/h1-segment-project-verdict-r16-20261013.md`（两份均经本轮直接核对，与 Manager 交接清单一致）；臂目录 `arms/parity-fused-r24/`。

### 26.1 H1 静态排除（h1-host-auditor）与时间线矛盾发现

**H1 静态排除**（`ggml_vk_segment_project` 函数体，ggml-vulkan.cpp 16938–17020，完整核对表）：

- **push constant 13 字段全匹配**：结构（1491–1505）vs host 填充（16975–16977）`pc{k, k, k, m, nelements(W), k, m, flags, 0, 1, n_rows, n_rows, 1}`——ncols/stride_a/stride_b=k、stride_d/batch_stride_d=m、batch_stride_a=nelements(W) 且 **batch_idx_a 恒 0**（get_offsets 推导：i13=0、i12=batch_idx、i02=0——权重不随 token 变的正确语义，a_offset=0、QUANT_K=1 无缩放）。
- **k/m 无填反**：k 恒取 weights->ne[0]（收缩维）、m 恒取 output->ne[0]（输出维）——down：k=hc_dim、m=hc_lr_slice（=320）；up：k=320、m=width。任务点名的「k/m 填反或错误 ne 索引」不存在。
- **descriptor 绑定正确**（17001–17003）：down 的 A=W_down / B=normalized / D=lo / fuse0=out 占位（extra=nullptr，down 不消费 fuse0）；up 的 A=W_up / B=lo / D=mixed / fuse0=normalized。subbuffer offset/size 有运行时直接证据（W_down {off=1049041280, size=6553600=320×10240×2}——BF16 全量字节宽度精确匹配张量形状）。
- **spec constants 正确**：down NUM_ROWS=1（6674–6676）、fold NUM_ROWS=4（6677–6679）——first_row 覆盖精确；y 维语义（down 的 wg.y=batch_idx 与 fold 的 wg.y=token_row）同构（都是 token 索引），差异在 fold 的 4× 因子（dispatch 约定，与 Q8_0 twin 对齐——Q8 生产证人）。
- **4*n_rows 张力定位为非缺陷**：y∈[T,4T) 的 over-dispatch，无害性依赖 robust buffer access（越界读零、越界写丢弃），`docs/TP5-BF16-UPFOLD-FIX.md` §3.3 已登记并交单 site 对拍裁定；若对拍后确认 4T 冗余，收窄为 n_rows 是零风险性能优化，不是正确性修复。

连续三轮的静态缺口（r15a 提出、r15c 复列、本轮闭合）至此清偿。**H1 出局。**

**时间线矛盾发现（本轮审查的副产品，交 r24 裁决）**：`TP5-BF16-UPFOLD-FIX.md` 记载 R2 轮曾实锤「W 全零」→ mmap 修复 → r6/r7 复查仍全零 → 归因「写入可见性层」；但 r22 判读 lo「量级合理」——若 W 真为零则 lo 必零，两者矛盾。该矛盾由 26.2 的 W 正确性终局裁决消解。

### 26.2 GPU 释放与 W 权重正确性终局裁决

- **GPU 释放确认**（strata 退出、KFD 进程列表空）；r24 fused 臂（`arms/parity-fused-r24/`）一次运行成功：不崩（断言 0）、全量 dump 采集成功（weightsA 6553600 bytes × 2 = 3276800 u16 全量——r23 增强生效）、bin 9a69607d（树状态确认无需重编）。
- **W 非零**：全量计数非零、first8=[15734,15269,48193,48298,48309,15524,15572,15622]——与 r16/r22 SYNCCMP 一致。
- **W 与 GGUF 逐字节一致**：blk.47.hc_ffn_down（ne=(10240,320)）与 hc_ffn_up（ne=(320,10240)）的 GGUF 原始字节 vs dump 前 4096 u16 **完全相同**——dispatch 绑定的权重视图 = 原始权重，无值错、无错位。
- **时间线矛盾终局裁决**：r6/r7 的「W 仍全零」历史观察是 dump deferred 伪影（r15 已证）；r22 的 lo/mixed「量级合理值错」判据有效——**W 在设备上从未为零，权重值正确**。写入可见性（V）嫌疑排除。
- 附带：手写 GGUF v3 解析器修正（type 8 = v3 string 非 uint64——v2/v3 类型编号差异，工具级修正）。

### 26.3 lo 对拍（down 段）分歧实锤

CPU 参照对拍（纯 python，公式与 AGENTS.md 工单 B 数学一致；形状自洽：norm=[10240]=4·width（width=2560）、W_down 320 行×10240（ne[0]=10240 连续）、lo=[320]、W_up ne=(320,10240)、mixed=[2560]=width）：

- **lo_ref=SiLU(0.25·W_down@norm) vs dump lo——巨大分歧**：lo[0] dump=+0.132684 vs ref=-0.014946（**符号相反**）；前 32 输出 worst rel err 3.61。
- **6 种公式变体全部不匹配**（0.25 位置/无 SiLU/4 倍缩放等——maxerr 0.44–26）：排除公式歧义。
- dump lo 与 r16/r22 **跨轮逐字节一致**：确定性错误，非随机。

### 26.4 mixed 双重判别（up 段）——自身也错

- 用 dump lo（kernel 错值）作输入：mixed worst rel 2.59（mixed[1] 差 2.6e-3——个别坐标碰巧接近）。
- **用 lo_ref（CPU 正确值）作输入重算：mixed worst rel 4.80——仍大分歧**——**up/gate 段自身数值也错**（不只是 down 错误的传播）。

### 26.5 综合裁决：两个 BF16 kernel 数值计算错误实锤

1. **W 正确**（GGUF 逐字节一致）——权重值错/绑定错位排除（V 排除）。
2. **normalized 假设成立（H2 排除）**：入口还原对照臂 fused 不激活即输出正确——norm 段两臂共享，norm 错无法解释该对照。
3. **lo 数值错（down 段）+ mixed 数值错（up 段，正确输入下仍错）**——**fused HC segment 的两个 BF16 变体 kernel（qwen4_hc_down_silu_bf16 与 qwen4_hc_project_fold_bf16）数值计算错误实锤**。乱码根因的运行时数值级定位完成：分歧不在权重、不在 norm、不在公式歧义——在两个 BF16 收缩 kernel 的计算过程内部。两 kernel 同错指向**共享的解码/累加路径**而非各自独立缺陷。
4. 与 r22 的「第 3 token 起分歧」自洽（HC 段整体数值损坏 → 状态污染从首个 decode 步开始）。

**从「最强解释间接推断」升级为「CPU 对拍直接证据」**（对照 24.4 的间接对拍边界——r23 全量 dump 增强使逐值对拍在本轮可行）。

**判读边界（如实标注）**：(a) 参照公式取自方案文档（bf16-kernel-auditor 从 qwen4exp.cpp 总结）+ AGENTS.md 工单 B 数学交叉印证，但 qwen4exp.cpp:382–400 的精确实现未由本轮直接读取核对；(b) normalized 未做独立数值验证（H2 靠对照臂逻辑排除而非逐值验证）。

### 26.6 静态排除与运行时实锤的关系（表面矛盾的澄清）

S1/S2 曾在源码语义层被静态排除（25.2/25.3），H1 本轮静态排除（26.1）——但 r24 对拍实锤 kernel 输出数值错。**两者不矛盾**：

- 静态排除的是**「调用侧与实现的参数语义不匹配」这一具体机理**（push constant 单位错位、aligned 索引错位、host k/m 填反——这些具体假说在源码层面确实不成立）。
- 运行时实锤的是**「kernel 输出数值错」这一事实**。
- 错误机理在**静态审查未覆盖的层面**：驱动/编译器实际行为（H3——理论等价无独立运行时证人）、audit 未覆盖的代码路径、或参照公式与实际 kernel 语义的偏差（判读边界 (a)——参照公式未从 qwen4exp.cpp 直接核对，若公式本身与实际 kernel 语义有偏差，对拍分歧的解释要重估）。
- 第十七轮的 FORCE_ITER 判别 + 静态深查正是裁决这些层面的下一步。

### 26.7 诚实边界与裁决影响

- **乱码根因修复未闭环**：kernel 数值错误是事实层实锤，但**具体缺陷机理未定位**（26.5 的「共享解码/累加路径」是排除法的交集推断 + 双段同错的指向性，非直接证据——与 12.3 的最终归因表述一致，本轮为其提供数值级支撑）；H3（编译器实际行为）未排除未证实；参照公式判读边界 (a) 如实保留。
- **off 臂三重判定连续第十九轮（r3/r5–r24）被前置阻断**——阻断只剩乱码数值缺陷本身的修复（崩溃维度 r21 已通过）。
- GAP 3 / GAP 4 残余 / GAP 5 状态不变；**h1 的 PROMOTE 状态（BEST_KNOWN_NOT_PROVEN_OPTIMAL）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关；理论 L = UNRESOLVED 不变。
- **r2→r24 归因链第十六轮增量**：V（写入可见性）排除（W 与 GGUF 逐字节一致——时间线矛盾消解）；H2（normalized 上游）排除（对照臂逻辑）；H1（host 值组装）静态排除（连续三轮缺口清偿）；kernel 侧静态假说域（S1/S2/H1）全部出局，但**运行时对拍把矛头收回 kernel 内部**（两个 BF16 kernel 数值错误实锤）——「静态排除 ≠ 运行正确」的警示（25.6）再次验证；r6/r7「W 全零」确认为 dump deferred 伪影（与 r15 SYNCCMP 裁决自洽，归因链的权重上传链假说域至此全部出清）。

---

## 二十七、第十七轮补录（2026-10-13）：静态深查全排除 + FORCE_ITER 判别实验无效

数据源：`artifacts/perf-gap/bf16-shared-path-verdict-r17-20261013.md`（kernel-defect-locator 静态深查裁决）与 `identity/quarantine-forceiter-r25/verdict.md`（三轮 GPU 判别）——两份均经本轮直接核对，与 Manager 交接清单一致。一处演进关系注明：静态深查文档写作时列 V/H3/C 三运行时嫌疑并要求澄清 r24 对拍注入集（C 嫌疑存废）；Manager 依 r24 既有证据裁定 V/C 出局（27.3）——文档与清单无事实冲突，是裁决演进的先后关系。

### 27.1 判读边界 (a) 清偿：公式一致，r24 判读无需重估

核对对象：`src/models/qwen4exp.cpp::build_hc_mix/build_hc_combine`（363–431，主仓与隔离树同源）。r24 CPU 参照公式逐项对照：

- **lo**（382–383）：`silu(scale(mm(w_down, xn), 1/hc))` → SiLU(0.25·W_down@xn)——**scale 在 SiLU 之前**，因子位置一致。
- **gate**（384）：`sigmoid(mm(w_up, lo))`——无额外缩放。
- **mixed**（387–400）：`gated = xn ⊙ gate` 按流切片（reshape_3d [n_embd, hc, nt]）求和后 `scale(1/hc)` → 0.25·Σ_b(xn_b⊙gate_b)——一致。
- **normalized**（377–379）：RMSNorm 单流 reduce、reshape_2d [hc_dim, nt] 行内 stream 连续、乘已折叠 (1+w) 的 [hc_dim] gamma——一致。
- **因子**：1.0f/hc = 0.25（非 1-hc）；2·sigmoid(inject/hc) 属 build_hc_combine 段、不进 lo/mixed。
- **W 形状**：W_down ne=(10240,320)（ne[0]=hc_dim=K、ne[1]=hc_lr=输出）、W_up ne=(320,10240)——ggml mul_mat 行优先（data[row·ne[0]+col]）与 shader `ib = row·ncols + col` 一致，**无偏差**。

**26.5 判读边界 (a) 清偿：公式无偏差，r24 的 lo/mixed 数值判读成立。**

### 27.2 共享层静态全排除（决定性结构事实 + 逐项核对）

**结构事实**（vulkan-shaders-gen.cpp:1112–1145）：Q8_0 与 BF16 两变体的编译字典中 `B_TYPE="float"`、`B_TYPEV4="vec4"`、`D_TYPE="float"`、`USE_SUBGROUP_ADD_NO_SHMEM` 完全相同；唯一差异 `DATA_A_Q8_0` vs `DATA_A_BF16`（→ QUANT_K=32/8-per-iter 的 iter() vs QUANT_K=1/4-per-iter 的 iter_aligned_nonquant + dequantize4_2aligned）。**B 读取骨架（get_offsets、batch_idx_a=0 推导、reduce_result 公共部分、D 写回、F32 B/D 类型匹配）有生产正确的 Q8_0 变体作运行时证人。**

BF16 特有层逐项静态核对（全部排除）：

| # | 嫌疑 | 核对结果 | 要点 |
|---|---|---|---|
| 1 | B vec4 视图 vs buffer 布局 | 排除 | B 是 F32（Q8 同声明生产正确）；col=4·(i·BLOCK_SIZE+tid) 恒 4 倍数、batch_stride_b（10240/320）与 b_offset 均 4 倍数——vec4 组索引整除、无错位 |
| 2 | A packed32 视图 vs subbuffer 对齐 | 排除 | W_down subbuffer offset=1049041280：mod 4/8/16 全零；W_up 顺延 +6553600 同对齐；ib=weight_row·ncols+col 恒 4 倍数——`(ib)/2` 无奇偶错位、u32 读字节 [2ib,2ib+3]=元素 {ib,ib+1} 正确 |
| 3 | BF16 位展开（dequantize4_2aligned） | 排除 | 小端 u32 低/高 16 位 = 元素 ib/ib+1；`(a & 0xffff) << 16` = BF16→F32 位模式补零——逐位正确（S2 排除维持并补上对齐证据） |
| 4 | subgroup 归约 vs wave64 | 排除 | RX 6800 wave64：BLOCK_SIZE=64、1 workgroup=1 wave=1 subgroup，subgroupAdd 跨全 64 lane 正确；无 gl_NumSubgroups>1 丢失路径 |
| 5 | 迭代结构/尾段 | 排除 | down：10240=40·256 整除全迭代无越界无遗漏；up：ncols=320 尾段 per-lane num_iters 覆盖恰 [0,320)；4T over-dispatch 由 robustness 吸收（Q8 twin 同 dispatch 生产正确——既有裁决 r16 维持） |
| 6 | D 写回 | 排除 | D_TYPE="float" 直接 F32 存；HC_DOWN_SILU 的 SiLU(0.25·temp) 与 CPU 公式一致；HC_UP_FOLD 的 0.25 折叠/sigmoid/fuse0 索引一致 |

### 27.3 归因收敛：H3 成为唯一剩余嫌疑（Manager 裁定）

静态深查文档写作时列三个运行时嫌疑：V（权重字节错）、H3（编译器）、C（上游 normalized 混入），并要求澄清 r24 对拍注入集（C 嫌疑存废）。**Manager 裁定（依 r24 既有证据）**：

- **V 排除**：r24 已实锤 W dump 与 GGUF 逐字节一致（26.2）——权重字节正确，V 层无缺陷空间。
- **C 排除**：r24 对拍已控制输入（同输入同权重同公式下 CPU≠GPU）——差异实锤在 kernel 计算过程，上游混入不能解释该对照。
- **H3 收敛为唯一剩余嫌疑**：驱动/编译器对 aligned 路径编译产物的实际行为。`mul_mat_vec.comp:20–31` 注释自证：两个 HC BF16 变体是树中仅有的 mul_mat_vec+DATA_A_BF16 实例，aligned 路径从未有独立执行证人。

### 27.4 r25 FORCE_ITER A/B 判别：实验无效（诚实结论）

三轮 GPU（bin 9a69607d 树状态不变）：

| 臂 | env | 32 token 正文 | dump |
|---|---|---|---|
| A（默认 aligned） | FORCE_ITER 未设 | 乱码（与 r24 逐字节一致——基线确认） | —（未开 dump） |
| B（判别） | FORCE_ITER=1（resolved env 确认传递） | **与 A 逐字节一致（乱码）** | — |
| B2（判别+dump） | FORCE_ITER=1 + DUMP_SITES=1 | 同乱码 | **lo/mixed/normalized 与 r24（默认路径）逐字节一致** |

**判读：实验无效——FORCE_ITER 的运行时效果未达 kernel**（非「iter() 也错」）：

- **机理推断（决定性）**：iter() 与 iter_aligned_nonquant 是两条几乎完全独立的数值实现（B 读取：标量 load_b vs vec4 data_b_v4；A 解码：dequantize1/dequantize4 vs dequantize4_2aligned 位展开）——浮点归约顺序不同的两条路径**不可能产出逐字节相同的输出**（即使都错，错的值也会有实现级差异）。B2 的 lo 与 r24 逐字节一致 ⇒ kernel 实际仍执行默认 aligned 路径。
- **H3 裁决无法给出**：本轮实验设计的前提（开关能切换路径）不成立。B/B2 的乱码与 A 一致本身不构成「iter() 也错」的证据（开关无效下 B=A 是必然）。
- env 传递已确认（result.json resolved env：GGML_VK_HC_BF16_FORCE_ITER="1"）+ 静态注入逻辑正确（switch-chain-audit 验证字段映射）——但运行时不活跃。**与 r13 轮 capture gate 教训同型：静态正确 ≠ 运行时活跃。**
- **判读边界（如实标注）**：「逐字节一致 ⇒ 开关无效」是机理推断而非直接观察（无 kernel 执行路径的直接证据）——插打印将提供直接证据。

**开关链断裂点候选**（verdict 落盘，三处）：(1) host 16951–16974 的 output_flags 组装中 pipeline 指针匹配条件（`force_iter_bf16 = hc_bf16_force_iter && (pipeline == pipeline_hc_down_silu_bf16 || pipeline == pipeline_hc_project_fold_bf16)`——若 fused 段 dispatch 实际选了别的 pipeline 指针，位不注入）；(2) shader 侧 0x200 运行时分支的位置与写法（mul_mat_vec.comp 的 FORCE_ITER 检查是否真的在 iter/iter_aligned 分叉点）；(3) push constant 的 fusion_flags 字段在 host→shader 的实际传递（pc 初始化第 8 字段 output_flags——绑定/布局偏移）。

### 27.5 诚实边界与裁决影响

- **乱码根因修复未闭环**：静态域全排除后错误在静态不可达层；H3 是唯一剩余嫌疑但**未证实**（FORCE_ITER 判别实验无效，H3 裁决悬置）；开关链断裂点未定位。
- **off 臂三重判定连续第二十轮（r3/r5–r25）被前置阻断。**
- GAP 3 / GAP 4 残余 / GAP 5 状态不变；**h1 的 PROMOTE 状态（BEST_KNOWN_NOT_PROVEN_OPTIMAL）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关；理论 L = UNRESOLVED 不变。
- **r2→r25 归因链第十七轮增量**：26.5 判读边界 (a) 清偿（公式一致——r24 判读成立，参照公式偏差假说出局）；kernel 源码语义层经穷尽后无可指认缺陷（共享层有 Q8_0 生产证人、BF16 特有层逐位正确）；V/C 由 Manager 依 r24 证据裁定出局；H3（编译器 aligned 实际行为）收敛为唯一剩余嫌疑——与第二轮补录 12.3 的最终归因（「iter_aligned_nonquant+DATA_A_BF16 组合全仓库只有 HC 两个变体、从诞生起无独立执行见证」）完全呼应，该判断经受住了 r24 数值级对拍与 r25 判别实验的检验；**FORCE_ITER 开关链断裂是本轮新发现的独立插桩缺陷**（阻断 H3 裁决，非乱码本身——第二轮 V-A 判别「两路径逐字相同」的旧观察由此获得新解释：当时开关同样未达 kernel，「V-A 原判据前提错误」的裁决方向正确、机理表述需修正）。【第十八轮修正：本条的「开关链断裂」「开关未达 kernel」表述被 r26 推翻——开关链一直全通，见 28.5。】

---

## 二十八、第十八轮补录（2026-10-13）：r26 插打印运行时裁决 + 归因重估

数据源：`identity/quarantine-traceflags-r26/verdict.md`（两跑打印行与裁决——本轮直接核对，与 Manager 交接清单一致；Engineer 侧插打印落盘细节——隔离树 ggml-vulkan.cpp 17015–17057、GGML_VK_HC_TRACE_FLAGS=1 门控（pipeline 身份布尔×7 + force_iter_env/match + pc.fusion_flags 十六进制 + grid，读回验证）、测试哨兵（1217 行 = 主树复制 + 62 行增补，主树未动）——按交接清单记录，verdict 为 DevOps 运行侧裁决——两者互补无冲突）。

### 28.1 插打印与静态裁决（switch-chain-tracer）

- **候选 (2) 静态排除**：mul_mat_vec.comp:511–515 的 `is_aligned_nonquant` 是**运行时 const bool**（读 `p.fusion_flags & 0x200`）——正确形态。**附带勘误**：第十七轮 27.4 的行号标注「base.glsl」张冠李戴——base.glsl 仅 500 行、判定实际在 mul_mat_vec.comp。
- **五环链静态全自洽**（env 解析 → pipeline 匹配 → flags 组装 → shader 宏与运行时分支 → 执行流到达）——修通与否依运行时证据裁决（任务设计如此）。
- 编译一次通过（bin 416922f2）；哨兵 52 项中 4 失败——**全部为历史哨兵与插桩演进的文本失配**（r4 mmap 删除 / r4 loadtrace 插桩 / r16 dump 改造的旧断言过期——非本轮缺陷，交 Engineer 同步）；**本轮 ForceIterTraceSentinelTests 三项全绿**（不在失败列表）。

### 28.2 跑 1（默认路径）：fused 段两算子的 BF16 身份确认

32565 行打印：`dispatch_is_down_bf16=1` 与 `dispatch_is_fold_bf16=1` 交替——**fused 段两算子都真实走 BF16 变体**；`dispatch_is_*_q8=0 q8dot=0 force_iter_env=0 force_iter_match=0 fusion_flags=0x00000000`；grid/k/m/n_rows 合理（down: grid(320,1,1) k=10240 m=320；fold: grid(2560,4,1) k=320 m=2560——**fold grid-y=4 证明 upfold 修复形态在位**）；17318 三元无 fallback。

### 28.3 跑 2（+FORCE_ITER）：开关链双侧全通——r25 推翻

32070 行：**`force_iter_env=1 force_iter_match=1 fusion_flags=0x00000200`**——0x200 注入**全部** dispatch；dispatch 布尔与跑 1 相同。shader 侧：is_aligned_nonquant 读 p.fusion_flags & 0x200 → 注入后为 false → **iter() 标量路径真实执行**。

**开关链 host + shader 双侧全通实锤。r25 的「开关无效」机理推断被本轮直接证据推翻**：路径确实切换了——而正文与 dump（r25 B2 的 lo/mixed 与默认路径逐字节一致）不变，即 **iter() 标量路径与 iter_aligned_nonquant 两条几乎独立的数值实现（标量 load_b + dequantize1/4 vs vec4 data_b_v4 + dequantize4_2aligned）产出逐字节相同的错值**。

**「推断被直接证据修正」的良性循环（如实记录）**：r25 verdict 当时明确标注判读边界——「逐字节一致 ⇒ 开关无效」是**机理推断而非直接观察**（27.4）——正是这条边界标注使 r26 插打印的推翻成为可能且无损：推断没有被当成事实固化，直接观察一到即被替换。这是「premise 先获得 provenance 再获得 architecture」的实践——r25 的推断便宜且方向明确（若成立可省一轮 GPU），r26 的插打印昂贵但 settle；两层证据各司其职，推断错了只损失一轮 GPU，不损失结论链的完整性。

### 28.4 归因重估：H3 排除，A 寻址契约层为最强嫌疑

1. **H3（aligned 编译产物缺陷）排除**——标量路径同样错，不是 aligned 编译产物的问题。
2. 错误在**两路径共享的层**。行映射 / reduce / B 语义被 Q8 证人排除（Q8 变体正常且共享这些）——**剩余共享嫌疑是 BF16 特有的 A 寻址契约**：QUANT_K=1 下的 `ib=(weight_row·ncols+col)`（元素索引）→ data_a 的实际寻址换算（packed32 u32 索引 `(a_offset+ib)/2` 与元素内偏移——**dequantize1 / dequantize4 / dequantize4_2aligned 三函数共享的 A 寻址层**）。
3. **静态审计矛盾的消解**：27.2 的「位展开逐位正确」覆盖的是**位运算**；**寻址/索引契约层未被覆盖**——「静态正确 + 运行时错」的矛盾在此消解（静态审查的覆盖域边界再次显形）。
4. **与 r24 CPU 对拍自洽**：W 字节正确（GGUF 逐字节）+ A 寻址错 → kernel 读错 A 位置 → lo/mixed 值错但量级合理（读到的是别处的真实权重）。
5. 乱码正文「健康做，，，，，，且，，，在rata…」跨 5 轮（r22–r26）逐字节稳定——确定性错误。

**修复指向（交 Engineer）**：深查 base glsl 三个 dequantize 函数的 BF16 A 寻址（ib / iqs / a_offset → data_a 偏移的换算——QUANT_K=1 语义下 packed32 对齐、a_offset 单位、元素内偏移的契约）——唯一能解释「两条解码路径同错同值」的层；修复后 r24 的 CPU 对拍脚本直接作回归门。

### 28.5 诚实边界与裁决影响

- **乱码根因修复未闭环**：A 寻址契约层是排除法收敛的**最强嫌疑（hypothesis）**，非定位实锤——第十九轮深查（含与 Q8 路径的寻址差异对比）才能定罪。
- **off 臂三重判定连续第二十一轮（r3/r5–r26）被前置阻断。**
- 4 项历史哨兵失配待 Engineer 文本级同步（非本轮缺陷——如实记录，不掩盖）。
- GAP 3 / GAP 4 残余 / GAP 5 状态不变；**h1 的 PROMOTE 状态（BEST_KNOWN_NOT_PROVEN_OPTIMAL）不变**——h1 的排名与资格证据基于冻结二进制正式臂，与 rebuild 树诊断运行无关；理论 L = UNRESOLVED 不变。
- **r2→r26 归因链第十八轮增量**：r25「开关无效」推断被推翻（**开关链一直全通**——第十七轮 27.4 的「开关链断裂点三候选」问题不存在，27.4/27.5 的相关表述已就地标注修正）；第二轮 V-A 判别「两路径逐字相同」的最终解释落定：**两路径同错同值**（A 寻址契约层共享）——第十七轮 27.5 的「当时开关同样未达 kernel」表述错误、一并修正（V-A 裁决「原判据前提错误」的方向仍正确——两路径数值并非同构，同值的原因是共享寻址层同错）；H3 出局后嫌疑收敛到 BF16 特有的 A 寻址契约层（三 dequantize 函数共享）——与 12.3「静态无错≠运行正确」的最终归因一致且更精确（缺陷域从「kernel 内部」收窄到「A 寻址契约」）。

---

**最终规则回执：** 先正确、再加速、再证明。h1 的速度证据（几何均值 1.05449，CI [1.0484, 1.0606]）与质量证据（V8/V9）不抵偿理论 UNRESOLVED 与乱码 OPEN；模型能力与状态门不以性能数字抵偿。

# Q2_0 新契约：首轮实机工单

预注册：2026-10-10。此文件先于本轮 GPU 实验写入。

目标：先建立 RERoT 关闭的强基线。有效理论下界 L 与合格实现耗时 U 必须同任务、同质量、同计时边界；U/L≤2 仅阶段成果，≤1.5 为最终门槛，向≤1.2收紧。本轮只做因果 pilot，不将探索最优点或旧成本预测记为有效理论界。

## 固定项

源模型为本机 GSQ-RCO-Q2_0 两分片。表示候选为已有 `models/perf-gap-q2-hcq8/` 两分片；转换证据引用 `../run-20261008T1845/hcq8-conversion.json`，不重复声称新转换贡献。实际全部分片身份、draft、可执行文件、实际加载库和物理 rank 映射随运行保存。

使用已有 `../frontier-20261009T154118Z/router-bf16-native-bin/` 冻结程序；先核验 identity.json 中全部文件哈希。不重编译生产程序，不改默认值。当前 checkout 源身份与该快照的构建来源分别记录，不把两者自动等同。

请求采用本目录 question.txt 与 system.txt。temperature=0、seed=42、thinking=false、cache_prompt=false；上限400只防止失控，不强制输出200。主输出自然结束、切题流畅、不明显重复或矛盾。160–240 token 为预先登记的长度检查区间；区间外的结果保留但不作本轮合格排名，不通过事后改提示词凑合格。既有哈希表 guard 保持原样；它不等于完整能力保护集。

CPU 22线程，五卡 tensor 分片、ctx1024、batch/ubatch32、单请求。所有臂显式固定 REPLICATE_ATTN=0、QSA_HEADMAP=1、GDN_HEADMAP=1、SMALL_Q8_COLUMNS=1、GREEDY_HOST=1、PARALLEL_SUBMIT=1、Q_WORKGROUP=256、NORM_WORKGROUP=256、NORM_PREFETCH=0、PACK_SINGLE_WG=0、SHARD_ROUTER=0。全部无性能插桩。

每进程1次冷warmup、2次measured、1次不同问题guard。两次measured不是两个独立进程样本。先汇总公开token/生成毫秒，另记完整请求毫秒、输出长度、结束原因、startup、fallback与退出。比较不同输出时不把归一化吞吐变化等同完整回答加速。

## 算法 P：顺序 pilot

P1. 验证源码工作区未被覆盖，冻结程序哈希一致，五卡空闲，无编译。
P2. 运行 A：原Q2_0，h0，LateBind off，MMVQ disabled。
P3. 运行 B：原Q2_0，h1，其余同A。发生执行故障则保留失败，不宣称MTP普遍无效。
P4. 运行 C：HC-Q8，h0，off/disabled；隔离表示变化。
P5. 运行 D：HC-Q8，h0，aggressive/disabled；隔离LateBind。
P6. 若相应MTP执行资格允许，运行 E：HC-Q8，h1，aggressive/disabled。
P7. 从无机械失败且长度合格的候选中，按池化生成成本选择一个点，比较 forced MMVQ。该选择及结果只作探索；若全部不合格，改做诊断，不重新定义通过门。
P8. 读取全部正文和对应日志。保留各相位失败、fallback、自然终止与预算终止区别。任何非正常退出停止该候选晋级，不扩大超时或改恢复机制。
P9. 保存短 decision：本轮观察、身份、质量/状态证据局限、下一项可证伪原型。未取得全局L时Gap记未证，不计算伪最优性比例。

每次GPU运行使用runner内的 `/var/tmp/tp5-gpu.lock`，外层不重复取锁。不并行编译，不结束无关进程，不commit/push。

## 开跑前身份修订

旧快照的 `libggml-vulkan.so.0.18.1` 实际SHA256为 `1373e69e…`，清单记为 `9d9969b1…`；另有当前build目录与旧快照的server-impl不同。差异原样保存于 frozen-identity-mismatch.json，不改旧清单、不猜测覆盖原因。

因此旧快照不作为本轮执行身份。改为在没有GPU实验时检查/构建当前 `build-perf-gap` 的 llama-server，并将可执行文件及配套库复制到本轮独立 `measured-bin/`，逐文件记录实际哈希；构建前后生产源身份另作核对。所有pilot统一使用该新快照。此修订先于任何GPU成绩。

首个A启动在加载模型前被另一轮 `-j12` 编译挡住；见 A-q2-h0/result.json 与 blocked-build-processes.json。它没有GPU成绩，不删除该目录。可用现场下的首次实际A另命名为 A-q2-h0-run。runner补充加载结束及每次请求前后的编译检查；发现并行编译则保留raw并判失败。边界检查不声称能发现两个检查点之间全部短命进程。

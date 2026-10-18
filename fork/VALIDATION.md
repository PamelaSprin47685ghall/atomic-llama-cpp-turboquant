# 验证记录

2026-10-18 批次起点为 `6de088714f35f84427d9952d713f9d6cdffc668b`；固定 A 为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。该批次新增 TQText、TQCalibration 和 imatrix C，扩展真实 batch C 的文本消费，并对 Vulkan 缓存的两个不同失效来源分别建立回归。六个 B 与五个 C 的职责见 [README.md](README.md)，旧 feature ID 与证据细分见 [audit/details-findings-20261018.md](audit/details-findings-20261018.md)。

本页先列 2026-10-19 TQRelay 批次已取得的结果，随后保留 2026-10-18 批次的 Final release gates 与更早历史；其中原模型摘要、HTTP、corpus、logits 与数值记录保持各自批次身份，没有被改写成本批次结果。

## 2026-10-19 批次：TQRelay（TP5 纯 CPU 合同）

本批次新增第六个 B 库 `fork/relay`（安装目标 `TQRelay::relay`），把《TP5-最新设计
保姆级指南》第 30–39、50、52、54 章中与设备无关的合同实现为 C++17：`channel_ledger`
（通道 bank 生命周期、精确票据、两种 credit、fail-stop 与排空事实）、
`run_submissions`（逐 rank 原生提交登记、封口与排空推导）、`compile_glu_plan`
（五 rank 合成 GLU 分片计划）。没有 A 文件变化，`fork/upstream-exceptions.json` 不变。
指南第 50、54 章内嵌的四个 Python 程序与四个 JSON 输入按记录 SHA-256 原样提取到
`fork/relay/tests/python-reference/`，全部匹配。

| 检查 | 实际结果 | 可以支持的结论 |
|---|---|---|
| Release 构建与测试 | `build-relay-20261018`，GCC 16.2.1，`-Wall -Wextra -Wpedantic -Werror`，4/4 ctest 通过 | ledger 所有权反例、提交登记边界、局部计划数字与 dense GLU 独立参考按代码路径通过 |
| ASan + UBSan | `build-relay-san-20261018`，4/4 ctest 通过 | 本库已执行用例无地址/未定义行为报告；不扩写到线程、GPU 或模型 |
| 完整构建流水线 | `build-clean-fork-relay-20261019`：A 固定 archive + 六个 B + 五个 C + a-seams，exit 0，末道边界复查通过 | 新库进入 `fork/build.py` 常规入口后整条流水线成立 |
| 完整来源隔离 | `build-isolation-relay-20261019`：63/63 条命令通过，`isolated_modules` 含 `relay`，relay 从自身源码副本构建、测试（含 Python 交叉对照）、安装并进入搬迁后交付集合 | 独立库边界与安装消费成立；不构成操作系统沙箱 |
| M0c 单卡平台探针（16 用例扩展版） | `fork/relay-probe`（HIP 构建，`build-relay-probe-20261019`）在本机 RX 6800（gfx1030，ROCm 7.2.4）实际运行 2000 轮：双向热复用、生产延迟排序、精确票据、多工作组最后到达、abort、五消费者输出 credit、延迟最后读者（H08）、双 bank 轮换、可变有效长度（H05）、单卡五流 RELAY 环全部 pass；host fetch_add 产生错误值（448 而非 7）记 fail，host CAS pass；五卡用例 not_run（本机 1 张）。证据报告 SHA-256 0cdd86bd…9cb8a | 单卡平台事实成立：单写者 load/store 协议可用且无需 host RMW；五消费者 credit 与双 bank 是软件协议层验证，不构成五卡或整链资格，换工具链/拓扑后需重跑 |
| 探针 bench 测量（`--bench`） | 独立 `tp5.relay.bench.v1` 文件（SHA-256 062c2806…a9de0ad），2000 轮 median：host 发布→设备 ack 9,690 ns、设备生产→host ready 7,770 ns、五流整圈 29,930 ns、host 五路折减 65.7 GB/s | 单设备含 launch 时间的观察值，供指南 14 成本模型使用；不是资格门，五卡数字待真机 |
| 设备等待预算修正 | 全量运行实测发现旧代码每次 ack 未命中把 2e9 ticks 预算一次性睡满（每轮 ~0.9 s，2000 轮预计数小时，600 s 上限内无产出）；改为累计预算等待（clock64 实测 ~2.3 GHz，预算 24e9 ticks ≈ 8 s，超过最长 5 s host 截止期），失败路径 abort+排空防晚写污染 | 全量 2000 轮从数小时级降到约 20 秒；防抖语义保留（设备等待超过 host 截止期，瞬时负载尖峰不再造成设备侧假超时） |
| 五卡跨设备用例实现加固（无新硬件证据） | 提交 35d41c8fc 之后的加固：`run_five_rank` 从单轮裸写升级为 guard 三明治 + 按 `--rounds` 多轮热复用（每轮 payload 随代次变化）+ 逐 rank/轮次失败归因 + 每轮同步后复位；`five_rank_cross_device` 现在登记进每张设备的证据记录（注册项可达）；`free_device_context` 与 `run_bench` 补上入口 `hipSetDevice`（五卡机上 `hipFree`/launch 作用于当前设备，缺失会解错设备）；`--rounds` 拒绝尾随垃圾与溢出。单卡 2000 轮 smoke 复跑 qualification=pass、five_rank=not_run、16 用例分布不变。机器可读索引见 [audit/relay-probe-evidence-20261019.json](audit/relay-probe-evidence-20261019.json) | 单设备路径无回归；跨设备路径第一次执行只能在真实五卡机器上发生，本行不含跨设备可见性证据。报告 SHA-256 0cdd86bd…9cb8a 是加固前的产物，其 five_rank_status=not_run 依然准确，不改写旧报告字节 |
| 探针交接批次：两个新用例 + 证据路径抗故障化 + 原子发布 | 注册表 16→18：新增 `offset_addressing`（指南 39.5 H06，字节 72、8 字节对齐但跨 cache line 的内部槽双向往返，arena 前后 guard 抓基址/越界算术）与 `multi_group_ordering`（指南 39.6 P01/P02，确定性反序延迟使最大编号组先完成，陷阱观察 ready 保持未发布，8 轮全部观察到陷阱窗口）。HIP 错误改为异常：逐用例捕获记 fail、坏设备 setup 失败仍以 `device_identity=fail` 留在报告、`run_five_rank` 任何 HIP 失败按 rank 归因且照常写报告、bench 失败不再丢失报告。证据文件改同目录临时文件 + fsync + 硬链接原子发布，已有文件/符号链接拒绝覆盖；CLI 拒绝 `--output` 与 `--bench` 同路径。单卡 2000 轮权威重跑 qualification=pass、17 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 3ff3a3b6…f03a6、bench SHA-256 e2a66eec…33de0；无 HIP 构建 config-test 1/1、relay release/san 4/4、check-upstream 通过、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-handoff.json](audit/relay-probe-evidence-20261019-handoff.json) | 单卡新增用例通过；故障路径由逐项代码评审与 CLI 拒绝验证保证，五卡机首次运行前不声称已实测。16 用例报告 0cdd86bd…9cb8a 保留为被取代历史，字节不改写 |
| 探针证据校验器与归因加固批次 | 新增 `fork/relay-probe/validate-report.py`：报告与 bench 的严格 JSON 解析（重复键、非有限数、未知字段、尾随数据、bool 冒充整数全拒绝）+ 指南 53.4 终态一致性机器检查（注册表精确覆盖、rank 与下标一致、five_rank 状态与设备数一致、qualification 与逐用例结果重算一致、无设备报告的合法形状、bench min≤median≤p90≤max 且测量名唯一），`--self-test` 覆盖 20 余个反例，两份权威证据实测 exit 0。五卡归因记录塑形抽成纯函数 `five_rank_case_records`（rank 0–4 记录、failed_rank==5 哨兵、失败详情只归因到该 rank），config-test 覆盖成功、rank 2 归因、哨兵三种形状——该路径真机上首次执行，先在此闭合其逻辑。stdout 改无缓冲（_IONBF，C 标准"尽快传输"保证）：config-test 新增 fork+SIGKILL 存活测试实测"杀进程不丢已打印证据行"，兑现 README 的可恢复承诺。`fork/build.py` 默认流水线接入探针模块（install=False：无 HIP 机器只构建并运行 config-test，有 ROCm 时顺带构建探针），AGENTS.md 例外条款相应修订；isolation.py 继续排除（无 HIP 时没有可安装 target）。校验器 `--self-test` 挂进 ctest（tq_relay_probe.validator），覆盖五卡 fail 报告形状及其不一致反例，流水线与真机 ctest 都先证明校验器。最终二进制 2000 轮权威重跑 qualification=pass、17 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 75dbd439…dec4c、bench SHA-256 29efb0c1…e80d，校验器对两份新证据 exit 0；relay release/san 4/4、无 HIP 构建 ctest 2/2、HIP 构建 ctest 2/2、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-validator.json](audit/relay-probe-evidence-20261019-validator.json) | 单卡平台事实与前批一致；五卡路径新增逻辑只经 CPU 测试闭合，跨设备可见性资格仍待真机首次运行。3ff3a3b6…f03a6 报告保留为被取代历史，字节不改写 |
| 探针有界同步批次：挂死设备不再让运行永不返回 | 全部 17 处裸 `hipDeviceSynchronize`/`hipStreamSynchronize`（逐用例检查、run_case/fail_case 排空、五卡逐 rank 排空与 teardown、bench 排空）替换为 `sync_with_deadline`：记录 event 后 host 截止期轮询，健康队列微秒级返回，停转队列 60 s 后变成该用例的 fail 并继续收集证据（指南 35.8；SPEC 7 的"只有设备枚举失败才可能无报告"由此补上"挂死也能出报告"）。超时分支不再是交班后首次执行的死代码：新增强制用例 `native_sync_deadline`（注册表 18→19）用 stall kernel 把队列停转超过 2 s 短截止期，实测触发有界失败、停止字恢复等待者、drain 完成，本机 pass。注册表 validator/config-test 同步；报告 limits 记入 60 s/2 s 两个截止期。最终二进制 2000 轮权威重跑 qualification=pass、18 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 c0a21839…2e1f、bench SHA-256 04473ee1…c375b，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记 HIP 误报外无告警。机器可读索引见 [audit/relay-probe-evidence-20261019-syncdeadline.json](audit/relay-probe-evidence-20261019-syncdeadline.json) | 有界同步语义在本机真实设备上验证（超时分支实际执行）；"挂死设备有界失败"是软件行为+event 查询活性的组合事实，五卡机上的最终确认仍待真机运行。75dbd439…dec4c 报告保留为被取代历史，字节不改写 |
| 探针 P 系列批次：同流逆序识别与双窗口独立前进（单卡可测子集） | 指南 39.6 的完成与前进必做用例在本机补上软件层覆盖：新增强制用例 `same_stream_inverse_dependency`（P04，消费者排在 payload 生产者之前的同流逆序，流序保证早读写入前载荷，验证字 0xDEADBEEF 确定性抓住早读而非静默消费垃圾，生产者后写精确，运行有界）与 `dual_window_independent_progress`（指南 31.1 与 P05/P06/P08 软件层：两个独立通道各自控制块/槽/种子，窗口 A 被扣留票据、64 个驻留等待工作组、延迟最后读者三种形态停摆时窗口 B 独立完成 528 代消费，逐代金丝雀证明 B 从未写 A 槽、A 的 ack 半读期始终扣留，释放后精确完成并复用）。device_context 增第二通道（cb2/slot_b），bare_waiter_kernel 用累计预算占驻留。注册表 19→21，validator/config-test 同步；单卡 2000 轮权威重跑 qualification=pass、20 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 b6baeb81…f5be、bench SHA-256 d51f2e08…fa16e，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-pcases.json](audit/relay-probe-evidence-20261019-pcases.json) | 两个新用例证明单设备上软件协议层的逆序识别与跨通道隔离；它们是五卡机上双窗口设备时间线的软件前提，跨设备资格仍由 five_rank_cross_device 给出。c0a21839…2e1f 报告保留为被取代历史，字节不改写 |
| 探针服务循环批次：P07/P09/P10、F04、F07（单卡可测的最后一格） | 指南 39.6 完成与前进表在本机闭合最后三个软件层格子：新增强制用例 `service_loop_independent_progress`（P07/P09/P10：独立 worker 线程驱动 B 通道 256 代而观察者只读状态；两个同时齐备的通道背靠背发布，确定性慢读者半读期内快通道完成其验证读，双向都验证；关停时带驻留 GPU 等待者，stop 字与两条流排空全在 worker 退出前完成，观察者只 join）、`nonfinite_reduction_gate`（F04/指南 41.6：host 五路折减新增发布前门——rank 权重模 2^64 和为 1，精确恒等式确定性抓住任何损坏的部分和，翻转位与 quiet-NaN 位型都在发布前被拒绝且输出槽零写入；five_stream 用例与 bench 共用同一门）与 `host_wait_deadline_no_reuse`（F07/指南 35.8：读完却永不确认的消费者触发 host 有界等待 400 ms 截止期——spin_until 超时返回分支从交班后死代码变成活设备实测路径；槽位字节不变，停止字恢复等待者）。注册表 21→24，validator/config-test 同步；单卡 2000 轮权威重跑 qualification=pass、23 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 01533600…fc73、bench SHA-256 05c3f207…f75e，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记 HIP 误报外无告警。机器可读索引见 [audit/relay-probe-evidence-20261019-serviceloop.json](audit/relay-probe-evidence-20261019-serviceloop.json) | 三个新用例证明单设备上服务循环进度独立、无队头串行与关停排空顺序，以及归约发布门与 host 有界等待；它们是真机双窗口调度与 CPU 服务循环的软件前提。剩余 F 系（F01/F03/F06/F08/F09/F11/F12）在健康单卡上不可注入，软件路径经代码评审、不上设备。b6baeb81…f5be 报告保留为被取代历史，字节不改写 |
| 探针定标与折减门批次：时钟标定入库 + 折减门下沉 + bench 多样本 | 三个收口：① `fold_partials` 与 rank 权重从 main.cpp 下沉到证据层（evidence.h/evidence.cpp），无 HIP 的 config-test 新增 `check_fold_gate` 钉住全部反例。审查中先抓出真缺陷再修正：权重原为 {1, 2^32, 2^32, 2^32, −3·2^32}——rank 1–3 同权，行复制对同权 rank 对不可见；改为一组两两互异且和为 1 mod 2^64 的奇数权重 {1, 3, 5, 7, 2^64−15}：奇乘是 64 位双射，任意位损坏（含幂权重会移出字的高位）必然改变折减，行复制对任意 rank 对可见；门改全有或全无（先验证全部词再写 out，拒绝不留部分结果）。活设备用例 `nonfinite_reduction_gate` 同步改钉高位翻转 + quiet-NaN 注入。② 定标方法入库：新增 HIP 门控诊断工具 `tq-clock-rate`（每设备 5 次固定 tick 自旋 + event 轮询墙钟截止期），本机三次独立运行 median 2320.1–2320.4 MHz，复核 2.3 GHz 预算锚；main.cpp 注释与运维 README 交叉引用。③ bench 折减带宽从两遍取优单样本改为 32 样本 min/median/p90/max（单样本被调度噪声支配，min 与 median 实测相差约 1.6 倍）。单卡 2000 轮权威重跑（全新树）qualification=pass、23 用例/设备（仅 rmw_add 预期 fail）、新报告 SHA-256 b40dd940…6b26、bench SHA-256 a1505a8c…7b2c，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 无新增真实告警。机器可读索引见 [audit/relay-probe-evidence-20261019-calib.json](audit/relay-probe-evidence-20261019-calib.json)
| 探针五卡审计批次：not_run 路径抓出真缺陷 + 五卡核索引单卡钉住 | 对唯一未在活设备跑过的五卡路径逐行审计，抓到三个缺陷并全部修复：① **真缺陷（活设备复现）**：`rank_write_kernel` 向基址发布 ready——五个 rank 全员踩写 `ready[0]`，rank 1–4 的 ready 永不到达，五卡机第一轮就会等满 5 s 并以虚假 stall 归因失败（/tmp 复现驱动实测 ready=[1,0,0,0,0]）；修复为 `release_store(&ready[rank], generation)`。② `run_five_rank` 的 setup 失败预检查是死条件：它检查 `record.ordinal < 0`，而 run_probe 的 catch 总会先把 ordinal 写正——五卡机上坏卡不会被识别，会向坏 context 发核再以误导性超时归因；改为按 `device_identity` 结果与 `mapped` 能力位判定，坏卡在核启动前被精确归因（bench 同补 mapped 守卫）。③ 零设备分支退出码为 0（与枚举失败分支的 1 不一致，qualification=fail 却退出 0 会骗过只看退出码的脚本）；改为与枚举失败一致的 1（本机 ROCm 枚举失败走报错分支，零设备分支为其他运行时的防御性一致化）。**新增单卡钉住用例 `five_rank_indexing`（注册表 24→25，validator/config-test 同步）**：五条流在一张设备上跑与五卡用例完全相同的核与 guard 布局，逐轮验证每个 rank 的 payload 与自己的 ready 字；两个用例共享同一套轮次协议 `run_rank_rounds`（抽取自 run_five_rank，五卡机上跑的循环正是单卡钉住的那份代码）。回注缺陷版的临时工作树实测：`five_rank_indexing=fail`，detail 精确归因 rank 1 5 s 超时，退出 1、运行有界——用例对缺陷的判别力在活设备上被证实。运维语义实测：HIP_VISIBLE_DEVICES 的空白/空值形态会让枚举报错（报告照写、退出 1），`0,1,2,3,4` 与尾随逗号可用，已写入运维 README。单卡 2000 轮权威重跑（全新树 build-relay-probe-20261019g）qualification=pass、24 用例/设备（仅 rmw_add 预期 fail）、five_rank=not_run、five_rank_indexing=pass、新报告 SHA-256 65146f98…b4533、bench SHA-256 70400732…03c3，校验器双通过；无 HIP 与 HIP ctest 各 2/2、relay release/san 4/4、check-upstream exit 0、tidy 除已登记误报外零新增（本班新增 rank_stream padding 与空指针推断两处即时修掉，既有 C++ 风格结论补登记进 AGENTS.md 例外清单）。机器可读索引见 [audit/relay-probe-evidence-20261019-fivecard.json](audit/relay-probe-evidence-20261019-fivecard.json) |

较早的单卡探针报告（SHA-256 9d1d1429…684112，12 用例版）按证据身份保留为被取代的历史记录；
新报告覆盖并取代其资格结论，不修改其字节。
| 实时跨语言对照 | `tq_relay.python_crosscheck` 通过：C++ 移植与已验证的 `tp5_local_plan.py`（SHA-256 59129982…59e657）在四个固定 fixture 上 decision、错误码、totals、逐 rank 字节/MAC、middle/output 区间全部一致 | C++ 端口与固定比较对象一致；不是 GPU 或真实量化资格 |
| Python 参考基线 | `tp5_protocol_spec.py` PASS 8 checks；`output_credit_model.py` broken 32 状态 16 条不安全边 / fixed 17 状态 0 条；`tp5_local_plan.py --self-test` 16 组通过 | 提取产物可运行且与指南记录输出一致 |
| 安装与搬迁消费 | 安装到 `/tmp/tqrelay-prefix-*`，从空目录 `find_package(TQRelay)` 构建消费者，搬迁前缀后重建再运行 `consumer ok` | 安装包自足，无绝对路径泄漏；不证明操作系统沙箱 |
| A 边界 | `fork/check-upstream.py` 通过，无未登记 A 变化 | 本批次未改 A |

本批次的通过范围是单线程可信调用方的 CPU 所有权模型与合成存储几何。它不构成
HIP 内存序、五卡前进性、模型数值或性能资格；指南 M0c 平台探针与 A 的最小执行
接口仍是下一步。通道账本的预算合同沿用 `TQScheduling::admission_budget`（指南
第 50 章场景组 5、6 的既有实现），不在 TQRelay 内重复。

## Final release gates

最终格式收尾仅删除 `vulkan-scratch-reuse.cpp` 与 `vulkan-set-rows-types.cpp` 各一个文件尾空行，逐字节验证其余内容不变。随后重新构建接缝测试，7/7 再次通过，日志为 `build-clean-vulkan/details-eof-seams-20261018.log`。交付源码摘要及这次重建单独记在 [machine-readable qualification](audit/details-qualification-20261018.json) 的 `eof_only_normalization`；此前 62 条 isolation、完整 GPU sweep 和 before/after 反例保留原始运行身份。A、B/C 生产源码与 exception manifest 均未因该格式收尾改变。

**当前状态：最终专项、完整来源隔离和 CPU / Vulkan 普通 tools 构建已完成；最终全量 GPU 仍有一个 FA 失败，较早 TOPK 失败归因未闭合。** 以下统一登记本批次实际身份与验证状态。较早失败保持原记录；它们不能被定向用例成功或一次 manifest 检查覆盖。

| 门槛 | 当前可回查事实 | 状态与边界 |
|---|---|---|
| 最终源码文件身份 | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` SHA-256 为 `6453d62fe781b6452406476f88ba4145fc55af7c7207b68357b1d8032a128c6a` | 这是一个源码文件的摘要，不是整 A 目录摘要或 commit。固定 A pin 不变，clean 提交历史以 Git log 为准。 |
| 最终 manifest 身份 | `fork/upstream-exceptions.json` SHA-256 为 `830a973a4e373d122d54f236984785de2d1b9cfeb3a8a67716a998c74b9f63dc` | 下述最终 archive 与 GPU 专项使用该内容身份；各文件集合和完整摘要以 manifest 为准。 |
| 完整 CPU 来源隔离 | `build-isolated-details-qualified-20261018`，62/62 条命令通过，2026-10-18 11:57:10 UTC 结束 | 从 pin archive 与经验证 overlay 重建；五 B / 五 C 独立安装，删除 source/build 副本及原前缀，搬迁后的实际 batch、logits、stream、imatrix 消费通过。没有使用 `reused_unverified` 代替。 |
| 最终 Vulkan 专项 | `build-clean-vulkan/details-final-gpu-pii1wep9`，7/7 接缝 CTest 通过 | 含 5 个 scratch case、2 个 source-alias case、16 次 SET_ROWS packed 比较；有真实重建重装后的执行记录，未转借较早版本的结果。 |
| 源数据原地修改 before/after | 冻结的 4-byte-offset probe 修复前第二个 matmul 64/64 错、最大绝对差 15.4662；修复后 control/inplace 均通过 | 每个 matmul 的全部 64 个输出，以及 SCALE/源存储各 514 个值的相应检查误差为 0；数值身份与较早零偏移 probe 分开保留。 |
| 最终 Vulkan 文本消费者 | 同一最终 GPU 目录：10 条命令，6 个模型配置各 3 个请求，脚本约 14.5 秒 | 18 个真实模型请求通过；脚本用时不是吞吐基准，也不提供 HTTP SSE 资格。 |
| 最终全量 GPU backend | 25932/25933，exit 1，用时 454.338607 秒 | 仅剩同一 mask+sinks、Q8_0 K/V、head dimension 256 的 FA case，ERR 0.009055835 > 0.0005；最终 TOPK 未复现，不能撤销 +12 full 的原失败或视为归因闭合。 |
| 普通 CPU tools 入口 | `build-clean-fork/details-build-entry-mob1rldf/result.json`，`build.py --tools` exit 0，用时 67.8365955 秒 | 这是实际常规构建入口的执行回执。 |
| 最终 A 差异统计 | 17 个 A 源码文件 +357/-43；另有 49 行自有 CI，共 18 路径 +406/-43；本批次新增 A 代码 29 行 | 最终 Git numstat 已核对；计数不代替内容摘要或运行资格。 |
| 普通 Vulkan tools 入口 | `build-clean-vulkan/details-build-entry-4byg2csx/result.json`，`build.py --tools` exit 0，用时 7.313101 秒 | 这是实际常规构建入口的执行回执。 |
| 可移交证据索引 | [audit/details-qualification-20261018.json](audit/details-qualification-20261018.json) | 本批次起点、源码与 manifest 摘要、命令、库/设备身份及结果分别记录；不借用旧 release receipt 取得新资格。全量 FA 失败和旧 TOPK 未定位结论继续保留。 |

“+12”只指下文远程日志使用的 scratch 修正阶段简称，不是最终 A 差异总数或提交身份。GitHub CI、CUDA 或发布流程未执行的部分保持未执行；成功的本地检查不能代替这些外部结果。

## 本批次已执行的检查

| 检查 | 实际结果 | 可以支持的结论 |
|---|---|---|
| 最终 CPU 来源隔离与安装 | 62/62 条命令通过，五个 B、五个 C 从各自源码副本构建、安装并搬迁消费 | 已在上述最终源码/manifest 身份上重建 archive + verified overlay，实际消费者通过。 |
| 新 B sanitizer | TQText、TQCalibration 的 ASan 与 UBSan 测试通过 | 两个新独立库的已执行用例通过；不扩写到全部项目、模型或早先未取得资格的 logits ASan。 |
| 生产 writer 互操作 | `2777543b2358c88cd1c2000f7fa342b764a0e213` 的原 calibration writer 两个文件逐字节相同，实际写出所有 flags 的 fixture；新 C 的 12 tensors 检查通过 | 旧生产格式与新 reader/exporter 互操作，而非历史真实模型采集质量。 |
| 实际 A imatrix loader | `llama-quantize` 读入 6 importance entries、9 chunks；模型文件被故意设置为不存在，后续阶段 exit 1 | 真实 A loader 接受并消费生成格式；整个量化命令没有成功，不宣称完成模型量化或质量验证。 |
| 最终 Vulkan 接缝 | 7/7 CTest 通过，含全部 5 个 scratch 用例、2 个 source-alias 用例与 16 次 SET_ROWS packed 比较 | 实际 RX 6800 Vulkan 操作的指定图与布局；不是全部缓存组合或全部设备。 |
| 最终 Vulkan stream/stop 模型运行 | 10 条命令，6 个模型配置各 3 个请求，共 18 个成功模型请求，脚本约 14.5 秒 | 实际模型 token、同配置 stream 开关、序列释放和文本事件合同；不是 10 次模型运行、吞吐基准或 HTTP SSE 资格。 |
| 已完成的全量 CPU backend ops | 27639/27639 通过 | 保留该次运行身份；最后广覆盖复验见统一门槛表，不能推导未运行平台。 |
| 最终全量 Vulkan backend ops | 25932/25933，exit 1，454.338607 秒 | 同一 FA case 仍失败；最终 TOPK 未复现，较早 +12 full 的 TOPK 失败与未定位状态保持记录。 |
| 普通 tools 构建 | CPU exit 0，67.8365955 秒；Vulkan exit 0，7.313101 秒 | 两个常规构建入口均实际完成；不把构建时间解释成推理吞吐。 |
| Vulkan K/V smoke | 9 个 K/V 组合的 pp/tg 共 18 行，exit 0，1 repeat | 这些配置的一次执行通过，不是统计可靠的性能比较或收益证明。 |
| CUDA | 没有可用 CUDA 设备 | 无运行资格；构建选项存在不代表执行通过。 |

各阶段构建与数值结果以实际运行身份为限。当前结果未提供新的吞吐、延迟分位数、峰值显存、模型 NLL 或量化质量测量。实际 GPU 资格来自 Vulkan 执行，不从 `--gpu-layers` 请求参数推断。

## 增量文本：独立合同与真实消费者

B `TQText` 拥有 UTF-8 尾部和 stop 匹配状态，不调用 A、I/O 或兄弟库。输入分块不会改变匹配结果；每个非法字节及最终未完成字节各转为一个 U+FFFD。停止字符串非空、必须是合法 UTF-8，限制为最多 32 个、每个 4096 bytes、总量 65536 bytes。同一字节位置完成的匹配选最长，相同字符串按配置顺序。

真实 C 用 A 返回的 token piece 推进 B。`--stream` 要求 `--json`；每个成功请求恰有一个 terminal text 事件，`offset_bytes` 连续，拼接增量等于 `result.text`。文本抑制不回删采样 token。匹配确定后才停止后续 decode；最终 UTF-8 修复可能在已提交过早先 token 后确定 stop，因此不宣称任何 stop 所在 token 都从未进入 A。C 消费本 batch 的全部输出、同步并删除序列后，才回收预算和槽位。

GPU 的六种实际模型配置为 serial-buffered、serial-stream、stop-buffered、stop-stream、parallel-buffered、parallel-stream，每种三个请求。其余命令检查参数失败，不能作为额外模型请求计数。stop 用实际模型基线文本形成，并验证跨 delta 的匹配；对照保持同配置模型提交形状。独立 B 单测负责分块、无效 UTF-8、匹配冲突、实例和结果所有权，不能将模型 fixture 的成功扩写为它单独覆盖所有字符路径。

上述真实模型 streaming 对照显式使用 `--ignore-eog` 和 `--no-phase-memory`。因此它不单独验证自然 EOG 时的最后文本边界，也不构成 stream/stop 与阶段释放组合的模型资格；finish-time UTF-8 与 stop 合同由 B 单测覆盖，阶段释放保留其独立实际回归。

该工作重建 `server:Q14` 的有用文本目的，不宣称复刻旧无 stop / 非 stream fastpath 或性能收益。`server:T09` 只涉及跨块 terminator 机制。`Q15/Q16/Q17`、完整 thinking/lane parser、HTTP SSE、网络送达、取消和重试仍有各自待迁合同。

## WXQCAL03：旧生产 writer 与真实 A loader

实际互操作使用旧提交 `2777543b2358c88cd1c2000f7fa342b764a0e213` 的 `wanxiangqi/lib/calib-dump.cpp` 与 `calib-dump.h`，不修改这两个文件。fixture driver 调用原 `wxq::write_calib` 写出两层小数据，包含全部三个统计 flags，decoded 与 declared total 均为 9。未找到可用于本次验证的历史真实校准样本；这个 fixture 是原生产 writer 生成的格式样本，不是训练或推理模型的测量。

新 C 的实际输出包含 12 个 F32 GGUF tensors。独立标准库 reader 检查形状、次序、typed metadata 和值；另由实际 A `llama-quantize` 读取为 6 个 importance entries 与 9 chunks。故意不存在的模型随后令命令退出 1，这是有意限定的 loader qualification。成功消费格式不证明 Tensor 命名适合任意模型、收集统计来自相同模型，或任何量化质量收益。

C 使用安装后的 `ggml::ggml-base` 与 TQCalibration；不加载模型，不创建 inference context，不初始化 GPU。实际 GGML 元数据 context 仍存在。A 的 package config 会继续发现该安装启用的 backend，因此不能删除 backend 安装文件来证明依赖缩减。

关键数值合同与 A 的实际消费保持一致：

- 保留每个 expert 的 raw sum2，不预先归一化；A 用 F32 sum2 / F32 count 求 importance。
- count 必须为精确 F32 整数且不超过 INT32_MAX；`chunk_count` 也为正且不超过 INT32_MAX，`chunk_size` 为 1。既考虑 F32 精度，也考虑 A 使用 `std::lround` 的可交换范围。
- 拒绝源非有限值、负平方和、F32 溢出与非零下溢，并检查实际 F32 除法不会把非零 importance 静默变成零。
- routing 计数与 expert activation 计数是不同统计人口，不能机械要求 routecounts 等于 `exp_n`。
- 默认拒绝不完整所选范围和未观测 expert。`--allow-partial` 保留所有实际原因，包括旧末 batch 的 overshoot；`--unseen global-mean` 明确标记 estimate、原始策略和精确填补 ID，不把填补 count 当作测量。

Reader 在分配前检查层大小，协方差分块验证后丢弃；C 一次持有一层、16 KiB 转换缓冲和私有 payload spool。输出完整验证、关闭和 sync 后以同目录 hard link 无覆盖发布。spool 与 staging 共存需要约两倍最终 payload 磁盘空间。平台为小端 POSIX、调用者控制目录；不承诺信号清理或断电目录持久性。stdout 后续失败准确报告完整文件已发布。

完整复现命令、旧 writer 源码摘要记录方式、可执行测试与失败合同见 [imatrix-compose/README.md](imatrix-compose/README.md)。本节不借用较早 corpus writer 的文件摘要作为 calibration writer 身份。`small:S04` 的格式与导出已重建；`small:S05` 的真实 callback 采集、DFS、checkpoint 和模型质量验证仍待迁。

## Vulkan：scratch 覆盖、同步与既有 SET_ROWS

本轮实际 probe 调用已安装 A 的公开 GGML/Vulkan API，直接运行 GPU，不使用 scheduler CPU fallback、私有头、模型或替代 kernel。原图使用同一个不连续 F32 activation view，依次执行 matmul、中间操作、第二个 matmul；两个 F16 权重内容相同但拥有不同 tensor。

原 scratch 修复前，small-softmax 和无 mask FA 控制通过。32768-column softmax 后的第二个 matmul 有 32/64 值不符，最大绝对差约 3380.01；masked one-head FA 后有 32/64 不符，最大绝对差约 `2.70e13`。这两个操作会覆盖 `prealloc_y`，必须同时清除转换缓存的 pipeline、source 和 decode-layout 三个身份字段。

执行顺序是另一项责任。large softmax 已标记最后 dispatch 后需要同步，split-K FA 已有读取 mask bitmap 后的 reduction barrier；no-split 路径需要在最后读取后标记 Y 待同步，让下一次覆盖 Y 的 matmul 使用已有 pre-copy barrier。数值 probe 与源码责任检查分别提供证据，没有声称数值测试复现了缺同步竞态或运行过 validation-layer 同步检查。

初版 no-split probe 用复杂 fractional V：全部 524288 值与同 GPU 独立中间操作相同，但其中 475136 值未通过闭式门限，最大偏差 0.0146484。该失败是独立控制的数值限制，不能归因为 scratch 同步。最终 fixture 采用 binary-scaled V、Q/K 为零和明确 mask，使预期值可精确构造，保留原 attention absolute tolerance `2e-4`。scratch 修正阶段全部五个用例通过，包含 source-write 修复的最终安装重新运行也通过；no-split 对全部 524288 输出做同 GPU 独立对照与闭式检查。公开 API 不暴露私有 FA tuning 状态，no-split 名称表达该设备与源码分析对应的目标条件，不是运行时私有状态探针。

`backend:G44` 的旧建议建立在“只编译 F32 source”前提上，固定 A 源码实际生成 F32/F16 到 Turbo2/3/4 的 I32/I64 SET_ROWS 变体。因此保留 A 已有支持。实际测试覆盖 Q8_0、Turbo2/3/4 四种目的类型及两种 index 类型，共八个配置，每个比较 F32 和 F16 scatter，共 16 次 packed 比较；同时检查其他行 sentinel。这个结果不独立量化 quantizer 误差，也不涵盖所有 stride、重复 index 或其他 GPU。

详细图、参数和源码责任见 [vulkan-scratch-reuse-protocol.md](tests/a-seams/vulkan-scratch-reuse-protocol.md)。`backend:G14` 的设备 pinned ownership 没有迁入，不能借本节取得多 GPU 资格。

## Vulkan：源数据原地修改是另一个反例

scratch 修正阶段的后续审查发现，同一 source view 指针的底层 storage 被 `ggml_scale_inplace` 改写，缓存 key 仍可命中旧转换结果。初始零偏移 probe 的 disjoint control 通过；inplace 图中 SCALE 和源存储全部 514 值正确变为两倍，但第二个 matmul 的 64/64 个值错误，最大绝对差约 16.0306。记录的一个值为预期 31.8711、实际 15.9355。

这建立了与 scratch 覆盖不同的公开图正确性问题：barrier 可以排序原始 storage 写入，不能刷新另一个 converted copy。冻结的后续 probe 使用 4-byte-offset 的同一 view、独立同 GPU 原始/预先加倍 reference，并比较全部 matmul 与 storage 值。其修复前第二个 matmul 有 64/64 值错误、最大绝对差 15.4662；重建重装最终 A 后，control 与 inplace 两个模式通过，每个 matmul 的全部 64 个输出，以及 SCALE/源存储各 514 个值的相应比较误差为 0。非零偏移改变了有效数据，所以两个 probe 的误差数字分别保留。实际最终来源见 [Final release gates](#final-release-gates)。

修复的 A 合同是：记录的 tensor write 只要与 cached source 的 storage range 重叠，就失效全部三个转换身份字段，包含 exact alias 和每个 fused-write 输出；不能在找到普通同步依赖后提前停止检查。无需引入新 B 或新的缓存类型。测试和范围见 [vulkan-activation-cache-protocol.md](tests/a-seams/vulkan-activation-cache-protocol.md)。该反例不证明某个模型或 recurrent clear 实际触发了这个序列，也不覆盖所有融合、stride、外部 host 修改或图捕获。

## 全量 Vulkan：最终仍失败，旧反例不撤销

scratch 修正的 +12 阶段全量 backend ops 为 25931/25933。FA 失败配置包含 mask、sinks、Q8_0 K/V 和 head dimension 256；对应单项在 baseline 与 +12 都失败。该次 TOPK 也在全量运行中失败，对应单项在两边均没有复现。

最终包含 source-write 修复的完整 GPU suite 为 25932/25933，exit 1，用时 454.338607 秒，只剩同一 FA case，测试报告 ERR 0.009055835 超过门限 0.0005。TOPK 在这次完整运行中没有复现。重复运行结果的变化没有定位旧 TOPK 的原因，原失败仍保留。

FA 两边失败是限定 case 的对照，TOPK 单项不复现保留了测试顺序、状态或非确定性等未定位可能；不能把它们合并成“已排除回归”。本页不猜测根因，也不以 99% 以上通过率替代明确的失败处置。最终测试状态和普通构建入口回执统一登记于 [Final release gates](#final-release-gates)。

## 同日较早发布候选记录：起点 3746ebf82585

以下保留原文。其“本轮”“当前”及原记录中的 18 项 manifest、三个 B / 四个 C、47 条隔离命令，仅指起点 `3746ebf82585f47c88fb48097f6ae45cb972b031` 的较早批次。本节内部还保留更早的 16 个 A 文件、三 B / 三 C、41 条命令的历史。旧 receipt 不承担本页新增能力和最终 A 的来源证明。

当前 A 固定为 `cda0f9256665845cd9d6f3a0176880883f653a3d`。登记例外共18个文件：17个 A 代码文件合计 +328/-43，另有49行自有 CI 工作流。本轮新增的生产 A 修改只有 Meta 异步偏移一个文件的 +4/-4；其余新行为由 B、C 或独立测试承担。

本轮可移交摘要为 [audit/release-qualification-20261018.json](audit/release-qualification-20261018.json)，记录关键源码摘要、模型身份、实际命令回执、反例和适用范围。原始 stdout、stderr、二进制 tape 与完整命令保留在对应 `build-*` 目录。本页后半保留同日上一轮记录，并明确标为历史验证，不能将其未复跑的部分自动当作本轮结果。

### 本轮实际参与者

- Linux x86-64，GCC 16.2.1；主测试 Python 3.14，CMake CLI 测试使用 Python 3.12.13。
- 模型：`/home/kunweiz/models/lfm25/LFM2.5-8B-A1B-UD-IQ4_NL.gguf`。
- 模型 SHA-256：`a1921216ded964e5c4f787226729e8c2c32bd587ac7b897742e0b7a7cc7eff6a`，本轮重新核对。
- Vulkan：单张 AMD Radeon RX 6800，RADV / Mesa 26.2.2。模型日志实际确认25层卸载；没有把请求的 GPU 参数当作执行证据。

### 本轮验收概览

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

### 准入、排队与真实状态释放

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

#### 保留的跨批形状反例

最初要求并行与串行 greedy token 完全一致的测试失败。随后在本轮准入修改前的原 C 二进制上，关闭 phase release、保持提示与模型相同，独立复现三路与单路的结果差异；单路 KV2048 与 KV256 则完全相同。

原 C 来自本轮起点 `3746ebf82585f47c88fb48097f6ae45cb972b031`。其 CPU、Vulkan二进制摘要、实际命令、确切 UTF-8 stdin 与返回 token 保存在本轮摘要的 `shape_counterexample_before_admission`。CPU组选取原fixture的3/4/5，Vulkan选取0/1/2；已将全部提示token行逐条对上原运行记录。它们后来被完整安装覆盖，复现时应从该提交重建。最初失败记录仍在 `build-release-admission-cpu/run-cga4dqjq` 与 `build-release-admission-vulkan/run-cc0q5fad`，没有改写成成功。

本轮队列测试的 CPU 在请求4的第5个生成 token 出现差异，Vulkan在请求1、4、7出现差异。开关 phase release不改变各自结果。这个证据建立了修改前已存在的批形状敏感性，**没有定位数值原因**，也不能据此认定差异无害。验收明确比较同配置形状、串行容量对照和同槽位重复；跨形状差异继续单列。

### logits：全行采集与有限的独立对照

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

### Meta：公开异步传输的 before / after

新回归直接创建两个真实CPU child backend，覆盖split axes0/1/2与mirrored、F32/F16、3+7不均分片、full/middle/tail、独立get和set，以及前后guard与未触及字节。

原A的零偏移16个控制通过，非零get、set单独运行均在旧 `offset == 0` 断言处以SIGABRT(-6)失败。A修复只移除该过度限制，并把逻辑chunk起点换算为每个child自己的偏移。修复后48例通过。CPU安装的全部A接缝为4/4，Vulkan链接安装为5/5；**Meta用例在两次运行中都使用CPU child**，没有取得GPU异步、多GPU或异构Meta的运行资格。

连续、单segment、chunk对齐与repeat条件保持。没有新增API、后台队列、cache、pin或所有权协议。完整形状与检查方式见 [meta-transfers-protocol.md](tests/a-seams/meta-transfers-protocol.md)。

### recurrent：直接复用当前 A

源码复查表明：当前A在构造与 `clear(true)` 时整buffer清零；`seq_rm` / `seq_keep` 修改元数据，通过空闲zero row与图内初始化/聚集取得新状态。旧M33性能记录减少的是旧fork自己加的eager clearing；把它搬到当前A会增加写入和fence，本轮没有修改A的recurrent代码。

公开接口测试使用上述LFM2.5 hybrid convolution/KV模型，实际 `n_rs_seq=0`。CPU、Vulkan各执行24次decode、189行输入；24个邻居快照与8个共享快照检查通过，合法恢复后同形状teacher-forced继续的最大绝对差为0。覆盖三个槽位两次复用、COW存活者、dirty槽位恢复、`clear(false)` / `clear(true)`再填充，以及不支持的部分回滚失败后原快照不变。batch输入在显式同步前始终存活。

这不等于一般GDN/recurrent行数、MTP、PLE、HTTP取消或跨进程snapshot兼容性证明。源码另发现坏restore可能在完整验证前改变目的状态；**没有宣称损坏snapshot恢复具有事务性，也没有在本轮修复它**。操作边界见 [recurrent-lifecycle-protocol.md](tests/a-seams/recurrent-lifecycle-protocol.md)。

### 独立交付、构建入口和 CI

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

### 本轮范围与后续候选

没有新的整模型吞吐、reset加速、峰值VRAM或自动容量适配结论。CUDA、HIP、Metal、Windows、macOS、多GPU与异构Meta没有取得本轮资格。

auto-fit/slack、PP reserve、idle LRU、物理recurrent槽、HTTP取消/恢复、流式stop/usage、在线corpus、状态envelope、校准/imatrix和其他局部算子继续逐项列在 [MIGRATION.md](MIGRATION.md)。714路径的审计分母与546完整改动内容/168选定范围的阅读记录保持不变，不能用这些测试覆盖所有历史实现。

### 同日上一轮验证记录（保留原证据边界）

以下段落中的“本次”指本轮起点之前的小功能补全；当时为16个A代码例外、三个B和三个C，隔离流程41步。本轮重跑项目以上面的记录为准。其可移交摘要仍是 [audit/validation-20261018.json](audit/validation-20261018.json)。保留旧数值用于回查，不将未重跑的HTTP、corpus互操作或采样测量写成本轮新结果。

#### Prefill / Decode：真正发生了什么

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

##### 分配器生命周期

`tests/a-seams/compute-memory.cpp` 使用真实 GGML CPU 后端：大图分配、排队异步读回、由 release 自己完成 fence、字节归零、重复 release、小图重建、再扩回大图。它检查保留输入/权重、scheduler 和 callback 身份，并执行重建后的计算。Release 构建中的失败检查始终有效。

API 前置检查拒绝 metadata-only context、活动 optimizer 和共享 CUDA phase arena；正常路径同步、失效两个可复用图结果，再释放分配器。模型 / KV / recurrent / output 的所有权保留由源代码边界与真实输出对照支持；没有宣称跨 context 共池或 target / draft 联合分配已经实现。

#### 原版 HTTP server

`tests/server-phase.py` 只启动并清理自己创建的 localhost server。真实设置为 logical 512、physical KV 1536、3 slots、Vulkan。两个 SSE 流各收到至少两个生成块后插入第三个长提示；初始两条流当时均未在客户端结束。

三条请求分别完成 48、48、8 个 token，槽位和 limit 结束原因正确。日志出现 `prefill → generation → prefill → generation`，实际卸载 25 层，`n_ctx_kv=1536`。关闭 Prefill 优先但开启阶段释放的组合，在加载不存在的模型前就报出专用参数错误。

初次 smoke 的三条请求也都完成，但默认日志隐藏 loader 信息，导致设备证据断言失败。最终脚本启用 `--verbose --log-colors off`，保留原来的正数卸载 / 物理容量断言，完整复跑通过，未放宽检查。详见 `build-server-phase-verified-20261018/summary.json` 与 `server.log`。

HTTP 接收时间不能证明内部批次严格暂停；逐批暂停 / 恢复的证据来自上一节 C 的真实批次输入。此次 server smoke 没有证明 MTP、语法、随机 RNG、取消或压力重试等全部协作组合。阶段释放后，A 析构时可能打印实际计算 buffer 与初始预留预测不同的诊断；当前观察接口和结果以实际分配及输出为准。

#### 两个小型正确性修复的 before / after

##### CPU：显式 F32 FlashAttention

针对 F16 V、显式 F32 precision 的实际 GGML CPU 操作，在相同测试中：

| KV 长度 | 固定 A 原实现最大绝对误差 | 修复后最大绝对误差 |
|---|---:|---:|
| 256 | 0.000388145447 | 0.00000011920929 |
| 1024 | 0.000388026237 | 0.00000011920929 |

原实现退出失败；四行级别的累加路径修复后通过。用例为 `tests/a-seams/flash-attn-precision.cpp`。这不是一次整模型质量评分，也不应归为无收益的量化实验。

##### Vulkan：公开 Turbo WHT

同一个 `tests/a-seams/vulkan-wht.cpp` 直接分别调用真实 CPU 与 Vulkan backend，没有混合调度器的 CPU fallback。12 例覆盖 group 32 / 64 / 128、正反变换、513 行跨工作组以及既有 scale 参数。

原 A 失败 10 / 12，仅未缩放 group128 的两个控制组通过。修复两个 A 文件后 12 / 12 通过，本设备最大绝对误差为 0。输出先填 NaN，遗漏写入会失败；所有数值须有限。当前支持仍限制在 128 对齐物理行和已验证 scale 布局，shader 越界返回保持整个 workgroup 一致。

记录：`build-clean-fork/wht-after.log`、`wht-vulkan-build.log`、`wht-vulkan-install.log`；完整复现合同见 [tests/a-seams/vulkan-wht-protocol.md](tests/a-seams/vulkan-wht-protocol.md)。

#### 生产 corpus 格式与旧入口复核

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

#### A / B / C 独立构建与搬迁

```sh
python3 fork/tests/isolation.py \
  --output build-isolation-small-20261018 --jobs 4 \
  --model /absolute/path/to/LFM2.5-8B-A1B-UD-IQ4_NL.gguf
```

以上为可替换模型路径的复现示例，实际命令保存在该目录 `commands.json`。本轮没有使用 `--a-prefix`：A 从 pin 的 Git archive 构建，只覆盖 manifest 已验证的 16 个完整文件，导出 A 不含 `fork/`，其自身 `test-sampling` 通过。

三个 B 分别从只含自身代码的副本构建 / 测试 / 安装；三个 C 分别只消费安装依赖。流程删除源码和构建副本、搬迁安装前缀，再编译实际包消费者，并执行搬迁后的真实 batch 模型比较。41 个命令全部成功。B 的输入 / 输出所有权和实例隔离测试不需要模型服务；C 的模型证明使用真正 A 和 B。

`--a-prefix` 是可选便捷路径，记录为 `reused_unverified`，不能拿来证明该次重新构建了对应 A。独立源码副本和安装消费也不等于操作系统沙箱。

#### 性能资料的边界

第一轮实际 CPU 采样微基准仍作为单一阶段证据保留：248,320 logits、top-k40、唯一最大值，每臂预热 10、计时 200 次，4 个 AB / BA 配对；完整行约 252–312 µs / row，compact 约 29.9–30.0 µs / row。每次使用真实 A sampler 检查 token。这不是整模型的加速倍数。

旧 fork 的 +3.66% compact、TP5 LateBind、router 与 MMVQ / MTP 数据是归档历史，不是这次新分支测量；具体取舍见 [MIGRATION.md](MIGRATION.md)。本轮阶段内存测试、WHT 与 F32 正确性测试都没有给出新吞吐结论。

#### 未覆盖与明确限制

没有证明全部模型、采样链、极长上下文或跨设备组合；CUDA、HIP、Metal、Windows、macOS 未取得本次运行资格。没有证明多 GPU / 五卡 TP5、RERoT 多 reader 共享 KV、FlashPrefill / XKV 整模型质量、设备隐藏层交接、自动显存 fit、压力抢占、跨 context 共池或随机 MTP 恢复。

Corpus 未提供 tokenizer 身份、并发录制、崩溃事务和多分片读取；比较器的文件格式也不携带完整构建 / 模型身份。逐文件静态审计中已识别但未运行的能力保留自己的证据等级，不能因本页中其他测试通过而自动获得资格。

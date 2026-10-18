# AXIOM

1. 在平台资格拿到之前，不能决定 strict mapped-relay 在这台机器上可不可做。
   Reason: “能分配”“能同步后比较”都不等于运行期间的双向可见与热复用
   （指南 31.1）；探针把每一项变成可以复核的平台事实。
2. 协议设计不得依赖 host 内存的系统 RMW。
   Reason: 实测 gfx1030 上 host fetch_add 产生错误值而 CAS 可用；依赖未证明的
   RMW 会把平台缺陷变成静默数值错误（指南 31.3）。
3. 探针的每一个结果都必须区分 pass / fail / unsupported / not_run。
   Reason: 单卡通过不能折算成五卡通过，没有运行的项目不能写成通过（指南 31.7）。

# LEMMA

1. [AXIOM 1] 热路径只用单写者 load/store 控制字：每 rank 独占自己的 ready，
   CPU 独占 generation；控制字是普通 uint64_t，host 与 device 用同一组
   `__atomic` 内建访问，不把 std::atomic 重解释为设备字（指南 31.6）。
2. [AXIOM 2] RMW 按操作独立记录能力位：fetch_add 与 CAS 分开判定；产生错误值
   记 fail，不执行记 unsupported，两者都不进入强制门。
3. [AXIOM 3] 每个等待 kernel 有有界预算与 abort 字，host 侧有 deadline；
   payload 模式由 (generation, index, seed) 决定，任何陈旧、撕裂或别名读取
   必然失配。
4. [AXIOM 1、3] 单卡多流用例验证的是协议软件层，不是五卡资格：五个并发
   consumers 的 credit 全收齐、延迟最后读者、双 bank 轮换、可变有效长度、五流
   RELAY 环、同流逆序依赖识别、双窗口独立前进与 CPU 服务循环（worker 驱动
   进度、双通道背靠背服务、关停排空在 worker 内），都只在证明“给定可见性
   语义成立时，软件协议正确”；跨设备资格仍只由恰好五张设备时运行的
   five_rank_cross_device 给出。
5. [AXIOM 1] 每个 HIP 入口函数在入口处 `hipSetDevice` 固定自己的设备：
   `hipFree` 与 kernel launch 都作用于当前设备，五卡用例结束后当前设备停在
   最后一个 rank，free 与 bench 若不各自固定就会在五卡机上解错设备、跑错
   设备；单卡机上这些路径肉眼不可见，只能靠入口纪律保证。
6. [AXIOM 1] 折减发布门是 host 侧可独立验证的纯算术合同：rank 权重是两两
   互异且和为 1 mod 2^64 的奇数——奇乘是 64 位双射，任意位损坏（含幂权重会
   移出字的高位）必然改变折减；互异使任何 rank 行的复制（缺失贡献）对任意
   rank 对可见。行顺序不属于该门（SUM 可交换，绑定由逐 rank ready 字证明），
   门是全有或全无：先验证全部词再写 out，拒绝的折减不在输出缓冲留下部分结果。
   Reason: 该门与 HIP 无关，无 HIP 的 config-test 必须能钉住全部反例；发布
   前校验是 F04 的核心，部分写穿透会把"已拒绝"变成可误读的中间态。

# SPEC

1. [LEMMA 1、3、4] 双向发布、2000 代热复用、精确票据、生产延迟排序、多工作组
   最后到达、确定性反序完成、abort、逐操作 RMW、五消费者 credit、延迟最后读者、
   双 bank 轮换、可变有效长度、非零偏移内部槽、五流 RELAY 环、有界同步超时路径、
   同流逆序依赖识别、双窗口独立前进、CPU 服务循环、非有限归约门、host 有界等待
   无复用、五卡核索引钉住与五卡用例分别实现并有独立结果。
   Proof: `src/main.cpp` 的各 case 函数；本机全量运行报告 SHA-256
   65146f98ff167bc475d93d8a84fe17208b75ea491affb8e055c3caa13b2e4533
   （RX 6800 / gfx1030，ROCm 7.2.4，2026-10-19，25 用例注册表）。24 用例报告
   b40dd940…6b26 与 01533600…fc73、21 用例报告 b6baeb81…f5be、19 用例报告
   c0a21839…2e1f、18 用例报告 75dbd439…dec4c、16 用例报告 0cdd86bd…9cb8a 与
   12 用例报告 9d1d1429…684112 保留为被取代的历史身份。
2. [AXIOM 3] 证据报告是严格 JSON、排他创建；设备数不等于 5 时五卡用例保持
   not_run；单机强制用例全部通过时 qualification=pass，且报告注明范围。
   Proof: `src/evidence.cpp` 与 `tests/config-test.cpp`；实际报告可由任意
   JSON 解析器读取。
3. [LEMMA 2] 证据配置、模式与序列化不需要 GPU 即可测试。
   Proof: `tests/config-test.cpp` 在无 GPU 机器上可构建运行，其中
   `check_fold_gate` 钉住折减门的全部反例（权重和、互异、干净折减、高位
   翻转、quiet-NaN 位型、行复制与全有或全无写穿透）。
4. [AXIOM 3] bench 测量（`--bench`）是成本模型观察值：延迟用 host steady
   clock 且含 launch 时间，折减带宽由可观察校验和锚定防止死代码消除并以
   32 样本的 min/median/p90/max 汇总（单样本被调度噪声支配）；
   测量结果写入独立的 `tp5.relay.bench.v1` 文件，不进入 qualification。
   Proof: `run_bench` 与 `summarize`；bench JSON 的严格序列化在
   `tests/config-test.cpp` 中覆盖；本机 bench 文件 SHA-256
   a1505a8c524f2fc29e2ff430f663e8e5f593cb8b5a2f017ec85dc6c7e08c7b2c。

5. [LEMMA 3] 设备等待是累计预算：clock64 实测 ~2.3 GHz（gfx1030，2026-10-19），
   预算 24e9 ticks ≈ 8 s，必须超过最长 host 截止期（5 s）；每轮只做短暂停后
   重查，不能把整段预算一次睡满。失败路径 abort+排空，防止超预算等待的
   kernel 在下一个用例复位后晚写共享控制块。定标方法入库：`tq-clock-rate`
   工具在每张可见设备上实测 clock64 速率（固定 tick 自旋 + event 轮询墙钟
   截止期），上真机后先跑一次复核预算假设。
   Proof: `wait_for_ack` / `wait_for_ticket` / `fail_case`；修复前 2000 轮
   预计数小时，修复后全量约 5 秒；`tools/tq-clock-rate.cpp` 本机实测
   median 2320.2 MHz（三次独立运行 2320.1–2320.4 MHz）。
6. [LEMMA 5、6] 五卡跨设备可见性在单卡机上不可能运行：它的第一次执行只能在
   真实五卡机器上发生（five_rank 保持 not_run）。但它的逐 rank 索引不是：新增
   单卡钉住用例 `five_rank_indexing` 用五条流跑与五卡用例完全相同的
   `rank_write_kernel` 与 guard 布局，逐轮验证每个 rank 的 payload 与自己的
   ready 字——两个用例共享同一套轮次协议 `run_rank_rounds`，五卡机上跑的循环
   正是单卡钉住的那份代码。本批次审计先抓到真缺陷并修复：
   `rank_write_kernel` 原先向基址发布 ready（全员踩写 ready[0]，rank 1–4 的
   ready 永不到达，五卡机第一轮就会以 5 s 虚假 stall 归因失败）；修复为
   `release_store(&ready[rank], generation)` 后，回注缺陷版被该用例在活设备上
   确定性拒绝（"rank 1 did not publish ready in round 1 within 5 s"，退出 1）。
   同时修掉两个相邻缺陷：run_five_rank 的 setup 失败预检查的是 setup 失败后
   必被写正的 `record.ordinal < 0`（死条件），改为按 `device_identity` 结果与
   `mapped` 能力位判定，坏卡在核启动前就被精确归因而不是跑出虚假超时；零设备
   分支的退出码改为与枚举失败分支一致的 1。每个 rank 的成功或失败都登记进对应
   设备记录，`five_rank_cross_device` 注册项因此可达。
   Proof: `run_five_rank` / `run_rank_rounds` / `case_five_rank_indexing` /
   `rank_write_kernel`；2026-10-19 单卡 2000 轮 qualification=pass、
   five_rank=not_run、five_rank_indexing=pass，回注缺陷版的确定性拒绝已实测，
   跨设备可见性仍不构成证据。
7. [AXIOM 1、3] 单个 rank 或设备的 HIP 失败不抹掉已收集证据：HIP 错误是异常，
   逐用例捕获并记为对应 case 的 fail，setup 失败的设备以
   `device_identity=fail` 留在报告里，五卡用例在任何 HIP 失败下仍按 rank 归因
   并写出报告；所有原生同步经 `sync_with_deadline` 有界化（event 轮询 +
   60 s 截止期），挂死的队列变成有界、可归因的失败而不是永不返回的运行，
   超时分支由 `native_sync_deadline` 用例在活设备上触发并验证（stall kernel +
   2 s 短截止期 + 停止字恢复）；只有设备枚举失败才可能无报告。
   Proof: `hip_error` / `sync_with_deadline` / `run_case` / `run_probe` /
   `run_five_rank` / `case_native_sync_deadline`；
   2026-10-19 单卡 2000 轮运行退出 0，故障路径由逐项代码评审保证，五卡机
   首次运行前不声称已实测。
8. [AXIOM 3] 证据文件经同目录临时文件写入、fsync、硬链接原子发布：目标路径
   要么完整出现要么不出现，已有文件与符号链接拒绝覆盖；临时名在任何路径下
   清除，链接成功后残留临时名不使证据失效。
   Proof: `evidence.cpp` 的 `write_file_exclusive`；
   `tests/config-test.cpp` 的 `check_exclusive_write` 覆盖首写成功、二次
   拒绝、内容不变与无残留（无 GPU 即可运行）。stdout 证据行同样按此精神
   保全：无缓冲输出 + config-test 的 fork/SIGKILL 存活测试，重定向运行被
   杀不丢已打印行。
9. [AXIOM 3] 证据文件的终态一致性由独立校验器机器检查（指南 53.4）：
   `validate-report.py` 对报告与 bench 执行严格 JSON 解析（拒绝重复键、
   非有限数、未知字段与尾随数据）并重算全部交叉不变量——注册表覆盖、
   rank 与下标一致、five_rank 状态与设备数一致、qualification 与逐用例
   结果一致、bench 序统计单调。未来字段变化必须升 schema 或改校验器，
   不允许静默漂移。
   Proof: `validate-report.py` 的 `--self-test` 与两份本机权威证据实测
   通过；真机证据归档前必须通过校验。

# Exceptions

- 本目录不是独立 B 库：它声明依赖 HIP 运行时与 ROCm 工具链，只在 GPU 机器上
  构建探针可执行文件；证据配置测试无此依赖。`fork/build.py` 默认流水线已构建
  并运行无 GPU 的 config-test（有 ROCm 工具链时顺带构建探针可执行文件）；
  isolation.py 仍排除本目录，因为无 HIP 时没有可安装 target，重装级隔离验收
  不覆盖平台工具。解除条件：无——平台资格探针天然依赖平台工具链。
- gfx1030 host fetch_add 的错误值行为按本机观察记录，不代表其他 ROCm 版本或
  拓扑；每次换工具链或拓扑后重跑，旧报告不延续。
- clang-tidy 在 HIP 代码上存在已知误报，不逐一 NOLINT：`misc-include-cleaner`
  穿不透 `<hip/hip_runtime.h>` 的 portable 头模型（“no header providing
  hip*/blockIdx/clock64”）；`readability-non-const-parameter` 看不见
  `__atomic_*` 内建对参数写的写（release_store/rmw_probe_kernel）；
  `readability-static-accessed-through-instance` 把 HIP 的 `blockIdx` 兼容
  实体当实例；`portability-avoid-pragma-once` 与本仓 `#pragma once` 惯例
  冲突（兄弟 B 库头同用）。此外有少量先于本目录存在、有明确理由的 C++ 风格
  结论按同样方式登记：`misc-misplaced-const`（`const auto` 措辞是刻意选择）；
  `bugprone-empty-catch`（两处 catch 主体是注释——注释即处理，检查器不看注释）；
  `performance-no-automatic-move`（run_relay_round 的 8 词 optional 返回路径，
  拷贝代价为零且 const 防误改）；`bugprone-suspicious-missing-comma`
  （limits 里一处两行字符串字面量，连接文本已核对无误）。真实告警（缺 include、
  未用 using、CLI 越界）已修完；换 LLVM 版本后按此清单复查。

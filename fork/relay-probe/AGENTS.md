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
   消费者的 credit 全收齐、延迟最后读者、双 bank 轮换、可变有效长度与五流
   RELAY 环，都只在证明“给定可见性语义成立时，软件协议正确”；跨设备资格
   仍只由恰好五张设备时运行的 five_rank_cross_device 给出。

# SPEC

1. [LEMMA 1、3、4] 双向发布、2000 代热复用、精确票据、生产延迟排序、多工作组
   最后到达、abort、逐操作 RMW、五消费者 credit、延迟最后读者、双 bank 轮换、
   可变有效长度、五流 RELAY 环与五卡用例分别实现并有独立结果。
   Proof: `src/main.cpp` 的各 case 函数；本机全量运行报告 SHA-256
   0cdd86bd4e2af7b6a2042fcdf64ba376e944b305539d791d8363a5f7c379cb8a
   （RX 6800 / gfx1030，ROCm 7.2.4，2026-10-19）。12 用例旧报告
   9d1d1429…684112 保留为被取代的历史身份。
2. [AXIOM 3] 证据报告是严格 JSON、排他创建；设备数不等于 5 时五卡用例保持
   not_run；单机强制用例全部通过时 qualification=pass，且报告注明范围。
   Proof: `src/evidence.cpp` 与 `tests/config-test.cpp`；实际报告可由任意
   JSON 解析器读取。
3. [LEMMA 2] 证据配置、模式与序列化不需要 GPU 即可测试。
   Proof: `tests/config-test.cpp` 在无 GPU 机器上可构建运行。
4. [AXIOM 3] bench 测量（`--bench`）是成本模型观察值：延迟用 host steady
   clock 且含 launch 时间，折减带宽由可观察校验和锚定防止死代码消除；
   测量结果写入独立的 `tp5.relay.bench.v1` 文件，不进入 qualification。
   Proof: `run_bench` 与 `summarize`；bench JSON 的严格序列化在
   `tests/config-test.cpp` 中覆盖；本机 bench 文件 SHA-256
   062c28060bb8d31a930f0c954a2aa3069ad3045f2f887618cec3e3713a9de0ad。

5. [LEMMA 3] 设备等待是累计预算：clock64 实测 ~2.3 GHz（gfx1030，2026-10-19），
   预算 24e9 ticks ≈ 8 s，必须超过最长 host 截止期（5 s）；每轮只做短暂停后
   重查，不能把整段预算一次睡满。失败路径 abort+排空，防止超预算等待的
   kernel 在下一个用例复位后晚写共享控制块。
   Proof: `wait_for_ack` / `wait_for_ticket` / `fail_case`；修复前 2000 轮
   预计数小时，修复后全量约 20 秒。

# Exceptions

- 本目录不是独立 B 库：它声明依赖 HIP 运行时与 ROCm 工具链，只在 GPU 机器上
  构建探针可执行文件；证据配置测试无此依赖。解除条件：无——平台资格探针天然
  依赖平台工具链；它不进入 fork/build.py 与 isolation.py 的默认流水线。
- gfx1030 host fetch_add 的错误值行为按本机观察记录，不代表其他 ROCm 版本或
  拓扑；每次换工具链或拓扑后重跑，旧报告不延续。

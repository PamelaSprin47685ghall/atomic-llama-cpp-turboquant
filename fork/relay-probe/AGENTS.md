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

# SPEC

1. [LEMMA 1、3] 双向发布、2000 代热复用、精确票据、生产延迟排序、多工作组
   最后到达、abort、逐操作 RMW 与五卡用例分别实现并有独立结果。
   Proof: `src/main.cpp` 的各 case 函数；2026-10-19 在 RX 6800（gfx1030，
   ROCm 7.2.4）上的实际运行报告 SHA-256
   9d1d14295dd0b8658f369be01f4fa17079ce2598c4bc62f98ce1c3d22f684112。
2. [AXIOM 3] 证据报告是严格 JSON、排他创建；设备数不等于 5 时五卡用例保持
   not_run；单机强制用例全部通过时 qualification=pass，且报告注明范围。
   Proof: `src/evidence.cpp` 与 `tests/config-test.cpp`；实际报告可由任意
   JSON 解析器读取。
3. [LEMMA 2] 证据配置、模式与序列化不需要 GPU 即可测试。
   Proof: `tests/config-test.cpp` 在无 GPU 机器上可构建运行。

# Exceptions

- 本目录不是独立 B 库：它声明依赖 HIP 运行时与 ROCm 工具链，只在 GPU 机器上
  构建探针可执行文件；证据配置测试无此依赖。解除条件：无——平台资格探针天然
  依赖平台工具链；它不进入 fork/build.py 与 isolation.py 的默认流水线。
- gfx1030 host fetch_add 的错误值行为按本机观察记录，不代表其他 ROCm 版本或
  拓扑；每次换工具链或拓扑后重跑，旧报告不延续。

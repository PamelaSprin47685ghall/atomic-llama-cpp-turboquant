# TQRelayProbe

TP5 mapped-relay 平台探针（指南 31、39.5 H-cases）。在每张现有设备上验证严格预提交
RELAY 依赖的映射主机内存能力：双向运行期可见握手、同一槽位跨多代热复用（精确票据
匹配）、生产者延迟下的发布顺序、多工作组最后到达发布、有界 abort，以及逐操作的
host 内存系统 RMW 能力位。在此之上再验证协议软件层：五个并发消费者的输出 credit
全收齐（H08 的延迟最后读者单列）、双 bank 轮换的热复用、可变有效长度的热复用，以及
单卡五流仿真的一整圈“五生产者部分和 → host 折减 → 发布 → 五消费者读回”。恰好五张
设备时另跑五卡跨设备用例；否则记录 NOT_RUN。`--bench` 另出一份纯测量 JSON（时延与
host 折减带宽），只供成本模型参考（指南 14），不是资格门。

这不是模型、调度器或整链 RELAY 的资格（指南 31.7）；它只回答“这个平台答应了什么”。

## 构建与运行

需要 ROCm HIP 工具链（CMake 的 `find_package(hip)`）。没有 HIP 时只构建证据配置
测试，探针可执行文件不生成。

```sh
cmake -S fork/relay-probe -B build-relay-probe \
  -DCMAKE_BUILD_TYPE=Release -DTQ_RELAY_PROBE_ARCHS=gfx1030
cmake --build build-relay-probe --parallel
ctest --test-dir build-relay-probe --output-on-failure
./build-relay-probe/tq-relay-probe --output probe-report.json --rounds 2000
./build-relay-probe/tq-relay-probe --output probe-report.json \
  --bench bench-report.json --rounds 2000
```

`--output` 使用排他创建，不会覆盖已有证据；重跑需换路径或先明确删除旧报告。退出码
0 表示本机范围内的强制用例全部通过，1 表示有失败，2 是用法或发布错误。五卡用例在
设备数不等于 5 时保持 `not_run`，不折算成通过。

## 关键设计

- 控制字是普通 `uint64_t`；host 与 device 用同一组 `__atomic` 内建按
  acquire/release 访问，不用 `std::atomic` 重解释（指南 31.6）。
- 热路径只用单写者 load/store 字：每 rank 独占自己的 ready，CPU 独占 generation。
  host 内存 RMW 是独立能力位，协议不得依赖（指南 31.3）。
- 每个等待 kernel 有有界预算与 abort 字，host 侧有 deadline；超时不释放任何
  payload。
- payload 模式由 (generation, index, seed) 决定，撕裂、陈旧或别名读取必然失配。

## 本机已观察事实（2026-10-19，RX 6800 / gfx1030，ROCm 7.2.4）

| 能力 | 结果 | 说明 |
|---|---|---|
| 双向运行期可见 + 2000 代热复用 | pass | 两个方向均无陈旧或撕裂读取 |
| 生产延迟发布顺序 | pass | ready 在半写标记后约 87 ms 才可见 |
| 精确票据语义 | pass | 旧票据继续等待，未来票据报告身份违例 |
| 多工作组最后到达发布 | pass | 8/8 与 2/8 活组两档 |
| abort 有界退出 | pass | 停止字后约 4 ms 退出 |
| host fetch_add（系统 RMW） | fail | 产生错误值（448 而非 7），不可用 |
| host CAS（系统 RMW） | pass | 正确执行 |
| 五卡跨设备 | not_run | 本机 1 张设备 |
| 五消费者输出 credit | pass | 各自完整读取验证后才有 ack |
| 延迟最后读者（H08） | pass | ack 在半读进度后约 200 ms 才出现 |
| 双 bank 轮换 | pass | 2000 代轮换，闲置 bank 从未被写 |
| 可变有效长度（H05） | pass | 2–7 词交替，live 范围外从未被写 |
| 单卡五流 RELAY 环 | pass | 2000 圈部分和折减+发布+五消费者 |

fetch_add 的失败是可复现的平台事实：gfx1030 经 PCIe 对 host 内存的部分 RMW 形式
不保证执行。这正是指南 31.3 要求协议只用单写者消息的原因。完整证据见本机运行产出的
报告 JSON（排他发布、严格 JSON）。

## bench 测量（`--bench`）

测量结果以 `tp5.relay.bench.v1` schema 写入独立文件，字段为整数的 min / median /
p90 / max（延迟单位 ns，带宽单位 bytes_per_second）。median 取排序后上中位，p90 取
`floor(samples*90/100)`。所有延迟含 kernel launch 时间，用 host steady clock；
单设备、单流（五流环单列）。这些数字用于指南第 14 章的成本模型与两窗口重叠预算，
不能当作五卡实测，也不进入 qualification。

本机 2000 轮观察（2026-10-19）：

| 测量 | 单位 | min | median | p90 | max |
|---|---|---:|---:|---:|---:|
| host 发布 → 设备消费 ack | ns | 9,320 | 9,690 | 10,020 | 30,499 |
| 设备生产 → host 可见 ready | ns | 7,220 | 7,770 | 7,780 | 17,770 |
| 单卡五流 RELAY 整圈 | ns | 28,520 | 29,930 | 31,209 | 133,040 |
| host 五路折减带宽 | bytes/s | 65,741,545,049（两遍取优，单样本） |

这些是本机单设备观察值：含 kernel launch 时间，供成本模型（指南 14）预算
双窗口重叠与 CPU 折减窗口使用；五卡数字要在真实五卡机器上重跑。设备侧等待
预算按 2026-10-19 实测 clock64 ≈ 2.3 GHz 定标（代码内注释记录测量方法）。

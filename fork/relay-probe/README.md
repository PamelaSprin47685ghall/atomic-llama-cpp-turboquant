# TQRelayProbe

TP5 mapped-relay 平台探针（指南 31、39.5 H-cases）。在每张现有设备上验证严格预提交
RELAY 依赖的映射主机内存能力：双向运行期可见握手、同一槽位跨多代热复用（精确票据
匹配）、生产者延迟下的发布顺序、多工作组最后到达发布、有界 abort，以及逐操作的
host 内存系统 RMW 能力位。在此之上再验证协议软件层：五个并发消费者的输出 credit
全收齐（H08 的延迟最后读者单列）、双 bank 轮换的热复用、可变有效长度的热复用，以及
单卡五流仿真的一整圈“五生产者部分和 → host 折减 → 发布 → 五消费者读回”。恰好五张
设备时另跑五卡跨设备用例：五个 rank 在同一块带 guard 三明治的 host 分配中各写一个
互不重叠的词，按 `--rounds` 多轮热复用（每轮 payload 随代次变化，陈旧写必然失配），
失败按 rank 与轮次归因；否则记录 NOT_RUN。另有两个单卡可跑的寻址与完成顺序用例：
非零偏移内部槽（指南 39.5 H06，字节 72 处、8 字节对齐但跨 cache line 的槽位双向往返，
arena 前后 guard 抓基址/越界算术错误）与确定性反序的多工作组完成（指南 39.6 P01/P02，
最大编号工作组先完成时 ready 必须保持未发布）。`--bench` 另出一份纯测量 JSON（时延与
host 折减带宽），只供成本模型参考（指南 14），不是资格门。

这不是模型、调度器或整链 RELAY 的资格（指南 31.7）；它只回答“这个平台答应了什么”。

任何单个 rank 或设备的 HIP 失败都不会抹掉已经收集的证据：单用例失败记录为该用例的
fail，setup 失败的设备仍以 `device_identity=fail` 出现在报告里，五卡用例按 rank 归因。
只有连设备枚举都失败时才可能拿不到报告。

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
设备数不等于 5 时保持 `not_run`，不折算成通过。`--rounds` 只接受 1 到 UINT32_MAX
的纯十进制整数；尾随垃圾、前导零之外的空串和溢出都在提交前拒绝（退出 2）。
`--output` 与 `--bench` 必须指向不同文件（退出 2）。

证据文件经同目录临时文件写入、`fsync`、再以硬链接发布（指南 16.3、42.5）：目标路径
要么完整出现，要么不出现；已有文件或符号链接一律拒绝覆盖。不承诺崩溃后的目录项持久
性，只保证进程正常完成时文件的完整性。

证据归档前用校验器做一次机器检查（指南 53.4 的终态一致性合同）：

```sh
python3 fork/relay-probe/validate-report.py probe-report.json bench-report.json
```

exit 0 表示结构、类型与全部交叉不变量成立；exit 1 给出第一条不成立的检查；exit 2
是用法错误。校验器拒绝重复键、非有限数、未知字段与尾随数据，未来字段变化必须升
schema 或改校验器。`--self-test` 在无 GPU、无证据文件的机器上验证校验器自身。
本目录的 config-test 已进入 `fork/build.py` 默认流水线：无 HIP 机器只构建并运行
config-test 与校验器 `--self-test`，有 ROCm 工具链时顺带构建探针可执行文件。

## 关键设计

- 控制字是普通 `uint64_t`；host 与 device 用同一组 `__atomic` 内建按
  acquire/release 访问，不用 `std::atomic` 重解释（指南 31.6）。
- 热路径只用单写者 load/store 字：每 rank 独占自己的 ready，CPU 独占 generation。
  host 内存 RMW 是独立能力位，协议不得依赖（指南 31.3）。
- 每个等待 kernel 有有界预算与 abort 字，host 侧有 deadline；超时不释放任何
  payload。
- payload 模式由 (generation, index, seed) 决定，撕裂、陈旧或别名读取必然失配。
- 每个 HIP 入口函数在入口处 `hipSetDevice` 固定自己的设备（分配、逐 rank 五卡
  用例、bench、释放都各自固定）：`hipFree` 与 kernel launch 作用于当前设备，
  五卡机上跨 rank 复用执行器时不能依赖调用方留下的设备状态。
- HIP 错误是异常而非进程退出：逐用例捕获、逐设备 setup 捕获、五卡路径全容错，
  首个错误保留在对应 case 的 detail 里，后续用例继续收集证据（指南 47.2）。
- stdout 无缓冲（`_IONBF`，C 标准"尽快传输"保证）：重定向运行被杀时已打印的
  证据行不丢；config-test 用 fork + SIGKILL 子进程把这条承诺钉在本机 libc 上。

## 运维交接（五卡真机）

真机上按以下步骤运行并保留证据：

```sh
HIP_VISIBLE_DEVICES=0,1,2,3,4 \
./tq-relay-probe --output probe-report-5gpu.json \
  --bench probe-bench-5gpu.json --rounds 2000 \
  > run-stdout.txt 2> run-stderr.txt
echo "exit=$?"
sha256sum probe-report-5gpu.json probe-bench-5gpu.json
python3 fork/relay-probe/validate-report.py probe-report-5gpu.json probe-bench-5gpu.json
```

机器上可枚举的 HIP 设备多于 5 个（例如另有 iGPU 或第六张卡）时，必须先以
`HIP_VISIBLE_DEVICES` 圈定目标五卡再运行，否则 `device_count != 5`，五卡用例保持
not_run，整趟白跑。设备数与逐卡 BDF 已写进报告，归档时一并核对。

健康五卡机的预期：退出码 0；stdout 五行各以 `device_identity=pass` 开头、以
`five_stream_relay_cycle=pass` 结尾；报告 `qualification=pass`、
`five_rank_status=pass`、每张设备 18 个用例（含 `five_rank_cross_device=pass`）；
`host_system_rmw_u64_add` 可能仍是本机观察到的 fail（平台事实，非强制门）。

任一张卡异常时：退出码 1，但报告照常写出——坏卡的对应用例为 fail 且 detail 带
rank/轮次/期望值/观察值；坏到连 setup 都过不去的卡以 `device_identity=fail` 出现，
其余卡证据完好。stdout 逐卡即时打印用例结果，进程中途被杀时最后一行之前的证据
仍可从 stdout 还原。把 stdout、stderr、退出码、两个 SHA 一起归档，旧报告字节
一律不改写。

## 本机已观察事实（2026-10-19，RX 6800 / gfx1030，ROCm 7.2.4）

| 能力 | 结果 | 说明 |
|---|---|---|
| 双向运行期可见 + 2000 代热复用 | pass | 两个方向均无陈旧或撕裂读取 |
| 生产延迟发布顺序 | pass | ready 在半写标记后约 87 ms 才可见 |
| 精确票据语义 | pass | 旧票据继续等待，未来票据报告身份违例 |
| 多工作组最后到达发布 | pass | 8/8 与 2/8 活组两档 |
| 多工作组确定性反序完成（P01/P02） | pass | 最大编号组先完成时 ready 保持未发布（8 轮陷阱观察） |
| abort 有界退出 | pass | 停止字后约 4 ms 退出 |
| host fetch_add（系统 RMW） | fail | 产生错误值（448 而非 7），不可用 |
| host CAS（系统 RMW） | pass | 正确执行 |
| 五卡跨设备 | not_run | 本机 1 张设备 |
| 五消费者输出 credit | pass | 各自完整读取验证后才有 ack |
| 延迟最后读者（H08） | pass | ack 在半读进度后约 200 ms 才出现 |
| 双 bank 轮换 | pass | 2000 代轮换，闲置 bank 从未被写 |
| 可变有效长度（H05） | pass | 2–7 词交替，live 范围外从未被写 |
| 单卡五流 RELAY 环 | pass | 2000 圈部分和折减+发布+五消费者 |
| 非零偏移内部槽（H06） | pass | 字节 72、跨 cache line，双向往返，arena guard 完好 |

fetch_add 的失败是可复现的平台事实：gfx1030 经 PCIe 对 host 内存的部分 RMW 形式
不保证执行。这正是指南 31.3 要求协议只用单写者消息的原因。完整证据见本机运行产出的
报告 JSON（排他发布、严格 JSON）。本批次权威报告 SHA-256
75dbd439ab9bc04a00fbc7f4dbb6a9da8e10f3f95e9918cf70d06c4c87ddec4c（18 用例注册表，
每设备 17 项，五卡项 not_run）；此前的 16 用例报告 0cdd86bd…9cb8a 保留为被取代的
历史身份，字节不改写。

## bench 测量（`--bench`）

测量结果以 `tp5.relay.bench.v1` schema 写入独立文件，字段为整数的 min / median /
p90 / max（延迟单位 ns，带宽单位 bytes_per_second）。median 取排序后上中位，p90 取
`floor(samples*90/100)`。所有延迟含 kernel launch 时间，用 host steady clock；
单设备、单流（五流环单列）。这些数字用于指南第 14 章的成本模型与两窗口重叠预算，
不能当作五卡实测，也不进入 qualification。

本机 2000 轮观察（2026-10-19，18 用例批次）：

| 测量 | 单位 | min | median | p90 | max |
|---|---|---:|---:|---:|---:|
| host 发布 → 设备消费 ack | ns | 9,730 | 10,300 | 11,060 | 159,090 |
| 设备生产 → host 可见 ready | ns | 7,480 | 7,880 | 8,380 | 1,251,431 |
| 单卡五流 RELAY 整圈 | ns | 30,380 | 32,580 | 34,860 | 2,985,484 |
| host 五路折减带宽 | bytes/s | 36,897,297,649（两遍取优，单样本） |

这些是本机单设备观察值：含 kernel launch 时间，供成本模型（指南 14）预算
双窗口重叠与 CPU 折减窗口使用；五卡数字要在真实五卡机器上重跑。设备侧等待
预算按 2026-10-19 实测 clock64 ≈ 2.3 GHz 定标（代码内注释记录测量方法）。

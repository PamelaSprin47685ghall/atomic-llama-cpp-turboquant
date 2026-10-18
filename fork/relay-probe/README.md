# TQRelayProbe

TP5 mapped-relay 平台探针（指南 31、39.5 H-cases）。在每张现有设备上验证严格预提交
RELAY 依赖的映射主机内存能力：双向运行期可见握手、同一槽位跨多代热复用（精确票据
匹配）、生产者延迟下的发布顺序、多工作组最后到达发布、有界 abort，以及逐操作的
host 内存系统 RMW 能力位。恰好五张设备时另跑五卡跨设备用例；否则记录 NOT_RUN。

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

fetch_add 的失败是可复现的平台事实：gfx1030 经 PCIe 对 host 内存的部分 RMW 形式
不保证执行。这正是指南 31.3 要求协议只用单写者消息的原因。完整证据见本机运行产出的
报告 JSON（排他发布、严格 JSON）。

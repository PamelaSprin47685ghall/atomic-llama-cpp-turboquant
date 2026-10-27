# TQRelay

TP5 五卡 CPU RELAY 的纯 CPU 记账与局部计划库。实现《TP5-最新设计保姆级指南》第
30–39、50、52、54 章中与设备无关的合同：通道 bank 生命周期、精确票据、两种
credit、fail-stop 与排空事实、逐 rank 原生提交登记，以及五 rank 合成 GLU 布局的
分片可行性检查。没有 HIP、没有 transport、没有模型；`feasible` 不是 GPU 资格。

## 组件

| 头文件 | 职责 | 指南章节 |
|---|---|---|
| `tq/relay/ledger.h` | `channel_ledger`：单通道 bank 状态机。一个 bank 承载整条消息（输入+输出，保守模型），只有输入读完、输出发布且全部消费者最后一次读取完成后才可复用。票据从 1 起精确匹配，不回绕。 | 32、50 |
| `tq/relay/submissions.h` | `run_submissions`：逐 rank 原生提交尝试登记。提交调用前先记 unknown，封口（sealed）与排空（drained）只能由真实完成事实推导。 | 35、52 |
| `tq/relay/plan.h` | `compile_glu_plan`：五 rank 局部 GLU 分片计划。覆盖、整块边界、字节与 MAC 记账、GATHER/SUM 两种代数、结构化拒绝。 | 51、54 |
| `tq/relay/detail/arithmetic.h` | 受检 u64 加法/乘法与票据推进；所有派生计数溢出即拒绝。 | 32.5、54.8 |

## 边界

- 单线程、可信调用方、单一签发者。`channel_ledger` 与 `run_submissions` 自身不做
  线程同步；跨线程使用由 C 串行化。
- rank 上限 32：生产者/消费者集合使用 32 位掩码（指南 35.6 的 `pending_rank_mask`
  同宽）。目标为 5。
- 账本只接收 transport 已经证明的事实。调用 `mark_drained(true)` 不会让设备停止；
  超时不证明存储可释放（指南 35.6）。
- 库不包含 HIP、llama 或任何 A 头文件；只依赖 C++17 标准库。
- `compile_glu_plan` 是已验证 Python 参考 `tp5_local_plan.py` 的 C++ 移植，只覆盖
  合成的逐行独立无 padding 块存储几何，不读取 GGUF，不接触真实量化内核。

## 构建与测试

```sh
cmake -S fork/relay -B build-relay -DCMAKE_BUILD_TYPE=Release
cmake --build build-relay --parallel
ctest --test-dir build-relay --output-on-failure
```

测试由四个目标组成：ledger 所有权反例（第 50 章场景组的 C++ 移植）、提交登记
边界、局部计划数字与 dense GLU 独立参考，以及一个与 Python 参考程序的实时
交叉对照（需要 Python 3；缺失时该项报告 skip，不构成通过）。

安装搬迁与无源码消费使用仓库的包边界入口：

```sh
python3 fork/tests/isolation.py --output build-relocated-packages --jobs 4
```

该入口删除本次创建的库源码、构建目录和原安装副本，再从搬迁后的前缀解析
`TQRelay::relay`。`installed-relay` 消费者只链接这个公开目标，验证预约、发布、
慢读者阻止复用及最后读取后的复用；不链接 llama 或其他 B 库。入口还校验导入库
位置，禁止回退到另一个安装前缀，并清除编译器与加载器的外部头文件、库搜索环境
覆盖；移除的覆盖项记录在 `environment.json`。这不是 OS 沙箱，也不提供 HIP 或模型资格。

## 为什么需要它

旧 fork 的 RELAY 收益历史记录在案，但整条私有 runtime 暂留归档。本库把其中与
设备无关、可独立审查的部分先落地：错误调用必须被拒绝、资源不能提前重新分配、
合法最后读者完成之后仍有可继续的路径（指南 50.10）。后续 C 与 HIP provider 消费
这些合同；平台探针与整模型资格按指南 39、46 章另行收集。

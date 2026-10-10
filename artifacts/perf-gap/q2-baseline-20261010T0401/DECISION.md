# 本轮结果：一个实机参考样本，一个 HC 分区代数原型

2026-10-10。目标仍为 RERoT 关闭、Q2_0 源模型、五张 RX6800、问题“如何减肥”、约200 token；有效最优性差距最终≤1.5，向≤1.2收紧。

**本轮没有取得 GPU 提速结论，也没有取得最终合格基线或全局必要性下界。** 已完成真实参考运行和代码原型，不把未执行的候选写为结果。

## 1. 实机参考

实际目录：`A-q2-h0-run2/`。同一新冻结程序，原Q2_0两分片，h0、LateBind off、MMVQ disabled，显式attention分片，22 CPU线程。用户消息严格为“如何减肥”；system及全部请求参数见 PLAN.md、system.txt 和各原始请求。

| 项目 | 本轮实测 |
|---|---:|
| 暖测量请求数 | 2，同一进程，非两个独立样本块 |
| 每次公开输出 | 140 token，自然 stop |
| 池化生成吞吐 | 50.8069412443 tok/s |
| 平均生成时间 | 2755.529 ms |
| 平均prompt时间 | 2455.8135 ms |
| 平均完整请求墙钟 | 5531.3701195 ms |
| 加载及服务准备 | 39395.534192 ms，另计 |
| measured fallback日志行 | 0 |
| warmup / guard fallback日志行 | 5 / 1，保留 |
| 退出 | returncode=0，无强杀 |

暖请求正文与冷warmup正文相同；实际映射DSO与新快照哈希核对无差异。原始证据为 request-0..3.json/log、result.json、mapped-libraries.json；复算在 A-raw-audit.json。

**资格：仅诊断。** 140低于本轮预注册160–240长度区间，不把它冒充约200-token最终排名。主回答语言流畅、切题，但这不是全面质量认证。guard输出“数组的访问速度取决于索引位置”表述不当，不能作为能力保留通过证据；这是原参考自身的输出，不归因于尚未执行的数值候选。冷相位与guard的fallback也不能因暖请求为0而删除。

## 2. 两个真实测量环境问题

旧 router-bf16-native-bin 的Vulkan库与其identity.json不一致；实际哈希与旧清单差异完整保留于 frozen-identity-mismatch.json。没有覆盖旧清单，也没有推定其原因。

本轮在无GPU实验时构建当前服务，并复制可执行文件与全部配套DSO到 measured-bin/。source-identity.json记录构建前源身份，binary-identity.json记录构建前后源未变及实际快照哈希。model-identity.json包含原模型、衍生模型各两分片及draft的完整SHA256；rank-identity.json来自实际新库的设备属性，不按DRM编号猜物理rank。

首次A在加载前被另一轮 `cmake --build ... -j12` 编译阻止，见 A-q2-h0/result.json 与 blocked-build-processes.json。第二次在取得GPU锁前受阻；后查锁时未发现持有者，没有猜测PID。空的 A-q2-h0-run/ 留存；该次异常为 `BlockingIOError: [Errno 11] Resource temporarily unavailable`，未启动服务。

runner因此补上请求边界的编译检查，并把取锁失败纳入result.json异常记录。未终止他人进程，未改时钟、功耗、驱动或恢复保护。边界检查仍不能证明不存在检查点之间的全部短命进程。

## 3. 已落地的结构原型

代码：`scripts/tp5-hc-coordinate-prototype.py`。
测试：`tests/test-tp5-hc-coordinate.py`。

这不是Vulkan内核，而是纯Python浮点的单HC site代数验证器，独立于GPU编译/测量。对照当前 qwen4exp.cpp 的 combine→mix，覆盖grouped RMSNorm、gamma、down、SiLU、up/sigmoid门、stream mean、inject，以及下一次2*sigmoid(inject/HC)残差更新。此前简化的“down/up”表达没有代替完整site。

算法 C：每个rank拥有所有HC stream中的同一坐标集合 J_r。

```text
C1. 在J_r上reduce-scatter完整Y，再计算 X_b=R_b+s_b*Y。
C2. 本地累加 X_b²、D_b*(gamma_b*X_b)、I_b*(gamma_b*X_b)。
C3. 归约这些小中间量；用各stream完整范数得到rho_b。
C4. 按rho_b合成down结果，除以HC，执行SiLU；合成完整inject小向量。
C5. 仅计算J_r对应的up输出行、sigmoid门、归一化输出和stream mean。
C6. 测试程序汇集结果，与物化参考及下一步残差对拍。
```

J_r不交且覆盖全部坐标，因此平方发生在完整Y之后，不会把 `||sum z||²` 错写为 `sum ||z||²`。down与inject的重排依据线性性，SiLU和sigmoid的位置保持不变。生产中C6的布局传播/汇集不是免费操作；本原型没有隐藏这部分成本为0。

6项测试通过，其中主属性测试覆盖36组行数/宽度/种子；另外覆盖非连续分区、空rank/尾段、重复/缺失坐标拒绝、正负partial抵消、零值/大有限值与非法输入。对拍容差为预设的float64 rel/abs=1e-11。

未取得的资格：真实模型权重/activation对拍、F32/量化误差、VkBuffer/epoch/双bank/active-capacity协议、KV/recurrent完整状态、完整target性能。该方案还可能因新增因果通信而败给LateBind；减少复制权重不等于必然更快。

## 4. 未执行与下一决策

B–E批量执行请求被平台拦截，没有运行原Q2 MTP1、HC-Q8/off、HC-Q8/LateBind、HC-Q8/MTP1对照。未拆分或改写该请求绕过拦截。没有这些候选的本轮速度、质量或胜负结论。MMVQ选择步骤F也未执行。

下一个技术决策仍需同契约候选结果，不能由旧APEX成绩填空。HC坐标分区现在有可运行的代数入口；其Vulkan单site实现应包含inject和实际下游布局，并先对拍再计完整成本。长度不合格的旧样本继续保留，不事后放宽门或改写本轮提示词来晋级。

全局L、U/L均未获证明。50.81不是硬件极限，也不是一个已经达到1.5倍门槛的基线。

## 5. 验证与改动范围

CPU测试：基准工具22项、HC坐标原型6项、必要性审计21项，共49项通过，见 cpu-tests.log。后者是审计与反例测试，不是自动数学证明器。

本轮只改基准工具、CPU原型、对应测试与本目录证据。没有修改生产C++/shader，没有commit/push，没有将其他会话的已有改动算为本轮贡献。

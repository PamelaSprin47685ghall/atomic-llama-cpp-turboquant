# r24 CPU 参照对拍 verdict（identity/quarantine-cpuparity-r24/）

## 执行状态

GPU 已释放（strata 退出，KFD 进程列表空）。r24 fused 臂（arms/parity-fused-r24/）一次运行成功：不崩（断言 0）、全量 dump 采集成功（weightsA 6553600 bytes×2 = 3276800 u16 全量——r23 增强生效）、bin 9a69607d（树状态确认无需重编）。

## W 字节复查——时间线矛盾终局裁决

- **W 非零**（全量计数非零、first8=[15734,15269,48193,48298,48309,15524,15572,15622]——与 r16/r22 SYNCCMP 一致）。
- **W 与 GGUF 逐字节一致**：blk.47.hc_ffn_down（ne=(10240,320)）与 hc_ffn_up（ne=(320,10240)）的 GGUF 原始字节 vs dump 前 4096 u16 **完全相同**——dispatch 绑定的权重视图=原始权重，**无值错、无错位**。
- **时间线矛盾裁决**：r6/r7 的「W 仍全零」历史观察是 dump deferred 伪影（r15 已证）；r22 的 lo/mixed「量级合理值错」判据有效——**W 在设备上从未为零，权重值正确**。写入可见性（V）嫌疑排除。

## CPU 参照对拍（纯 python，公式与 AGENTS.md 工单 B 数学一致）

形状自洽：norm=[10240]=4·width（width=2560）、W_down 320 行×10240（ne[0]=10240 连续）、lo=[320]、W_up ne=(320,10240)、mixed=[2560]=width。

**lo 对拍（down 段）**：lo_ref=SiLU(0.25·W_down@norm) vs dump lo——**巨大分歧**（lo[0] dump=+0.132684 vs ref=-0.014946，符号相反；前 32 输出 worst rel err 3.61）。6 种公式变体（0.25 位置/无 SiLU/4 倍缩放）**全部不匹配**（maxerr 0.44-26）——排除公式歧义。dump lo 与 r16/r22 跨轮逐字节一致（确定性，非随机）。

**mixed 对拍（up 段，双重判别）**：
- 用 dump lo（kernel 错值）作输入：mixed worst rel 2.59（mixed[1] 差 2.6e-3——个别坐标碰巧接近）。
- **用 lo_ref（CPU 正确值）作输入：mixed worst rel 4.80——仍大分歧**——**up/gate 段自身数值也错**（不只是 down 错误的传播）。

## 综合裁决

1. **W 正确**（GGUF 逐字节）——权重值错/绑定错位排除。
2. **normalized 假设成立**（H2 排除：入口还原对照臂 fused 不激活即输出正确——norm 段两臂共享，norm 错无法解释该对照）。
3. **lo 数值错（down 段）+ mixed 数值错（up 段，正确输入下仍错）**——**fused HC segment 的两个 BF16 变体 kernel（qwen4_hc_down_silu_bf16 与 qwen4_hc_project_fold_bf16）数值计算错误实锤**。乱码根因的运行时数值级定位完成：分歧不在权重、不在 norm、不在公式歧义——在两个 BF16 收缩 kernel 的计算过程内部。
4. 与 r22 的「第 3 token 起分歧」自洽（HC 段整体数值损坏 → 状态污染从首个 decode 步开始）。

**判读边界（如实标注）**：(a) 参照公式取自方案文档（bf16-kernel-auditor 从 qwen4exp.cpp 总结）+ AGENTS.md 工单 B 数学交叉印证，但 qwen4exp.cpp:382-400 的精确实现未由本轮直接读取核对；(b) normalized 未做独立数值验证（H2 靠对照臂逻辑排除而非逐值验证）。

## 下一步（交 Manager/Engineer）

1. **Engineer 静态深查两个 BF16 kernel 的计算过程**（S2 候选维持：dequantize4_2aligned 的 packed32/iqs 组合语义——两个 kernel 都错指向共享的解码/累加路径而非各自独立缺陷）。
2. **FORCE_ITER 判别**（GGML_VK_HC_BF16_FORCE_ITER=1 强制 iter() 标量路径跑 off 臂）：乱码消失→错在 aligned 专属路径（S2 实锤）；仍乱→共享路径（行映射/reduce）。一轮可裁。
3. 修复后验证路径已就绪：本对拍脚本可直接复用（dump → CPU 复算 → 逐元素对拍）作为回归门。

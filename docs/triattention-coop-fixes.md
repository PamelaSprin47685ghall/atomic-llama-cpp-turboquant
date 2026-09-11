# TriAttention 组合兼容修复

本修复针对存储坐标、回收提交顺序、draft 容量规划和跨请求状态，不修改已有数值测试的容差。

## 行为

| 组合 | 修复 |
| --- | --- |
| Turbo K + context shift | 不再跳过位移。逆存储变换、相对 RoPE 相位、重新量化；未发生位移的行保持原字节。 |
| Tri + CPU | host buffer 使用原生 memmove，整批边界检查在写入前完成。 |
| Tri + Vulkan / 转置 F16 V | 对齐范围保留 transfer 路径；未对齐范围通过设备端 scratch 和按字节复制 shader 搬移，保留边界外字节。描述符在 preflight 分配并复用。 |
| 回收失败 | 先在元数据副本上规划淘汰和所有 stream 的搬移，全部 preflight 成功后才搬移和提交。能力拒绝及评分/规划异常不删除 live refs。设备执行失败仍属于不可恢复的后端错误。 |
| Tri + 额外 K Hadamard | writer、scorer、shift 共用创建 cache 时确定的旋转块；scorer 撤销额外旋转后才逆 RoPE。 |
| Tri + 普通 q4/q5 K | 使用实际类型的解码器，不再打印错误后以全零分数继续回收。普通类型不使用 Turbo 的 128 维 padding。 |
| Tri + 未校准 nextn draft | target 不允许校准层集合零覆盖；MTP draft 零覆盖时明确记录并使用 draft-only 最近项保留策略。不是 nextn Tri 校准，也不是 target 策略替换。 |
| 只有 draft 存在 KV 压力 | draft 回收不再依赖 target.changed。使用 draft 自己的执行槽位 hints，而不是 target 的 archive/parked ID。 |
| prompt cache 原地复用 | 未加载新状态时保留 compressed 标记；只有实际反序列化成功才重算。 |
| 多序列 / 多 stream / shared prefix | 无 hints 时枚举所有驻留序列，评分读取对应 stream。仅删除引用而未释放物理 cell 也报告 changed，触发视图失效。 |

Turbo K-shift 当前使用 host codec 和按层/stream 的快照，不宣称是 GPU 融合 kernel；Tri 的 Vulkan compaction 仍在设备端完成。缺失校准的 nextn draft 明确退回最近项保留，不能据此宣称已完成 nextn importance 校准或接受率验收。

## 回归测试

- `test-triattention-score`：同一份量化字节的独立 dense-Hadamard 对照；q4/q5/q8/Turbo、64/128/192/256 head dimension、额外旋转、部分 RoPE、Turbo padding，以及零校准覆盖策略。
- `test-triattention-kv`：真实 cache 对象；seq 1、非 unified stream、共享前缀、规划异常、后端能力拒绝、K/V 搬移字节、Turbo2/3/4 位移和转置 V。
- `test-backend-memmove`：CPU 必跑；有 Vulkan 时测试其实际设备路径。覆盖重叠、strided、未对齐边界和整批错误检查。
- `test-server-triattention`：当前 prompt 的 sticky 状态、实际加载后的重算、target 无进展时仍调用 draft 回收。

示例构建：

```sh
cmake -S . -B build-tri-fix -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-tri-fix --target \
  llama-server test-triattention-score test-triattention-kv \
  test-server-triattention test-backend-memmove -j 4
ctest --test-dir build-tri-fix --output-on-failure \
  -R 'test-(triattention|server-triattention|backend-memmove)'
```

纯 CPU 对照使用另一个构建目录并设置 `GGML_VULKAN=OFF`。RERoT、FlashPrefill、server-task 和 Vulkan MoE threshold 的原有测试也应运行。

这些是组合正确性回归，不代替真实模型的长轨迹、MTP 接受率、多请求压力或长稳验收。

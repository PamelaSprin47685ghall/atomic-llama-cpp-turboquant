# TP5 正交优化：输出与依赖契约

这份记录对应 2026-09-21 的源码增补。没有在本轮编译、运行 CPU 测试或启动 GPU；不据此宣称性能已经达到最优。

## 分开三种决定

矩阵内部采用 native contraction 还是 MMVQ，不应决定外层融合是否合法；输出采用本地 F16 companion、P1 或 direct-host，也不应决定后续算子能否继续读取本地 partial。算术、写出方式、张量生命周期是三个不同的决定。

共享工作树中的 MMVQ region 和 direct-host 向量化实现保持原样。本次补齐的是这些实现相互组合时的输出契约，不修改 CPU speculative、模型切分或其他协作者的图定义工作。

## 省略本地 partial 必须有图上的依据

旧的动态 route 只知道是否启用了 LateBind，却不能证明原图里没有其他读取者。现在在录制具体图时执行一次 `ggml_vk_tp5_output_can_elide_local`，检查 producer 之后的实际读写范围，包含非零 offset 的 view 和不同 tensor 身份的重叠存储。

检查结果记录在该 dispatch 的 push constants 中：`MAT_VEC_FUSION_FLAGS_TP5_ELIDE_LOCAL = 0x100`。普通 replay 和预定义 command IR 都直接复用该许可，不在每 token 重新遍历图。

Shader 的最终规则是：

```text
保留本地 z_p = 没有图级删除许可 OR route 要求保留
```

| 情况 | 本地 F32 partial |
|---|---|
| 原图存在后续读取者，或无法确认存储范围 | 保留 |
| 一个 stage 不是五个 rank 全部 direct，仍需 P1 | 保留 |
| LateBind precompute 需要 z_p | 保留 |
| 已证明无读取者，且整个 stage direct、无 LateBind | 允许省略 |
| 本地 F16 companion 路径 | 始终保留正常 F32 输出 |

该许可同时接入普通 Q5_K、MMVQ Q5_K、MoE fold/shared-down 与 K128。K128 的 push constants 从只有 `width` 的 4 字节变为 `width + fusion_flags` 的 8 字节，主机结构及 shader 同步更新；所有 pipeline 创建处仍使用同一个 `sizeof`。

混合 rank 的 stage 也更新直接写出 rank 的 route，而不是因为整个 stage 不能去掉 P1 就跳过更新。其 bank、epoch、ready 都随当前调用更新，并强制保留 P1 要读取的本地结果，避免复用上一轮的 `keep_local=false`。

新增图外消费者时，必须把它需要本地 partial 的事实传入 route；不能仅凭“terminal”名称推定该结果无人读取。

## 本地 wire 和 direct-host 共用 packet 写出

`mul_mat_vec_iface.glsl` 为两种路径提供 `tp5_write_wire4` / `tp5_write_wire2`，完整的连续块分别采用四元素或双元素向量写出，剩余单元素使用标量写出。

本地 F16 companion 使用 `f16vec4` / `f16vec2`，不继承 host-memory 的 coherence 要求。direct-host 根据 wire 类型使用 F32 或 F16 向量。非对齐起点回到精确的标量范围，不对邻居数据做 read-modify-write，也不越界补写 padding。

共享的 `mul_mat_vec_base.glsl` 在 contraction、bias/scale 和非线性 epilogue 完成之后组织输出块。native 与 MMVQ 因而使用同一套写出规则，保留各自的量化和累加方式，不增加 dispatch、跨 workgroup 自旋或新的全局 barrier。

向量写出是否比驱动合并的标量写出更快，仍须由 GPU 测量回答；这里没有把 GLSL 向量宽度当成 PCIe 事务计数的实测证据。

## Descriptor 查询不应顺带要求 BDA

`ggml_vk_tp5_tensor_dev_ref` 现在只取得 buffer、offset、size 和 owner，并检查实际范围。它不再因为 `bda_addr == 0` 就调用 `vkGetBufferDeviceAddress`，因此检查图和使用 storage descriptor 不会偷偷引入设备地址需求。

显式需要地址的调用仍使用 `ggml_vk_tp5_get_tensor_bda`。缺少设备能力或有效地址时返回 0，不能把 `0 + view_offset` 当作有效地址；地址加法和 view 范围都采用防溢出检查。设备创建时的 BDA、descriptor indexing 等能力没有被本补丁关闭。

## 统一构建后的核验入口

新增 CPU-only target：`test-vulkan-tp5-output-liveness`。测试不初始化 Vulkan、不加载模型、不向 GPU 提交工作。覆盖 metadata-only tail、直接读取者、别名读取者、非零 view offset、其他 writer 复用范围、无法解析的存储、重复节点、循环 metadata、范围与地址溢出。

该 fixture 验证的是图/存储策略，不是 GPU 浮点结果、host-device 可见性或速度。后续 GPU 对照仍需检查 MMVQ 与 native、P1 与 direct、F16 与 F32、LateBind 与非 LateBind，以及 mixed-rank fallback 的组合；特别要确认重放第二次及 bank 交替后仍保持正确。

本补丁不新增用户环境开关。没有 stage、commit 或 push 共享工作树，避免将尚未完成的并行改动打包提交。

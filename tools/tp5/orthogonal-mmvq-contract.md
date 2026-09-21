# TP5：MMVQ 与融合、通信输出的组合契约

本记录覆盖 QSA/GDN projection 与 Q5_K 输出的源码整合。按用户要求，本轮没有编译、运行测试、加载模型或控制其他工作进程。性能和数值结果均待统一构建后验证；不能据源码结构宣称已达到硬件最优。

## 三个决定互相独立

1. 图上的 shape、alias、边界和存储生命周期决定融合是否合法。
2. 权重类型、设备能力和现有策略决定 contraction 使用 native F32 activation 还是 MMVQ 的 Q8_1 activation。
3. producer 输出选择普通 tensor/P1、本地 F16 companion 或 direct-host。

局部 MMVQ 建议不再作为整个 HC/MoE/Attention/GDN region 的否决条件。也不能为了直接写通信 payload 而撤销已经选中的 MMVQ。某个传输 epilogue 缺失时，保留已经选好的计算，正常写 tensor 后由 P1 处理通信；不是把 MMVQ 关掉。

这不意味着所有算子、精度和 shape 都有同一个融合整数实现。当前新增的组合是下面列出的范围，不能将未实现的格式计作 MMVQ 命中。

## 四投影 region 内真正使用 MMVQ

`qwen4_attention_projections.comp` 保留原四投影 region。新增的 `qwen4_attention_mmvq.glsl` 使用项目已有的 `mul_mat_vecq_funcs.glsl` 权重解包和 integer-dot 实现，而不是复制另一套量化格式解释。

| region | 可选 MMVQ 成员 | 保留原算术的成员 |
|---|---|---|
| GDN projections | 两个 Q5_K、成对 Q6_K | 没被当前策略选中的成员 |
| QSA projections | 一个 Q5_K、成对 Q6_K | BF16，以及没被选中的成员 |

一次 GPU Q8_1 输入准备供同一 region 的所有整数 contraction 共用。量化输出是原生 `block_q8_1_x4` 布局，使用已有 quantizer。原四投影 fusion 不拆成四个 generic GEMV。

选择采用 specialization constant `QWEN_MMVQ_MASK`，不是额外的 runtime push field：bit 0 为 Q5[0]，bit 1 为 GDN 的 Q5[1]，bit 2 为 paired Q6。GDN 注册 mask 1..7，QSA 注册 1、4、5；mask 0 使用原 native pipeline。所有组合共用原 16-byte push-constant ABI。pipeline 仍通过系统既有的 lazy registration/request 路径使用，未新增运行时 autotune、GPU 探测或同步。

整数循环保留原 MMVQ 的 4/2/1 展开和每个 lane 的 K 尾部处理。workgroup/subgroup 的既有资格检查仍保留，不允许把多个 subgroup 的结果误当作一个 subgroup 的完整结果。

## Q5_K 输出和系统内置 GEMV 共用 epilogue

`ggml_vk_tp5_q5k_output` 是公共输出 lowering，同时由以下入口调用：

- QSA/GDN fused region 中的输出 projection。
- 原生 `ggml_vk_mul_mat_vec_q_f16` 的合格单向量 Q5_K 分支，包括其已有的 bias epilogue。

它不重新决定 activation 精度。输入已经是 Q8_1 时只选择整数 contraction 的输出 variant；输入仍是 F32 时只选择相应 native variant。不满足单向量、shape、buffer 范围或传输 variant 条件时返回原计算/P1 路径。

direct-host 与本地 companion 共用其他协作者维护的向量写出和本地 partial 生存期规则。公共 lowering 不自行推断“terminal 一定无人读取”；只有图上的 `wire_local_elision_safe` 许可才能设置 `MAT_VEC_FUSION_FLAGS_TP5_ELIDE_LOCAL`，shader 还会结合 route 的 keep-local 要求。LateBind 或 mixed-rank fallback 需要的局部值必须继续保留。F32 wire 不能回退成一个未被消费的 F16 companion。

## Q8 scratch 不得偷偷提交正在录制的图

融合内的输入准备不再调用 `ggml_vk_preallocate_buffers(ctx, subctx)`。该函数接受活动 subctx 时会结束、提交并同步当前命令；不能把这种行为藏在一个看似普通的量化 scratch 扩容里。

现在 region 有独立的 device-local scratch，按容量增长。旧 allocation 通过 shared owners 保留，当前记录使用的精确 allocation 同时进入现有 `recording_quant_owners`，随后进入 replay/program 的 buffer owners。普通异步记录的旧 allocation 在 backend 的正常同步清理之后才释放。它不占用或污染 standalone matvec 的 `prealloc_y` 有效性记录。

每个录制的 region 都包含自己的输入量化。不能单凭相同 tensor 指针断言数据没变，也不能跨 token 缓存量化值。region 内的四个 sibling 才共享这一次结果。scratch 重用的读写依赖显式录入命令，量化后的 RAW 依赖复用原 helper；没有 host callback、额外 queue submit 或新增自旋。

## 精度边界

MMVQ 先将 F32 activation 量化成 Q8_1，和保持 F32 activation 的 contraction 不是逐位等价的数学路径。“正交”指合法实现可以组合，不是允许把额外量化隐去，也不是保证所有开关的收益相加。

数值回归要分别比较：native fused 对 native standalone；MMVQ fused 对同一 Q8_1 算术的 standalone。native/BF16 未改算术的部分保留 bitwise 核验；整数 region 允许既定的 F32 accumulation tolerance。不能用贪婪输出相同代替全部 logits 或中间值一致。

## 统一构建后可用的回归入口（本轮未执行）

```bash
# 保持 activation 为 F32 的旧数值对照
env -u GGML_VK_FORCE_MMVQ GGML_VK_DISABLE_MMVQ=1 \
  build-tp5/bin/test-vulkan-command-replay --attention-projections-only

# 默认策略：检查当前设备实际选择的融合/native/MMVQ 组合
env -u GGML_VK_FORCE_MMVQ -u GGML_VK_DISABLE_MMVQ \
  build-tp5/bin/test-vulkan-command-replay --attention-projections-only

# 仅用于数值覆盖，不是性能策略：强制覆盖 Q5 与 paired Q6 的整数组合
build-tp5/bin/test-vulkan-command-replay --attention-mmvq-only
```

新入口在 backend 初始化前设置 force，使用一个设备上的小型合成张量，不加载模型。除原有 offset、odd rows、fallback shapes 和变动输入外，加入 768/1280/3072/3584 的 K 尾部；在小图仍然存活时用 8192-wide 图扩大 scratch，再核验旧图真正 replay 且结果跟随新输入。

只读 proc `ggml_backend_vk_get_region_mmvq_stats` 返回的是**录制计数**，不是运行次数或计时。测试用它核验一组四投影只有一次 Q8_1 准备，以及 GDN/QSA 的整数 variant 确实命中；能力不支持时明确 SKIP，不能把 native fallback 算作 MMVQ 通过。

已有 `GGML_TP5_RELAY_DEBUG=1` 下，一种 region/mask 只报告一次 `[vulkan-region-policy]`，不要求用户再增加一套开关，也不在正常 warm token 上逐轮输出。

后续性能 A/B 仍须固定同一模型、请求、attention replicate、wire 与 transport。先在两种 activation 算术各自的参照中完成正确性核验，再计入端到端吞吐。本次未声称已消除全部瓶颈，未 stage、commit 或 push 并发工作树。

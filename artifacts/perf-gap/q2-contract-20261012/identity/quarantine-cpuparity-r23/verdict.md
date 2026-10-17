# r23 CPU 参照对拍 + S1 静态核查 verdict（identity/quarantine-cpuparity-r23/）

## 执行状态

- **dump 增强已落地**：weights dump 的 copy_bytes 从 64 元素改为全量 subbuffer（r23 插桩级修改——方案文档 §3.1 的「dispatch 实际绑定权重」全量需求），编译成功（libggml-vulkan 9a69607d…）。
- **CPU 参照对拍被 GPU 阻塞**：fused 臂运行失败于 runner 前置检查——`RuntimeError('GPUs not idle; no reset or recovery is attempted')`。占用者：**strata（PID 1450245，5 卡 71.4GB VRAM，非我方进程）**。按 AGENTS.md 红线 2（不杀别人进程、遇锁冲突停止自己的计划）停止 GPU 计划。**全量 dump 未采集、numpy 对拍未执行**——如实报告，交 GPU 释放后重跑。

## S1 静态核查——决定性排除（本轮实质成果）

S1 假说（host 侧 push constant 单位契约随 QUANT_K 变化——BF16 的 A 索引错位 32 倍或 2 倍）经三层静态证据排除：

1. **变体注册**（vulkan-shaders-gen.cpp:1133-1147）：qwen4_hc_down_silu_bf16 / qwen4_hc_project_fold_bf16 的宏字典只含 DATA_A_BF16=1、B_TYPE=float、D_TYPE=float、HC_DOWN_SILU/HC_UP_FOLD——**不含 QUANT_K**（无量化参数注入）。
2. **QUANT_K 语义实锤**（vulkan-shaders-gen.cpp:1130 注释原文）：「**QUANT_K=1/QUANT_R=1 via DATA_A_BF16**; dequantize4 handles the…」——BF16 变体的 QUANT_K=1（元素语义）——A 索引 `ib=(weight_row*ncols+col)/QUANT_K = 元素偏移`（正确）。
3. **push constants 单位一致**（ggml_vk_segment_project:16970）：`pc{ k, k, k, m, nelements, k, m, … }`——k=weights->ne[0]（元素单位）、stride_a/b=k、batch_stride_a=nelements——与 shader 侧 B 的 `blocks_per_row=ncols/B_QUANT_K`（B_QUANT_K=32 仅 HC_DOWN_Q8_DOT 变体定义，BF16 变体不涉及）及 A 的元素索引语义全部一致。K_PER_ITER 对非量化族（F32/F16/BF16）=4——元素单位自洽。

**S1 排除**：BF16 的 host↔shader 单位契约正确，不存在 32 倍/2 倍错位。

## 剩余嫌疑格局（S1 排除后）

- **S2（最强剩余嫌疑）**：iter_aligned_nonquant 的 dequantize4_2aligned packed32 索引（`(a_offset+ib)/2` 整除截断 + iqs 元素内偏移语义）——bf16-kernel-audit 列为「S2（中）」未排除；BF16 的位展开审计（§1 排除项）覆盖的是 dequantize 函数族，但 aligned 路径的 packed32/iqs 组合语义待深查（dequantize4_2aligned 实现在 mul_mat_vec_base.glsl）。
- FORCE_ITER 判别（GGML_VK_HC_BF16_FORCE_ITER=1 强制 iter() 标量路径）：**待跑**（GPU 被阻）——乱码消失→S2 实锤（aligned 专属路径）；仍乱→共享路径。
- CPU 参照对拍（全量 dump + numpy 复算 lo/mixed）：**待跑**（同上）——对拍是 S2 的数值级判定。

## 下一步（GPU 释放后，一轮可完成）

1. 重跑 fused 臂（全量 weights dump 已就绪的 9a69607d 二进制）→ numpy 对拍（方案 §3.2/3.4 判读规则）。
2. FORCE_ITER 对照臂（同二进制 + GGML_VK_HC_BF16_FORCE_ITER=1）。
3. 两判据合并裁决 S2（或转向 base glsl 的 dequantize4_2aligned 实现静态审查——Engineer 侧）。

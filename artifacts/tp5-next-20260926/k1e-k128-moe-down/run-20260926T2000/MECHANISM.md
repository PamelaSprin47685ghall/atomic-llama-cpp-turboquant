# K1e 机制文档：配对专家 K=128 MoE down 路径的验收与资格（run-20260926T2000）

日期：2026-09-26。工单定位：AGENTS §5.2 MoE 分组成员的**现成增量验收**（K1a 模式：
不写新 kernel，验证已存在但从未开启的实现）。

## 发现

`qwen4_moe_down_k128.comp` 及 8 个管线变体（plain/wire/relay/relay_f32 × shared）
已完整实现并接入 `ggml_vk_moe_output_segment`（dispatch ~16830），但被
`GGML_VK_MOE_DOWN_K128=1` 显式 opt-in 关闭，注释明言"until paired validation"。
`tests/test-vulkan-command-replay.cpp` 的 MoE fixture 注释确认："fixture runs
identically on the native path **without the flag**"——**该路径从未被开启验证**。

## 几何与收益模型

现行 `moe_down_fold*`：每 4 行 tile 一个 320 线程 WG（10 波×32，每波 1 专家，
16 活跃 dot lane + 16 零 lane）。K128 变体：**两个选中专家共享一个 wave32 的
两个 16-lane 半波**，同样 4 行 tile 只需 160 线程（5 波）——**线程减半、每波
活跃 lane 从 16/32 提到 32/32**。

真实模型命中 shape（K1b segments dump 证实）：routed down K=128（ffn_down
ne0=128 per-expert 分片）、10 selected experts、shared K=128（fuse_shared_down
在 shared_k%32==0 时已默认开）——`use_k128` 门 `k==128 && experts==10 &&
(!fuse_shared_down||shared_k==128)` 全部满足。48 层 MoE × 每 verify 1 次
down fold × 19-25 cycle/请求。

预期收益上限：MoE down fold 是重 stage ~325 µs 的组成部分。线程减半主要省
调度/零 lane 空转，带宽不变（权重仍全读）。**诚实预期：与 K1c/K1d 同级
（~0-5 ms/请求，可能 ABBA 不可检出）**；若 RADV 对 160 线程 WG 的占用率
更优则更高。不预设收益，由 ABBA 裁决。

## 数值契约（shader 头注释自述 + 我的核对）

1. 每专家 partial 只在自己 16 lane 内做 XOR 蝶形（步 1,2,4,8）——**独立半波
   归约**，绝不跨半波 subgroupAdd（否则两专家互相污染）。
2. 16-lane 蝶形 ≡ native 32-lane subgroupAdd(16 partial + 16 零) 的低半树：
   native 步 16 合并恰加 0.0。**ASSUMED not proven：RADV 把 subgroupAdd 降为
   该蝶形顺序**——这是 GPU 对拍要证的命题。
3. 10 专家 fold、shared-down 融合、F16 wire 输出镜像 `mul_mat_vec.comp` 的
   `MOE_DOWN_FOLD reduce_result`：有序专家和（`precise`）+
   `gate=1/(1+exp(-shared_gate[0]))` + `folded+shared`。

## 验收阶梯（预注册）

1. **GPU 对拍**：`GGML_VK_MOE_DOWN_K128=1` 下 `--moe-output-only`（fixture 已
   含 K=128、10 experts、相邻专家 ID 同波用例——正是为 K128 写的定向污染
   检测）+ `--moe-only` + 全套 28/28。对拍对象 = 同 GPU native 路径
   （K1c/K1d 教训：GPU 归约序 ≠ CPU 序）。
2. **真实模型 smoke**：canonical 12/12 精确 + dispatch 账（K128 命中时
   moe_down_fold 换 moe_down_k128，dispatch 数不变但线程减半）。
3. **ABBA**：A=`GGML_VK_MOE_DOWN_K128=0`（off），B=on。同二进制、5 blocks ×
   2 repeats、`--count-to 60`。预注册裁决：decode ratio CI 下限 ≥0.99 非劣
   保留；CI 整体 >1 才宣传提速。

## 风险与回退

- RADV subgroupAdd 降阶顺序若与假设不符 → 对拍失败 → **fail-closed**：保持
  opt-in 关闭，记录负结果（这正是注释等待的"paired validation"）。
- 160 线程 WG 在 RDNA2 的占用率若低于 320 线程 → ABBA 非劣失败 → 撤。
- `subgroup_size16 == subgroup_size` 才创建 shared 变体（6775 行）——RX 6800
  subgroup=32，`subgroup_size16=max(32,16)=32`，条件满足；RADV 驱动确认。
- 回退零成本：env 关闭即回现行 fold 路径，无代码删除。

## 证据锚点

- 管线创建：`ggml-vulkan.cpp:6759-6805`（opt-in 门 + 8 变体）
- dispatch：`ggml_vk_moe_output_segment` ~16820（`use_k128` shape 门）
- shader：`qwen4_moe_down_k128.comp`（353 行，契约注释 30-46 行）
- fixture：`test-vulkan-command-replay.cpp:1979+`（K128 定向用例 2085-2095）
- 真实 shape：K1b `segments.tsv.gz` ffn_moe_down ne0=128/ne2=10

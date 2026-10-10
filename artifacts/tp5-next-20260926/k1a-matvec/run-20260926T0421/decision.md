# K1a decision

日期：2026-09-26。工单：K1a（多行 matvec 权重复用最小增量）。

## 改了什么

从整合提交 `9b3c33881` 分离出的**单文件 shader 增量**：`ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`
（121 行 diff，`git apply` 到 `16e28b8ac` 基座的 candidate worktree `/var/tmp/tp5-candidate-k1a`）。

- 基座行为：`K_PER_ITER==8 && !MUL_MAT_ID && !HC_DOWN_Q8_DOT` 时的权块复用快路只在
  `num_rows == 1` 触发；量化管线以 `NUM_ROWS = 2*rm_stdq`（stdq）或 `rm_kq`（K-quants）创建，
  RDNA2 上均为 2，因此七行 target verify 的 skinny/普通 batch matvec（`NUM_COLS=7`）
  从不命中快路，每个 (列,行) 对重复读并解量化同一权块。
- K1a 行为：快路条件放宽为 `NUM_COLS > 1`；`NUM_COLS <= 4` 时把全部列的激活 staging 进
  寄存器；`NUM_COLS > 4`（含 7）走行优先循环——权块解量化一次，7 列点积从寄存器发射。
- **不含**整合版的另一改动（`rm_wide` PQ2_0 输出行 tile，`ggml-vulkan.cpp`）——K1a 是
  单因素实验，PQ2_0 行 tile 属 K1b 范畴。
- 附加修复：`ggml-vulkan-tp5-rows.h` 补 `#include <cstddef>`（基座预存干净构建缺陷，
  `test-tp5-row-program` 从未在基座编译过；文档 `TP5-MTP-PREDEFINED.md` 明言
  "已登记但未编译/执行"）。control 侧同样只加这一行，保证 A/B 同构。

## 此前为什么失败 / 本次机制何处不同

- 不是重启旧失败路线。整合提交的 shader 增量从未单独测过（战报 §4 明言整合版未验证）；
  本工单按手册 §5.2 要求把它作为独立候选 K1a 从基座出发评估。
- G1 账证明命中的收益面存在：96-stage 链中 stage 0–9 的最慢 rank compute
  964–2237 us（其余 stage 200–300 us），ready 等待头 10 stage 合计 ~12.9 ms/链。

## 实际命中什么

- 命中面（静态推导 + 代码 guard 核对，见 `k1-hit-analysis.txt`）：七行 target verify 的
  普通 `MUL_MAT` skinny/batch proj（attention Wq/Wk/Wv/Wo、HC fold 权重等 Q5_K/Q8_0 投影，
  `mat_vec_cols=7`、`NUM_ROWS=2`）。`MUL_MAT_ID`（MoE）、QSA/GDN 专用 pipeline、LM head
  wire pipeline 不命中（guard 或专用 pipeline 排除）。
- 数值对拍全部通过：skinny、argmax、rows、transfer、attention-projections（F32 bitwise +
  MMVQ 容差）、MoE、HC fold（bitwise CPU 对拍）、rope、GDN region、tp5-mesh、
  rank-local readback `--vulkan`、GDN multistep；CPU 套件 11/11。
- 行为门：20 trial 全部逐字精确、prompt31、171/171、自然 stop；GPU 身份前后一致。

## 节省哪个互斥时间项

- **未检出**。五块 ABBA（A=基座 `7bc567b4…`，B=K1a `9dc93522…`，各 5 进程 20 trial）：
  - Decode：A mean **83.394**（stdev 0.428），B mean **83.526**（stdev 0.299）
  - B−A = **+0.131 tok/s**，95% CI **[−0.566, +0.828]**（跨零）
  - 比值 **1.0016**，95% CI **[0.9932, 1.0100]**；非劣性下界 0.9932 ≥ 0.99 **通过**
  - client wall 比值 1.0020（同向，无显著性）
- Pilot（2 blocks）B/A=0.9993，与正式 ABBA 一致：收益量级 < 噪声。

## 新增多少成本

- 无新增驻留内存（同 pipeline 数、同 descriptor 集）；shader 寄存器压力变化未致
  spill（`GGML_VK_PIPELINE_STATS` 检查的 q8_0 管线 VGPR=32/SGPR=108/0 spill，与 control
  相同 inventory）；构建产物大小差异仅在 libggml-vulkan.so。

## 裁决

**保留为非劣融合成员候选，不宣称提速。** 按手册 §10.3：小于噪声只称"未检出收益"。
正确性/维护价值成立（消除 7 列路径上 6/7 的重复权重读的代码债，为 K1b staging/行 tile
铺路），非劣性达标，无回归。不晋升为默认性能优化宣传；不关闭 K1（K1b 的
NUM_COLS<=4→7/8 staging 或 rm_wide 行 tile 扩展仍待真实高占比 shape 实测，但 G1 账显示
大头在 stage 0–9 的 producer compute，未来 K1b 应针对这些 stage 的具体 shape 再立项）。

## 证据

- pilot：`k1a-pilot-abba2.json`（2 blocks，PILOT_COMPLETE_NOT_ACCEPTANCE）
- 正式：`k1a-abba5.json`（5 blocks/20 trials，status=success）
- 日志：`k1a-abba5_logs/`（20 个 server log + SHA256 in JSON）
- 构建产物：control `7bc567b4c2b2…`、candidate `9dc935223bcf…`（llama-server），
  libggml-vulkan control `92d3f577…`、candidate `be69be3e…`
- candidate diff：2 文件（mul_mat_vec.comp 93+/28−；tp5-rows.h +1 include）

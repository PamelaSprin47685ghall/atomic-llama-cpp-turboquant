# K1b 裁决：stage 0–9 慢的机制指认与量化（稳态）

日期：2026-09-26。run-id：20260926T1130。二进制：control `16e28b8ac` 干净重建
（`7bc567b4…`）。诊断运行（插桩），不作吞吐声明。

## 方法修正（推翻我此前两个错误假设）

1. **"stage 0–9 是首链预热"——错误。** 稳态捕获（`GGML_TP5_GPU_TIMING=85`，
   chain 85 ≈ 第 5 个请求内）与首链（chain 6）对比：stage 1 稳态 2214.9 us vs
   首链 2236.8 us——**几乎无差**。慢是稳态结构，不是预热。
2. **"stage 1 与 stage 9 同 op 同 shape 所以慢在等待"——部分错误。** compute 括号
   （时间戳 5s+1→5s+2，`tp5_poll_gpu_timing` 源码确认）只含模型 dispatch 段，
   不含 P2 等待（在 5s+3→5s+4 段）。stage 1 的 2215 us 是真实计算时间。

## 机制指认（segment dump + 稳态 timing 交叉）

`GGML_TP5_SEGMENT_DUMP`（meta chain 既有功能）给出每 stage 每 node 的 op/shape：

- **96 stages = 48 层 × 2**：偶数 stage = GDN 状态 + PLE 分支，奇数 stage =
  QSA attention（SOFT_MAX）+ MoE（MUL_MAT_ID）；stage ≡6 (mod 8) 是 12 个
  FLASH_ATTN_EXT 层（KV attention）。
- **stage 1（layer 0）比 stage 9（layer 4）多 17 个 node**：`cache_ple_r_l1`、
  `conv_state_at`、`ple_conv1d` 视图、PLE 门控 MUL/SCALE——**layer 0–3 是 PLE 层**
  （`hparams.is_ple(il)`，qwen4exp.cpp:290/474 `build_ple` 分支：ple_key/ple_value
  投影、分组 norm、门控点积、`build_conv_state_at` 卷积历史、kern 次 shifted
  conv tap、SILU）。
- GDN 状态（cache_r/cache_s）每层同尺寸（27648/163840 元素），PLE 层与非 PLE 层
  无差；MoE 专家 shape（512×5120 ×2 + 10240）逐层相同。

## 稳态量化（chain 85，slowest-rank compute）

| stage 组 | 内容 | 稳态 us | 非对照层均 us |
|---|---|---:|---:|
| 0–7（layer 0–3，PLE） | PLE+GDN+QSA+MoE | 965/2215/1337/1177/837/1180/897/1168 | — |
| 8–9（layer 4，非 PLE） | GDN+QSA+MoE | 591/331 | 591/331（过渡） |
| 10+（layer 5+，非 PLE） | 同构 | 229/325 均值 | **~230/325** |
| **合计** | | **stages 0–9 = 10.70 ms / 32.97 ms（32.4%）** | |

- PLE 层均 ~2.07 ms/层（两 stage），非 PLE 层 ~0.61 ms/层。
- **PLE 超额 ≈ 4 层 × 1.46 ms ≈ 5.8 ms/链**；加 layer 4 的过渡（591 vs 229）≈
  总可归因 **~6.4 ms/链 × 24–25 链 ≈ 150–160 ms/请求**。
  （我此前 8.4 ms/链 的估算把 layer 4 过渡误归 PLE，修正。）

## 与 K1a 的关系（为何 K1a 无收益）

K1a 改的七列 matvec 权重复用命中 QSA/HC 投影（stage 1–7 内），但 PLE 层的额外
时间在 `build_ple` 的**非 matvec 算子**：门控点积链（SUM_ROWS/SQRT/SIGN/SIGMOID）、
`build_conv_state_at` 的 per-slot CPY（n_rs_seq+1 份卷积历史回写）、kern 次
shifted-view CONT+TRANSPOSE+MUL、REPEAT_4D 广播。七列 matvec 快路无法触及。

## 候选方向（按手册 §5.2 单因素原则，未实现）

1. **PLE conv tap 消除 view/copy**：kern 次 `ggml_cont(ggml_transpose(view))`
   每次物化整行（hc_dim×F32×rows）；可改为单次 staging + 索引乘加（一次物化）。
2. **conv_state_at per-slot CPY**：`n_rs_seq+1` 份回写是快照语义（rollback 槽）；
   属 S1 范畴（状态快照），须先过 S1 门，不在此动。
3. **PLE 门控点积链融合**：SUM_ROWS+SQRT+SGN+SIGMOID 是小张量逐元素链，
   dispatch 数多但字节小——先数 dispatch 占比再定。

## 下一实验窗口的第一项

对 PLE conv tap（方向 1）做单因素补丁：`build_ple` 内 kern 次 shifted 物化改为
一次 staging。验收：同形数值对拍（PLE 层输出逐元素）、七行 verify 命中、
pilot → ABBA。预期收益上界 ~150 ms/请求（PLE 超额的 matvec 外部分）；
若 conv tap 只占其中小头，退选并记录。

## 证据

- `server-steady85.log`（480 条 tp5-gpu-stage + 5 条 tp5-relay-gpu 稳态账）
- `segments.tsv`（97045 行，97 stages 全 node dump）
- `segdump-request.json` / `s85-r*.json`（请求正确性：全部 171/171 exact）
- 首链对照：`/var/tmp/tp5-g1-20260926/manual/server-instrumented.log`（chain 6）

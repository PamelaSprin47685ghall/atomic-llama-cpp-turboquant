# MMVQ × MTP 四格深井（run-20260928T1700+，`/var/tmp/tp5-mmvq/`）

## 命中证明（先证"测到了"，再谈有没有用）

`GGML_TP5_RELAY_DEBUG=1`（dbg1.log）：
```
[vulkan-region-policy] backend=Vulkan0..4 family=gdn-projections fused=1 mmvq_mask=3 shared_q8_1=1
[vulkan-region-policy] backend=Vulkan0..4 family=qsa-projections  fused=1 mmvq_mask=1 shared_q8_1=1
```
**融合 attention 投影的 MMVQ 真实命中**（gdn: Q/K+V 三个投影量化；qsa: 一个）。
账本差：auto 臂 `dispatches=3024 vs 2879`（+145 = 量化 kernel）。
注意：`[vulkan-region-policy]` 打印本身被 RELAY_DEBUG 门控——region=0 行数不代表未命中。

## 四格结果（同二进制、GPU 锁、每臂 3 请求 + 镜像、arm.sh 环境纪律）

| 格 | 配置 | ms/cyc 中位 | 配对比值 | 输出 |
|---|---|---|---|---|
| A0 | no-MTP + off | 29.5/29.1 | — | md5 `add18ad37ff7`（=历史 R） |
| **A1** | no-MTP + auto | 31.2/31.4 | **A1/A0 = 1.067（慢 6.7%）** | md5 `671ea4f7fab7`（变） |
| B0 | aggressive h=1 + off | 28.2/28.5 | — | md5 `ebf49ef9d1ca`（=历史 A） |
| **B1** | aggressive h=1 + auto | 28.7/27.4 | **B1/B0 = 0.990（不可分）** | md5 `4a21e639425b`（变） |
| B0(h2) | aggressive h=2 + off | 38.6/38.1 | — | h=2 本身比 h=1 慢 35% |
| **B1(h2)** | aggressive h=2 + auto | 43.7/43.3 | **B1/B0 = 1.136（慢 13.6%）** | 接受率 0.57-0.63 |

## 判决

1. **MMVQ 单独（A1）在 no-MTP 参照上慢 6.7%**：activation Q8_1 量化 + 145 个额外
   dispatch 的成本超过收缩收益。RX 6800/RADV 上 Q8_1 dot 对 Q5_K/Q6_K 权重不占优。
2. **MMVQ×MTP（B1, h=1）与 B0 不可分（0.99）**：无正交互。
   I = S_B - S_A = (B0-B1) - (A0-A1) ≈ 0 - (-0.29) = **+0.29ms/cyc 名义正交互**，
   但 B1 仍不快于 B0 本身——交互只是"MMVQ 在 MTP 下不那么亏"，不是收益。
3. **h=2 上 MMVQ 恶化到 -13.6%**：多行验证让量化成本随行数增长（每行都要量化），
   与 §14 经济账预测一致。h=2 本身（B0）也比 h=1 慢 35%——短 horizon 判决再次成立。
4. **数值红线确认**：A1/B1 输出 md5 均改变（activation 量化改变 logits），
   文本仍连贯（A1: cjk=202 连贯；B1 h=1: cjk=202）。
5. **结论对应手册 §18 第 8 行**："所有合格候选都不赢"——在本机 RADV/RX6800、
   Qwen4EXP Q5_K/Q6_K 投影、当前 LateBind 刀锋上，MMVQ（含融合 region 命中）
   不提供可检定收益。TP5 recipe 默认 DISABLE_MMVQ=1 是正确默认。

## 边界（不外推）

- 未测：Q8_0 主导权重（本模型投影是 Q5_K/Q6_K）、Intel/NVIDIA、h>=3、
  非 LateBind 布局。IQ 专家类型本就不在 MMVQ 支持表内。
- B1 h=1 的 0.99 在 ±3-5% 会话漂移带内，正确表述是"未检出"而非"证明无用"。

## h 扫描补充（h3 目录）

| h | B0 tok/s | B1 tok/s | B1/B0 |
|---|---|---|---|
| 1 | 62.5/61.9 | 62.0/65.1 | 0.99（不可分） |
| 2 | 57.6/58.3 | 48.3/48.7 | **1.136（慢 13.6%）** |
| 3 | 53.6 | 52.2 | 0.95*（样本内输出不稳定，md5 漂移，不可判） |

*h3 的 B0/B1 输出自身 md5 不稳定（接受率 0.43-0.55 摆动，cjk 85-22 摆动），
该格不可判。h=1 仍是本负载最优 horizon；h 增大单调变差。

## 终局

四格 + h 扫描全部收敛：**MMVQ 在本机/本模型/本刀锋上无可检定收益**。
最优配置仍是 aggressive-q8 h=1 + MMVQ disabled（TP5 recipe 现默认）。

## 补充：A1（no-MTP, auto）命中证明（dbg2.log，RELAY_DEBUG=1）

```
family=gdn-projections fused=1 mmvq_mask=3   (x5 rank)
family=qsa-projections  fused=1 mmvq_mask=1  (x5 rank)
```

**两臂（A1 与 B1）的融合投影 MMVQ 均真实命中**。A1 的 +238 dispatch
（4327 vs 4089）= 参照刀锋上的 standalone Q8_1 量化 + 整数收缩。
四格的"测到了"门在两臂均成立；判决维持。

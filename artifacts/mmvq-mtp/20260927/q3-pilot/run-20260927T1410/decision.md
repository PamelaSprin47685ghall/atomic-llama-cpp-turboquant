# QMP 四格 pilot：MMVQ×MTP 交互为负；B0（h1+off）胜出——G8 的 2.9% 被证伪（2026-09-27，run-20260927T1410）

## 前置门（全部通过）

- qmp-* 六个 cell 注册（h1/h2），CPU 断言全过（对内单因素、A 无 MTP、B h 正确、无基础变体污染）；dry-run NOT_MEASURED。
- 身份冻结：server `b92fda12`、libggml-vulkan `b81cc51b`；无新 wedge（journalctl 0 事件）；GPU 独占。
- auto 策略在五 rank 全部 resolve（`mode=auto` ×5）。

## 四格交错 pilot（2 轮 × 4 cell × (1 warmup+3 稳态)，镜像轮序）

|Cell|配置|ms/tok（n=6）|tokens|acceptance|
|---|---|---:|---:|---:|
|A0|noMTP + MMVQ off|24.861 ± 0.071|177|—|
|A1|noMTP + MMVQ auto|24.858 ± 0.059|179|—|
|B0|h1 MTP + off|**22.536 ± 0.332**|178|0.77 / mean_len 1.77|
|B1|h1 MTP + auto|23.459 ± 0.467|**165**|**0.72** / 1.72|

预注册指标（Plan §3.3）：

```
S_A = +0.003   （MMVQ 单独：零效应）
S_B = −0.923   （MTP 中开 MMVQ：更慢）
I   = −0.926   （交互为负）
J   = −0.0403
G   = +5.6%    （B1 比最优 no-MTP 差 5.6%）
```

## 三个结论

1. **MTP h1 在该自然问题上真实有价值**：B0 比 A0 快 2.33 ms/tok（**9.4%**）。
   ——G8 的"2.9%"被证伪；MTP 黑锅审计（289ms 跨臂缺口）预言的方向
   得到确认：G8 低估了 MTP。
2. **MMVQ 单独零效应**（A0=A1，与历史 E4 一致）。
3. **MMVQ×MTP 组合交互为负**（I=−0.93）：auto 使 acceptance 从 0.77 降至
   0.72、输出变短（165 vs 178 tok）、且更慢——**activation Q8_1 量化
   扰动 target logits，draft 匹配变差**（Plan §0.2 红线 3 预言的机制，
   方向与用户假设相反：组合不是大用，是互相拖累）。
4. B1 的 165-token 输出是合法变体（MMVQ 改变近 tie 措辞，同 G7 变体集
   机制）；acceptance 下降是真实的机制证据，不是噪声。

## 爬山基线更新

**当前最优 cell = B0（h1 MTP + MMVQ off）：22.54 ms/tok。**
旧"no-MTP 最优"（A0=24.86）不再是基线——**基线随爬山移动**。

下一步（相邻 cell）：
- B0-h2（h2+off）：h2 的 3 行 verify 是否进一步摊薄？（G7 曾显示
  h2=3691 vs h1=3233 更慢——但那是 n1-varargmax 旧配置，B0 现在是
  新基线，且 G7 的排名被标注"平衡敏感"。重测。）
- B0-h1 的 auto-draft 变体（draft-only MMVQ）需要 per-role 策略（Q5，
  代码改动）——暂缓，先测 h2 这个纯配置步。

## 证据

- `qmp-pilot.json`（8 臂全部，含 warmup）、`server-{A0,A1,B0,B1}-r{1,2}.log`
- `pilot-nomtp.json`（首次 harness 跑，A1 变体 fail-closed 触发变体集机制）
- 身份：`../q0-baseline/run-20260927T1400/identity.sha256`

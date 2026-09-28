# LateBind 深井交接手册（TP5 / 5×RX 6800 / Qwen4EXP）

更新：2026-09-28（晚，lbfin 终局）。适用：五张 RX 6800、Qwen4EXP TP5 + MTP，仓库 `~/atomic-llama-cpp-turboquant`。
本文件替换 `AGENTS-2026-09-26-TP5-EXECUTION-MANUAL.md`（保留为旧路线图，不再作为执行依据）。
历史性能账与失败记录仍查 `TP5.md`、`TP5_FUSION_HANDOFF_2026-09-26.md`、
`TP5-LATEBIND-LINEAR.md`、`交接.md` 与 `artifacts/`。

**LateBind 路线已做到无可做（本轮终局判决，§终局表）。
aggressive 刀锋显著超参照 -4.2% ms/cyc（CI 不含 1）；完整 L2 与参照打平。
剩余效应 < 测量分辨率。下一轮若继续，方向是会话漂移或 GPU 化 LO 合成，
不是 LateBind 本身。**

## 0. 开工规则

遇到冲突按此序处理：用户最新要求；当前源码与原始证据；本文件；历史叙述。
**提交已合并不等于已测；单次 smoke 不等于性能资格。**

- 实验树 `/var/tmp/tp5-candidate-k1a`（提交 `f27ba5a98`，基 `16e28b8ac`），
  主仓已同步（`5c4dac988` + `cff10a2bd`，逐字节核对 diff=0）。
  构建在 `/var/tmp/tp5-final-build`（RelWithDebInfo/-O2）。
- 一次只允许一组 GPU 实验；测量期间不并行编译。GPU 锁 `/var/tmp/tp5-gpu.lock`（`flock 9`）。
- 保留所有回退路径；不删共享 Vulkan 的保活、tensor 身份与同步契约。
- 证据归档 `artifacts/tp5-next-20260926/latebind-deep-well/run-20260928T0330/decision.md`
  （~3000 行，**§12 是本轮终局，先读 §12 执行摘要**）。

**启动**：`--tp5 qwen4exp-af -md <mtp.gguf> --spec-type draft-mtp --tp5-latebind aggressive`。
测量环境只给 `RADV_DEBUG=nobolist`（`env -i` 式）；**不设 `GGML_VK_HC_DOT`**（静默失臂）。
L2 需要 `GGML_TP5_LATEBIND_L2=td GGML_TP5_LATE_TRANSPORT=gtt-push GGML_TP5_LATEBIND_ALLOW_APPROX=1`。

## 1. 终局判决（ABBA run-20260928T1550，`/var/tmp/tp5-lbfin/abba2/`）

| 臂 | 配置 | tok/s 中位 | ms/cyc/R（95% CI） | 输出 |
|---|---|---|---|---|
| R | `--tp5-latebind off` | 61.34 | 1.0 | 12/12 同 md5 |
| **A** | aggressive-q8 | 63.41 | **0.9579 [0.9178, 0.9979]** | 12/12 同 md5 |
| Ap | A + q8first + pack-once | 62.83 | 0.9606 [0.9301, 0.9911] | 与 A 同 |
| T | A + L2=td | 62.19 | 1.0049 [0.9745, 1.0353] | 12/12 同 md5，cjk=255 |

- **A 显著超参照**（-4.2% ms/cyc，CI 不含 1）——LateBind 净收益正式成立。
- **T（完整 L2）与参照打平**：L2 的协议+CPU 代价被链内并行掩盖；T/A=1.05。
- Ap 与 A 不可分：q8first/pack-once 的旧 +2.5% 未复现，不再主张。
- T 输出连贯度（cjk=255）**超过参照**（241）；L2_CHECK `max_rel=2.17e-7`。

## 2. 本轮全部根因（勿重做）

1. **九个深井根因**（孔径族、量化器字索引、`cc` 偏移、统计区基址、跨 rank d 折叠、
   `TRANSFER_READ`、vec2 步长、统计量折叠绑错张量、`mmvq_dot_multi` 守卫）
   ——详见 decision.md §2/§11。
2. **L2 尺度失配 3×10⁵**：已解。L2=td 重设计（ACT_Q8 内生成 `(a_b,d_b)`、
   relay 合成 `rho=1/sqrt(max(0,ss)/width+eps)`），链内 L2_CHECK 对拍 2.17e-7。
3. **tag 协议**：Q ready word 不保证 host 见到 payload（实测 ~0-4 次/19k 交接）。
   GPU 用 epoch+z_p 生成 tag，relay 自旋等 tag 匹配。
4. **【本轮新修，关键】回退循环**：decode↔prefill 相位切换 free compute chunk
   → 静默抹掉 vk replay 缓存（`ggml_backend_vk_buffer_free_buffer` 的
   `buffer_owners` 连坐擦除）→ 每请求 3×480-subgraph 回退 + 计划重录
   （T/R 曾达 1.414）。**修复**：`ggml-alloc.c` trim 路径 `GGML_GALLOC_KEEP_CHUNKS`
   （默认 1，保留 chunk，增长不受影响）→ T 42.2→62.2 tok/s、回退 15→6、
   L2_CHECK 2.4e-3→2.17e-7、R 无回归。VRAM 代价：两相位 chunk 常驻。

## 3. 已收敛的消融（勿重跑）

- **transport**：L2 硬约束 gtt-push（bar-pull fail-closed）；A/Ap 用 gtt-push。
- **overlap**：aggressive 刀锋上无作用点（六臂账本全同，§10f.2-a#）。
- **q8first/pack-once**：与 L2 不叠加（63.2 vs 66.2 单因素）；与 A 不可分（ABBA2）。
- **L2+q8first 组合**：劣于 L2 单独。**T 臂最优 = L2=td + legacy + per-site pack**。

## 4. 剩余空间与不可检定性

- T/A = 1.05（每周期 ~1.4ms）= L2 relay CPU 段（~5µs×96 handoff）+ 计划差异。
  吃掉它需 GPU 化 LO 合成（Q8DOT/LO_U 已有全部输入：rho 在 stats 区、Q 在孔径）。
- **本 harness 分辨率 ±3-5%（会话漂移）**：R 自身 60.2-61.8 漂移。任何 <5% 的
  改进不可检定，除非先修会话漂移（GPU 时钟锁定？）。
- **路线判定：做到无可做。** 单因素全部收敛，剩余效应低于测量分辨率。

## 5. 已验证的纪律（违反过，代价已付）

1. 单一来源是正确性（七次同义变量各自维护的教训）。
2. 标量指纹对偏移类缺陷是盲的——先 dump 原始字。
3. 写下"未证实"之后不能再用它下结论。
4. 带门控的插桩必须先确认它真的在输出。
5. 后台作业未回传时不得从中间产物下结论。
6. fail-closed 会掩盖它本该掩盖的问题，并伪造"被测对象失败"。
7. **【新】buffer free 会静默抹掉 replay 缓存**——性能异常先查
   `GGML_VK_TRACE_CACHE_FREE=1`，再查指纹。
8. **【新】回退路径自身会改变资源生命周期**（分配新 buffer→free→缓存连坐），
   形成"回退自我维持循环"。修复点在资源策略，不在被测机制。

## 6. 源码索引

- 主文件：`ggml/src/ggml-vulkan/ggml-vulkan-collective.cpp`（L2=td：`:207`、
  form_lo `:6174`、L2_CHECK `:6147`、tag 协议 `:6206-6280`）
- 着色器：`tp5_hc_latebind.comp`（Q8DOT/精确/统计量）、`tp5_hc_resume.comp`（norm/LO/LO_U）
- **chunk 保留修复：`ggml/src/ggml-alloc.c`（`GGML_GALLOC_KEEP_CHUNKS`）**
- 偏移契约：`ggml-vulkan-tp5-rows.h`
- 加载门：`common/common.cpp`（`--tp5-latebind` 三档 + `ALLOW_APPROX`）
- 测量：`/var/tmp/tp5-lbfin/arm.sh`（单臂）、`abba.sh`（预注册四臂）、
  `abba_report.py`（配对统计）
- 归档：`artifacts/tp5-next-20260926/latebind-deep-well/run-20260928T0330/decision.md`

**下一轮若开工：不是 LateBind。是会话漂移（harness 分辨率）或 GPU 化 LO 合成（<5%，先修漂移才可检定）。**

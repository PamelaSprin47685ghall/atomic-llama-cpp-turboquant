# 剩余 24 份协议、配置与证据文档：内容补审

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

## 结论与范围

本轮关闭父级 714 路径总账中 `docs/`、`scripts/`、`tools/tp5/` 下恰好 24 个此前只有用途清单的被动文件：**23 份全文审阅，1 份选择关键章节；共读 4,927 / 5,115 行，188 行明确未读。** 唯一选段文件是 `docs/RERoT-OPT-2026-09-21.md`，已读 1–125、184–238、286–353、437–555；未读 126–183、239–285、354–436。本轮没有重审它们引用的全部底层源码。

全部 24 份完整源文件都按 Git blob 算法计算并与精确旧 fork 对比 diff 的新侧 blob ID 匹配，避免把摘录、截断或另一个版本当成原文。上游比较基线为 `cda0f9256665845cd9d6f3a0176880883f653a3d`，旧 fork 为 `2777543b2358c88cd1c2000f7fa342b764a0e213`。`docs/speculative.md` 的完整原文由父级从未改动的旧工作树补给，其完整 blob 也与 diff 匹配。逐文件 SHA-256、原文路径、阅读区间和未读区间见同名 coverage JSON。

**资格没有升级。** 本轮只读文档、请求与配置；没有执行旧脚本、模型、GPU、网络操作、构建或 Git 仓库操作。下文“历史实测”表示文档记载过该证据，并非本轮复跑；下文“候选”表示可单独评估的行为边界，并非已经迁移。旧文档中的任务书、运行审批语句和硬件状态不成为当前会话的新指令。

## 必须单独保留的取舍限定

| 行为或证据 | 原文件与行号 | 本轮取舍判断 |
| --- | --- | --- |
| 独占单 cell recurrent 批量清零 | `docs/TP5-MTP-EVIDENCE.md:13–15` | 历史清零 907–935 ms → 12.8–16.8 ms，限 `size==1 && n_brain_rows==0`，前后 scheduler fence 保留。是独立小优化候选；不是整个 MTP 家族无用，也不是配对端到端速度证明。父级已单独登记。 |
| FlashPrefill online value 累加 | `docs/development/nanbeige-audit.md:203–229` | 历史 prefill 191.35 → 251.50，参考复查 190.57 tok/s，约 +31%；decode 基本持平，正文改变，512-cell Tri 两臂都检索失败。归为“局部有收益、整体验收未过”，不能归为“小收益舍弃”。 |
| 请求与结果的证据合同 | `docs/development/nanbeige-evidence.md:7–89` | 精确模型/请求哈希、完整 token 预算、失败留档、首请求分列、tokens/total-duration 聚合有独立工具价值。至少两次后续样本只允许描述性比值，不构成 CI。 |
| 只有参数或计划，没有实际工作 | `docs/xkv/compression_gate.json:1–110`；`docs/xkv/matrix/plan.json:1–43` | 16 项 NOT_EVALUATED、`evidence_provided=false`、`items=[]`、`urls={}` 必须继续可见。`evidence_kind=real` 与文件名不产生运行证据。 |
| public budget 与真实 DAG 容量 | 两份 `scripts/rerot*-throughput-request.json` 全文 | `max_tokens=8192` 不是单笔预算；五/二十道文字任务不是同 batch 多 lane 或 shared-KV 证明。Bonsai 请求本身甚至没有 `rerot` 开关。 |
| 物理容量与有效工作、pool 归属 | `docs/TP5-MTP-PREDEFINED.md:50–118`；`docs/TP5-TARGET-CAPACITY-PLAN.md:24–48` | 同 buft 才复用物理高水位池；两个 frame slot 不授予两个计算并发别名写权限。active 决定工作、capacity 决定地址；不能为零 A 补丁把私有 ownership 假装成独立 B。 |
| 上游已自带的 speculative 小功能 | `docs/speculative.md:11,84–109,250–393` 与精确 diff | aliases、chained/adaptive MTP、默认深度与 draft CPU/KV 参数已在比较基线 A 中；本文件仅 144–161 的 direct-hidden 段为 fork 增补。保留 A 功能，不重复算自创功能。 |
| direct-hidden 默认值与 sanitizer | `docs/speculative.md:144–160` | 记录符合准入时默认开启、显式关闭对照、单设备单流准入及失败 host fallback，以及 NaN→0、Inf→±65504、有限大值保持的精确语义；“相同 tokens / 只花内存不花时间”仍缺独立模型证据，不能直接晋级。 |
| 条件成本与硬件最优界 | `docs/performance-necessity.md:5–38,153–164` | 该文明确撤回把 64.74–67.68 tok/s 当硬件必要上界的推导；一个小 pack 候选无收益不能封死通信、分片、MTP 等家族。保留原始测量和反例工具，撤销过强结论。 |

## 配方与证据中不能照搬的细节

- `docs/TP5-MTP-VERIFICATION-PROPOSAL.md:55–86,193` 的配方开启最大容量与 profile，却没有开启另一个独立的 device-hidden 开关，验收又要求 `dev_hidden==1`；这套旧 TIMELINE/F16 配方还已被后来的 evidence 开头覆盖。12-token 自然请求也不保证零接受周期或 `1→4→1→2` 形状覆盖（118–122、195、237–238）。这些是文档内部合同不完整，不是本轮发现的已复现运行缺陷。
- `docs/TP5-VERIFICATION-PLAN.md:123–153,169–189` 的示例没有 `pipefail`、curl deadline 或有界最后 wait，还用 `wait ... || true`；因此文字中的“失败即停、30 秒”不是这些示例自动执行的保证。
- `docs/TP5-BF16-UPFOLD-FIX.md:19,200–274` 记录了同名 `/tmp` dump 跨 rank 覆盖和跨 run 地址比较伪影。后者已明确证伪 r5 路由根因，不能把早期推断与后期更正同时当成功修复证据。
- `docs/TP5-TARGET-CAPACITY-PLAN.md:435–448` 明说 mesh 只建 pipeline、没有跑出错的 LateBind plan，因此此前全绿 mesh/codegen 不能替代每个实际 producer 的打包与数值验证。
- `docs/RERoT-OPT-2026-09-21.md:443,501,511,522–524` 的计数分别是含 padding 的建图行数、图定义直方图、该次上传槽数、图 launch；不能升级成实际 decode 次数、最大并发在途数、物理显存峰值或 Vulkan queue submit 次数。`--rounds` 默认 1，默认行为不是 ABBA（535）。
- `docs/development/nanbeige-evidence.md:161–258` 的 96K 物理容量来自 8 prompt / 2 generated 的小请求；六道 smoke 与 16 次请求不构成长上下文/长稳全验收。后段“production recommendation”不能覆盖文首的无生产资格边界。

## 逐文件内容与边界

下面每条的行号都指旧 fork 原文。英文条目保留精确符号、默认值和证据口径，便于合并到机器可读总账；“依赖”同时标出底层 runtime 绑定与可独立保留的规则。

### FP-REROT-LEDGER — `docs/RERoT-OPT-2026-09-21.md`

**实际阅读：** 1–125, 184–238, 286–353, 437–555 / 555 行；未读：126–183, 239–285, 354–436。

**证据状态：** Selected contract and late evidence sections of a555-line historical execution ledger; no new run.

**功能、默认值与协议：**

- 1–125: baseline2e4bb881b2c13f49a83c35522064835711862429, P0–P14 behaviors, separate baseline/capacity/ON-OFF comparisons, ABBA full samples,5% median engineering heuristic explicitly not statistical law.
- 184–238: production metrics and gated256-event profile have separate ownership; Q-prep fusion deferred until actual padding cost; prefix/recurrent clearing retains synchronization rather than assuming duplicate waits redundant.
- 286–353: unsupported PQ2 fusion stays safe standalone; physical shader/module split, shared metadata, snapshot upload and Hadamard memo are distinct archived candidates with no automatic performance qualification.
- 437–477: q_prep_rows includes padded groups, normal_prefill filled from slot timing including parked wait, completed generation uses synchronized perf counts; live distribution still needs actual run.
- 479–526: first_valid preserves a real zero-duration first event, histograms count graph definitions not decode steps; scratch/in-flight/submit metric meanings are narrowly defined.
- 530–555: --rounds defaults1, only N>1 alternates arm order, JSONv2 retains raw samples and descriptive percentiles; replay float comparison switched to1e-4, mesh missing-devices uses exit77 and CTest SKIP.

**依赖与 A/B/C 边界：**

- Private RERoT server DAG/shared-KV and graph builders for runtime behavior.
- Stdlib-only counter/statistics and evidence-report policy can be separated; device counters keep their actual owner.

**缺陷证据或资格限制：**

- 100 versus later dated incident documents: broad hardware-safe statement is historical and unverified, not a present reliability assertion.
- 443,501,511,522–524: padded/graph-defined/upload-slot/graph-launch counts are not live GPU work, real peak concurrent submissions, physical allocation peak or Vulkan queue submits.
- 483: first event is first accumulator winner, not necessarily earliest wall-clock start under concurrency; denominator count-1 needs count>1 in any consumer.
- 535–540: default rounds1 does not supply ABBA evidence; descriptive medians/p95 alone are not confidence intervals.
- 542–555: relaxed numeric tolerance and9pass+1skip must remain explicit; cannot advertise strict bitwise equivalence or ten exercised GPU paths.
- Unreviewed126–183,239–285,354–436 remain excluded from content findings.

**候选保留：** Retain independent no-allocation metrics math, exact first/steady/count scopes and full-sample protocol candidates. Shared-KV DAG, shaders, parking and private graph ownership stay deferred; no blanket low-value ruling from the selected text.

### FP-BF16-LATE — `docs/TP5-BF16-LATEBIND-VERIFICATION.md`

**实际阅读：** 1–130 / 130 行；未读：无。

**证据状态：** Source proposal and unexecuted validation plan, explicitly NOT_RUN.

**功能、默认值与协议：**

- 8–26: distinguishes original BF16-weight target from Q8-derived approximate target; BF16 native expansion avoids new weight quantization but accepts higher weight bandwidth.
- 34–55: gate requires matching down/up families, Q8 inject, BF16 pipeline, row/stream limits; BF16+RHO fails visibly before plan construction rather than parsing BF16 as Q8 blocks.
- 69–114: NO_HIT is not speedup, teacher-forced full-logit/site checks precede natural-EOS performance; proposed 1e-3 tolerance cannot be substituted for strict bitwise eligibility.
- 125–130: inject type is static inference and all actual compile/numerical validation are absent.

**依赖与 A/B/C 边界：**

- Original-versus-derived model identity and fixed token/logit fixtures.
- Private Vulkan HC capture, BF16/Q8 shader dispatch and TP5 collective plan.

**缺陷证据或资格限制：**

- 101–104: exact BF16 weight expansion does not prove bitwise full-logit equivalence after reordered reductions.
- 127: unprobed inject type can produce expected NO_HIT; no benefit can be claimed from enabling an environment variable alone.

**候选保留：** Keep separate strict/approximate identities, fail-closed type and hit qualification requirements; defer the private LateBind implementation, not classify all TP5 as useless.

### FP-BF16-HISTORY — `docs/TP5-BF16-UPFOLD-FIX.md`

**实际阅读：** 1–274 / 274 行；未读：无。

**证据状态：** Evolving static and historical diagnostic ledger; later sections supersede earlier root-cause claims.

**功能、默认值与协议：**

- 25–76: aligned BF16 HC_UP_FOLD indexing needs weight_row=(first_row%stride_d)+n*stride_d; ordinary variants remain linear; includes a concrete out-of-range old-index example.
- 78–150: BF16 dispatch_rows aligned with Q8 4*n_rows, but document explicitly records unresolved over-dispatch/robustness tension; pure Python mirrors do not prove GPU semantics.
- 16–21: dump filenames collide across ranks; sigmoid(0)=0.5 means down-zero alone cannot explain every mixed-zero result.
- 153–197: mmap/meta fan-out proposed as a candidate loader issue, with static preallocation explanation unconfirmed.
- 200–274: r7 explicitly falsifies r5 container-routing root cause by same-run per-rank identity; previous compared addresses came from different runs, and guards are defense rather than demonstrated repair.

**依赖与 A/B/C 边界：**

- Private HC_UP_FOLD shader/dispatch and meta model-upload lifecycle.
- Pure integer indexing/provenance checks can be independent; raw dumps require actual per-rank ownership.

**缺陷证据或资格限制：**

- 53–56 and 100–104: historical confident root-cause language is later qualified; do not import over-dispatch on assumption robustness supplies correctness.
- 153–184 versus 200–274: evidence evolves and final W dumps still zero; these paragraphs cannot all be cited as a closed successful fix.
- 19 and 202–210: shared dump names and cross-run pointer comparisons are concrete provenance defects, not numerical evidence.

**候选保留：** Preserve red/green integer indexing oracle, identity/provenance requirements and falsified-hypothesis history. Backend-specific fixes remain conditional archived candidates, with no renewed performance qualification.

### FP-HC-DUMP — `docs/TP5-HC-SITE-DUMP.md`

**实际阅读：** 1–146 / 146 行；未读：无。

**证据状态：** Instrumentation manual explicitly not execution evidence.

**功能、默认值与协议：**

- 15–55: opt-in dump count defaults to zero, copies normalized/lo/mixed only after dispatch barriers using mid-graph synchronize, then resumes recording.
- 61–89: raw F32 files and shape/name stderr, cold path only; active replay/predefined recording and invalid/non-F32 tensors explicitly skip.
- 91–146: first divergent intermediate helps localize a defect; no timing from instrumented runs; source sentinels prove wiring presence only.

**依赖与 A/B/C 边界：**

- Private Vulkan HC graph insertion and staging synchronization.
- Raw F32 parsing/comparison can remain a separate tool with shape, rank and run metadata.

**缺陷证据或资格限制：**

- 65–69: fixed /tmp site/tensor names lack run/rank identity and can overwrite each other; UPFOLD:19 confirms the multi-rank collision.
- 138–145: cold-path mid-graph synchronization cannot observe warm replay behavior or establish an uninstrumented performance result.
- 94–107: tolerance 0 and later allowed ULP differences require a frozen numeric policy before judging runs.

**候选保留：** Retain dump schema and isolated finite-F32 comparator as tools; archive private mid-graph hooks and require per-run/rank provenance before reusing raw evidence.

### FP-MTP-EVIDENCE — `docs/TP5-MTP-EVIDENCE.md`

**实际阅读：** 1–184 / 184 行；未读：无。

**证据状态：** Historical controlled runs plus a superseded verification recipe; no new run in this audit.

**功能、默认值与协议：**

- 3–29: old TIMELINE/F16 recipe superseded by opt-in RELAY/F32; production MTP default off, real requests distinguish committed/client-wall-time TPS from server decode TPS, faults and reset attribution remain bounded.
- 13–15: narrow exclusive single-cell non-grouped recurrent clearing (size=1,n_brain_rows=0) retains scheduler fences, documented 907–935 ms ->12.8–16.8 ms; cancellation/reuse and two-cell noninterference evidence described, not a paired end-to-end estimate.
- 17–27: exact MTP+LateBind has controlled correct requests but no speed promotion; BO isolation A/B pilot 35.28 vs35.41 committed TPS lacks CI and cannot become default.
- 34–51 and 96–112: only type=3 MTP graph reuse, valid rows/outputs, generation and cycle conservation have specified log gates; phase/reset/status-heartbeat absent logs and handoff<=200us/catchup<=1500us are manual reference, not hard script checks.
- 127–184: RAII exactly-once cycle settlement resets error/zero-accept accumulators; witnessed generation can avoid redundant synchronization with an explicit conservative fallback.

**依赖与 A/B/C 边界：**

- Private MTP/RELAY and recurrent memory state for the described runtime.
- Narrow recurrent bulk-clear optimization has separate value; public request metrics and offline ledgers can be C/tools.

**缺陷证据或资格限制：**

- 5,13,17,23–27: acceptance ratio, target microtime and short pilots cannot be relabeled complete-request speedups or >100 TPS qualification.
- 111 and 117–123: missing direct phase/status logs and manual budgets mean the historic checker cannot prove those runtime invariants.
- 137–158: settlement on an error must remain distinct from successfully committed output; the doc formula final=accepted+1 is not proof an error produced a final token.
- 180–184: a safe fallback sync is correctness-preserving; a performance gate requiring zero fallback needs an explicit warmup/measured interval.

**候选保留：** Retain the narrow recurrent bulk-clear candidate, request/ledger reset rules and evidence contracts separately from deferred private MTP. Do not blanket-discard the small reset optimization because of its document family.

### FP-PREDEFINED — `docs/TP5-MTP-PREDEFINED.md`

**实际阅读：** 1–454 / 454 行；未读：无。

**证据状态：** Layered implementation/design narrative; initial batches explicitly uncompiled/unrun, later source-completeness claims are not fresh runtime evidence.

**功能、默认值与协议：**

- 14–48: MAX_CAPACITY defaults off, single-sequence Qwen4EXP only, reject oversize instead of resize or shortening drafts; 64-byte frame and 12-byte indirect args separate physical capacity from active input/output rows.
- 50–70: retained high-water pool and sealed growth checks are failure-atomic; physical reuse is only within equal backend buffer types, not all models on the same GPU.
- 72–118: RAII token+embedding ownership, verified/seed/carry lifetimes, per-owner graph definitions and unique 48-bit parent IDs; two parameter slots retire only actually submitted rank/timeline values and do not license overlapping scratch writes.
- 155–219: DEVICE_HIDDEN is a separate opt-in; RESULT has nonzero generation and valid rows, CARRY/SEED expect zero, reset must not rewind generation; copy failure can leave queued work that still needs retirement.
- 221–291: four-column Q5 transport, active payload header, descriptor bounds excluding status, scalar/tail guards before reads; generic Q5 selector max18 columns and unsupported ADD layouts preserve their original path.
- 293–340: MTP-specific fixed execution identity avoids false phase churn; target/prefill stay exact and keep lifecycle invalidation, not a generic exemption based solely on row count.
- 342–454: full-cycle timing and five semantic coverage categories, unresolved dynamic work blocked; target trunk, raw copies and noncontiguous views remain explicit gaps.

**依赖与 A/B/C 边界：**

- Private graph identity, meta buffer types, frame/route ownership and device-copy APIs.
- Checked capacity/slot arithmetic and owning host workspaces are separable B candidates; C must consume a supported A seam.

**缺陷证据或资格限制：**

- 66–70: same device != same model-owned meta buffer type; cannot claim a single physical pool for every sidecar.
- 8–17,137–153,219 and441–454: later strong completion/zero-overhead statements do not erase the unexecuted and target-incomplete boundaries.
- 285–287: dynamic transport rows alone do not prove an entire fixed-capacity model graph.
- 386–391 vs MTP-EVIDENCE:51: a cycle-summary snapshot can precede the final commit; aggregate all completed cycle rows for a full response.

**候选保留：** Preserve independent capacity/checked-layout/epoch-slot math and RAII workspace concepts as B candidates. A-owned graph/copy/hidden plumbing remains deferred; C should use explicitly exposed A phase and capacity seams rather than borrow private contexts.

### FP-MTP-PROPOSAL — `docs/TP5-MTP-VERIFICATION-PROPOSAL.md`

**实际阅读：** 1–304 / 304 行；未读：无。

**证据状态：** Historical proposal only; candidate commits and fixtures are not execution receipts.

**功能、默认值与协议：**

- 3–23 and252–270: candidate 737aed5f813df4dc5d18a43429c0ab4866ade7e2 and11f8c7eb23e3ceef0d8b4dc8b88f328ff267dac7 over target base14d9b321a2a44539ce3b2e42ae98c5237148beb0; short functional scope explicitly distinct from throughput.
- 42–92: old TIMELINE/F16, capacity/profile/replay flags, draft max3,pmin0,ctx512,batch/ubatch32,local port8099.
- 112–219: 12-token request, 30-second stated deadline, distinct log gates for type3 UID/hidden/cycle/mode, manual phase/status/mean budgets.
- 243–304: old approval and machine-state statements are dated context; not current session authorization or a validation result.

**依赖与 A/B/C 边界：**

- Old TP5 launcher/private flags, fixed local model/hardware paths and profile schema.
- Explicit identity, workload and process failure contracts are transferable to a new actual C test.

**缺陷证据或资格限制：**

- 55–57,64–86 versus193: command enables MAX_CAPACITY but not DEVICE_HIDDEN while gate requires dev_hidden==1; referenced PREDEFINED:157–163 says the latter is a separate opt-in. Later EVIDENCE supersedes the recipe entirely.
- 118–122,195,237–238: a single natural 12-token completion cannot guarantee a zero-accept cycle or exact1->4->1->2 shape coverage.
- 115 and164: curl has no --max-time; 30 seconds is an operator rule, not enforced by this literal command.
- 276: broad hardware-all-fixed wording is contradicted by later bounded incident records and must not become a current safety or reliability conclusion.

**候选保留：** Archive recipe and preserve explicit mode/identity/counter/budget contracts. Rebuild any future actual C test from the supported API and controlled shapes, not a copied obsolete launcher.

### FP-TARGET-CAPACITY — `docs/TP5-TARGET-CAPACITY-PLAN.md`

**实际阅读：** 1–476 / 476 行；未读：无。

**证据状态：** Evolving plan with bounded CPU/operator/mesh historical evidence; whole-model recurrent TP5 admission remains closed.

**功能、默认值与协议：**

- 3–48: target verify capacity equals session verify_tokens, outputs=tokens, single sequence; prefill exact; active controls work, capacity controls stable stride/address, inactive state writes forbidden.
- 50–149: explicit input/position, attention, recurrent/rollback, MoE, output and relay consumer contracts; MTP output semantics and phase exemption cannot be reused blindly for target.
- 175–213: named commits and local single-card GDN A={1,2,4},C=4 poison-tail/reference checks, but GDN/recurrent production remains refused; multisequence,capacity>8,cached/RBB/chunking combinations remain unsupported.
- 215–282: header word5 publishes active rows per bank, seven late kernels consume it; sidecar format follows producer, full-capacity barriers and norm-ready command emission correct cross-stage consumption.
- 284–310,346–448: packed weights36B/32 elements plus initialized done flag; activation packed layout changes f16 scale to f32, later ACT_Q8 packing OOB discovered despite mesh success; ABI constants and every producer need the same word packing contract.
- 311–344: pipelineExecutableInfo must be enabled for requested statistics; pipeline code/VGPR/LDS facts are creation-time observations, not model throughput.
- 450–476: scalar lane/subgroup geometry and checked packed range; pack-clear fence changed from UINT64_MAX to2 seconds, a small failure-bounding behavior inside private TP5.

**依赖与 A/B/C 边界：**

- Private target graph lowering, active-row kernels, recurrent rollback and RELAY bank publication.
- Pure checked layout/ABI functions, bounded waits and generic statistics capability wiring need individual ownership decisions.

**缺陷证据或资格限制：**

- 5 and200–205: independent regional passes cannot be combined into full-model state or speed acceptance.
- 361–363: wider scale changes numeric identity; "dequantization error strictly shrinks" is not a proven end-to-end quality guarantee.
- 435–448: mesh builds pipelines but did not execute the faulty latebind plan; previous15/15 or codegen statistics did not prove its correctness.
- 341–344,458–464: zero spill, smaller code and VGPR changes do not by themselves prove occupancy or throughput gain.
- 100–105,224–239: payload length and fixed capacity are separate producer/consumer publication fields; a library holding only scalar dimensions cannot replace the actual GPU ownership protocol.

**候选保留：** Keep checked row/ABI arithmetic, explicit capability/coverage failures, producer-wide packing tests and bounded cleanup requirements. Generic Vulkan statistics feature-enable fix merits its own conditional candidate; archive private full-target graph implementation pending real end-to-end evidence.

### FP-VERIFY-PLAN — `docs/TP5-VERIFICATION-PLAN.md`

**实际阅读：** 1–257 / 257 行；未读：无。

**证据状态：** Historical v1.0 draft explicitly superseded by its2026-09-23 coverage note.

**功能、默认值与协议：**

- 3–16: superseding note rejects blanket all-ctest/hardware-safe assumptions and old TIMELINE sequence; distinguish model-reachable GPU suites from focused mesh/operator checks.
- 21–55: fixed spin upper bounds, submitted-work retirement, fail-closed resource ownership and independent numeric-mode/FD evidence.
- 61–109: six CPU contracts and dry configuration print are distinct from actual model execution.
- 117–219: proposed aggressive/exact probes and MTP cycle observability, with numeric-mode and hidden-generation requirements; runtime caller wiring is still listed as an open point.
- 225–253: corrected sidecar offsets and old qheader retirement, wave32 capability fallback and default11-slot workspace isolation requirement.

**依赖与 A/B/C 边界：**

- Historical TP5 mesh/RELAY/MTP tools and launcher defaults.
- Public process/HTTP checks can be independent but this literal old recipe is not self-enforcing.

**缺陷证据或资格限制：**

- 123–129,147–153: tee pipelines omit pipefail; the literal shell status need not be the tested process status.
- 169–189: fixed15-second sleep is not readiness; curl has no deadline and wait||true hides exit failures with no bounded final wait.
- 25–26 vs TARGET-CAPACITY:445–448: the mesh creates latebind pipelines but is not an actual latebind-plan execution proof.
- 111 versus40: two different idle VRAM thresholds occur; they are historical operating criteria rather than transferable product defaults.
- 175,252–253: launcher defaults can be11 slots; a single HTTP request does not prove a single-slot configuration or workspace isolation.

**候选保留：** Archive obsolete operating commands; retain failure ownership, finite waits, capability-hit evidence and explicit slot isolation as requirements for a future actual consumer.

### FP-NAN-AUDIT — `docs/development/nanbeige-audit.md`

**实际阅读：** 1–243 / 243 行；未读：无。

**证据状态：** Detailed runner contracts plus bounded historical operator/model evidence; explicitly no production approval.

**功能、默认值与协议：**

- 28–44: exact Vulkan device selection avoids success-with-zero-cases; required-backend routes fail instead of substituting CPU, planner tests use physical key sets and offsets.
- 46–84: owned loopback/auth server, finally cleanup and pre-cleanup exit status; dependency/link response-file freshness, runtime commit and before/after artifact hashes have explicit incomplete-coverage limits.
- 104–130: incomplete output/timing fails with saved evidence; early/middle/late needle fixtures share source tokens, duplicates invalid; first request separate from repeats and no synthetic warm metric for one request.
- 132–185: real drain/scoring/FlashPrefill plan gates, floor-only score0 vs scorer>0, private training-trie limits8x1024 tokens; logical44 layers and physical KV capacities must be accounted separately.
- 187–242: online value accumulation measured prefill191.35->251.50 with reference recheck190.57 tok/s, decode~58 unchanged and sampled VRAM2281 MiB; output differs and512-cell Tri retrieval fails both arms.

**依赖与 A/B/C 边界：**

- Stdlib HTTP/process/filesystem probe plus exact old server response schema.
- Private Tri/FlashPrefill runtime counters for pressure/plan checks; model-specific calibration and token IDs.

**缺陷证据或资格限制：**

- 61–77: checked_links=0/dirty source/current version text do not prove build freshness, reproducibility or ABI consistency.
- 132–166: floor-only reclaim cannot be counted as calibrated scorer coverage.
- 218–229: local~31% prefill improvement is real historical bounded evidence, but text drift and quality failures block whole-model qualification.
- 231–236:22-second process-first versus0.35-second repeats need distinct reporting, not erased warmup.

**候选保留：** Retain strict public HTTP probe/identity/report contracts and prefix-trie preparation boundary as independent tool/C candidates. Archive FlashPrefill as locally beneficial but unqualified, not worthless.

### FP-NAN-EVIDENCE — `docs/development/nanbeige-evidence.md`

**实际阅读：** 1–258 / 258 行；未读：无。

**证据状态：** Runner/report protocol with historical small probes; later recommendation wording exceeds bounded evidence.

**功能、默认值与协议：**

- 7–33: exact canonical request/token/model hashes, stable files, untruncated uncached workloads and consistent finite counts/rates; hashing is outside timings but warms file cache and covers only the named GGUF.
- 35–89: first request separated, aggregate tokens/total duration, at least2 subsequent samples for a descriptive ratio, mandatory identity/completeness; optional --require-output-match, duplicate JSON keys/nonfinite numbers/failures rejected.
- 104–159: startup crashes remain unresolved despite matching artifact identities and later short successes; GCC16 PRE workaround is function-local, dirty-source build explicitly not clean-HEAD.
- 161–230:32K/64K/96K capacity probes use only8prompt/2generated tokens; actual Tri scoring pressure4096->384 cells has score_ms24269 and pack_ms25.717.
- 232–258: six numeric/needle checks, four concurrent requests and16-request stress record are finite scenarios, followed by a historical configuration recommendation with FlashPrefill off.

**依赖与 A/B/C 边界：**

- Offline stdlib JSON report, canonical request/model/artifact hashes and raw runner responses.
- Old private Tri/physical-KV configurations for historical measurements; no implied clean-fork feature availability.

**缺陷证据或资格限制：**

- 72–83: output agreement is optional for approximate experiments, so a report ratio does not imply exact-output equivalence or broad quality.
- 163–223: allocated96K physical capacity plus an8/2-token workload does not establish96K-context retrieval/long generation stability; device-wide sampled VRAM is not an exact peak.
- 240 and246–258: "100%" and production recommendation are stronger than six smoke items/16 requests; retain bounded observations and the header no-approval limit.
- 251–252: old logical65536/physical32768+Tri recipe is not compatible with the clean fork merely by copying flags; capacity and omitted eviction semantics must remain explicit.

**候选保留：** Retain strict comparison and failed-evidence preservation rules as independent report candidates. Preserve capacity/scorer facts with exact workload limits; do not import historical deployment recommendation.

### FP-PERF-NECESSITY — `docs/performance-necessity.md`

**实际阅读：** 1–182 / 182 行；未读：无。

**证据状态：** Explicit withdrawal of a theoretical optimality claim; no new GPU run.

**功能、默认值与协议：**

- 5–38: measured implementation time gives attainable U, not universal lower bound L; fixed numerical contracts and allowed strategies must be declared.
- 40–58:1224 tensors inspected/1222 counted,4,614,087,680 unique-name encoded bytes vs9,198,287,872 current-replica bytes, not mandatory information-theoretic traffic.
- 60–101: prefetch/overlap/critical-path counterexamples, independent kernel timing and predicted67TPS do not prove optimality.
- 103–151: explicit I/O/cut-set premises, MTP public-token denominator and separate achievable/necessary/explanatory ledgers.
- 153–164: historical24 responses/eight exits rechecked; baseline67.5681447 vs pack67.4015693TPS, source/logic tests only.

**依赖与 A/B/C 边界：**

- GGUF/manifest metadata and independent Python arithmetic/evidence classification.
- Hardware, workload and numerical premises required for any conditional bound; no live A or GPU needed to review the argument.

**缺陷证据或资格限制：**

- 14 and159:67.15 observed or accurate prediction cannot justify declaring little optimization potential; pack micro-optimization negative result only constrains that candidate.
- 118–127 and155–157: labels and abstract counterexamples are not an automatic theorem prover or new hardware speed evidence.
- Full references read as part of provided source; external links were not opened or independently verified in this bounded local audit.

**候选保留：** Keep necessity-versus-measurement distinction and pure counterexample/audit tool candidates; retire false optimality claims while preserving original measurements.

### FP-QWEN-UPSTREAM — `docs/qwen4exp-upstream-optimizations.md`

**实际阅读：** 1–193 / 193 行；未读：无。

**证据状态：** Dated upstream-port tracking and bounded historical evidence, not a current upstream status lookup.

**功能、默认值与协议：**

- 3–19: superseding2026-09-14 scope correction: a sidecar path/load is not evidence MTP drafted; older3.02TPS baseline invalid and full quality/performance open.
- 42–62: sparse FA still pays O(mask length) compaction and overflows to dense, indexer-V removal does not halve total bandwidth, fusion requires allocator lifetime dependency.
- 66–158: seven externally authored PR-derived features tracked with local CPU/operator/model scopes, commits311d4211b,6788edb4f,50182a53f; speeds explicitly UNVERIFIED HERE.
- 162–193: multi-rank replica copy/state-restoration boundary, operator correctness and full-model performance kept distinct; unresolved checkpoints/replay/transfer tests remain listed.

**依赖与 A/B/C 边界：**

- External PR provenance as recorded in the provided historical document.
- Current-pin A source comparison is needed before deciding any old port is still absent; not performed anew by this document pass.

**缺陷证据或资格限制：**

- 5 versus158: old ordinary TP5 outputs were later disqualified as MTP participation evidence; apply the leading correction.
- 68–78: "ported and compiled" is not authorship and does not show absence from the new upstream pin; do not re-credit or blindly transplant upstream PR work.
- External PR/issue URLs and hardware notices are historical text; this pass does not verify their current status.

**候选保留：** Preserve provenance and scope corrections. Keep upstream-supported model/runtime functionality through A; any old fork-specific extension requires separate current-pin diff evidence, not this tracker alone.

### FP-SPEC-HANDOFF — `docs/speculative.md`

**实际阅读：** 1–498 / 498 行；未读：无。

**证据状态：** Full old-source document read; exact diff shows only144–161 device-hidden paragraph is fork-added in this comparison.

**功能、默认值与协议：**

- 11,84–109,250–393: aliases, chained/adaptive MTP, default depths, CPU scheduling, KV overrides and ngram parameters were already in the compared A version; no fork-authorship claim.
- 144–160: fork addition says GGML_SPEC_DIRECT_HIDDEN defaults1 for one stream/one common non-CPU device with copy+finite-sanitizer support,0 forces host reference; capability probe before first decode, fallback explicit.
- 156–160: sanitizer contract NaN->0, infinity->+/-65504, finite out-of-F16-range values unchanged; persistent storage is F32 n_embd*n_batch per extracted layer plus nextn.
- 162–248,469–498: upstream draftless precedence/shared ngram pool and draft/accepted counters must not be confused with complete-request net throughput.

**依赖与 A/B/C 边界：**

- Most documented speculative modes/options already belong to compared upstream A.
- Fork-added direct hidden path needs owned persistent device buffers, real range-copy/sanitizer capability and same-device single-stream semantics.

**缺陷证据或资格限制：**

- 146–159: unconditional "both paths identical tokens" and "cost memory,not time" are documentation assertions, not independent DFlash/DSpark model evidence; device copy/probe/allocation still have costs.
- 152–155: matching device name or successful model load alone does not prove supported range-copy/sanitization or safe stream/owner lifetime.
- No external model links were browsed; supported-model listings are provided historical source context only.

**候选保留：** Keep the exact sanitizer/default/fallback/memory contract as an archived direct-hidden requirement; retain existing speculative features in A. Do not import an unverified private hidden-state fast path or count A aliases/controllers as new B features.

### FP-TRI-COMPAT — `docs/triattention-coop-fixes.md`

**实际阅读：** 1–43 / 43 行；未读：无。

**证据状态：** Combination-correctness contract and test recipe, not new runtime acceptance.

**功能、默认值与协议：**

- 9–14: Turbo shift inverses storage rotation/RoPE then requantizes shifted rows only, CPU batch-preflight memmove, Vulkan unaligned byte scratch preserving boundaries, real quant-type decoder instead of zero scores.
- 12–18: plan all stream moves before committing live refs; draft pressure runs independently of target.changed with its own slot hints; actual deserialization governs compressed-state recomputation.
- 15–20: target zero calibration coverage refused, uncalibrated nextn draft gets explicit recent-only fallback; no-hints enumerate all resident sequences/streams, ref-only changes invalidate views.
- 22–43: model-free real-cache/memmove/score tests separated from long trajectory, MTP acceptance, concurrency and soak acceptance.

**依赖与 A/B/C 边界：**

- Private Tri cache metadata, quantized codecs, per-stream compaction and server draft-state hooks.
- Transactional planning, exact type decoding and memmove/range oracles are individually separable candidates.

**缺陷证据或资格限制：**

- 12: preflight failure is transactional but device execution failure is still unrecoverable; do not label all failures rollback-safe.
- 15–20: draft recent-only fallback is not learned importance calibration or target policy equivalence.
- 20: host codec K-shift is not a GPU fused kernel.

**候选保留：** Retain candidate transaction, decoder, shift and request-state invariants as separate behaviors; archive private Tri cache semantics until an actual supported A boundary exists.

### FP-XKV-BASE — `docs/xkv/baseline.json`

**实际阅读：** 1–102 / 102 行；未读：无。

**证据状态：** Historical frozen candidate specification; every local evidence field is NOT_EVALUATED.

**功能、默认值与协议：**

- 1–43: target commit 5298ea46061c4780d173de866bdbba42f5918135, server/calibration historical hashes and candidate 40-layer model layout are separated from actual local verification.
- 46–69: old Vulkan K=turbo4_0/V=turbo2_0, Tri 3/32 with recent 128, RERoT strong and MTP draft max 2 form a full-stack baseline.
- 70–82: admission to acceptance requires coverage >=75%, net saving >=10%, no dense mirror, peak <=1.15x, PPL degradation <=5%, context ratio >=95%, state/allocator/fingerprint gates.
- 85–100: auto-fit, shared-prefix union, recurrent pressure, RAM/swap, soak, codecs and landmark-quality evidence are explicitly absent.

**依赖与 A/B/C 边界：**

- Frozen old target/model/calibration/server identities and private XKV/Tri/RERoT/MTP profile.
- Plain configuration/evidence schema can be handled outside A.

**缺陷证据或资格限制：**

- 8–20: candidate architecture/head dimensions are unverified; other profile JSON concrete dimensions cannot promote them to measured model facts.
- Historical hashes are not fingerprints of the clean fork or current running process.

**候选保留：** Archive the frozen specification and explicit missing-evidence states; reuse numeric acceptance contracts in a separately scoped C experiment, not the old private runtime defaults.

### FP-XKV-GATE — `docs/xkv/compression_gate.json`

**实际阅读：** 1–110 / 110 行；未读：无。

**证据状态：** Saved unqualified report: compression_goal_met=false, 16 NOT_EVALUATED checks, evidence_provided=false.

**功能、默认值与协议：**

- 1–13 and 95–110: target E1, zero PASS/FAIL, sixteen NOT_EVALUATED, thresholds preserved.
- 14–94: missing calibration/profile/codecs/coverage/mirror/saving/peak/allocator/fingerprint/quality and state interoperability remain explicit.

**依赖与 A/B/C 边界：**

- Offline compression gate schema and referenced frozen baseline thresholds.
- Contains no runtime evidence to depend on or promote.

**缺陷证据或资格限制：**

- 4 and 109: evidence_kind="real" is a schema category while evidence_provided=false; it does not establish a real accepted run.

**候选保留：** Keep as a negative or incomplete historical receipt; never cite it as a successful compression result.

### FP-XKV-PLAN — `docs/xkv/matrix/plan.json`

**实际阅读：** 1–43 / 43 行；未读：无。

**证据状态：** Plan only: empty items, empty server URLs and launch=false.

**功能、默认值与协议：**

- 1–20: B0 and E1 profile roles, sampling max 512, seed 42, greedy temperature 0/top_k 1/top_p 1.
- 21–43: old single-server context 262144, GPU all, total_kv="auto", no n_parallel/device resolved.

**依赖与 A/B/C 边界：**

- Offline matrix runner with referenced B0/E1 profiles; workload and URLs are empty.
- Old total_kv=auto policy is not the current explicit physical-capacity seam.

**缺陷证据或资格限制：**

- 3 and 21–43: no workload or destination is defined, so a plan fingerprint is not execution evidence.
- B0/E1 in the referenced profile change RERoT and MTP along with XKV; a comparison is full-stack rather than isolated XKV.
- Old total_kv="auto" does not define the new explicit physical-capacity API.

**候选保留：** Reusable C experiment schema with explicit workload and identity checks; archive historical defaults.

### FP-XKV-PROFILES — `docs/xkv/profiles/ornith-r2.json`

**实际阅读：** 1–368 / 368 行；未读：无。

**证据状态：** Configuration-only profile matrix; no model probe or evaluation performed.

**功能、默认值与协议：**

- 1–65: fixed model groups/dimensions and quality/memory gates; candidate model identity still requires validation against baseline.json.
- 66–368: ten profiles B0, R0–R5, E0–E2 distinguish dense/factor/sparse storage, F16/Turbo4 codecs, Q8/Turbo4 landmarks, CPU-reference/Vulkan factorizer and full-stack six-slot variants.
- 66–368: active profiles expose segment 4096, chunk 8, workspace 256 MiB, decode cache 64 MiB, minimum net saving 10%, runtime factor coverage 50%, SR budget 256 and boundary max rows 128.

**依赖与 A/B/C 边界：**

- Private XKV codecs/factorizer/SR/Tri/RERoT/MTP runtime choices and model dimensions.
- Budgets/profile roles and exact integer seed are transferable configuration contracts, not an independent implementation.

**缺陷证据或资格限制：**

- R3/R4 descriptions imply one-factor ablations but also switch CPU-reference versus Vulkan factorizer; retain that additional changed variable.
- Runtime minimum coverage 50% and acceptance gate 75% are different thresholds; documentation must not conflate them.
- Seed 6362273814452121649 exceeds exact JavaScript Number integer range; preserve integer serialization if transferred through a JS consumer.
- Tail group ranks 192/288 coexist with per-profile global ranks 384/576; this JSON alone does not establish which consumer override wins.

**候选保留：** Retain explicit budgets and controlled ablation roles as C configuration candidates; do not import every private runtime flag or label profiles as validated.

### FP-FP-STATE — `scripts/flashprefill-state-matrix.json`

**实际阅读：** 1–132 / 132 行；未读：无。

**证据状态：** Explicit NOT_RUN/unvalidated HTTP scenario specification.

**功能、默认值与协议：**

- 1–25: required-mode FlashPrefill plus mean correction and trusted Tri; RERoT off; required five requests have 600/600/120/180/120 second deadlines.
- 17–23: GEN:256 strings are documented examples requiring operator materialization, not an implemented token-count macro.
- 32–36: MTP partial rollback, RERoT publication, context shift and pen suspend have no corresponding HTTP endpoint; C++/context tests are needed.
- 52–128: slot save/erase/restore and cancellation legs define metric deltas and visible>=exact/sparse-implies-corrected invariants; C and F are explicitly partial.
- 130–132: no owned server launch and comparisons=[]; this is not an A/B quality matrix.

**依赖与 A/B/C 边界：**

- Actual HTTP slot-state/stream/cancel endpoints plus exact private metrics schema.
- Unsupported state transitions require real lower-level tests; literal GEN example requires token materialization.

**缺陷证据或资格限制：**

- 52–73 and 115–128: FULL labels describe the exposed HTTP leg; neither actual resource drain nor a cancellation before natural completion is ensured by this JSON.
- Known separately reviewed runner can accept natural completion or HTTPError in cancellation; recipe success alone cannot prove reclaim/cancel semantics.
- 25: hard-abort metric availability is version-specific and differs from other old runner contracts; absence must not silently pass.

**候选保留：** Keep public slot-state/cancel scenario requirements and honest unsupported-operation boundaries; archive private FlashPrefill metrics until implemented consumer and actual workload evidence exist.

### FP-BONSAI-REQUEST — `scripts/rerot-bonsai-throughput-request.json`

**实际阅读：** 1–13 / 13 行；未读：无。

**证据状态：** Input fixture only.

**功能、默认值与协议：**

- 1–13: hardcoded model path, twenty square tasks, temperature 0.4, seed 20260923, max_tokens 8192, non-streaming.

**依赖与 A/B/C 边界：**

- Ordinary chat request JSON and caller-resolved model/experiment arm.
- Actual RERoT/DAG/lane semantics are external to this fixture.

**缺陷证据或资格限制：**

- No rerot flag is present; an external runner must set the actual experiment arm.
- Twenty requested text tasks do not imply twenty active pens or six lanes in a batch.
- Nonzero temperature is a different sampling contract from the Ornith greedy fixture.

**候选保留：** Retain prompt fixture after C resolves model path and variant; do not enable private runtime implicitly.

### FP-REROT-REQUEST — `scripts/rerot-throughput-request.json`

**实际阅读：** 1–15 / 15 行；未读：无。

**证据状态：** Input fixture only.

**功能、默认值与协议：**

- 1–15: Ornith model alias, five simple independent text tasks, temperature 0, seed 424242, public max_tokens 8192, non-streaming RERoT on with trace off.

**依赖与 A/B/C 边界：**

- Chat request consumer with old RERoT flag and model alias.
- Public token budget is separate from private per-pen scheduling/capacity.

**缺陷证据或资格限制：**

- The public 8192 cap does not define per-pen budgets or actual token count.
- Five tasks in prompt prose do not establish DAG topology, same-batch lanes or shared KV; those require runtime counters.
- Model alias is not a model-file fingerprint.

**候选保留：** Retain as a C request fixture with explicit model/arm identity; no runtime capability claim.

### FP-MMVQ-CONTRACT — `tools/tp5/orthogonal-mmvq-contract.md`

**实际阅读：** 1–76 / 76 行；未读：无。

**证据状态：** Authored source-integration contract explicitly uncompiled/unrun.

**功能、默认值与协议：**

- 5–28: fusion legality, activation arithmetic and transport epilogue are independent; QSA/GDN masks use existing Q8_1 layout and one preparation per recorded region; missing transport epilogue falls back to tensor/P1 without disabling chosen MMVQ.
- 30–47: output lowering preserves arithmetic; graph-proven local elision and recording buffer ownership prevent hidden scratch realloc submit/sync or cross-token stale-quant reuse.
- 49–76: compare native against native and MMVQ against the same Q8_1 reference; greedy token agreement is insufficient, record counts are not runtime invocation counts or timing; capability misses are SKIP.

**依赖与 A/B/C 边界：**

- Private Vulkan fusion/MMVQ pipelines, Q8_1 layout, recording owners and TP5 output routes.
- Arithmetic-identity and safe fallback requirements can be preserved independently.

**缺陷证据或资格限制：**

- 3,55–76: proposed synthetic coverage and source structure are not GPU numerical or end-to-end performance acceptance.
- Native F32 and Q8_1 activation paths are not bitwise equivalent; composition does not prove additive speedups.

**候选保留：** Keep arithmetic/ownership/fallback contracts as archived TP5 requirements and independent test-oracle candidates; private recording/pipeline plumbing stays deferred.

### FP-OUTPUT-CONTRACT — `tools/tp5/orthogonal-output-contract.md`

**实际阅读：** 1–59 / 59 行；未读：无。

**证据状态：** Authored source-integration contract explicitly uncompiled/unrun.

**功能、默认值与协议：**

- 11–35: local partial elision requires real graph read-range/liveness proof and route permission; aliases, mixed ranks and LateBind keep local data; K128 push ABI grows from 4 to 8 bytes.
- 37–45: vector writes use exact scalar tails/unaligned fallback and no neighboring read-modify-write; vector width is not measured PCIe transactions.
- 47–59: descriptor lookup must not request BDA; explicit address lookup checks zero/overflow/view ranges. CPU liveness fixture covers aliases/unknown ranges/metadata cycles, not GPU visibility or speed.

**依赖与 A/B/C 边界：**

- Private Vulkan producer liveness, route permissions, descriptor/BDA helpers and shared shader ABI.
- Pure range/liveness/overflow policy can form B tests only if C/A supplies real ownership facts.

**缺陷证据或资格限制：**

- 3 and 55–57: source contract and CPU liveness policy cannot prove host/device coherence or replay correctness.
- Future external consumers must declare local-data need; a terminal name alone is not a liveness proof.

**候选保留：** Preserve liveness/range/zero-address contracts and independent CPU oracle candidates; do not move private Vulkan graph traversal into a nominally independent B.

## 收口

24 个限定 residual 均有实际内容记录，其中 23 个可从“仅用途清单”提升为“文档全文已审”，RERoT 台账提升为“指定章节已审”。该深度只描述阅读覆盖，所有 `runtime_validation_performed`、`new_execution_qualification`、`migration_performed_by_this_record` 都为 `false`。188 行未读、底层实现引用、外部 PR 链接、历史工件与原机配置没有被擅自纳入已验证范围。本轮至此停止，没有追加实现或模型运行。

# scripts / tools/tp5 / pocs / docs：实际内容补审

> 整合说明：这是固定旧提交的分册审阅记录。原 coverage 文件中的逐路径、逐 hunk 和函数范围已并入 [files.json](files.json)，保留 `source_stage`；下文原始 diff / checkpoint / patch / 生成器名称是审阅资料标识，不是已迁入功能或当前构建入口。后续补审见同目录其他分册；最终实现状态以 [MIGRATION.md](../MIGRATION.md) 和 [VALIDATION.md](../VALIDATION.md) 为准。所有旧侧 / 新侧行号都指旧 fork 与固定 A 的差异，不是 clean 分支行号；记录明确标注 `audit-extra/clean` 的补充片段例外。

本轮把可执行小工具逐一从“用途/符号盘点”推进到实际内容审阅。**建议保留若干独立数值、协议和证据能力；下列候选尚未在本轮迁移。旧验证器的缺陷只说明旧验收证据不足，不推出 TP5、RERoT 或 XKV 本身没有价值。**

## 范围与证据边界

- 基线 A：`cda0f9256665845cd9d6f3a0176880883f653a3d`；旧 fork：`2777543b2358c88cd1c2000f7fa342b764a0e213`。
- 原始证据：`audit-scripts-tests.diff`，SHA256 `ed00f769609b23934f39e63ef743edf2b64c396334262badb5acb6db802749b9`。文件/行号均指旧 fork b-side 源文件。
- 总清单 4,634 路径；去除 artifacts/reports 为 714。本分册覆盖指定四个前缀中的 110 路径，其余 604 个属于其他分册，不声称由本分册完整审阅。
- 新增内容审阅 72 路径：**54 个完整源码文件、17 个大文件的指定函数/区间、1 个仅已提供的 CMake diff 区间**。其中 71 个为代码/构建文件，另 1 个为 4 行 sudoers 策略。
- 原有 14 个内容审阅只归属引用，未重读、未升级深度。24 个纯文档/JSON 配置仍标成 purpose inventory，未作内容取舍。这不是对所有 110 文件的全文审计声明。
- 全部工作只读本地 diff/source。没有远端调用、没有编译模型/运行服务器、没有 GPU 或恢复脚本执行，没有修改原仓库实现。
- JSON 中每条都有 `path`、`review_level`、`read_scope`、`ranges`、`finding_ids`；大文件另有未读区间和所读 Python 函数。此报告与 JSON 一起构成边界明确的证据。

## 已用纯本地反例确认的问题

执行方式是从原文件 AST 选取已读纯函数体，以及一个 bash `read` 内建命令；不执行 runner 顶层。原始输入、输出和源码 hash 见 `audit-script-content-counterexamples.json`。

| 证据 ID | 文件 / 函数 | 实际观察 | 取舍含义 |
| --- | --- | --- | --- |
| SC-Q01 | `flashprefill-quality.py:960–984` / check_pair_equality | 一侧缺键、另一侧为 null，payload 不同但 violations 为空 | 保留请求对拍的意图；修成真正 canonical payload equality 后再作为验收门。 |
| SC-Q02 | `flashprefill-quality.py:494–524` / score_numeric | 输出 1、参考 1e309 被判正确 | grader 必须拒绝非有限数值和非法 tolerance。 |
| SC-N01 | `tp5-necessity-certificate.py:278–304` / verify_lp_dual | 真正最优值 0 的 LP 返回正下界 5e-10 | tol-feasible 不是严格 dual-feasible；数学证书需要误差修正，不能直接升级为 PROVEN。 |
| SC-S01 | `xkv/run-evaluation-matrix.py:230–313` / parse_sse_stream | 坏 JSON 行后跟 stop=true，malformed=1 但 error=None | 终止事件正确不代表全部事件合法；上层必须检查 malformed，不据此单独断言整个 matrix 错判。 |
| SC-P01 | `gpu-hard-unlock.sh:421–423` | 两行 PID 经 read -a 只读第一行 | 全部客户端 preflight 的说法不成立；恢复代码未运行，应单独修复再人工使用。 |

## 优先保留的小能力及 A+B+C 边界

| 能力 | 具体依据 | 最小独立边界 / 仍需条件 |
| --- | --- | --- |
| 真实请求对拍与测量分级 | tp5-gap-bench 的非空/预算校验、自然 stop、原始 body、实际 /proc DSO hash、NOT_COLLECTED 与 MEASURED_NOT_QUALIFIED；pack/stage-y 配对原文等价 | C 管 owned server 与公开 HTTP，请求/计数/原文作为值传入判定；私有 TP5 参数和日志生产者留为可选配方。 |
| 普通服务端 multi-input / streaming usage 契约 | rerot-simple-continuation-smoke:140–177 的两输入×两输出索引、一次 usage/DONE、逐输出终止 | 普通请求测试可直接作为 C 测试保留；probe/RERoT 段与公共 HTTP 测试分开。 |
| 缺证据与未执行状态 | rerot-throughput-gate:503–642 区分 FAIL / EVIDENCE_INCOMPLETE / NOT_EXERCISED，gauge 不能冒充并行 work；真实调用 gate.main 的 mock HTTP 回归 | reader 可独立 C；旧 A 没有 producer counter 时应明确未执行，mock 不替代真实同批 decode。 |
| 数值参考与内存算术 | HC coordinate 原型、RBB stride replay、XKV checked integer/row-padding/owner-dedup 算术 | B 只接收自有数值与结构；C 从 A 输出/公开 type 信息构造 owned 值。估算与实际分配/压缩分开。 |
| 条件性性能必要性推理 | necessity-audit 的同前提 max，certificate 的全分支 min/未解决拒绝 | 条件性 B 数学工具有价值；修 LP tolerance 和溢出，不把非空论证文字当硬件必要性证明。 |
| 证据文件队列和独占锁 | experiment-queue 的 QUEUED/NOT_RUN、fcntl 与临时替换 | 独立 C 的离线登记工具；它没有执行模型，不称 GPU scheduler。 |
| 可执行答案 grader / 最终答案提取 | flashprefill-quality / XKV matrix 的 exact、boxed、numeric、regex、external grader；manual 保留 NOT_EVALUATED | B 提取/规范化，C 调用受控 grader 并冻结身份；修 finite、版本、命名和缺数据，不混入模型 runtime。 |
| 真正独立硬件 POC | Vulkan p2p allreduce cpp 不消费 llama 私有 API，有真实 51200 元素检查和 100 周期计时 | 可以成为独立诊断 C，但需修能力/内存 type/等待/返回码/非恒定数据/资源清理；不是模型性能证明。 |
| 手动 GPU 诊断 | gpu-idle-audit 的 readonly busy/VRAM/dmesg 状态 | 单独 C 工具，修 missing-data 与日志游标。root reset/reboot/sudoers 与推理发行物分开归档，不自动接入。 |

## 旧 PASS / “已验证”文案不能直接沿用的具体位置

- `rerot-capacity-matrix.py:136–147,174–185` 两/三请求实际串行；ThreadPool 在 return 后；263/273 先写 PASS，281 才检查 abort。不能作为真实并发 KV 池容量证据。
- `rerot-target-ornith-multi-lane.py:237–259` 并行 counter 可缺席或为 0，末尾仍声称 multi-lane verified；文本节点位置不能替代因果事件。
- `bench-bonsai-6pens.py:142` 的 speedup 分母是 server decode TPS，分子是 client end-to-end TPS，口径不同；单次 OFF/ON 也没有配对统计。
- `flashprefill-matrix.py:2003–2073` 可以把 HTTPError、自然提前结束或仅 client close 标为 cancel ok；真实服务端取消/slot 释放仍须事件或状态证据。
- `test-t1c-publish-order-mock.py:101–102` 的 cpu_reduce 不读取 payload，常数结果不能证明 Q8/传输算术。其顺序契约仍可保留。
- `xkv/off-regression.py:364–407` 若干登记选项没有消费；resume 未冻结模型/calibration/DSO，基线零值可使性能门通过。应修 checker，不把这类旧 PASS 当 OFF 全语义证明。
- 多个小 micro runner 只看 rc=0 或允许空 JSON。实际 gap-bench 自己有 HTTP/teardown 超时，但外层 runner 或独立 micro 进程没有普遍的 wall-clock 上限；逐文件记录区分了这两层。

## 逐文件内容审阅清单

“可保留”均为审计候选；本轮没有执行迁移。大文件的取舍只针对已读行为，不覆盖未读主程序、规则表或模型执行路径。

### SCF001 — `pocs/CMakeLists.txt`

**实际阅读：** 11–15；级别 `all_provided_diff_hunks`。仅实际 diff b-side 11–15 可用并全读；未提供的原文件区间不作判断。

**行为与依赖：** 在现有 POC 构建分支新增 vulkan-p2p 子目录。 A 内部 POC 构建图；新增子目录要求 Vulkan。

**保留边界：** 新独立诊断工程不需这处 A 构建接入。

Original complete source is unavailable; only b-side 11–15 was supplied. Do not infer other lines.

实际缺陷或适用限制：

- 启用该 POC 分支引入 REQUIRED Vulkan 依赖；独立构建需从 A 图脱离。

### SCF002 — `pocs/vulkan-p2p/CMakeLists.txt`

**实际阅读：** 1–8；级别 `full_source`。1–8 全文。

**行为与依赖：** Vulkan::Vulkan 链接的 C++17 独立诊断可执行 target。 CMake/Vulkan；仍列出 A/src/common include 路径。

**保留边界：** 可改成独立 build/install 工程；实际 cpp 没有消费 A include。

实际缺陷或适用限制：

- 无自身 minimum/project/install/test；当前文件本身不能证明 standalone build。

### SCF003 — `pocs/vulkan-p2p/test-vulkan-p2p-allreduce.cpp`

**实际阅读：** 1–472；级别 `full_source`。1–472 全文；386–430 全输出检查；432–459 计时；461–469 条件性成本外推。

**行为与依赖：** 真实五设备 DMA-BUF full mesh 两阶段 ReduceScatter/本地计算/AllGather，CPU 检查 51200 浮点输出并计时 100 周期。 Vulkan SDK、Linux fd、glslc；没有 llama 私有 API 依赖。

**保留边界：** 有独立硬件诊断价值，可单独成为 C；先补设备选择/能力/超时/错误/非恒定数据/清理，不原样迁移。

实际缺陷或适用限制：

- 113–139 固定前五枚举设备/queue family 0；167–171 无匹配 memory type 时仍使用 0。
- 182–203 import 不查询 buffer memory requirements 或与 fd memoryTypeBits 交集，allocation size 使用逻辑 TOTAL_BYTES。
- 固定 /tmp shader 产物可并发冲突，读取 SPIR-V 缺完整错误校验。
- UINT64_MAX fence wait 无界，多处 Vulkan 返回码未检查；进程结束前未显式释放资源。
- 输入按 rank 为常数，坐标/chunk 置换错误可能不被发现；只在计时前验证一次。
- 跨设备外部内存 ownership/visibility 仍需规格与真实平台验证，现有 host fence+局部 barrier 不构成可移植性证明；96 倍周期外推不是模型吞吐。

### SCF004 — `scripts/99-gpu-reset.sudoers`

**实际阅读：** 1–4；级别 `full_source`。1–4 全文。

**行为与依赖：** 特定主机的免密 root GPU 恢复入口和 PCI rescan 写入权限。 sudo、指定用户名及用户 home 内的 repo 路径。

**保留边界：** 机器管理策略独立归档；不自动作为推理库/C 安装内容。

实际缺陷或适用限制：

- 允许 root 执行用户仓库内可变脚本，不是固定 root-owned artifact 的窄接口。

### SCF005 — `scripts/bench-bonsai-6pens.py`

**实际阅读：** 1–155；级别 `full_source`。1–155 全文；10–39 HTTP/metrics；126 fallback；142 倍数。

**行为与依赖：** stdlib HTTP/API key 请求、客户端单调时钟完整响应计时、OFF/ON 同 prompt/seed/budget 与服务端 metrics。 HTTP server；私有 RERoT/P6 指标和请求参数。

**保留边界：** HTTP 计时与明确边界的指标采集可保留；现有比较值需修正并做真正配对验收。

实际缺陷或适用限制：

- 142 speedup_tps 分子为 ON 客户端 TPS、分母为 OFF 服务端 decode TPS，口径混合。
- metrics 失败返回空字典并默认 0，没有真实六路 active decode 或完成/语义门。
- 只有单次 OFF/ON，不构成配对统计或稳定性能收益证明。

### SCF006 — `scripts/bench-bonsai-raft-6pens.py`

**实际阅读：** 1–186；级别 `full_source`。1–186 全文；75 health；107 reuse；152/165 子工具；176–181 清理。

**行为与依赖：** 管理 owned server，等待 health，保存配置/日志，调用 benchmark 与 throughput gate。 stdlib HTTP/subprocess；旧 server 私有 RERoT 参数与 sibling tools。

**保留边界：** 进程所有权和回执可以独立 C 保留；旧配方修正后才能当验收。

实际缺陷或适用限制：

- --reuse-server 留 proc=None，但 wait_health 75 无条件 proc.poll，静态可见 AttributeError。
- 两个子工具无 timeout，KILL 后 wait 无界；模型只记路径/HEAD，缺内容身份。

### SCF007 — `scripts/check-tp5-mtp-evidence.sh`

**实际阅读：** 1–317；级别 `full_source`。1–317 全文；149–150 字段默认；208–224 CHAIN；306–309 明确未检项目。

**行为与依赖：** 检查 graph reuse、cycle/token 守恒、零接收、CHAIN、数值参考、错误日志，并明说未验证人工 phase release/heartbeat。 bash/awk、私有 TP5/MTP 日志 schema。

**保留边界：** 严格 counter ledger/证据 reader 可独立 C 保留，但要补字段必填和正向证据；不能把历史日志检查等同运行时实现。

实际缺陷或适用限制：

- 149–150 将缺失 draft_us/catchup_us/handoff_us/draft_tokens/accepted_tokens 默认为 0，可漏关键证据。
- AVOIDED_SYNC_COUNT 可为 0 仍打印 PASS；hidden 检查主要是没有 mismatch，缺正向完成证据。
- CHAIN 只要求至少一行且无 FAILED，没有验证文案所称 continuous hits。

### SCF008 — `scripts/eagle-gpu-watchdog.sh`

**实际阅读：** 63–113, 293–430；级别 `selected_functions`。完整 log/save_incident_evidence、recover_card、card_operational、request_early_reboot、clock_pin_is_correct/restore_clock_card；触发扫描、主循环和参数未读。

**行为与依赖：** 按卡记录 incident 后才启动 helper，保留失败证据，区分恢复与 DPM 复位，持久冷却标记限制自动 reboot。 bash/Linux sysfs/systemd/root、固定 BDF/helper；恢复动作是机器管理。

**保留边界：** 默认不迁移自动 reset/reboot；独立手动诊断 C 可以保留证据先行/目标显式/失败保留的习惯。

未读区间：1–62, 114–292, 431–694；未对这些区间作内容结论。

实际缺陷或适用限制：

- 323 timeout 没有 kill-after，忽略 TERM 的 helper 仍可无限运行；426 同样不能仅凭 timeout 字样声称有硬上限。
- save_incident_evidence 94–109 未 fsync，durable 是过强表述；reboot guard 才显式 sync。
- 365 冷却只在 now>=previous 时生效，系统时钟回退会绕过 24h 判定；多实例互斥未在已读函数中体现。

### SCF009 — `scripts/experiment-queue.py`

**实际阅读：** 1–60, 141–231, 352–390, 396–466, 506–545；级别 `selected_functions`。完整 validate_envelope、ExclusiveLock、load/save/enqueue_pack、drain/free-safe mock。未读 CLI/dry-run layout/main。

**行为与依赖：** 离线 pack QUEUED/NOT_RUN 登记、fcntl 独占锁、临时文件替换；结构化 envelope 与完成证据/释放约束模型。 stdlib+Linux fcntl；envelope schema、MockTp5Comm，不是真实 GPU scheduler。

**保留边界：** 独立 C 可保留证据队列/锁；mock 作为 B 规格，不能宣称执行器或 GPU free-safe 已迁移。

未读区间：61–140, 232–351, 391–395, 467–505, 546–763；未对这些区间作内容结论。

实际缺陷或适用限制：

- envelope 只验证若干键和布尔值，未解析 expires_at/身份匹配；不是实际授权真实性/到期验证。
- _save_packs 406–411 rename 原子替换但没有 fsync 文件/目录，不能宣称掉电持久队列。
- tp5_drain_submitted 511–517 只查长度和 membership，不验证注释所说的已提交前缀顺序；模型不证明实际 fence。

### SCF010 — `scripts/flashprefill-matrix.py`

**实际阅读：** 848–1020, 1084–1219, 1222–1326, 1347–1444, 2003–2073；级别 `selected_functions`。完整 metrics helper、read_sse_stream/assemble_stream/extract_nonstream_text、OwnedServer、run_cancel_stream；矩阵 schema/main/evaluator 未读。

**行为与依赖：** 带标签 metrics 选择/差分、SSE 文本/终止/trace 泄漏检查、owned server 有界停止、客户端取消。 stdlib HTTP/subprocess；FlashPrefill/RERoT 私有指标；通用 HTTP 部分可与 stock A 协作。

**保留边界：** 独立 C 保留生命周期/transport 证据和指定 parser；取消需另验真实服务端 abort/slot 释放，不将 close 当完整证明。

未读区间：1–847, 1021–1083, 1220–1221, 1327–1346, 1445–2002, 2074–3001；未对这些区间作内容结论。

实际缺陷或适用限制：

- 1189–1195 每个 data 行立即 flush，与“multi-line data blocks correct”注释不符；合法多行 JSON 事件拆坏。
- 2054–2063 HTTPError 仍 ok=True；2046–2052 连提前自然完成也记取消证据，不能证明取消中的工作。
- 2030 read(4096) 可阻塞超过 cancel_after_s，直到大 read/timeout 才关闭。
- parse_labels_block 912–914 跳过坏 matcher；unescape_label 的顺序替换、canon_labels 无重新转义可能改变反斜线/换行标签。
- OwnedServer start 打开日志后 spawn 失败，stop 在 proc=None 时直接返回，log_fp 未关闭。

### SCF011 — `scripts/flashprefill-quality.py`

**实际阅读：** 312–314, 399–437, 444–585, 951–1017；级别 `selected_functions`。完整 wilson、paired_diff_ci、extract_span、normalize_text、parse_last_number、三 scorer、score_external、assert_no_rerot_keys、check_pair_equality、config_fingerprint。

**行为与依赖：** Wilson/配对准确率区间、boxed/answer-tag 提取、exact/numeric/regex/external 评分、A/B 请求一致性及配置指纹。 纯数值/文本 stdlib B；外部 grader 与 HTTP 结果为 C 输入。

**保留边界：** 独立评分与可追踪的 grader C 有用户价值；先修一致性/finite/小样本区间边界，不能据旧报告宣称质量门已通过。

未读区间：1–311, 315–398, 438–443, 586–950, 1018–1924；未对这些区间作内容结论。

实际缺陷或适用限制：

- check_pair_equality 975–981 用 .get 比较：一侧缺 key、另一侧显式 null 时整个 payload 不同却可返回零 violations。
- score_numeric 494–524 允许 float 溢出为 infinity，未检 finite/tolerance；例如有限输出 vs 1e309 参考可能 inf<=inf 错判通过。
- paired_diff_ci 426–427 在所有配对一致时给零宽 CI；小样本无法据此证明总体差异严格为零。
- score_external 584–585 接受 grader 自报不同 version，未在本函数强制与指定版本一致；fingerprint 的 model/startup 是声明，不是自动读取运行进程。

本地反例：SC-Q01, SC-Q02。

### SCF012 — `scripts/gpu-hard-unlock.sh`

**实际阅读：** 255–577；级别 `selected_functions`。完整 get_pids_for_nodes、pid_has_node_handles、is_known_compute_process、preflight block、quiesce_target；PCI mutation/恢复本体与参数未读。

**行为与依赖：** fuser 错误区分、按真实 exe basename 检查持设备节点的进程、all-target preflight、TERM/再检查/KILL/排空后才允许恢复。 bash/root/Linux proc、fuser、PCI 拓扑；固定机器恢复程序。

**保留边界：** 人工管理工具归档；新 C 只回收自己启动的进程，不能沿用“所有同名计算进程”的所有权规则。

未读区间：1–254, 578–1036；未对这些区间作内容结论。

实际缺陷或适用限制：

- 283 输出逐行 PID，但 381/422/472/498/522/546 用 read -a <<< 多行字符串，只读取首行，不能声称已检查/处理全部 PID。
- basename whitelist 接受任意同名 llama-server/microbench/benchmark，不等于当前 C owned PID；exe 再检查也不是无竞态进程句柄。
- 所谓 5s/3s 等待由轮数×sleep 构成，内部 fuser/readlink 未限时，不是完整 wall-clock 上界。

本地反例：SC-P01。

### SCF013 — `scripts/gpu-idle-audit.sh`

**实际阅读：** 1–249；级别 `full_source`。1–249 全文；27–42 设备发现；183–197 增量日志与状态；217 总结。

**行为与依赖：** 只读检查 GPU busy/VRAM、计算进程、增量 dmesg，区分 unknown 状态并保存扫描游标。 bash/Linux sysfs/dmesg/pgrep；状态文件。

**保留边界：** 可独立保留为 C 的手动诊断工具，修复 missing-data 与游标语义。

实际缺陷或适用限制：

- 103 VRAM 读取失败替换 0，114 缺节点只 WARN，可能仍总 PASS。
- PREV_TIMESTAMP 在 179 读取却未使用；183–193 仅按行数增量，同大小环回或重启可漏新错误。
- 阈值无数值验证、状态写无锁/原子性、全局 pgrep 可能命中非本 runner 工作负载。

### SCF014 — `scripts/install-gpu-recovery.sh`

**实际阅读：** 1–51；级别 `full_source`。1–51 全文。

**行为与依赖：** 检查 shell 语法、备份、分阶段复制并比较两个恢复文件，重启并验证服务。 root、指定 systemd 单元与本机路径。

**保留边界：** 只保留显式人工管理配方；不作为运行时内存/调度功能。

实际缺陷或适用限制：

- 两次 rename 不是同一原子事务；第一文件替换或服务 restart 失败后无回滚。
- root wrapper 后续执行用户 repo 的 source，部署所有权边界需另行明确。

### SCF015 — `scripts/make-tp5-mtp-evidence-fixtures.sh`

**实际阅读：** 1–118；级别 `full_source`。1–118 全文。

**行为与依赖：** 明确标成 synthetic 的通过/七种失败 parser fixtures。 bash；check-tp5-mtp-evidence 日志格式。

**保留边界：** 保留为 parser 数据，增加真实返回码断言和缺字段案例；不算 GPU 证据。

实际缺陷或适用限制：

- 本文件只写日志，不执行/断言 checker；多个 failure fixture 同时违反多个条件，不能证明目标分支被测。
- mkdir -p 加重定向可覆盖旧同名证据文件，应保证单独 synthetic 输出目录。

### SCF016 — `scripts/rerot-audit-replay.py`

**实际阅读：** 1–96；级别 `full_source`。1–96 全文；21–26 snapshot shape；33–34 q/k；64–79 结果比较。

**行为与依赖：** 读取捕获 tensor 的字节 stride，NumPy float64 重放 RBB 线性求解、条件数和误差。 NumPy、旧 RERoT tensor dump producer；可无模型运行。

**保留边界：** 独立数学 oracle B 和格式 reader C 可保留；不等于 shared-KV 运行时已迁移。

实际缺陷或适用限制：

- 重复 TSV 名会覆盖；assert 维度检查在 python -O 下消失；结果只取前段且缺严格完整形状约束。
- 检查实际输出 finite，但未明确要求参考中间结果 finite；需要避免非有限 oracle/tolerance 导致错误判定。

### SCF017 — `scripts/rerot-capacity-matrix.py`

**实际阅读：** 1–290；级别 `full_source`。1–290 全文；136–147/174–185 serial loops；197 return；199–216 unreachable pool；263–281 verdict。

**行为与依赖：** 意图测试 1B+3P、两请求与三请求容量形状，并保存请求/指标/abort 门。 HTTP、旧 RERoT/P6 拓扑参数与 metrics。

**保留边界：** 保留容量场景规格，重写真实并发 C；旧 PASS 不能当统一 KV/并发证据。

实际缺陷或适用限制：

- 两/三“并发”分支实际串行逐请求发送，ThreadPool 位于 return 后不可达。
- 263/273 已写 status=PASS，281 才检查 abort；失败退出仍留下 PASS JSON。
- 未验证指标确实观察到声明 B/P 拓扑，缺 metrics 默认为 0；没有质量/finish 门。

### SCF018 — `scripts/rerot-continents-benchmark.py`

**实际阅读：** 243–398；级别 `selected_functions`。完整 check_heading_anomalies、verify_continents_quality；map、HTTP、root parser 和 main 未读。

**行为与依赖：** 检测嵌套/重复/失控标题，按区域别名及关键词覆盖验证指定八分区格式。 stdlib regex；特定 prompt 的 CONTINENT_MAP 与 root-ol parser。

**保留边界：** 可保留明确标成格式/关键词 heuristic 的 B grader；不能称通用答案事实质量或 DAG 并发验收。

未读区间：1–242, 399–651；未对这些区间作内容结论。

实际缺陷或适用限制：

- 316 将 reasoning+content 合并评分，最终可见答案为空也可能靠 reasoning 中关键词满足覆盖。
- 363–366 只按全文关键词数量，未验证地区归属或论述正确性。
- 263–271 先收全 HTML 标题再收 Markdown 标题，274 的“连续”不一定是真实文档顺序。

### SCF019 — `scripts/rerot-mindmap-bench.py`

**实际阅读：** 275–395；级别 `selected_functions`。完整 check_reader_order 及内部 emit/oracle/path helper；未审 wire/parser/server/prompt 主流程。

**行为与依赖：** 300 个确定性随机树上，比较层次 cyclic DFS 与路径 rank-sort，检查读者最后、兄弟子树连续。 纯 Python 树值；descendants 辅助函数未读，其他模型执行流程未读。

**保留边界：** 有用的独立 B 顺序规格/参考测试候选；后续需把真实 B 实现接入对照。

未读区间：1–274, 396–827；未对这些区间作内容结论。

实际缺陷或适用限制：

- 两个算法共用 path_of/path_step 和可变 reader_ref，oracle 独立性有限。
- 这里只调用 production-shaped Python 算法，不调用生产 C++ 读取器；300 棵树不是任意图/环/畸形输入的证明。

### SCF020 — `scripts/rerot-perf-gate.sh`

**实际阅读：** 1–256；级别 `full_source`。1–256 全文；114 二次 preidle；148–150 PIPESTATUS；186 dmesg；245 post gate。

**行为与依赖：** GPU 前后 idle 门、元信息/进程/dmesg 证据、保存真实命令 rc，工具自身不 reset 设备。 bash/Linux/gpu-idle-audit.sh；命令作为 argv。

**保留边界：** 可独立保留 C 的诊断 gate，修复日志/锁/owned cleanup；区分设备只读与输出文件写入。

实际缺陷或适用限制：

- 第二次即时 preidle 返回码未使用；无独占锁，空闲检查不保留设备。
- 执行本身无 owned timeout/trap；只看 PIPESTATUS[0] 忽略 tee 写失败，可能丢证据仍成功。
- 秒级目录 mkdir -p 可冲突覆盖；按行数做 dmesg 增量可漏同长度环回。

### SCF021 — `scripts/rerot-phase8-quality.py`

**实际阅读：** 87–168, 517–593；级别 `selected_functions`。完整 start_server/stop_server 与三 verify_*；query/数据汇总/main 未读。

**行为与依赖：** owned 进程组启动/health/清理；执行生成的 palindrome、coin-change 和 spiral 程序的有限用例。 Python stdlib、prlimit/g++、旧 RERoT model flags；生成代码由本机子进程执行。

**保留边界：** 保留有限题目的可执行评分规格和 C 进程所有权；要独立隔离 grader，不能把这一小套例子当全模型质量验收。

未读区间：1–86, 169–516；未对这些区间作内容结论。

实际缺陷或适用限制：

- 165 KILL 后无界 wait；583 C++ 编译无 timeout；spawn 失败/health 异常清理依赖未读调用者。
- 生成代码和断言在同一程序，返回码0+stdout含OK不保证断言真正执行（被测程序可先退出）；并非受控独立测试协议。
- Python 每题4例、C++2例，不能外推一般代码正确率；本轮没有运行生成代码。

### SCF022 — `scripts/rerot-semantic-smoke.py`

**实际阅读：** 1–255；级别 `full_source`。1–255 全文；23–40 capacity；79–84 auto 参数；198–209 maps；230–231 并发。

**行为与依赖：** 不偷偷扩大手动容量，校验 full-auto 参数所有权；真实多线程请求与实际加载 DSO /proc/maps 身份。 stdlib HTTP/ThreadPool/Linux proc；旧 RERoT server。

**保留边界：** 保留 C 的手动容量拒绝/身份/显式诊断模式/进程生命周期；RERoT 私有运行时另行暂存。

实际缺陷或适用限制：

- 固定 DSO 清单在可选库未构建时即失败；146 capture 无 timeout，242 KILL 后 wait 无界。
- 最终通过只按 response，forced shutdown 没有设失败；metrics HTTP 状态未验收。

### SCF023 — `scripts/rerot-simple-continuation-smoke.py`

**实际阅读：** 1–223；级别 `full_source`。1–223 全文；35–83 simple/stream；86–137 multiple；140–177 普通 input groups；201–215 grammar。

**行为与依赖：** 普通/probe 单输出和流式等价、n=2 输出索引/usage/DONE、普通两输入×两输出计数、JSON grammar。 stdlib HTTP/SSE；部分测试是普通公开 server 行为，部分依赖 RERoT probe 参数。

**保留边界：** 立即可候选保留普通 multi-input/SSE/usage/grammar 的 C 测试；probe 部分等待真实 RERoT seam。

实际缺陷或适用限制：

- 多 choice streaming 没像单输出分支检查 tool_calls 泄漏。
- 只认 data: 后带一个空格、无 multiline SSE；这是特定 producer parser，不是通用 SSE；无 auth 选项。

### SCF024 — `scripts/rerot-target-ornith-multi-lane.py`

**实际阅读：** 145–259；级别 `selected_functions`。完整 verify_dag_topology_and_multilane；HTTP/main/答案 grader 未读。

**行为与依赖：** 检查 RERoT usage 正数、文本节点标记与前后指标。 HTTP response/私有 RERoT counters，文本 reasoning trace。

**保留边界：** 保留 token-accounting reader 作为 C 候选；重写并行和因果验收，不沿用旧 success 文案。

未读区间：1–144, 260–527；未对这些区间作内容结论。

实际缺陷或适用限制：

- 237 metrics 可完全缺席；254/256 仅在正数时打印，不要求 completed/parallel>0，259 却总称 multi-lane verified。
- 229 只有 pred/succ 文本位置均找到才检查；plan: 名称可以满足节点存在而依赖顺序检查被跳过。
- 文本出现顺序不是任务完成→后继开始的真实因果事件，更不证明同时 decode。

### SCF025 — `scripts/rerot-throughput-gate.py`

**实际阅读：** 73–154, 479–642；级别 `selected_functions`。完整 validate_dag_payload、extract_rerot_token_breakdown、evaluate_round_checks；HTTP/main/最终配对统计未读。

**行为与依赖：** 每轮 episode/abort/final-fence/token 守恒，缺指标分 evidence-incomplete，六路要求真实 six-row counter；明确非流请求不能测 TTFT。 stdlib、私有 RERoT counters；数据验证本身独立。

**保留边界：** 保留证据状态与 counter/gauge 区分的 C reader；私有 producer 尚未迁移，普通 A 没有这些 counter 时必须不通过。

未读区间：1–72, 155–478, 643–978；未对这些区间作内容结论。

实际缺陷或适用限制：

- validate_dag_payload 135–149 只是关键词/条目数启发式，缺 opt-in 也可能通过；显式 DAG 分支不验唯一 ID/依赖/环，不是图语义 validator。
- evaluate_round_checks 对 supplied counter 未统一检 finite/nonnegative；NaN 的 <=0 判定可绕过 positive parallel-work 检查。
- whole-server counter 差分要有请求隔离/唯一完成关联；本函数本身不证明来自该请求。

### SCF026 — `scripts/reset-gpu-pci.sh`

**实际阅读：** 1–71；级别 `full_source`。1–71 全文；40–41 DPM 写；67 installed/source 字节比较。

**行为与依赖：** 显式目标白名单、时钟恢复或调用匹配版本的 hard-unlock helper。 root、固定 BDF/card、已安装 helper 与源仓库。

**保留边界：** 机器恢复工具单独归档；不放进默认推理路径。

实际缺陷或适用限制：

- check-clocks-only 仍写 manual DPM 和最高 index，readback 只检查模式而非实际时钟。
- 错误文案称 root-owned helper，但只 cmp 内容，没有验证 owner/mode。

### SCF027 — `scripts/reset-gpu.sh`

**实际阅读：** 1–19；级别 `full_source`。1–19 全文。

**行为与依赖：** 默认只读 idle audit；只有显式目标才 sudo -n 调用恢复 helper。 bash、同目录脚本、本机 sudo 策略/设备映射。

**保留边界：** 保留默认诊断与显式人工动作分开的入口习惯；设备恢复本体归档。

实际缺陷或适用限制：

- 目标/权限依赖固定机器；不能直接当通用 C 自动故障恢复。

### SCF028 — `scripts/run-qwen38-flash-tp5-server.sh`

**实际阅读：** 1–282；级别 `full_source`。1–282 全文；35–44 模式；170–172 print-config；212–225 清理；240–254 argv。

**行为与依赖：** golden/control/standard 运行配方、无 GPU print-config、源/二进制/DSO 身份、限时 idle/health 和 owned PID 清理。 bash、五卡 Linux 环境、私有 TP5 flags/模型。

**保留边界：** 保留 C 的显式参数/身份/子进程所有权；旧配方不冒充独立运行时特性。

实际缺陷或适用限制：

- golden 与 control 可同时被选择，未按文档互斥拒绝。
- MTP_MODEL 60 / KV_SIZE 61 未消费；TP5_ARGS/EXTRA_ARGS 以 shell word splitting 展开。
- TERM 超时后 KILL 的最终 wait 224 仍无上界。

### SCF029 — `scripts/run-ternary-bonsai-2-27b-server.sh`

**实际阅读：** 1–176；级别 `full_source`。1–176 全文；65 设备 glob；110–114 清理；138–145 argv。

**行为与依赖：** 模型/二进制存在检查、配置展示、health 轮询、owned PID 回收。 bash/curl/sysfs；特定模型和旧运行参数。

**保留边界：** launcher 行为可作为小 C 配方；模型专属采样默认不算已证明调优。

实际缺陷或适用限制：

- card[0-9] 漏 card10+，且 idle 检查全部卡而非请求设备。
- parallel/-np 重复；EXTRA_ARGS 字符串拆词；KILL 后 wait 无界。
- provenance 只有 HEAD，缺 binary 身份；health 日志秒数与实际 sleep 不符。

### SCF030 — `scripts/run-ternary-bonsai-2-27b.sh`

**实际阅读：** 1–68；级别 `full_source`。1–68 全文。

**行为与依赖：** 薄启动器，检查输入、引用实际参数，用 exec 保留退出码/信号并透传用户 argv。 bash、模型路径、旧 CLI 选项。

**保留边界：** 可保留简洁 C launcher 作为已实现功能入口，不能把 launcher 本身称为新调度算法。

实际缺陷或适用限制：

- 模型固定采样值与 -kvauto 等旧参数需逐项对新 A 的公开 CLI 核对。

### SCF031 — `scripts/test-experiment-queue-safe-teardown-mock.py`

**实际阅读：** 1–46；级别 `full_source`。1–46 全文；18/33 CLI。

**行为与依赖：** 通过真实 CLI 子进程调用 mock teardown/envelope validator，确认保留未完成资源且不伪造完成。 Python stdlib、experiment-queue.py 的模型。

**保留边界：** 保留明确标为 contract model 的测试；不能作真实 GPU teardown 证据。

实际缺陷或适用限制：

- 两次 subprocess 没有 timeout；被测对象是 Python mock，非 Vulkan/queue 生产实现。

### SCF032 — `scripts/test-safe-teardown-mock.py`

**实际阅读：** 1–142；级别 `full_source`。1–142 全文；117 CLI。

**行为与依赖：** MockTp5Comm 的 drain 失败、不完整、伪造序号、成功、null 与 CLI roundtrip 测试。 Python unittest/importlib；experiment-queue 模型。

**保留边界：** 保留完成证据与释放约束的规格测试，并配真实公共接口测试防止模型漂移。

实际缺陷或适用限制：

- 117 subprocess 没 timeout；不证明实际 GPU fence、payload 或设备恢复。

### SCF033 — `scripts/test-t1c-publish-order-mock.py`

**实际阅读：** 1–202；级别 `full_source`。1–202 全文；101–102 cpu_reduce；123–138 same arithmetic 测试。

**行为与依赖：** generation/payload hash/event tick 的 Python ready 发布顺序模型与 ABA 场景。 纯 Python mock。

**保留边界：** 顺序契约可保留为 B 规格；固定输出的 arithmetic mock 不作为 Q8/传输验证。

实际缺陷或适用限制：

- cpu_reduce 直接返回 target_generation*10，不消费 payload 或实际传输，等值测试不能证明算术正确。
- publisher 不要求 generation 单调，所测 ABA 仅 old/new 两值案例。

### SCF034 — `scripts/test-throughput-gate-mock.py`

**实际阅读：** 1–502；级别 `full_source`。1–502 全文；191–205 调用实际 main；207–498 九种场景。

**行为与依赖：** 真实调用 gate.main 的 mock HTTP 回归：坏首轮不被干净后轮盖过、缺 counter、gauge 不顶替并行、route 未落地、forced frame 不冒充有效吞吐、六路必须同批。 Python unittest/mock 与实际 rerot-throughput-gate 模块；没有模型或 GPU。

**保留边界：** 这些是有效 reader/判定逻辑回归候选，可独立保留；明确 proof scope=mock HTTP，不是六路运行时性能证据。

实际缺陷或适用限制：

- metrics mock 用 min(call_count,last) 重复末次值，意外多 scrape 不必失败；没有真实 HTTP framing/concurrency/设备执行。

### SCF035 — `scripts/tp5-domain-check.py`

**实际阅读：** 1–49；级别 `full_source`。12–49全main；18检查设备唯一与warmup界；31–38有界idle门。

**行为与依赖：** 按真实 mailbox domain、device、逐dispatch/批时间戳组合运行独立shader，并记录argv/env/hash/空闲状态。 Python stdlib+Linux fcntl/sysfs；导入 tp5-gap-bench.py；五张卡和私有micro/shader ABI。

**保留边界：** 提取参数化C实验runner的锁/证据编排；历史五卡job表归档。

实际缺陷或适用限制：

- 41 subprocess.run没有timeout；有界idle循环不使GPU子进程有界。
- 43–46允许零条JSON且rc=0时继续，未证明输出已被校验。

### SCF036 — `scripts/tp5-f32-protocol-check.py`

**实际阅读：** 1–38；级别 `full_source`。6–38全main；14–19列具体relay/consumer消费。

**行为与依赖：** F32 Y交接与ready-consumer单体测量的配对job和两种时间戳scheme。 tp5-gap-bench.py、Linux五卡、私有relay/micro。

**保留边界：** 保留编排规则候选；原协议实验依赖未迁移runtime。

实际缺陷或适用限制：

- 31子进程没有timeout；33允许result=None而rc=0。

### SCF037 — `scripts/tp5-gap-bench.py`

**实际阅读：** 83–117, 120–133, 395–470, 473–697；级别 `selected_functions`。完整 api/library_identity/request_payload/evaluate_length_gate/audit_minimum_fields/main；cycle/rank-stage reader 未读。

**行为与依赖：** 冻结非空请求/预算、真实 loaded DSO hash、独占执行、HTTP错误原文、有界 owned teardown、length gate 与 NOT_COLLECTED 最低字段清单。 stdlib HTTP/fcntl/proc/sysfs；私有 TP5/LateBind/MMVQ/MTP 参数和日志 reader。

**保留边界：** 真实独立 C runner 的请求/证据/owned teardown 候选；保留 MEASURED_NOT_QUALIFIED 与 DIAGNOSTIC_NOT_RANKED 分级，私有配方暂存。

未读区间：1–82, 118–119, 134–394, 471–472；未对这些区间作内容结论。

实际缺陷或适用限制：

- 672 第二次 wait(timeout=10) 若仍失败，finally 在 691 写最终 result 前抛异常，会丢最终失败回执；应 best-effort 保存。
- 模型/draft 只有 stat identity，文件 SHA256 明确 NOT_COLLECTED，须附外部回执才能完整冻结。
- health 仅看端口响应且无 port owner handshake；新进程尚未退出时可能短暂误认已有 listener。
- length_gate false 仍 rc0 是明确设计：只是测量成功，不是质量/性能 promotion。

### SCF038 — `scripts/tp5-gap-report.py`

**实际阅读：** 1–273；级别 `full_source`。16–56 inventory；59–82 effective_profile/summarize；85–132 profile；135–197 paired_report；200–219 unique_texts；222–273 CLI。

**行为与依赖：** 未插桩资格分类、raw profile按rank/rows汇总、固定四块配对log-ratio CI、输出内容去重。 stdlib；旧result/manifest/mapped-libraries/TP5 log格式；inventory绑定10专家及模型布局。

**保留边界：** 纯统计、证据归并与去重可独立B/C；源模型成本表单独归档。

实际缺陷或适用限制：

- 147的配对identity只包含bin和DSO，没有model/request/config；157收集输出hash但从不比较A/B，不能单独证明相同工作量。
- 151–170不先验证token/time有限正，直接除法/log。
- 128–129对缺失profile字段补0，仅适合已确认producer schema，不宜作为完整计数证明。

### SCF039 — `scripts/tp5-hc-coordinate-prototype.py`

**实际阅读：** 1–148；级别 `full_source`。32–67 stable sigmoid与shape/partition前置；70–88 reference；91–140 coordinate_owned；143–148 combine_next。

**行为与依赖：** 完整与坐标分区HC代数reference：先合并Y再平方，ragged partitions、norm/down/inject充分统计、后续residual combine。 Python stdlib math/NamedTuple；无A/B私有接口。

**保留边界：** 值得保留为独立数学参考B和真实kernel对照的oracle；不称GPU/时序证明。

实际缺陷或适用限制：

- 39–47允许bool充当数值；仅检查输入finite，极端有限输入的中间乘方/乘法仍可能溢出。
- 143–148信任传入Result内部shape，仅验证next_block；单独公共化需明确Result来源契约。

### SCF040 — `scripts/tp5-hc-envelope.py`

**实际阅读：** 1–93；级别 `full_source`。15–26 last_json/hc_ms；28–93 fullmain。

**行为与依赖：** 带显式串并行假设的HC临界路径预算及剩余权重敏感性；明确非硬件上界。 stdlib；旧模型94sites/23hidden和若干验证log/metadata/roofline文件。

**保留边界：** hc_ms纯算术和带前提报告可独立保留；固定模型配方归档。

实际缺陷或适用限制：

- 65–68读取的Q64/norm512额外输入没有进入89的sources哈希清单，报告provenance不完整。
- 35 exists检查和90写入分离，不是排他/原子输出。

### SCF041 — `scripts/tp5-hc-q8-variant.py`

**实际阅读：** 1–162；级别 `full_source`。27–63 copy/hash/quantize；74–89 shard/native ABI；101–151 rewrite和逐未变tensor验证；157明确未质量验收。

**行为与依赖：** GGUF分片HC浮点矩阵转Q8，重写type/offset，对每个未变tensor逐字节SHA验证。 ctypes公开ggml row-size和私有tp5_hc_quantize helper；tp5-gguf-metadata目录读取；旧raw-type映射。

**保留边界：** 模型派生转换器C有独立价值，先修输出保护/完整shard/版本类型契约；质量与速度另验收。

实际缺陷或适用限制：

- 95–97对任意--report直接write_text，未拒绝输入GGUF或库同路径；可违背Never edits source承诺。
- 78–80只检查glob数量而不核验分片序号全集；继承目录reader的字段校验缺口。
- 111将raw type42硬映射47用于row_size，依赖旧fork数值枚举，不能作为跨A版本通用转换。

### SCF042 — `scripts/tp5-host-sidecar-run.py`

**实际阅读：** 1–26；级别 `full_source`。1–26 全文；14–24 case 编排与返回码传播。

**行为与依赖：** 按外部 preregistration 运行 host/thread/sidecar 配置，记录精确 argv，失败即停。 stdlib；真实执行委托 tp5-gap-bench.py；私有 TP5 参数、固定 GPU 环境。

**保留边界：** 保留 C 的显式登记/无隐式重试模式；历史 case 表归档。

实际缺陷或适用限制：

- 21 subprocess.run 无 timeout；case 名称/extra 参数没有严格 schema 或输出路径 containment。

### SCF043 — `scripts/tp5-independent-envelope.py`

**实际阅读：** 1–135；级别 `full_source`。17–27算术；29–124 build；126–135 CLI。

**行为与依赖：** HC+fusedMoE service替换原weight预算，保留剩余native proxy、明确缺项及非硬件界限。 stdlib+tp5-roofline.py、固定五卡micro/模型94sites/48层目录。

**保留边界：** 纯hc_budget/replace_bytes与显式前提预算候选；模型实例参考归档。

实际缺陷或适用限制：

- 49–51、55–57、92–94只数5行而不验证5个唯一物理device/identity，重复记录可满足数量。
- HC/MoE来源execution.json未逐一纳入最终digest；110仅metadata/layout，native日志81另有hash。

### SCF044 — `scripts/tp5-independent-service.py`

**实际阅读：** 1–75；级别 `full_source`。1–75 全文；49–64 三次 idle 稳定检查；65–73 执行与日志。

**行为与依赖：** resident/stream 服务微测量矩阵，白名单环境、独占锁、有界 idle 检查、独占创建日志。 stdlib+fcntl/sysfs；TP5 私有 micro/relay 程序，固定类型/形状/五卡。

**保留边界：** 锁、参数校验、原始日志所有权可提取为小 C runner；私有 kernel job 配方暂存。

实际缺陷或适用限制：

- 67 实际子进程无 timeout；只看返回码，没有结果正确性字段验收；relay elements 未做正数/上界校验。

### SCF045 — `scripts/tp5-late-service-run.py`

**实际阅读：** 1–43；级别 `full_source`。1–43 全文；36–40 执行后写身份/结果。

**行为与依赖：** 逐项测量 inject/act/q/up 的 local/system 服务，记录二进制和 shader hash。 stdlib、gap-bench helper、固定五卡与私有 late-bind 程序。

**保留边界：** 保留参数化执行/产物身份 C 模式；旧模型专用 job 作为实验配方。

实际缺陷或适用限制：

- 36 子进程无 timeout；hash 在运行后采集，未锁定运行时文件；没有解析正确性结果。

### SCF046 — `scripts/tp5-match-audit.py`

**实际阅读：** 1–73；级别 `full_source`。12–63 fullaudit；23–36运行/作用域/eligibility；38–60原始响应与A/B比较；65–73 CLI。

**行为与依赖：** 逐原始响应核对payload/输出SHA/token计数/时长/吞吐、相同模型stat及DSO身份、只改变greedy-host策略。 stdlib；固定7对历史目录和eligibility日志字段。

**保留边界：** 抽取可参数化的离线raw A/B一致性门；对当前sampler改动仍有用户价值。

实际缺陷或适用限制：

- 38–59未要求requests非空或有measured样本；构造全空requests且相应日志0次可返回RAW_AUDIT_PASS。
- 27–28 model_stat_identity不是内容hash，所证明身份范围有限。

### SCF047 — `scripts/tp5-match-interactions.py`

**实际阅读：** 1–35；级别 `full_source`。9–35全main。

**行为与依赖：** 固定八组latebind/MMVQ/MTP交互screen，首失败即停止并保存argv/返回码。 tp5-gap-bench.py和私有TP5 env/CLI。

**保留边界：** 留可复用实验矩阵/stop-on-failure编排候选，原八组配方归档。

实际缺陷或适用限制：

- 27外层run无timeout且29写execution前未mkdir root；依赖子runner创建root和有界性。

### SCF048 — `scripts/tp5-match-model.py`

**实际阅读：** 1–208；级别 `full_source`。27–33 identity；35–70 request_cost；72–102 calibrate；104–138 predict/counterfactual；140–184 validate；186–208 CLI。

**行为与依赖：** 独立计时区间建模、冻结基线、CPU微测counterfactual、held-out预测误差检验；明确不证明最优。 stdlib；decode-cost/sample-cost producer、旧manifest/result。

**保留边界：** 纯解析/预测/冻结与验证C候选；A内计时producer另列耦合。

实际缺陷或适用限制：

- 147按字符串比较UTC时间，通用化需解析同一时区/精度格式。
- 145只排除status==FAILED，未要求唯一完成状态；应采用明确状态集合。
- request_cost按出现顺序zip ubatch/decode/sample并信任scope格式；不能独立验证跨线程trace关联。

### SCF049 — `scripts/tp5-match-sweep.py`

**实际阅读：** 1–47；级别 `full_source`。1–47 全文；38–45 运行、回执和错误退出。

**行为与依赖：** 十种固定配置/线程/缓存组合，预先记录 argv，失败即停。 stdlib、gap-bench 与私有 TP5 选项。

**保留边界：** 通用 C 的少量配置列表即可表达，避免迁移多个重复框架；旧 recipe 归档。

实际缺陷或适用限制：

- 38 无 timeout；每文件质量、模型身份、配对统计未在本层验收。

### SCF050 — `scripts/tp5-moe-service-run.py`

**实际阅读：** 1–51；级别 `full_source`。1–51 全文；18 唯一设备；45–48 hash 和非空 JSON 门。

**行为与依赖：** 验证唯一物理设备、warmup 范围，测 moe2/moe3 的 resident/stream 与时间戳组合。 stdlib、gap-bench helper、五卡私有 fused-MoE benchmark。

**保留边界：** C 可保留设备/参数检查、缺结果即失败及证据写入；内核配方暂存。

实际缺陷或适用限制：

- 41 subprocess.run 无 timeout；只取最后一条结果 JSON，具体 all_outputs_checked 语义委托被测程序。

### SCF051 — `scripts/tp5-necessity-audit.py`

**实际阅读：** 1–315；级别 `full_source`。26–77 primitives；80–119 counterexamples；132–207 inventory；210–239 theoretical report；242–291 observations；294–315 CLI。

**行为与依赖：** 把条件必要工作/容量上界与实测分离；max合并资源下界、cache减法、反例、精确payload目录与旧原始样本核验。 纯算术stdlib；inventory C导入A gguf constants和旧tp5-roofline；历史artifact schema。

**保留边界：** NecessaryResource/joint_lower/mandatory_io及证据分类可独立B；模型目录/历史观测C分别保留候选。

实际缺陷或适用限制：

- work_basis/capacity_basis/premise是调用者提供标签，函数不证明前提；源码已明确此边界。
- 195/200在全被排除或空有效weight时可能除零；正式工具需要非空工作量校验。
- 262–283原始观测比较输出和计数但不比较raw payload/config，不是完整受控A/B身份。

### SCF052 — `scripts/tp5-necessity-certificate.py`

**实际阅读：** 269–275, 278–304, 312–350, 363–523；级别 `selected_functions`。完整 verify_lp_dual、_verify_claim/_claim_summary、evaluate_branch、merge_lower_bounds、gap_assessment。

**行为与依赖：** 验证 LP 对偶候选，按同一前提 max 合并资源下界，跨全分支 min，未解决分支不制造全局界或 U/L。 stdlib 数值/数据类+tp5-necessity-audit；具体硬件/模型 derive 和主程序未读。

**保留边界：** 保留数学条件与证据状态 B/C；非空论证文字是用户前提，不能自动证明硬件必要性。

未读区间：1–268, 276–277, 305–311, 351–362, 524–767；未对这些区间作内容结论。

实际缺陷或适用限制：

- verify_lp_dual 302 容许 tol 违反对偶约束，却直接返回未经修正的目标值；严格 lower-bound 需要误差界/可验证舍入，不能仅称 PROVEN。
- 有限输入相乘相加仍可 overflow/NaN，col_sum 与最终 objective 未检 finite。
- merge_lower_bounds 438–480 只看传入结果，不核对预期分支全集/重复 ID；全集覆盖是调用者责任。
- PROVEN 标签只验证数值结构和非空论证，未验证 counting argument 或 relaxation mapping 的真实性。

本地反例：SC-N01。

### SCF053 — `scripts/tp5-p1-fold-pilot.py`

**实际阅读：** 1–25；级别 `full_source`。1–25 全文；20–23 启动和回执。

**行为与依赖：** 固定 A/B 对比 relay-vector-copy-fold，并保存运行回执。 stdlib、私有 TP5 flag 与 gap-bench。

**保留边界：** 登记式小 runner 可保留；配方不等于已通过性能门。

实际缺陷或适用限制：

- 20 无 timeout；本文件不比较输出、模型身份或配对统计。

### SCF054 — `scripts/tp5-pack-confirm.py`

**实际阅读：** 1–38；级别 `full_source`。1–38 全文；27 teardown；29–35 配对比较。

**行为与依赖：** 四对 AB/BA 配对，正常 teardown 与 fallback 检查，逐请求 phase/count/body 完全一致。 stdlib、gap-bench schema、私有 pack 开关。

**保留边界：** 保留 C 中的平衡顺序与原始输出等价门；针对旧 pack 的配置归档。

实际缺陷或适用限制：

- 23 无 timeout；同为空的 requests 可通过本层 equality，非空/身份必须由上层保证。

### SCF055 — `scripts/tp5-pack-single-check.py`

**实际阅读：** 1–36；级别 `full_source`。1–36 全文；28–33 执行、解析、hash 与返回码。

**行为与依赖：** 对 1/255/256/257/102400/102403 字节边界、五设备、guard on/off 做矩阵。 stdlib、私有 pack micro/shader。

**保留边界：** 保留边界尺寸/guard 组合的验证规格；独立真实内核消费者具备后再执行。

实际缺陷或适用限制：

- 28 无 timeout；29 允许空结果列表，33 仅要求 rc=0，不证明实际边界比较发生。

### SCF056 — `scripts/tp5-prefetch-check.py`

**实际阅读：** 1–62；级别 `full_source`。1–62 全文；20 有界 idle；50–58 实际执行与记录。

**行为与依赖：** norm local/system、rows、prefetch、withhold generation 与 relay 并行组合测试。 stdlib+fcntl/sysfs、私有 norm/relay 程序与 shader。

**保留边界：** 保留 withheld-ready 的显式参数矩阵和原始证据 C；私有协议测试不能当通用 A 测试。

实际缺陷或适用限制：

- 50 无 timeout；仅 rc 判断，没有检查预期 withheld 结果；空闲门不是执行期限。

### SCF057 — `scripts/tp5-queue-confirm.py`

**实际阅读：** 1–47；级别 `full_source`。1–47 全文；24–27 native probe；36–43 配对。

**行为与依赖：** 先执行 full/partial native queue probe，再执行四对 AB/BA 全模型并行提交对照。 stdlib、私有 TP5 queue probe 与 gap-bench。

**保留边界：** 保留部分提交/排空这一验证规格；真实 probe 与模型证据仍须分开。

实际缺陷或适用限制：

- 26 只看 rc 后 27 打印 NATIVE_PROOF_PASS，没有在本层解析 fence/payload 证据。
- root 目录没有在写 23 的日志前创建；24/40 subprocess 无 timeout。

### SCF058 — `scripts/tp5-resolved-config.sh`

**实际阅读：** 1–25；级别 `full_source`。1–25 全文；11–22 shell heredoc 构造 JSON。

**行为与依赖：** 无需 GPU，输出源提交、二进制路径和一组环境默认配置。 bash/git；选定 TP5 环境变量名单。

**保留边界：** 保留 JSON-safe 的请求配置快照 C；应明确不是执行器解析后的有效配置。

实际缺陷或适用限制：

- 路径/环境值未经 JSON 转义，含引号、反斜线或换行会产生错误 JSON。
- 没有向 binary 查询有效配置或记录 binary hash；resolved 名称超出实际证据。

### SCF059 — `scripts/tp5-resource-inventory.py`

**实际阅读：** 1–24；级别 `full_source`。9–24整个CLI。

**行为与依赖：** 按去layer前缀的name/shape/type/extent统计张量家族并列最后独立micro JSON。 stdlib；metadata目录和固定log glob。

**保留边界：** 只读模型家族目录汇总可独立保留。

实际缺陷或适用限制：

- 15只读前6个records；24展示最后JSON不验输出正确性/退出码，不能作为验收。

### SCF060 — `scripts/tp5-roofline.py`

**实际阅读：** 1–118；级别 `full_source`。17–51 selected/rank_fraction；54–98 estimate；101–118 CLI。

**行为与依赖：** 选中专家比例/复制与分片布局的逻辑权重预算、分阶段max及MTP专家重用敏感性。 stdlib；GGUF目录元数据与固定五rank Qwen4/TP5布局。

**保留边界：** 条件成本算术可保留为离线B/C；不是通用实际流量或硬件上界。

实际缺陷或适用限制：

- 硬编码10专家、5rank、head分配，模型/布局变化必须重审。
- 除bandwidth/committed外未校验shape/extent和总字节正有限；117输出允许覆盖输入。

### SCF061 — `scripts/tp5-sparse-summary.py`

**实际阅读：** 1–60；级别 `full_source`。10–51 summarize；32–39完整window和stage/rank人数；53–60 CLI。

**行为与依赖：** 完整400..415观测window及97stage×5rank×15steady样本检查，首样本分列，不把跨rank时间相加。 stdlib；TP5 sparse log与固定诊断目录命名。

**保留边界：** 离线计数完整性/窗口聚合规则可独立保留；具体producer保持归档。

实际缺陷或适用限制：

- 28读取latest[rank]要求span先出现，乱序日志报KeyError，格式前置要明示。
- 29对缺少p2/compute/p1/late_pre字段补0；36只验证stage/rank键，不能证明每项时间都存在。
- 21只解析非负普通小数，科学记数/负数不属支持输入。

### SCF062 — `scripts/tp5-stage-y-check.py`

**实际阅读：** 1–51；级别 `full_source`。1–51 全文；42–46 所选 shader hash；48 非空结果门。

**行为与依赖：** norm variant/domain/rows/withheld-generation 组合与五卡 warmed 服务测量。 stdlib、私有 norm shader/benchmark。

**保留边界：** 保留缺 JSON 即失败、精确 shader 选择 hash 的 C 验证模式；旧矩阵暂存。

实际缺陷或适用限制：

- 40 无 timeout；JSON 中具体语义正确性字段仍委托被测程序。

### SCF063 — `scripts/tp5-stage-y-run.py`

**实际阅读：** 1–42；级别 `full_source`。1–42 全文；31 teardown；33–39 请求对应比较。

**行为与依赖：** pilot 或四对 AB/BA，检查 teardown/fallback 和逐请求原始输出一致。 stdlib、gap-bench schema、私有 stage-Y 开关。

**保留边界：** 保留 C 配对输出门；不凭此封装推定 stage-Y 有净收益。

实际缺陷或适用限制：

- 26 无 timeout；输入非空、配置相同和模型身份未在本文件独立验证。

### SCF064 — `scripts/tp5-steady-stage-run.py`

**实际阅读：** 1–36；级别 `full_source`。1–36 全文；30–34 执行与 rc 传播。

**行为与依赖：** 固定 400/16 稀疏窗口及线程/缓存/direct-producer 组合，写回执且无隐式重试。 stdlib、私有 TP5 stage trace 参数与 gap-bench。

**保留边界：** 保留诊断窗口明确标记和失败传播；此类 instrumentation 不作为无扰动排名。

实际缺陷或适用限制：

- 30 无 timeout；本 runner 不证明 trace population/输出质量，需配套 reader。

### SCF065 — `scripts/tp5-tighten-run.py`

**实际阅读：** 1–36；级别 `full_source`。1–36 全文；23 拒绝既有目标；28–32 执行。

**行为与依赖：** 按外部登记的 Q/norm workgroup/parallel case 执行，拒绝既有 case 目录。 stdlib、私有 TP5 配置与 gap-bench。

**保留边界：** 保留登记 case 和拒绝复用结果的 C 模式；硬件调优值归档。

实际缺陷或适用限制：

- case schema、名称路径与数值范围未校验；28 无 timeout。

### SCF066 — `scripts/verify-vbios-flash.sh`

**实际阅读：** 1–38；级别 `full_source`。1–38 全文；22–24 asort 中位数；末尾提示。

**行为与依赖：** 展示 BIOS/BDF、限时 ALU 测试的中位数与相关 journal 日志。 本机固定卡/历史 BIOS、GPU 测试程序、GNU awk asort。

**保留边界：** 可保留明确标成诊断展示的配方；不当验收门。

实际缺陷或适用限制：

- 不强制 BIOS 相等或 0.85 阈值，失败工具/缺数据也可在最后 echo 后返回 0；无 pipefail。

### SCF067 — `scripts/xkv/collect-runtime-evidence.py`

**实际阅读：** 154–310；级别 `selected_functions`。完整 parse_prometheus_text/load_snapshot/delta_metrics；runtime build_evidence/main/self-tests 未读。

**行为与依赖：** Prometheus/JSON snapshot 解析、namespace 统一、nonfinite 标记、info 身份字段、双端都有有限值才生成 delta。 Python stdlib；私有 XKV/Tri info metrics，JSON snapshots。

**保留边界：** 独立 C snapshot reader 候选；先修格式/身份有效性，不能仅 reader 完整就宣称 XKV 已运行。

未读区间：1–153, 311–878；未对这些区间作内容结论。

实际缺陷或适用限制：

- 191 简单按逗号拆标签，不支持字符串内逗号/转义；175 rsplit 不支持可选样本 timestamp。
- 200–224 在 sample 数值验证前写 info，NaN/非法值样本的 fingerprint 仍可保留，非严格 fail-closed 身份证据。
- 235 对所有同名标签 series 直接求和，丢标签维度且求和可溢出；counter/gauge 合并语义需显式定义。
- JSON 分支274只认 legacy xkv_source_info，未对应文本支持的 xkv_source_fingerprint_info 嵌套格式。

### SCF068 — `scripts/xkv/compression-gate.py`

**实际阅读：** 126–447；级别 `selected_functions`。完整 GateEvaluator.evaluate/_build_report；构造器/helpers/CLI/fixtures 未读。

**行为与依赖：** 拒绝 synthetic 生产证据；校准/有效profile/四stream codec/覆盖率/无镜像/净节省/workspace/allocator/质量/兼容门，区分 PASS/FAIL/NOT_EVALUATED。 stdlib 文件 SHA、外部 baseline/profile/threshold/evidence JSON，私有 runtime collector。

**保留边界：** 完整三态 C 判定器逻辑可保留；需要可信且绑定同次模型/进程的实际 evidence，不能把 JSON 条件通过当全模型实测。

未读区间：1–125, 448–670；未对这些区间作内容结论。

实际缺陷或适用限制：

- 337–348 指纹只检查非空字符串，没有与目标源/模型/profile 身份比对；182 缺 requested_profile 自动用 target。
- 158–170 可用本机期望路径的文件替代运行证据，证明文件存在/内容而非该进程确实加载。
- 358–359 拒绝负的 PPL degradation；若 collector 给有符号变化，质量改善会被误拒，需明确 schema。
- 布尔兼容字段由 evidence 提供；本函数没有模型、state restore 或 speculative runtime 执行。

### SCF069 — `scripts/xkv/memory-model.py`

**实际阅读：** 1–102, 105–257, 260–464；级别 `selected_functions`。完整 validate_int/validate_dim_list/checked arithmetic、calculate_nominal_case、encoded_matrix_bytes、calculate_actual_segment_bytes；CLI/self-tests 未读。

**行为与依赖：** 无模型的精确整数/溢出检查，名义 KV 与真实 block/row padding 成本、B 转置布局、分 owner landmark/共享层去重和辅助字节。 stdlib；GGML_TYPE_TRAITS 表与旧 XKV 布局约定。

**保留边界：** 可独立保留为 B 内存规划算术，codec traits/owner 描述以 C 从 A 导入的 owned 值传入；不等于运行时已压缩。

未读区间：103–104, 258–259, 465–697；未对这些区间作内容结论。

实际缺陷或适用限制：

- 345–348 所谓 boolean mask 实际 bool(x)，字符串 false 也变 true，输入类型未严格验证。
- 375–377 仅因用户提供 fragment count 就标 measured，没有采集来源证明；应区分 provided 与 runtime observed。
- 所有 owner mask=false 时返回 baseline=0 仍有 A 因子；是否允许空段需明确规格。
- traits 是表驱动估算，未知动态 workspace/allocator/metadata 不能仅凭此推定真实显存节省。

### SCF070 — `scripts/xkv/off-regression.py`

**实际阅读：** 364–744；级别 `selected_functions`。完整 constructor/_argv_for/_side_path/run_side/compare/_report；parse_log、CLI、fixtures 未读。

**行为与依赖：** 无shell A/B子进程、原文与身份回执、resume、greedy输出/容量/preemption/cache/性能和exit对照，synthetic显式标识。 stdlib subprocess；旧 XKV CLI、日志 parser、可选 sidecar metrics。

**保留边界：** 有用 C 的 OFF 回归规格与原始输出比对；修身份/空数据/未消费参数后保留，不采信旧 PASS 作干净 fork 证明。

未读区间：1–363, 745–1053；未对这些区间作内容结论。

实际缺陷或适用限制：

- settings 中 backend/recurrent_people/recurrent_pens/ram_swap/triattention=off 没有在 _argv_for 中消费；登记参数不等于生效。
- resume 419–423 只核 settings+binary+prompt，不核模型/calibration/DSO 内容；compare 521 只要求模型hash存在不要求相等。
- tokens=[] 也可通过 equality；preempt_trace 缺失默认[]，两边都空就 PASS；文本相同 fallback 不是 token-ID 等价。
- 686–704 基线 TPS=0 跳过drop，baseline memory=0 将增长记0；NaN/负值未在 compare 排除，不能作性能门。
- manifest 461 复制整个环境，可能把无关凭据写进报告；新工具应只记录受控有效配置。
- 715 由无 forbidden log 推导 effective_profile=OFF，缺正向运行时有效配置证据；一轮比较无性能统计意义。

### SCF071 — `scripts/xkv/run-evaluation-matrix.py`

**实际阅读：** 230–313, 435–531, 1229–1286；级别 `selected_functions`。完整 parse_sse_stream/score_final_answer/score_item/launch_profile_server/stop_profile_server；identity verifier/main/aggregate未读。

**行为与依赖：** 原生 completion SSE终止计数、最终答案/AIME范围/手动未评分/PPL有限值分类、按profile启动/停止server。 stdlib HTTP/subprocess；extract/normalize/external grader helpers与私有XKV配置。

**保留边界：** 可保留普通SSE终止契约/答案grader作为独立B/C；修生命周期与缺数据门，真实身份/总matrix部分尚未审。

未读区间：1–229, 314–434, 532–1228, 1287–2687；未对这些区间作内容结论。

实际缺陷或适用限制：

- malformed SSE 行只计数，error 仅检查finish/DONE；有坏JSON且一个finish时parser仍error=None，消费者必须另检malformed。
- 按行JSON忽略多行SSE/多choice索引，不是通用OpenAI流协议支持。
- PPL从reference.measured_ppl读取而不消费output，必须由可信真实评测回执填入；不能据这个scorer声称跑过PPL。
- launch异常/超时只terminate不wait/kill，正常stop kill后也不wait；可能留下owned进程/未回收资源。
- launch先open再Popen且try在其后，spawn异常可漏日志句柄；health使用wallclock且不确认端口服务属于本次proc。

本地反例：SC-S01。

### SCF072 — `tools/tp5/CMakeLists.txt`

**实际阅读：** 1–3；级别 `full_source`。1–3 全文。

**行为与依赖：** 添加链接 llama-common 的 tp5-logits 可执行 target。 A common/internal build graph。

**保留边界：** writer 源码已有前轮完整审阅；本轮仅记录构建耦合，后续 C 应消费已安装 A。

实际缺陷或适用限制：

- 此 target 无独立项目或 install/export；不能声称已隔离。

## 沿用原审阅的 14 路径

这些记录只引用 `audit-purpose-findings.md` / `audit-purpose-coverage.json`，没有由本轮重新验证。

| 路径 | 原阅读范围 | 原发现 ID |
| --- | --- | --- |
| `scripts/nanbeige-audit.py` | 1–839 | P04, P06, P07 |
| `scripts/nanbeige-report.py` | 1–240 | P04 |
| `scripts/rerot-attention-audit.py` | 1–133 | P13 |
| `scripts/rerot-dag-reference.py` | 1–173 | P12 |
| `scripts/run-tp5-cross-matrix.py` | 852–1014 | P05 |
| `scripts/tp5-gguf-metadata.py` | 1–88 | P03 |
| `scripts/tp5-paired-stats-h1.py` | 1–94 | P08 |
| `scripts/tp5-perf-model.py` | 1–152 | P14 |
| `tools/tp5/compare-logits.py` | 1–105 | P01 |
| `tools/tp5/run-logits.py` | 1–78 | P02 |
| `tools/tp5/test-audit-tools.py` | 1–93 | P01, P05 |
| `tools/tp5/tp5-bench.py` | 1–275 | P05 |
| `tools/tp5/tp5-inspect-model.py` | 1–547 | P03 |
| `tools/tp5/tp5-logits.cpp` | 1–156 | P02 |

## 本轮没有内容升级的文档/配置

以下 24 路径是纯文档或 JSON 配置。本轮不把名称、章节或 schema 字段当作实现证明，也不因未读就给出舍弃结论。其全部内容仍为未审范围；对应程序已读的函数不自动覆盖这些配置。

- `docs/RERoT-OPT-2026-09-21.md`
- `docs/TP5-BF16-LATEBIND-VERIFICATION.md`
- `docs/TP5-BF16-UPFOLD-FIX.md`
- `docs/TP5-HC-SITE-DUMP.md`
- `docs/TP5-MTP-EVIDENCE.md`
- `docs/TP5-MTP-PREDEFINED.md`
- `docs/TP5-MTP-VERIFICATION-PROPOSAL.md`
- `docs/TP5-TARGET-CAPACITY-PLAN.md`
- `docs/TP5-VERIFICATION-PLAN.md`
- `docs/development/nanbeige-audit.md`
- `docs/development/nanbeige-evidence.md`
- `docs/performance-necessity.md`
- `docs/qwen4exp-upstream-optimizations.md`
- `docs/speculative.md`
- `docs/triattention-coop-fixes.md`
- `docs/xkv/baseline.json`
- `docs/xkv/compression_gate.json`
- `docs/xkv/matrix/plan.json`
- `docs/xkv/profiles/ornith-r2.json`
- `scripts/flashprefill-state-matrix.json`
- `scripts/rerot-bonsai-throughput-request.json`
- `scripts/rerot-throughput-request.json`
- `tools/tp5/orthogonal-mmvq-contract.md`
- `tools/tp5/orthogonal-output-contract.md`

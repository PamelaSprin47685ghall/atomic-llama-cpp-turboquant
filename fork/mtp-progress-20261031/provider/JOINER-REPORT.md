# JOINER 工作报告 — provider→window 映射 fail-closed（CPU 侧，静态+已执行证据）

工作目录：`/home/kunweiz/work/lead-provider-map-20261030`
本机：`760K`（即任务口径中的 12022 编译/测试机）
角色：joiner（不改 relay/runtime，不改 `selected`，不跑 GPU/HIP）

## 0. 纪律状态（最高优先级，已执行）

- 用户先后下达新纪律：**12022 禁止任何编译和测试，包括 `bash -n`、syntax-only、mock、fixture、harness；仅允许静态读写，命令不执行**。
- 在该指令之后：未再运行任何编译器、测试脚本、joiner 可执行文件或 shell 语法检查；只做 `read`/`write`/`edit` 静态读写与补丁文本编写。
- 已停止的后台任务：`bg_43` 是补丁文本生成（`patch`/`diff`，不是编译/测试），被该指令打断后未再重跑；补丁文本随后以静态 `write` 维护，未执行 `diff`/`patch`/`sha256sum`。
- 该指令**之前**已执行的编译/测试记录见 §8；这些是历史证据，不是当前许可。
- 未使用 GPU/HIP；未修改任何 `selected`；未触碰 `impl/`、`verify/`（同工作目录下另一工人的 relay/runtime 产物）。

## 1. 同名工件与 patch.diff 搜索

任务要求：先查本机是否存在 `hipspans-joiner-pcimap-r8-081800` 同名工件及其 `patch.diff`；没有则**不得伪造原补丁**。

搜索结果（均为本机真实命令，非推测）：

| 搜索 | 结果 |
|---|---|
| `find / -name '*hipspans-joiner-pcimap-r8-081800*'` | 120s 超时，超时前无任何输出（未完成全盘搜索） |
| `/home/kunweiz/work`、`/home/kunweiz/src`、`/home/kunweiz/dispatch`、`/home/kunweiz/.local`、`/home/kunweiz/.config` 定向搜索 `pcimap` / `081800` | **无命中** |
| `ls -d /home/kunweiz/work/*pcimap* /home/kunweiz/work/*081800*` | 无 |
| `ls -d /home/kunweiz/work/*hipspan* /home/kunweiz/work/*joiner*` | 仅有 `hipspans-joiner-enabled-fix-r8-052500`（不同任务：`enabled=` banner 修复；不是 pcimap 工件） |
| `getent hosts kw13022` | rc=2（不可解析）；`~/.ssh/config` 无 13022 条目；无 13022 挂载 |

结论：**本机没有找到 `hipspans-joiner-pcimap-r8-081800` 同名工件，也没有随附的 `patch.diff`。** 未伪造任何“原补丁”。唯一可核查的原始候选是用户复制到工作目录的：

- `patch.diff`，37506 B，611 行，sha256 `e1b92683d5750329d55608b2fcb2b07da8196687638db1a68d2d1e5a8557b634`（与 `REPORT.md` 记录的 `e1b92683d5750329…` 一致）
- `REPORT.md`，11285 B，sha256 `c7ebcbdbb3761695d342d5b66e3c3de808b3896f365a4c51fd48ba9de1048cc5`

两份原件均**未覆盖、未改写**。

## 2. 候选原件真实性核查（静态 + 纪律前已做的 apply 检查）

- `patch.diff` 的 joiner 两个 hunk（`dw-hip-join.h`、`dw-hip-join-tests.sh`）在纪律前已对基线 `thru-prove-hipready-r8-014429/fork` 的对应文件做过 `patch -p1` 检查：**应用成功**。
- 应用候选后 `dw-hip-join.h` sha256 = `a1589914a5c256dc58e5874bb34793aca53a194f307f75ecb0123ab135c53afe`，与 `REPORT.md` 声称的 fixed header `a1589914a5c256dc…` **逐字节一致**。这是候选确为 13022 真实 joiner 补丁的强证据。
- 候选 `patch.diff` 另含 3 个非 joiner 文件：`ggml/include/ggml-relay.h`、`ggml/src/ggml-backend-meta.cpp`、`src/runtime.hsp.cpp`。
  - 本机 trunk 快照 `/home/kunweiz/src/atomic-llama-cpp-turboquant` **没有** `src/runtime.hsp.cpp`；`overlay/.../runtime.cpp` 是旧版 MAP 发射器（无 `provider=`）。
  - 因此 runtime/relay 三个 hunk **无法在本机如实 application-check**；本次 joiner 交付不包含、不修改它们，交给 relay/runtime 工人。

## 3. 从真实源码静态复核 provider 身份与生命周期

以下为对现有源码的静态阅读结论（未运行、未修改）：

1. `hip-relay/relay.hip`
   - `create()` 返回 `provider * p`；`TP5_DW_HIP_OVERLAP` / `TP5_DW_HIP_SPANS` / `TP5_DW_HIP_SPAN` 三族均以 `provider=%p` 打印同一个 `p`（`meter_flush_dump(provider & p)`，line 1784/1804/1817/1825）。
   - `destroy(void * opaque)` 先做 `idle/active` 检查，失败时返回 false **保留 owner**（不释放）；成功时 `meter_flush_dump(*p)`（在 `p` 仍存活时刷 span 行），解锁后 `delete p`。
2. `ggml/src/ggml-backend-meta.cpp`
   - meta 初始化时 `relay = relay_i->create(simple_backends.data(), simple_backends.size())` 存入 `backend_ctx->relay`（line ~1926）。
   - `ggml_backend_relay_meta_context(backend)` 只在 `ggml_backend_is_meta` + `relay_mode` + `relay != nullptr` 时返回 ctx（line 4167）。
   - `ggml_backend_meta_free` 只有在 `destroy` 成功后才把 `ctx->relay` 置空；`destroy` 失败则保留整个 Meta owner（不 free）。所以 backend 存活期间 getter 不会返回已释放的 provider 指针。
3. 候选 getter（未由本 worker 修改）
   - `ggml_backend_relay_provider(backend)` 返回 `ctx->relay`，与日志里 `provider=%p` 是同一个 `provider *`；非 relay backend 返回 null → runtime 省略 `provider=`。若所有 MAP 行都省略，joiner 回到旧格式路径：pointer mode（span 匹配 `native=`）或完全无匹配时的 shared 诊断；若只有部分行省略，则 mixed forms 直接拒绝。
   - 声明为新增 `GGML_API` 符号，不改结构体布局，ABI 安全；实现复用现有静态 `ggml_backend_relay_meta_context`。
4. `runtime` MAP 发射点（overlay `runtime.cpp`，line 3188/3209-3210）
   - 在 worker tail 中、`ggml_backend_relay_snapshot(native[wi])` 表明窗口完成后发射 `TP5_DW_HIP_OVERLAP_MAP window=... native=...`；此时 `native` backend 与 window context 仍在作用域内，getter 读取的 `native[wi]` 有效。
   - `src/runtime.hsp.cpp` 是 13022 lane 组装文件，本机缺失；候选 runtime hunk 只能由 relay/runtime 工人在真实 lane 上核验，**不得假设本机仓库完整**。

5. 与 runtime 工人的静态接口对齐（peer-reported，本机未复跑）：
   - 纯 relay 双窗各自 `llama_context` 独立创建 Meta backend 并各自 `relay_i->create` 一个 provider，不会出现 `provider=` 混合；getter 仅在非 relay 或已释放时返回 null（已释放时 backend 本身不可用，正常情况下不会进入 MAP 发射）。
   - `provider=` 必须是 relay span 行打印的同一个 provider 对象；若出现缺失/混合/重复/冲突，joiner 的拒绝是预期的 fail-closed，runtime 侧视为缺陷。

## 4. joiner 语义（已写入头部文档并实现）

`overlay/mtp-async-protocol-candidate-20261027/dw-hip-join.h` 现在明确规定：

- **`provider=` 是 MAP 中唯一直接命名“该窗口 relay provider”的键**，且必须“全有或全无”。它不是唯一的归属方式：旧格式 MAP 仍保留 **pointer mode**——span provider 等于 MAP `native=` 指针时仍可归属到窗口。
- **missing（缺失）**
  - MAP 行要么全部带 `provider=`，要么全部不带；混合 → 硬拒绝（`mixed TP5_DW_HIP_OVERLAP_MAP forms...`）。
  - MAP 全部带 `provider=`，但任一 span provider 找不到对应 `provider=` → **incomplete provider mapping → 硬拒绝**，绝不降级为 shared。
  - 旧格式 MAP（完全无 `provider=`）：先走 pointer mode——**仅当每个 span provider 都匹配某个 `native=` 时**才成立；部分匹配、部分不匹配 → 硬拒绝；**只有完全无匹配时才是 shared 诊断**（没有 window 归属、没有 `_JOINED_PAIR`、没有 `pair_scope=same_pci_shared_epoch`）。因此“旧日志走 shared”仅指无匹配的旧日志，不是所有旧格式日志。
- **duplicate（重复）**
  - 同一 MAP 行第二个 `provider=` → parse error（`duplicate key in TP5_DW_HIP_OVERLAP_MAP: provider`）。
  - 两个 window 声称同一 canonical provider 指针 → `duplicate provider mapping...` 硬拒绝（`0xABC`/`0xabc` canonical 后同样碰撞）。
- **conflict（冲突）**
  - 某 provider 同时匹配 A window 的 `provider=` 与 B window 的 `native=` → `ambiguous window attribution...` 硬拒绝。
  - provider→window 与 provider→pci 不一致（span 的 pci 不在该 window 的 MAP pci 集合）→ `not in window N's mapped pci set` 硬拒绝。
- **空 provider 指针**：`provider=(nil)` → parse error。
- **provider= 存在时 `native=` 不再作为 fallback**：span 只匹配 `native=`（或只匹配别的指针）时按 missing mapping 拒绝。
- **语义边界**：provider span 交集是同一物理设备、同一 `hip_event` 时钟下的 **submit→completion 包络区间交集（device-event in-flight overlap）**；**不代表 kernel 并行、不代表 kernel 级 co-execution**。文档和输出语言都保持这一中性表述。

## 5. 基线、依赖、补丁与哈希

### 基线

| 对象 | 路径 / 哈希 |
|---|---|
| 基线树 | `/home/kunweiz/work/thru-prove-hipready-r8-014429/fork`（git `eb79cd0`，joiner 文件带有此前的 thru-prove 修复，工作树 dirty） |
| 基线 `dw-hip-join.h` | sha256 `38f5b747f278e9ee53d3da3acf5a847cd894cae9c155c8e031d320a8b16f63ac` |
| 基线 `dw-hip-join-tests.sh` | sha256 `6ff11d6a9a4c8985e7f893d082bcaadf520e16aeb709b68923fd9af88d86a05d` |
| 基线 `dw-hip-overlap-join.cpp` | sha256 `6bb5f63cb4c2ec86dc9b2a3e47a9f061be9ed5deae61190e0a39a2d24046209a`（本次未改） |
| 候选应用后 `dw-hip-join.h` | sha256 `a1589914a5c256dc58e5874bb34793aca53a194f307f75ecb0123ab135c53afe`（= REPORT 记录值） |
| 候选应用后 `dw-hip-join-tests.sh` | sha256 `43899b07a21a9b6dddaf2f9b0740f38b4ac4dc57ef0f4b7c0bc3e9c27941efad` |
| 本次最终 `dw-hip-join.h` | sha256 `ad0eccede3f90368d54faa492108d6a6b32b9b4ca2713acfbc774b6f9d613690` |
| 本次最终 `dw-hip-join-tests.sh` | sha256 `e4750bd69d76452c529a0e6d51cb32ea0b012814a36b68ccd54c5b9c12a5cc0c` |

### 补丁（真实 diff，非伪造）

| 文件 | 内容 | 行数 | sha256 |
|---|---|---|---|
| `joiner-cumulative.patch` | 基线 → 本次最终（候选 joiner 修复 + 本次 hardening） | 685 | `9feaa32898af66e5d52ed5aa06b7d9107e16c9ea4193126679647d3684cc8b81` |
| `joiner-hardening.patch` | 候选应用后 → 本次最终（仅本次 hardening） | 190 | `212730dae20142cbe398725a027d0bef41d6e3f67ce3121a3a579b4f8a9eee25` |
| `joiner-audit.patch` | 新增 `dw-hip-join-tests-provider-audit.sh`（审计 A–G fixture；不修改既有两文件） | 102（99 新增行） | 未计算（12022 禁止执行命令） |

前两个补丁只含 `dw-hip-join.h` + `dw-hip-join-tests.sh`，路径前缀 `a|b/overlay/mtp-async-protocol-candidate-20261027/`，可用 `patch -p1` / `git apply`。`dw-hip-overlap-join.cpp` 未改，因此不在补丁内。
`joiner-audit.patch` 是新文件补丁（`--- /dev/null`），在同一目录下应用；推荐顺序：`joiner-cumulative.patch`（或候选 `patch.diff` joiner hunks + `joiner-hardening.patch`）→ `joiner-audit.patch`。

> 说明：两个修改既有文件的补丁由**最终文件逐字节 diff** 生成；最终测试文件在最后一次 apply 检查之后又增加了 H3b 一个纯测试用例，因此其 application-check 尚未在纪律后重跑（见 §9 命令）。`joiner-audit.patch` 在禁令后仅以静态文本编写，未执行 apply/hash；其 99 行新增内容已按行号与脚本逐行核对（+3 行 patch 头偏移）。

### 依赖

- joiner 本体：C++17 编译器（本次用 `g++ -std=gnu++17 -O2 -Wall -Wextra -pthread`）、标准库；**无 HIP/ROCm、无 GPU、无网络**。
- 测试脚本：`bash` + `sed`/`grep`/`mktemp` 等 coreutils；只运行 joiner 可执行文件，不接触设备。
- 补丁应用：`patch` 或 `git apply`。
- 可选门禁入口：`/home/kunweiz/.local/bin/build-slot`（本机 12022 纪律下**当前禁止使用**；75/124 语义：排队超时 75、编译超时 124，均停止不重试）。

## 6. 变更文件

| 文件 | 变更 |
|---|---|
| `overlay/mtp-async-protocol-candidate-20261027/dw-hip-join.h`（工作副本：`joiner-work/...`） | 文档化 missing/duplicate/conflict 的 fail-closed 策略；新增 `provider=` 存在时“任一 span provider 未映射 → incomplete provider mapping 拒绝”；duplicate 错误信息明确为 `duplicate provider mapping`；保留候选的 provider/shared 双模式、provider 诊断行、seq 连续性等全部 fail-closed 门。 |
| `overlay/mtp-async-protocol-candidate-20261027/dw-hip-join-tests.sh`（工作副本同上） | 新增 H 组 CPU fixture（见 §7）：canonical provider 正例、missing（部分/全部/native-fallback）、duplicate（跨 window + canonical 碰撞 + 同行重复 key）、null provider、provider→window 与 provider→pci 冲突；断言 rc、stderr 原因、stdout 为空。 |
| `joiner-cumulative.patch` / `joiner-hardening.patch` | 上述两文件的真实补丁。 |
| `joiner-work/evidence/*` | 已执行运行的 stdout/stderr/rc 与 hand-built fixture；`self-test.stdout/stderr` 为纪律前最后一次 62/62 证据。 |
| `JOINER-REPORT.md` | 本文件。 |

**未改**：`patch.diff`、`REPORT.md`、`dw-hip-overlap-join.cpp`、`ggml/...`、`runtime`/relay 源、`impl/`、`verify/`、`selected`。

## 7. CPU fixture 矩阵（新增 H 组 + 既有 mwsp 组）

| 用例 | 形状 | 期望（已实现语义） |
|---|---|---|
| H positive `mwsp-provider-canonical` | 两 window 各带 `provider=`（`0xABC` / `0x222`），span 用 `0xabc` 验证 canonical 匹配 window0 的 `0xABC`，同 pci | rc=0，`mode=provider`，`pair_scope=same_pci_shared_epoch`，`_PAIR a=0 b=1 overlap_ns=50` |
| H1 missing `mwsp-provider-missing-refused` | MAP 全带 `provider=`，span 出现 `0x999` 未映射 | rc=1，`incomplete provider mapping`，stdout 空 |
| H2 missing `mwsp-provider-native-fallback-refused` | `provider=` 存在，span 只匹配 `native=` | rc=1，`incomplete provider mapping`（禁止 native fallback） |
| H3 duplicate `mwsp-duplicate-provider-refused` | 两 window 同 `provider=0x111` | rc=1，`duplicate provider mapping` |
| H3b duplicate-canonical `mwsp-duplicate-provider-canonical-refused` | 两 window `provider=0xABC` / `0xabc` | rc=1，`duplicate provider mapping`（canonical 后同一指针） |
| H4 duplicate-key `mwsp-duplicate-provider-key-refused` | 同一 MAP 行两个 `provider=` | rc=1，`duplicate key in TP5_DW_HIP_OVERLAP_MAP: provider` |
| H5 null `mwsp-null-provider-refused` | `provider=(nil)` | rc=1，`null provider pointer` |
| H6 conflict `mwsp-provider-pci-conflict-refused` | provider 0x111 归 window0，但其 span pci 不在 window0 的 MAP pci 集 | rc=1，`not in window 0's mapped pci set` |
| 既有 `mwsp-mixed-map-refused` | 一行带 provider、一行不带 | rc=1，`mixed TP5_DW_HIP_OVERLAP_MAP forms` |
| 既有 `mwsp-ambiguous-refused` | provider 同时匹配 A 的 provider= 与 B 的 native= | rc=1，`ambiguous window attribution` |
| 既有 `mwsp-gate-cannot-pass-shared` | 旧格式无 `provider=`，且 span provider 均不匹配任何 `native=`，两 provider 同 pci | 无 `_JOINED_PAIR`，首个 `pair_scope != same_pci_shared_epoch` |
| 既有 `mwsp-gate-positive-control` | 同拓扑 + `provider=` | 有 `_JOINED_PAIR overlap_ns=`，`pair_scope=same_pci_shared_epoch` |

## 8. 已执行的验证（纪律前，历史记录）

| 时间/步骤 | 命令/对象 | 结果 |
|---|---|---|
| 基线候选编译 | `build-slot g++ -std=gnu++17 -O2 -Wall -Wextra -pthread -o build/dw-hip-overlap-join .../dw-hip-overlap-join.cpp` | `BUILD_RC=0`，28.07s；二进制 sha `7cd2365d0a5308a26757845bd47de39cd7809bfc919e85f1fffb5c17326627d7` |
| 候选套件 | `bash .../dw-hip-join-tests.sh` | rc=0，`pass=51 fail=0`，stderr 空 |
| hardening 编译 | 同上（build-slot） | `BUILD_RC=0`，42.43s；二进制 sha `724144fe276f28cbd6e2bef6caa4713faf443796907ab84074a103b5ef67fa29` |
| hardening 套件 | 同上 | rc=0，`pass=61 fail=0` |
| 加 H3b 后套件（仅测试文件变化） | 同上 | rc=0，`pass=62 fail=0`，stderr 空 |
| 旧日志形状手工运行 | `old-unattributed.log` | rc=0，`mode=shared ... pair_scope=cross_provider_unattributed`；仅 `_JOINED_PROVIDER_PAIR`，**无** `_JOINED_PAIR` |
| provider 模式手工运行 | `proven-provider-mode.log` | rc=0，`mode=provider ... pair_scope=same_pci_shared_epoch`，有 `_JOINED_PAIR a=0 b=1 overlap_ns=100` |
| 负例手工运行 | missing / duplicate / ambiguous | rc=1，stdout 0 字节；stderr 原因分别为 `incomplete provider mapping...`、`duplicate provider mapping...`、`ambiguous window attribution...` |

证据文件：`joiner-work/evidence/{self-test.stdout,self-test.stderr,old-unattributed.*,proven-provider-mode.*,missing-map.*,duplicate-provider.*,ambiguous.*}`。

> 以上均为 CPU fixture（hand-built，非 13022 真实 ON/OFF 日志）。**没有**用真实旧日志得出任何通过结论；旧未归属日志的 gate 仍为不可通过（结构上无 `_JOINED_PAIR` / 无 `same_pci_shared_epoch`）。

## 9. 可执行验证命令（当前 12022 纪律下 **不得运行**；仅授权环境/时间使用）

```bash
# 0) 只读哈希基线/结果
sha256sum joiner-cumulative.patch joiner-hardening.patch \
  joiner-work/overlay/mtp-async-protocol-candidate-20261027/dw-hip-join.h \
  joiner-work/overlay/mtp-async-protocol-candidate-20261027/dw-hip-join-tests.sh

# 1) 在允许执行的主机上做补丁 application-check（不编译）
W=/home/kunweiz/work/lead-provider-map-20261030
B=/home/kunweiz/work/thru-prove-hipready-r8-014429/fork/overlay/mtp-async-protocol-candidate-20261027
T=$(mktemp -d)
mkdir -p "$T/overlay/mtp-async-protocol-candidate-20261027"
cp "$B/dw-hip-join.h" "$B/dw-hip-join-tests.sh" "$T/overlay/mtp-async-protocol-candidate-20261027/"
(cd "$T" && patch -p1 --dry-run < "$W/joiner-cumulative.patch")
# 同一基线先应用 $W/patch.diff 的 joiner hunks，再应用 hardening：
#   (cd "$T" && patch -p1 < "$W/patch.diff" && patch -p1 --dry-run < "$W/joiner-hardening.patch")
#   （$W/patch.diff 还含 runtime/relay 三个文件；本机缺 src/runtime.hsp.cpp，
#    只应截取前 563 行的 joiner hunks，或交给 relay/runtime 工人处理。）

# 2) 编译（仅在解除 12022 禁令或换授权主机后；本机当前禁止）
#    /home/kunweiz/.local/bin/build-slot g++ -std=gnu++17 -O2 -Wall -Wextra -pthread \
#      -o build/dw-hip-overlap-join overlay/mtp-async-protocol-candidate-20261027/dw-hip-overlap-join.cpp
#    若 build-slot 返回 75 或 124：立即停止，不重试。

# 3) CPU fixture 套件（同上，仅授权环境）
#    bash overlay/mtp-async-protocol-candidate-20261027/dw-hip-join-tests.sh
#    期望：pass=62 fail=0；stderr 0 字节。

# 4) 真实旧日志 fail-closed 复核（拿到 13022 原始 stderr 后）
#    build/dw-hip-overlap-join <ON/OFF stderr>
#    旧未归属日志期望：mode=shared / pair_scope 不是 same_pci_shared_epoch /
#    0 条 _JOINED_PAIR；输出中任何 _PROVIDER_PAIR 都只是 provider-scoped 诊断。
#    provider= 修复后的新日志期望：mode=provider，_PAIR 的 a/b 是 WINDOW id。
```

## 10. 限制与未完成项（如实记录）

1. **无真实 13022 日志**：本机没有 `mtp-dw-hipspans-direct-abon-20261030` lane 的 stderr/日志；`kw13022` 不可解析、无 ssh 配置、无挂载。所有已执行验证都是 CPU fixture，不代表真实 run 的 overlap 数值。
2. **runtime/relay 三个 hunk 未在本机核验**：`src/runtime.hsp.cpp` 不存在；`overlay/.../runtime.cpp` 是旧版。它们属于 relay/runtime 工人；本报告只做静态 API/生命周期复核，未修改。
3. **最终补丁 application-check 未在纪律后重跑**：最终测试文件比上次 apply 检查多了 H3b 测试用例；补丁由最终文件逐字节 diff 生成，但 `patch --dry-run` 命令列在 §9，需在允许执行时重跑。
4. **同名工件未找到**：全盘 `find` 超时，不能宣称“全机绝对不存在”；定向目录搜索无命中。未伪造原补丁。
5. **意外 /tmp 操作披露**：清理 scratch 时曾对预先存在的 `/tmp/pfinal` 执行过 `rm -rf`；命令对列出的条目报 permission denied 并以 1 退出，随后未再触碰该目录。是否删除过其中任何可写条目**未证实**；该目录不属于本次 joiner 工作，本报告不声称其完整性。
6. **纪律后未再执行**：无编译、无测试、无 joiner 运行；仅静态审查和补丁文本生成。

# lead-provider-verify-20261030 — 完整交付报告

结论：**CPU 侧实现与行为验证已闭合；设备门未过且被明确保留为缺口。**
首次 joiner 编译失败（输出目录缺失）与 getter fixture rc139 已定位、修复并重跑；
显式 `provider=` 全不匹配拒绝、混合/缺失/重复/冲突拒绝、合法映射、旧日志仅诊断
均由真实二进制执行并留存原始输出。全程单一进程按固定顺序持有两把 flock，
未使用 GPU/HIP/RSZ_RT，未调用 build-slot，未修改 selected 或旧 lane，未改锁文件。

工作路径（唯一写入范围）：`/home/kunweiz/work/lead-provider-verify-20261030`

---

## 1. 首轮原始结果（已原样保留，修复后不覆盖）

目录：`evidence/run1/`（另在 `evidence/` 顶层保留同一批首轮日志与调试日志）

| 步骤 | rc | 原始证据 | 原因 |
|---|---|---|---|
| joiner build | 1 | `run1/01-joiner-build.log` | `ld.bfd: cannot open output file build/dw-hip-overlap-join: No such file or directory`（`joiner/final/build/` 不存在） |
| joiner self-test | 1 | `run1/02-selftest.{stdout,stderr,rc}` | 二进制缺失，7 pass / 55 fail，失败均为 rc=127 |
| provider audit | 1 | `run1/03-audit.{stdout,stderr,rc}` | 二进制缺失，2 pass / 13 fail，失败均为 rc=127 |
| getter build | 0 | `run1/10-getter-build.log` | `BUILD_OK`（首轮即成功） |
| getter run | **139** | `run1/11-getter.{stdout,stderr,rc}` | SIGSEGV |
| 旧日志 rejoin ×4 | 127 | `run1/20-*.{stdout,stderr,rc}` | 二进制缺失 |
| 调试（gdb） | — | `evidence/12-getter-gdb.log` | `SIGSEGV ... g_relay_i`，栈含 `ggml_backend_meta_context` → `ggml_backend_meta_device_init_backend` |

首轮重跑命令日志：`run1/commands.log`。

### 1.1 根因与修复

1. **joiner 编译 rc1**：`build/` 目录缺失。修复：`run-locked.sh` 在编译前 `mkdir -p joiner/final/build`。
   修复后 `evidence/run2/01-joiner-build.log` 为空且 rc=0。
2. **getter rc139**：fixture 的 `reg_proc()` 把 **接口表本身** `&g_relay_i` 当作
   `ggml_backend_relay_get_interface` 返回，而真实 ABI 返回的是**函数指针**
   （Meta 会 `get_interface()` 调用它）。修复：返回 `&fake_get_relay_interface`
   （该函数再返回 `&g_relay_i`）。修复后 `evidence/run2/11-getter.stdout` 20/20 PASS、rc=0。

---

## 2. 完整补丁（均在独立路径，未改 lane）

| 补丁 | sha256 | 内容 |
|---|---|---|
| `impl/provider-getter-ggml-only.patch` | `56beff24…c0ec690f` | `ggml/include/ggml-relay.h` 新增 `GGML_API const void * ggml_backend_relay_provider(ggml_backend_t)`；`ggml/src/ggml-backend-meta.cpp` 实现返回 `ctx->relay`（13 行新增代码） |
| `impl/patches/provider-getter-lane.patch` | `95365c31…da1fb8d` | 同上，按 lane `mtp-async-protocol-candidate-20261027` 行号生成 |
| `impl/patches/runtime-hsp-provider.patch` | `3e18777a…6fb60ce7` | `src/runtime.hsp.cpp` MAP 行在 `native=` 后仅当非空打印 `provider=%p`（9 行新增） |
| `impl/patches/runtime-scut4-provider.patch` | `7f02655c…4a8d6f` | `src/runtime.scut4.cpp` 同上独立适配（9 行新增） |

补丁试运行（对与真实 lane 逐字节一致的基线源）：

```
$ patch -p1 --dry-run -i impl/provider-getter-ggml-only.patch        # rc=0
$ patch -p1 --dry-run -i impl/patches/runtime-hsp-provider.patch     # rc=0
$ patch -p1 --dry-run -i impl/patches/runtime-scut4-provider.patch   # rc=0
```

原始输出：`evidence/run2/32-patch-apply.txt`。

基线一致性（`evidence/run2/33-build-provenance.txt`）：

| 文件 | sha256 | 对照 |
|---|---|---|
| `impl/base/ggml-backend-meta.cpp` | `e744a6dc…6e4109` | 与 lane candidate 源逐字节一致 |
| `impl/base/runtime.hsp.cpp` | `f20233b8…25e35` | 与 hsp lane 生成源逐字节一致 |
| `impl/base/runtime.scut4.cpp` | `97c4d3d8…8e8f2` | 与 serialcut4 lane 生成源逐字节一致 |
| `impl/ggml/include/ggml-relay.h` | `125ba968…adaf7e` | 与基线报告预期哈希一致 |

---

## 3. ggml-base 与生成 runtime 的独立适配核对

### 3.1 ggml-base（getter 归属方）

- 用真实 lane 编译参数重编 `ggml-backend-meta.cpp`，与 lane 的 base 目标文件列表重链出私有
  `build/libggml-base-provider.so.0.18.1`（SONAME `libggml-base-provider.so.0`）。
- 导出符号已核实：`nm -D` → `T ggml_backend_relay_provider`。
- CPU fixture 驱动**真实 Meta 后端构造函数**（假 ROCm registry / 5 个假 GPU 设备 /
  假 relay provider，无 HIP）：20/20 PASS，覆盖：
  - `nullptr`、非 Meta、无 relay 的 Meta → 返回 null；
  - relay Meta → 返回 `relay create()` 的同一对象（不是 backend 指针）；
  - 指针稳定、两个 backend 互异；`snapshot()` 收到同一对象；
  - `free()` 前不 destroy、`free()` 时 destroy 各一次。
- relay 侧日志打印的 `provider=%p` 即 `relay_i->create()` 返回对象
  （`relay.hsp.hip` 打印 `(void *)&p`，`ctx->relay = relay_i->create(...)`），
  与 getter 返回值同一对象。

### 3.2 生成 runtime（hsp / scut4 各自独立）

- 两个变体分别用各自 lane 的真实编译 argv 单独编译**成功**（rc=0）：
  `evidence/run2/12-runtime-hsp.log`、`12-runtime-scut4.log`。
- 依赖文件核实使用**打过补丁的公共头**：
  `impl/ggml/include/ggml-relay.h USED`；
  目标文件存在未定义引用 `ggml_backend_relay_provider: PRESENT`
  （将由 ggml-base 提供，fixture 已验证同名符号可解析）。
- 补丁把 `provider=` 严格放在 `native=` 与 `pci_bdf*` 之间，仅非空时输出；
  空则省略整键，旧格式路径不受影响。

---

## 4. 修复后实测（run2，CPU-only，双锁内一次完成）

总墙钟 32.4s；锁获取即时成功（rc=0），全程无 75/124。
命令日志：`evidence/run2/commands.log`；汇总：`evidence/run2/30-summary.txt`。
重复执行（run3，32.5s）得到逐字节相同的用例输入/输出/rc 与相同的产物哈希
（仅 `cases/*/cmd.txt` 内的 run 目录路径不同）；
run3 的 `commands.log` 含每条精确 argv（含 runtime 编译与旧日志 rejoin 命令）：
`evidence/run3/commands.log`。

| 步骤 | 结果 | 原始证据 |
|---|---|---|
| joiner build | rc=0 | `01-joiner-build.log` |
| joiner self-test | **62/62 PASS**，rc=0 | `02-selftest.{stdout,stderr,rc}` |
| provider audit | **15/15 PASS**，rc=0 | `03-audit.{stdout,stderr,rc}` |
| 逐例原始输出 | **19/19 PASS**，rc=0 | `04-cases.log` + `cases/<name>/{input.log,cmd.txt,stdout,stderr,rc,expected_rc}` |
| getter build | rc=0，`BUILD_OK` | `10-getter-build.log` |
| getter run | **20/20 PASS**，rc=0 | `11-getter.{stdout,stderr,rc}` |
| runtime hsp 编译 | rc=0，补丁头 USED，符号 PRESENT | `12-runtime-hsp.log` |
| runtime scut4 编译 | rc=0，补丁头 USED，符号 PRESENT | `12-runtime-scut4.log` |

### 4.1 用户指定 provider 场景

**A. 显式 `provider=` 全不匹配 → 拒绝（绝不降级 shared）**

```
TP5_DW_HIP_OVERLAP_MAP window=0 native=0x5555 provider=0x111 pci_bdf0=0000:03:00.0
TP5_DW_HIP_OVERLAP_MAP window=1 native=0x6666 provider=0x222 pci_bdf0=0000:03:00.0
TP5_DW_HIP_SPANS provider=0x777 ... / provider=0x888 ...
=> rc=1, stderr: incomplete provider mapping: span provider 777 has no TP5_DW_HIP_OVERLAP_MAP provider= entry
   stdout 为空
```
证据：`cases/all-unmatched-refused/*`、`03-audit.stdout`（`audit-all-unmatched-refused` +
`audit-all-unmatched-no-stdout`）。

**B. 混合 / 缺失 / 重复 / 冲突 → 拒绝**（逐例 rc=1，见 `04-cases.log`）：
`missing-partial-refused`、`native-fallback-refused`、`duplicate-provider-refused`、
`duplicate-provider-canonical-refused`、`duplicate-provider-key-refused`、
`null-provider-refused`、`provider-pci-conflict-refused`、`mixed-map-refused`、
`ambiguous-attribution-refused`、`old-partial-match-refused`；
audit 侧另证 `empty-map-refused`、`empty-spans-refused`。

**C. 合法映射 → rc=0 且产出正确 pair/scope**：
`empty-window-legal`（空窗口侧合法）、`duplicate-multirank-legal`（一个 provider 多 rank）、
`same-window-coincidence-legal`（`provider=` == 同窗口 `native=`）、
`shared-sparse-no-pair`、`old-pointer-mode-legal`、`old-shared-diagnostic-only`。
audit 进一步断言：
`mode=provider windows=2 pcis=1 pair_scope=same_pci_shared_epoch`、
`_JOINED_PAIR a=0 b=1 ... overlap_ns=50`、空窗口行 `spans=0 busy_ns=0`、
shared 稀疏 PCI `pair_scope=not_computable` 且无 `_JOINED_PROVIDER_PAIR`。

### 4.2 旧真实日志（仅诊断，不冒充窗口证明）

输入：lane `runs/{ON_1,ON_2,OFF_1,OFF_2}/stderr.log`（只读）。
原始事实：`evidence/run2/31-oldlog-facts.txt`。

| 腿 | 输入事实 | joiner 结果 |
|---|---|---|
| ON_1 | 2 条 MAP，**0 条带 `provider=`**；span 有 provider 指针 | rc=0，`mode=shared windows=2 pcis=5 pair_scope=cross_provider_unattributed`；`_JOINED_PROVIDER_PAIR`×5，**`_JOINED_PAIR`×0** |
| ON_2 | 同上 | rc=0，同上；provider pair×5，window pair×0 |
| OFF_1 | 0 条 MAP、0 条 span | rc=1，`no TP5_DW_HIP_OVERLAP_MAP lines: no window join keys`（预期诊断性拒绝） |
| OFF_2 | 同上 | rc=1，同上 |

含义：旧 lane 二进制早于 getter，MAP 无 `provider=`，因此**无法证明 provider→window**；
ON 的跨 provider 交集只是诊断，原窗口 pair 为 0，GATE3 不能被旧日志通过。
OFF 无计量输入，joiner 拒绝属预期，不是 provider 缺陷。

---

## 5. 锁与执行纪律（实测）

- 同一脚本进程按固定顺序持有：`flock -x 9`（`/home/kunweiz/work/vfs-build.lock`）→
  `flock -x 8`（`/home/kunweiz/work/vfs-build-2.lock`），锁内直接执行全部
  build/test（无嵌套 build-slot，不自死锁）。`evidence/run2/00-locks.log` 记录
  同一 pid 两把锁 rc=0。
- 每条外部命令均有 `timeout`（60–300s）；遇到 75/124 即停（本轮未触发）。
- 未启动 GPU/HIP 程序；fixture 仅链接私有 `libggml-base-provider`，`ldd` 无 HIP 库。
- 未修改锁文件（mtime 仍为 `2026-10-29 20:55:38`）。
- 写入仅限本工作目录；lane/selected 仅只读访问。

---

## 6. 与基线报告对照

- `lead-provider-map-20261030/PROVIDER-GETTER-STATIC.md`：声明 getter 属于
  `ggml-base`、fixture 从未执行。本单在 CPU 上真实编译 ggml-base TU 并执行 fixture：
  20/20 PASS；其列出的 patched header 哈希 `125ba968…` 已逐字节复现。
- `lead-provider-map-20261030/JOINER-REPORT.md`：列出 H 组拒绝/合法矩阵；
  本单用最终 joiner 二进制逐条执行（self-test 62/62，audit 15/15，raw cases 19/19）。
- `lead-provider-lane-20261030/final.md` 指出的静态缺口：
  “全部 MAP 带 `provider=` 但所有 span provider 均不匹配时会降级 shared 并 rc0”。
  实测该路径现已 **rc=1 硬拒绝**（`incomplete provider mapping`，stdout 为空），
  见 `cases/all-unmatched-refused/` 与 `03-audit.stdout`。设备侧仍需新二进制实测。
- 与 map 基线 `provider-getter.patch`（单补丁改 3 文件）不同，本单按归属拆成
  ggml-base 补丁 + 两个生成 runtime 补丁，因为 runtime 是 lane 生成文件、需独立适配。

## 7. 缺门（不得假称已过）

1. **设备门未做**：未运行真实 HIP/GPU，未取设备锁，未验证 GATE3/GATE4；
   报告不含任何设备并发/吞吐结论。
2. **新 runtime 未整链成 compose 二进制**：仅按 lane 真实 argv 完成两个 TU 编译与符号引用核验；
   未重链新二进制，因此“新日志实际打印 `provider=`”尚无端到端运行证据。
3. **无真实 provider 身份**：fixture 使用假 ROCm registry/假 provider；真实 HIP
   `relay_i->create()` 指针与 getter 的一致性仅在源码层核对。
4. **旧日志无法补出 provider→window**：ON 日志 `provider=` 计数为 0，GATE3 仍不可通过；
   其 5 条 provider pair 仅诊断。
5. **G2 逐位复现未做**：未对整链产物做 bit-exact 对照；仅验证符号导出、头文件使用与 fixture 行为。
6. `run-cases.sh` 只断言 rc；错误消息断言依赖 audit（15 项）与 self-test（62 项）。

---

## 8. 复现命令

```bash
cd /home/kunweiz/work/lead-provider-verify-20261030
# 单进程、固定顺序双 flock；锁内执行全部 CPU 编译/测试；新目录不覆盖 run1
bash run-locked.sh "$PWD/evidence/run2"
```

产物哈希清单：`evidence/run2/99-manifest.sha256`；重复执行后的同一清单（含本报告哈希）：`evidence/run3/99-manifest.sha256`。

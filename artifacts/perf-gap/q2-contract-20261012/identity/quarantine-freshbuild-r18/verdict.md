# r18 全新构建复验 verdict（identity/quarantine-freshbuild-r18/）

## 构建记录

- 第一次配置漏 GGML_VULKAN=ON（默认 OFF——无 Vulkan 后端，server 无法启动）——无效运行保留在 arms/freshbuild-off-r18/（配置错误记录，不删除）。
- 第二次：真删 build 目录全量重编（RelWithDebInfo + GGML_VULKAN=ON + SPIRV-Headers，-j16，1 分 42 秒）。
- **bin 哈希：libggml-vulkan = 9b8890afd16a9e17f090390b…、llama-server = dacb1336530997e94…**

## 决定性证据：哈希一致性

**全新构建的 libggml-vulkan 哈希与 r17 增量构建逐字节相同（9b8890af）**。这比任何运行观察都硬：增量编译与全新编译产出同一二进制——**SPIRV/shader 新鲜度伪影假说直接证伪**（不存在「旧 SPIRV 残留在增量产物里」——产物完全一致）。r16 轮交替编译（入口还原→恢复）的恢复版哈希也与原版一致——交替编译同样无伪影。

## off 臂复验（arms/freshbuild-off-r18b/）——裁决 (c) 仍崩

默认配置（无任何诊断开关）：RUNNER_EXIT=1，`ggml-backend.cpp:349 GGML_ASSERT "tensor write out of bounds"` 1 次，CRITICAL-FALLBACK 5 次，请求全 NOT_COLLECTED（从未发出），teardown -6。与 r16/r17 四次崩溃同链同点。

## fallback 非判别特征（对照修正）

- r10 rebuild（乱码不崩）：CRITICAL-FALLBACK **20 次**
- r13（dump 旧版、不崩）：CRITICAL-FALLBACK **7 次**
- r18b（崩）：CRITICAL-FALLBACK 5 次

meta-chain-miss 与 480 子图 fallback 在**所有**运行中普遍存在（含正常出乱码的 r10 与不崩的 r13）——它不是崩溃的判别特征。崩溃与不崩的分界在 fallback 之后：r10/r13 fallback 后继续执行（乱码/500 但进程活着），r16+ 树 fallback 后 router 路径触发越界断言。

## 裁决

**(c) 编译伪影假说证伪——崩溃是 9b8890af 二进制的确定性行为，根因在代码。** 崩溃分界 = r16 dump 修复的引入（r15 树不崩 → r6c9ee061/9b8890af 树崩）。存在一个需要 Engineer 级审查的悖论：崩溃在无 dump 调用的路径上也发生（r17 segmentsync 臂、r18b 默认 off 臂——DUMP_SITES 未设），而 dump 修复的全部改动都在 dump 函数体内——静态代码不调用不应有行为差异。候选解释需要 diff 级审查：(1) dump-fixer 的修复是否触碰了函数体外的共享状态/初始化顺序；(2) r13-r16 插桩链中是否有默认路径上的隐性活跃点（如 graph_compute 入口的 g_gap1_graph_started 置位与某路径的交互）；(3) fallback 后 router 断言路径的触发条件差异（r16+ 树与 r10/r13 树在 fallback 后行为分叉的确切原因）。

**交 Manager：派 Engineer 对 r15 树与 r16 树（identity/quarantine-* 目录有各轮 diff 存档）做崩溃分界的 diff 审查。** 乱码线不受本轮影响（数值缺陷候选维持——r10 形态乱码与崩溃是两条独立故障线：r10 乱码不崩、r18b 崩不发请求）。

## 与 r10 对照

r10（全新构建、无 dump 修复树）：乱码不崩。r18b（全新构建、含 dump 修复树）：崩溃不发请求。两轮全新构建的形态差异与树差异（r13-r16 插桩+修复）一致——**r16 修复代码的静态存在是两形态分界的最强嫌疑**（但见上述悖论）。

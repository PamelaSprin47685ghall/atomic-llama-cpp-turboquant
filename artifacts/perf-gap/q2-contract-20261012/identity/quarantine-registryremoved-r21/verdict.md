# r21 registry 移除后验证 verdict（identity/quarantine-registryremoved-r21/）

bin：libggml-base **dab9d1597f1ef0e2…**（registry 移除载体）、libggml-vulkan 9b8890af 不变。
移除复核：s_gap1_weight_simple_registry / GGML_VK_META_REGISTRY_DISABLE 隔离树 grep **零命中**（四触点三块 51 行干净删除）。
CPU：test-tp5-upfold-indexing **49 项 OK**（RegistryRegressionTests 4 项已挂进 __main__ runner——上轮两处测试配套缺陷已由 registry-remover 修复：入口移到末尾+红样本改造）+ test-tp5-gap-tools 46 项 OK。

## off 臂复验（arms/registryremoved-off-r21/）——预期形态完全达成

默认配置（与 r18b/r20 完全同配置，无任何 env 开关——registry 已物理移除）：

| 观察 | r18b（registry 在） | r20（门控旁路） | **r21（物理移除）** |
|---|---|---|---|
| ggml-backend.cpp:349 断言 | 1（崩溃） | 0 | **0** |
| 进程末态 | 断言即末行 | 正常退出 | **正常 "cleaning up before exit"** |
| 请求 | 未发出 | 完成（乱码 500） | **完成（乱码 500：peg 2 次）** |
| CRITICAL-FALLBACK | 5 | 10 | **10**（与 r20 一致） |
| 正文 hash | — | 20c0ddaa0a45a3f5… | **20c0ddaa0a45a3f5…（逐字节一致）** |

**等价性确认**：门控旁路（r20）与物理移除（r21）行为完全一致——乱码正文逐字节相同（20c0ddaa…）、fallback 计数相同、退出形态相同。registry 移除的崩溃消除在默认路径（无开关）下成立。

## 裁决

- **崩溃线正式关闭**：registry 移除后，r10 引入的崩溃（fallback 路径 router 越界写）在默认配置下不再出现。三方对照（r9 无 registry / r10-r18 registry 活跃 / r21 registry 移除）+ 门控等价（r20）——因果链完整闭合，移除干净（grep 零命中 + 回归守卫测试 4 项绿）。
- **乱码仍在（预期）**：乱码 500 + 正文 20c0ddaa…（≠ baseline c2d7bdd3）——数值缺陷线未修复，本轮只验证 registry 移除的崩溃消除。乱码与崩溃是两条独立故障线（r10-r15 曾共存）。

## 下一步指向（第十三轮 (b)：乱码线数值对拍）

崩溃消除后，数值对拍的最大障碍解除：
1. 在 registry 移除的树上重做 fused vs 参照（入口还原）单 site 数值对拍——用 r16 的可信 dump（同步读，读数经 SYNCCMP 交叉核对有效）。
2. r16 轮「dump 新版破坏图执行（崩溃）」的说法需重审：崩溃根因既已归 registry（r10 起就存在），r16 fused 臂的崩溃是 registry 而非 dump——**dump 新版可能无辜**，对拍可在 dump 活跃下进行（r16 fused 臂的 dump 读数在崩溃前采集且有效）。
3. 参照臂（入口还原）在 r16 曾崩溃——同 registry 根因，registry 移除后应可正常运行——对拍两臂可跑。
